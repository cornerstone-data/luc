import dataclasses
import functools
import hashlib
import inspect
import logging
import os
import threading
import typing

import pandas
import xarray

logger = logging.getLogger(__name__)

COMPUTE_LOCK = threading.RLock()


def path_exists(uri: str) -> bool:
    import fsspec  # type: ignore[import-untyped]

    fs, path = fsspec.url_to_fs(url=uri)
    return fs.exists(path)


def put_file(local_path: str, uri: str) -> None:
    import fsspec  # type: ignore[import-untyped]

    logger.info(f"Writing file {local_path=:s} to {uri=:s}")
    fs, path = fsspec.url_to_fs(url=uri)
    fs.makedirs(os.path.dirname(p=path), exist_ok=True)
    fs.put_file(local_path, path)


def to_gdal_path(uri: str) -> str:
    if uri.startswith("gs://"):
        return "/vsigs/" + uri.removeprefix("gs://")
    else:
        return uri


def join_uri(root: str, prefix: str) -> str:
    return f"{root.rstrip('/'):s}/{prefix:s}"


def write_dask_dataset_to_zarr(dset: xarray.Dataset, path_to_zarr: str) -> None:
    assert dset.chunks is not None, f"{dset=:} is not a chunked dask array"

    import dask.config
    import distributed
    import rasterio

    from jdluc.config import Config

    num_workers = Config.from_dot_env().number_of_dask_workers

    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("distributed.scheduler").setLevel(logging.CRITICAL)
    logging.getLogger("gcsfs").setLevel(logging.WARNING)

    with (
        COMPUTE_LOCK,
        rasterio.Env(
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.vrt",
            # 256 MiB
            CPL_VSIL_CURL_CACHE_SIZE=1 << 28,
            # 512 MiB
            GDAL_CACHEMAX=1 << 29,
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        ),
        dask.config.set(
            {
                "distributed.admin.large-graph-warning-threshold": "300MiB",
                "distributed.comm.timeouts.connect": "90s",
            }
        ),
        distributed.LocalCluster(
            processes=False,
            n_workers=1,
            threads_per_worker=num_workers,
        ) as cluster,
        distributed.Client(
            cluster,
            # NB: avoid race by not sharing across threads
            set_as_default=False,
        ) as client,
    ):
        logger.info(
            f"Saving to {path_to_zarr=:s} with {num_workers=:d} and {dset.chunksizes=:}"
        )
        logger.info(f"Dask dashboard at {client.dashboard_link:s}")
        logger.info("Building graph and writing zarr metadata+coords")
        to_write = dset.drop_vars("spatial_ref", errors="ignore")
        for variable in to_write.variables.values():
            variable.encoding.pop("chunks", None)
        delayed = to_write.to_zarr(
            compute=False, consolidated=False, group=None, store=path_to_zarr
        )
        logger.info("Submitting task graph to dask scheduler")
        future = client.compute(delayed, retries=5)
        distributed.progress(future, scheduler=client.scheduler.address)
        future.result()
        logger.info(f"Finished writing to {path_to_zarr=:s}")


def open_zarr_to_dask_dataset(path_to_zarr: str) -> xarray.Dataset:
    import rioxarray  # noqa

    path_to_zarr = path_to_zarr.rstrip("/")
    logger.info(f"Loading {path_to_zarr=:s}")
    dset = xarray.open_zarr(store=path_to_zarr, consolidated=False)
    assert dset.chunks is not None
    assert isinstance(dset, xarray.Dataset)
    assert dset.encoding["source"] == path_to_zarr
    return dset.rio.write_crs(4326)


P = typing.ParamSpec("P")
R = typing.TypeVar("R")


class CacherProtocol(typing.Protocol[R]):
    def __init__(self, *, hash_key: str, root: str) -> None: ...
    @property
    def uri(self) -> str: ...
    @property
    def exists_uri(self) -> str: ...
    def deserialize(self) -> R: ...
    def serialize(self, value: R) -> None: ...


@dataclasses.dataclass
class ParquetCacher:
    hash_key: str
    root: str

    @property
    def uri(self) -> str:
        return join_uri(root=self.root, prefix=self.hash_key + ".parquet")

    exists_uri = uri

    def deserialize(self) -> pandas.DataFrame:
        logger.info(f"Loading from {self.uri=:s}")
        return pandas.read_parquet(path=self.uri)

    def serialize(self, df: pandas.DataFrame) -> None:
        logger.info(f"Saving to {self.uri=:s}")
        df.to_parquet(path=self.uri)


@dataclasses.dataclass
class ZarrCacher:
    hash_key: str
    root: str

    @property
    def uri(self) -> str:
        return join_uri(root=self.root, prefix=self.hash_key + ".zarr")

    @property
    def exists_uri(self) -> str:
        return f"{self.uri:s}/zarr.json"

    def deserialize(self) -> xarray.Dataset:
        return open_zarr_to_dask_dataset(path_to_zarr=self.uri)

    def serialize(self, dset: xarray.Dataset) -> None:
        write_dask_dataset_to_zarr(dset=dset, path_to_zarr=self.uri)


class CacherDecoratorProtocol(typing.Protocol[R]):
    def __call__(self, func: typing.Callable[P, R]) -> typing.Callable[P, R]: ...


def get_cache_decorator(
    cacher_cls: type[ParquetCacher] | type[ZarrCacher],
    version: int,
    ignored_args: list[str] | None = None,
) -> CacherDecoratorProtocol[typing.Any]:
    def get_module_for_func(func: typing.Callable[..., object]) -> str:
        import git

        relative_path = os.path.relpath(
            path=inspect.getfile(func),
            start=git.Repo(search_parent_directories=True).working_dir,
        )
        return relative_path.removesuffix(".py").replace(os.sep, ".")

    def decorator(
        func: typing.Callable[P, R],
    ) -> typing.Callable[P, R]:
        if ignored_args is not None:
            assert set(inspect.signature(func).parameters).issuperset(ignored_args)

        from jdluc.config import Config

        @functools.cache
        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            ignored = set(ignored_args or ())
            hash_tokens = (
                get_module_for_func(func=func),
                func.__qualname__,
                version,
                *(
                    (name, value)
                    for name, value in bound.arguments.items()
                    if name not in ignored
                ),
            )
            data = "|".join(map(str, hash_tokens)).encode()
            hash_key = hashlib.sha1(data=data).hexdigest()[:12]
            config = Config.from_dot_env()
            cacher = typing.cast(
                CacherProtocol[R],
                cacher_cls(hash_key=hash_key, root=config.scratch_root),
            )
            if not path_exists(uri=cacher.exists_uri):
                cacher.serialize(func(*args, **kwargs))
            # Always return the deserialized cache so cache hits and misses
            # look identical and so downstream operations start with a fresh
            # compute graph
            return cacher.deserialize()

        return typing.cast(typing.Callable[P, R], inner)

    return decorator


def cache_to_parquet(
    version: int, ignored_args: list[str] | None = None
) -> CacherDecoratorProtocol[pandas.DataFrame]:
    return get_cache_decorator(
        cacher_cls=ParquetCacher, ignored_args=ignored_args, version=version
    )


def cache_to_zarr(
    version: int, ignored_args: list[str] | None = None
) -> CacherDecoratorProtocol[xarray.Dataset]:
    return get_cache_decorator(
        cacher_cls=ZarrCacher, ignored_args=ignored_args, version=version
    )
