"""Write a pipeline step's output under EXPORT_ROOT, a raster per tile or a table.

A utility, like `storage`: it knows nothing about the steps, and nothing calls it
automatically. A workflow opts in by passing its own lazy dataset to `write_tile`, which
writes the chosen variables of one tile to `{EXPORT_ROOT}/{name}/{tile_id}.{tif|zarr}`,
computing only those, as it writes:

- COG: every variable as float32 2-D bands, NaN no-data. A variable with dimensions beyond
  the pixel grid is flattened to one band per combination of their values, named
  `<variable>:<value>[:<value>...]`;
- zarr: the variables as they are, with their dtypes and extra dimensions. A rewrite
  replaces the zarr, and the `.complete` marker beside it says the write finished.

For example:

    export.write_tile(
        dset=emit.workflow(tile_id=tile_id),
        format=export.Format.COG,
        name="emit",
        tile_id=tile_id,
    )
    export.write_tile(
        dset=statistical.get_downscaled_luc_emissions(tile_id=tile_id),
        format=export.Format.ZARR,
        name="statistical-emissions",
        tile_id=tile_id,
    )

A tabular deliverable (see `trace.export_workflow`) is one parquet per set of countries,
written by `write_table` to `{EXPORT_ROOT}/{deliverable}/{name}.parquet`.

Either may be given a `version`, such as a run id or a release label, which puts the whole
export under `{EXPORT_ROOT}/{version}/` and stamps it in the metadata. The default, blank,
writes directly under EXPORT_ROOT, and a rerun writes over the last one.

Either way, provenance is stamped in the output's metadata. Unlike SCRATCH_ROOT, EXPORT_ROOT
is not a cache that the pipeline reads back: what lands there is a terminal output.
"""

import collections.abc
import enum
import itertools
import logging
import os
import tempfile

import numpy
import pandas
import xarray

from jdluc import config, geo, storage, utils

logger = logging.getLogger(__name__)

PIXEL_DIMS = ("y", "x")


class Format(enum.StrEnum):
    COG = "cog"
    ZARR = "zarr"


FORMAT_TO_EXTENSION = {Format.COG: ".tif", Format.ZARR: ".zarr"}


def get_metadata(
    product_name: str, source_name: str, version: str = ""
) -> dict[str, str]:
    """Provenance, as `datasets.base` stamps on ingested COGs, and the export's version."""
    return {
        "cornerstone-processing-time": utils.get_utc_timestamp(),
        "cornerstone-processing-version": utils.get_git_version(),
        "cornerstone-product-name": product_name,
        "cornerstone-remote-url": utils.get_git_remote_url(),
        "cornerstone-source-name": source_name,
    } | ({"cornerstone-export-version": version} if version else {})


def get_export_root(output_root: str | None = None, version: str = "") -> str:
    """`{output_root}/{version}`, or `output_root` itself for a blank version;
    `output_root` defaults to EXPORT_ROOT."""
    if output_root is None:
        output_root = config.Config.from_dot_env().export_root
    if not version:
        return output_root
    if "/" in version or version in (".", ".."):
        raise ValueError(f"{version=:s} is not one path segment")
    return storage.join_uri(root=output_root, prefix=version)


def get_tile_output_uri(
    format: Format,
    name: str,
    tile_id: str,
    output_root: str | None = None,
    version: str = "",
) -> str:
    """`{output_root}[/{version}]/{name}/{tile_id}.{tif|zarr}`; see `get_export_root`."""
    return storage.join_uri(
        root=get_export_root(output_root=output_root, version=version),
        prefix=f"{name:s}/{tile_id:s}{FORMAT_TO_EXTENSION[format]:s}",
    )


def iter_bands(
    darray: xarray.DataArray,
) -> collections.abc.Iterator[tuple[str, xarray.DataArray]]:
    """Each 2-D band of a variable and its name: one per combination of values on its dims
    beyond the pixel grid, taken in the variable's dim order and each dim's coord order.

    A value is named by `str`, so it reads well for string or integer coords; a datetime or
    float coord gives a name like `variable:2020-01-01T00:00:00.000000000`."""
    dims = [dim for dim in map(str, darray.dims) if dim not in PIXEL_DIMS]
    for indexes in itertools.product(*(range(darray.sizes[dim]) for dim in dims)):
        point = dict(zip(dims, indexes, strict=True))
        values = (str(darray[dim].values[index]) for dim, index in point.items())
        yield ":".join((str(darray.name), *values)), darray.isel(point, drop=True)


def to_cog_dataset(dset: xarray.Dataset) -> xarray.Dataset:
    """`dset`'s variables as 2-D (y, x) bands, float32, NaN no-data, in order.

    Chunked so one chunk of every band fits in memory together, as `emit` writes its output.
    """
    name_and_darrays = [
        band for darray in dset.data_vars.values() for band in iter_bands(darray=darray)
    ]
    name_to_darray = dict(name_and_darrays)
    if len(name_to_darray) < len(name_and_darrays):
        raise ValueError(f"Band names repeat: {[name for name, _ in name_and_darrays]}")
    chunk_size = geo.get_chunk_size(
        dtypes=[numpy.dtype("float32")] * len(name_to_darray)
    )
    return xarray.Dataset(
        {
            name: geo.unify_dtype_and_no_data(darray=darray.rename(None)).chunk(
                chunks=chunk_size
            )
            for name, darray in name_to_darray.items()
        }
    )


def to_zarr_dataset(dset: xarray.Dataset) -> xarray.Dataset:
    # NB: zarr v3 specifies variable-length strings, not numpy's fixed-length ones. Only
    # coords are converted: they are in memory, whereas an `object` dask variable is loaded
    # whole to infer its dtype, so a string data variable is left fixed-length
    return dset.assign_coords(
        {
            name: coord.astype(object)
            for name, coord in dset.coords.items()
            if coord.dtype.kind == "U"
        }
    )


def write_zarr(dset: xarray.Dataset, uri: str) -> None:
    """Write over any zarr at `uri`, marking it complete only once the write is."""
    complete_uri = storage.get_zarr_complete_uri(path_to_zarr=uri)
    storage.remove_file(uri=complete_uri)
    storage.write_dask_dataset_to_zarr(dset=dset, mode="w", path_to_zarr=uri)
    storage.touch_file(uri=complete_uri)


def write_tile(
    dset: xarray.Dataset,
    format: Format,
    name: str,
    tile_id: str,
    output_root: str | None = None,
    source_name: str | None = None,
    variables: collections.abc.Sequence[str] | None = None,
    version: str = "",
) -> str:
    """Write `variables` of the (lazy) `dset` for one tile; return the URI.

    `variables` defaults to every data variable, in order. Computes only those, as it writes.
    A `version` puts the tile under `{output_root}/{version}/` (see `get_export_root`).
    """
    if isinstance(variables, str):
        raise TypeError(f"{variables=:s} is one name; pass a sequence of names")
    if variables is None:
        variables = list(map(str, dset.data_vars))
    if missing := [
        variable for variable in variables if variable not in dset.data_vars
    ]:
        raise ValueError(f"{name=:s} has no variables {missing}")
    uri = get_tile_output_uri(
        format=format,
        name=name,
        output_root=output_root,
        tile_id=tile_id,
        version=version,
    )
    metadata = get_metadata(
        product_name=name, source_name=source_name or name, version=version
    )
    selected = dset[list(variables)]
    logger.info(f"Writing {name=:s} for {tile_id=:s} as {format!s}")
    match format:
        case Format.COG:
            storage.write_dask_dataset_to_cog(
                dset=to_cog_dataset(dset=selected), metadata=metadata, uri=uri
            )
        case Format.ZARR:
            write_zarr(
                dset=to_zarr_dataset(dset=selected).assign_attrs(metadata), uri=uri
            )
    logger.info(f"Wrote {name=:s} for {tile_id=:s} to {uri=:s}")
    return uri


def get_table_output_uri(
    deliverable_name: str,
    name: str,
    output_root: str | None = None,
    version: str = "",
) -> str:
    """`{output_root}[/{version}]/{deliverable}/{name}.parquet`; see `get_export_root`."""
    return storage.join_uri(
        root=get_export_root(output_root=output_root, version=version),
        prefix=f"{deliverable_name:s}/{name:s}.parquet",
    )


def write_table(
    df: pandas.DataFrame,
    deliverable_name: str,
    name: str,
    source_name: str,
    output_root: str | None = None,
    version: str = "",
) -> str:
    """Write a tabular deliverable as one parquet, provenance in its `attrs`; return the URI."""
    uri = get_table_output_uri(
        deliverable_name=deliverable_name,
        name=name,
        output_root=output_root,
        version=version,
    )
    df = df.copy()
    df.attrs |= get_metadata(
        product_name=deliverable_name, source_name=source_name, version=version
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, os.path.basename(uri))
        df.to_parquet(path=local_path)
        storage.put_file(local_path=local_path, uri=uri)
    logger.info(f"Wrote {deliverable_name=:s} ({len(df):d} rows) to {uri=:s}")
    return uri
