import collections.abc
import datetime
import enum
import functools
import logging
import threading
import time
import typing

import git
import requests
import requests.adapters
import xarray

from jdluc import config, storage

logger = logging.getLogger(__name__)


@functools.cache
def get_requests_session() -> requests.Session:
    retry = requests.adapters.Retry(
        allowed_methods=["GET"],
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        total=5,
    )
    logger.info("Creating a fresh session for requests")
    session = requests.Session()
    session.mount(
        "https://",
        requests.adapters.HTTPAdapter(max_retries=retry, pool_maxsize=16),
    )
    return session


def save_remote_url_to_local_path(
    local_path: str,
    params: dict[str, str] | str,
    remote_url: str,
    # 4 MiB
    chunk_size: int = 1 << 22,
    # Bounds the gap between chunks rather than the whole transfer: a publisher that stops sending
    # is the failure mode, and without this the read blocks forever
    read_timeout_seconds: int = 120,
    body_retries: int = 3,
    cache_key: str | None = None,
) -> None:
    """Stream `remote_url` to `local_path`.

    With a `cache_key`, and SOURCE_CACHE_ROOT set, the download is shared through
    `{SOURCE_CACHE_ROOT}/{cache_key}`: a process that finds it there copies it rather than
    asking the publisher, and one that does not downloads as usual and leaves a copy for the
    next. Pass a key only for a URL whose content never changes, as nothing refreshes the copy.
    """
    cache_uri: str | None = None
    if cache_key is not None:
        source_cache_root = config.Config.from_dot_env().source_cache_root
        if source_cache_root:
            cache_uri = storage.join_uri(prefix=cache_key, root=source_cache_root)
            if storage.path_exists(uri=cache_uri):
                storage.get_file(local_path=local_path, uri=cache_uri)
                return
    for attempt in range(body_retries):
        logger.info(f"GET'ing from {remote_url=:s} with {params=:}")
        try:
            with get_requests_session().request(
                method="GET",
                params=params,
                stream=True,
                timeout=read_timeout_seconds,
                url=remote_url,
            ) as response:
                response.raise_for_status()
                logger.info(f"Streaming data from {remote_url=:s} to {local_path=:s}")
                # NB: reopened per attempt, so a truncated body is discarded rather than appended
                with open(file=local_path, mode="wb") as fp:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fp.write(chunk)
        except requests.HTTPError, requests.exceptions.RetryError:
            # A status RETRY does not list, or one it already spent all its attempts on
            raise
        except requests.RequestException as exc:
            logger.warning(f"Retrying {remote_url=:s} after {attempt=:d}: {exc}")
            if attempt + 1 == body_retries:
                raise
            time.sleep(2**attempt)
        else:
            if cache_uri is not None:
                storage.put_file(local_path=local_path, uri=cache_uri)
            return


@functools.cache
def get_git_version(default_branch_name: str = "main") -> str:
    repo = git.Repo(__file__, search_parent_directories=True)
    if (branch_name := repo.active_branch.name) == default_branch_name:
        return f"{branch_name:s}-{repo.head.commit.hexsha[:8]:s}"
    else:
        return branch_name


@functools.cache
def get_git_remote_url(default_remote_name: str = "origin") -> str:
    repo = git.Repo(__file__, search_parent_directories=True)
    (remote,) = (
        remote for remote in repo.remotes if remote.name == default_remote_name
    )
    return next(iter(remote.urls))


def get_utc_timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def threadsafe_cache[**P, R](func: typing.Callable[P, R]) -> typing.Callable[P, R]:
    cached = functools.cache(func)

    # each cached function has its own lock
    lock = threading.Lock()

    @functools.wraps(cached)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with lock:
            return cached(*args, **kwargs)  # type: ignore

    inner.cache_clear = cached.cache_clear  # type: ignore
    inner.cache_info = cached.cache_info  # type: ignore
    inner.cache_parameters = cached.cache_parameters  # type: ignore
    return inner


def get_sum_totals[E: enum.Enum](
    enum_to_name_to_darray: dict[E, dict[str, xarray.DataArray]],
) -> dict[str, dict[str, float]]:
    enum_name_to_darray = {
        (e, name): darray
        for e, name_to_darray in enum_to_name_to_darray.items()
        for name, darray in name_to_darray.items()
    }
    batched_sum = xarray.Dataset(enum_name_to_darray).sum().compute()

    ret: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for e, name in enum_name_to_darray:
        ret[e.name][name] = float(batched_sum[(e, name)])
    return ret


def iter_sharded[T: str | int](
    modulus: int,
    residues: collections.abc.Collection[int] | None,
    values: collections.abc.Iterable[T],
) -> collections.abc.Iterator[T]:
    """Shard values by modulus; helpful for avoiding collisions during chunked backfills"""
    assert modulus >= 1, f"{modulus=:d} must be nonzero"
    if residues is None:
        yield from sorted(values)
    else:
        assert set(range(modulus)).issuperset(residues), (
            f"{residues=:} exceed {modulus=:d}"
        )
        for idx, value in enumerate(sorted(values)):
            if idx % modulus in residues:
                yield value
