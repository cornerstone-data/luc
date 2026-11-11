import dataclasses
import pathlib
import typing

import numpy
import pytest
import xarray

from jdluc import storage
from jdluc.config import Config

# The fields a unit test needs a real value for
TEST_ENV = {"number_of_dask_workers": 2}


def get_unit_test_config() -> Config:
    """TEST_ENV over every other field left empty, so a test that reaches for a root fails rather than touching real storage."""
    types = typing.get_type_hints(Config)
    empty = {field.name: types[field.name]() for field in dataclasses.fields(Config)}
    return Config(**{**empty, **TEST_ENV})


@pytest.fixture(autouse=True)
def config_without_dot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Writing a zarr reads the config; these tests need no .env."""
    config = get_unit_test_config()
    monkeypatch.setattr(Config, "from_dot_env", classmethod(lambda cls: config))


def get_dset() -> xarray.Dataset:
    return xarray.Dataset(
        {"band": (("y", "x"), numpy.arange(16, dtype="float32").reshape(4, 4))}
    ).chunk({"x": 2, "y": 2})


def assert_reads_back(cacher: storage.ZarrCacher) -> None:
    xarray.testing.assert_equal(
        cacher.deserialize()["band"].drop_vars("spatial_ref"), get_dset()["band"]
    )


def write_without_marker(cacher: storage.ZarrCacher) -> None:
    """A write killed part-way: zarr.json, but no marker."""
    storage.write_dask_dataset_to_zarr(dset=get_dset(), path_to_zarr=cacher.uri)
    assert storage.path_exists(uri=cacher.exists_uri)
    assert not storage.path_exists(uri=cacher.complete_uri)


def test_zarr_cacher_marks_a_finished_write_complete(tmp_path: pathlib.Path) -> None:
    cacher = storage.ZarrCacher(hash_key="abc", root=str(tmp_path))
    assert not storage.path_exists(uri=cacher.exists_uri)
    assert not storage.path_exists(uri=cacher.complete_uri)

    cacher.serialize(get_dset())

    assert storage.path_exists(uri=cacher.exists_uri)
    assert storage.path_exists(uri=cacher.complete_uri)
    assert_reads_back(cacher)


def test_zarr_cacher_refuses_to_write_over_an_incomplete_zarr(
    tmp_path: pathlib.Path,
) -> None:
    cacher = storage.ZarrCacher(hash_key="abc", root=str(tmp_path))
    write_without_marker(cacher)

    with pytest.raises(storage.ZarrExistsAndIncompleteException):
        cacher.serialize(get_dset())

    assert not storage.path_exists(uri=cacher.complete_uri)


def test_zarr_cacher_refuses_to_write_over_a_complete_zarr(
    tmp_path: pathlib.Path,
) -> None:
    cacher = storage.ZarrCacher(hash_key="abc", root=str(tmp_path))
    cacher.serialize(get_dset())

    with pytest.raises(storage.ZarrExistsAndCompleteException):
        cacher.serialize(get_dset())

    assert_reads_back(cacher)


def test_mark_existing_zarrs_complete(tmp_path: pathlib.Path) -> None:
    unmarked = storage.ZarrCacher(hash_key="unmarked", root=str(tmp_path))
    write_without_marker(unmarked)
    complete = storage.ZarrCacher(hash_key="complete", root=str(tmp_path))
    complete.serialize(get_dset())

    assert storage.mark_existing_zarrs_complete(root=str(tmp_path)) == [unmarked.uri]
    assert not storage.path_exists(uri=unmarked.complete_uri)

    assert storage.mark_existing_zarrs_complete(root=str(tmp_path), dry_run=False) == [
        unmarked.uri
    ]
    assert storage.path_exists(uri=unmarked.complete_uri)
    assert_reads_back(unmarked)

    assert storage.mark_existing_zarrs_complete(root=str(tmp_path), dry_run=False) == []
