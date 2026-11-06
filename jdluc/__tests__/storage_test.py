import pathlib

import numpy
import pytest
import xarray

from jdluc import storage


def get_dset() -> xarray.Dataset:
    return xarray.Dataset(
        {"band": (("y", "x"), numpy.arange(16, dtype="float32").reshape(4, 4))}
    ).chunk({"x": 2, "y": 2})


def assert_reads_back(cacher: storage.ZarrCacher) -> None:
    xarray.testing.assert_equal(
        cacher.deserialize()["band"].drop_vars("spatial_ref"), get_dset()["band"]
    )


def test_zarr_cacher_marks_a_finished_write_complete(tmp_path: pathlib.Path) -> None:
    cacher = storage.ZarrCacher(hash_key="abc", root=str(tmp_path))
    assert not storage.path_exists(uri=cacher.exists_uri)

    cacher.serialize(get_dset())

    assert storage.path_exists(uri=cacher.exists_uri)
    assert_reads_back(cacher)


def test_zarr_cacher_refuses_to_write_over_an_unfinished_zarr(
    tmp_path: pathlib.Path,
) -> None:
    cacher = storage.ZarrCacher(hash_key="abc", root=str(tmp_path))
    # A write killed part-way leaves zarr.json but no marker
    storage.write_dask_dataset_to_zarr(dset=get_dset(), path_to_zarr=cacher.uri)
    assert storage.path_exists(uri=f"{cacher.uri:s}/zarr.json")
    assert not storage.path_exists(uri=cacher.exists_uri)

    with pytest.raises(FileExistsError):
        cacher.serialize(get_dset())

    assert not storage.path_exists(uri=cacher.exists_uri)
