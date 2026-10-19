import dataclasses
import pathlib

import numpy
import pytest
import rasterio
import xarray

from jdluc import config, export, storage

TILE_ID = "00N_000E"


def get_darray(value: float) -> xarray.DataArray:
    return xarray.DataArray(
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]},  # north up, as a raster
        data=numpy.full((2, 3), value, dtype=numpy.float32),
        dims=("y", "x"),
    )


def get_dset() -> xarray.Dataset:
    """A step's output: an integer (y, x) variable, and one that varies by `scheme`."""
    by_scheme = xarray.concat(
        [get_darray(value=1), get_darray(value=2), get_darray(value=3)], dim="scheme"
    ).assign_coords(scheme=["c", "a", "b"])
    return xarray.Dataset(
        {
            "total:t": by_scheme,
            "area:ha": get_darray(value=7).astype(numpy.int16),
            "count": get_darray(value=4),
        }
    ).chunk({"x": 2})


def write_tile(format: export.Format, tmp_path: pathlib.Path, **kwargs: object) -> str:
    return export.write_tile(
        dset=get_dset(),
        format=format,
        name="example",
        output_root=str(tmp_path),
        tile_id=TILE_ID,
        **kwargs,  # type: ignore[arg-type]
    )


def test_get_tile_output_uri_is_under_the_name_by_tile(tmp_path: pathlib.Path) -> None:
    uri = export.get_tile_output_uri(
        format=export.Format.COG,
        name="example",
        output_root=str(tmp_path),
        tile_id=TILE_ID,
    )
    assert uri == str(tmp_path / "example" / f"{TILE_ID:s}.tif")


def test_get_tile_output_uri_defaults_to_the_export_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_root = dataclasses.replace(
        config.Config.from_dot_env(), export_root=str(tmp_path)
    )
    monkeypatch.setattr(
        config.Config, "from_dot_env", classmethod(lambda cls: with_root)
    )
    uri = export.get_tile_output_uri(
        format=export.Format.ZARR, name="example", tile_id=TILE_ID
    )
    assert uri == str(tmp_path / "example" / f"{TILE_ID:s}.zarr")


def test_get_tile_output_uri_puts_a_version_above_the_name(
    tmp_path: pathlib.Path,
) -> None:
    uri = export.get_tile_output_uri(
        format=export.Format.COG,
        name="example",
        output_root=str(tmp_path),
        tile_id=TILE_ID,
        version="run-1",
    )
    assert uri == str(tmp_path / "run-1" / "example" / f"{TILE_ID:s}.tif")


def test_get_table_output_uri_puts_a_version_above_the_deliverable(
    tmp_path: pathlib.Path,
) -> None:
    uri = export.get_table_output_uri(
        deliverable_name="factors",
        name="BRA",
        output_root=str(tmp_path),
        version="run-1",
    )
    assert uri == str(tmp_path / "run-1" / "factors" / "BRA.parquet")


@pytest.mark.parametrize("version", ["a/b", ".", ".."])
def test_get_export_root_rejects_a_version_that_is_not_one_segment(
    version: str, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(ValueError, match="one path segment"):
        export.get_export_root(output_root=str(tmp_path), version=version)


def test_iter_bands_flattens_extra_dims_in_coord_order() -> None:
    names = [name for name, _ in export.iter_bands(darray=get_dset()["total:t"])]
    assert names == ["total:t:c", "total:t:a", "total:t:b"]


@pytest.mark.parametrize("format", list(export.Format))
def test_write_tile_rejects_a_missing_variable_before_writing(
    format: export.Format, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(ValueError, match="missing"):
        write_tile(format=format, tmp_path=tmp_path, variables=["count", "missing"])
    assert not list(tmp_path.iterdir())


def test_write_tile_rejects_one_name_as_variables(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="sequence of names"):
        write_tile(format=export.Format.ZARR, tmp_path=tmp_path, variables="count")
    assert not list(tmp_path.iterdir())


def test_write_tile_as_cog_writes_every_variable_by_default(
    tmp_path: pathlib.Path,
) -> None:
    uri = write_tile(format=export.Format.COG, tmp_path=tmp_path)
    assert uri == str(tmp_path / "example" / f"{TILE_ID:s}.tif")
    with rasterio.open(uri) as dataset:
        assert list(dataset.descriptions) == [
            "total:t:c",
            "total:t:a",
            "total:t:b",
            "area:ha",
            "count",
        ]
        assert set(dataset.dtypes) == {"float32"}
        assert numpy.isnan(dataset.nodata)
        numpy.testing.assert_array_equal(dataset.read()[:, 0, 0], [1, 2, 3, 7, 4])
        tags = dataset.tags()
    assert tags["cornerstone-product-name"] == "example"
    assert tags["cornerstone-source-name"] == "example"
    assert "cornerstone-processing-version" in tags
    assert "cornerstone-export-version" not in tags


def test_write_tile_with_a_version_stamps_it(tmp_path: pathlib.Path) -> None:
    uri = write_tile(format=export.Format.COG, tmp_path=tmp_path, version="run-1")
    assert uri == str(tmp_path / "run-1" / "example" / f"{TILE_ID:s}.tif")
    with rasterio.open(uri) as dataset:
        assert dataset.tags()["cornerstone-export-version"] == "run-1"


def test_write_tile_as_cog_writes_a_subset_in_the_given_order(
    tmp_path: pathlib.Path,
) -> None:
    uri = write_tile(
        format=export.Format.COG,
        source_name="emit",
        tmp_path=tmp_path,
        variables=["count", "area:ha"],
    )
    with rasterio.open(uri) as dataset:
        assert list(dataset.descriptions) == ["count", "area:ha"]
        assert dataset.tags()["cornerstone-source-name"] == "emit"


def test_write_tile_as_cog_marks_no_data_as_nan(tmp_path: pathlib.Path) -> None:
    dset = get_dset()
    dset["count"] = dset["count"].where(dset["x"] > 1)
    uri = export.write_tile(
        dset=dset,
        format=export.Format.COG,
        name="example",
        output_root=str(tmp_path),
        tile_id=TILE_ID,
        variables=["count"],
    )
    with rasterio.open(uri) as dataset:
        band = dataset.read(1, masked=True)
    assert band.mask.tolist() == [[True, False, False], [True, False, False]]


def test_write_tile_as_zarr_keeps_dtypes_and_extra_dims(
    tmp_path: pathlib.Path,
) -> None:
    uri = write_tile(format=export.Format.ZARR, tmp_path=tmp_path)
    assert uri == str(tmp_path / "example" / f"{TILE_ID:s}.zarr")
    written = xarray.open_zarr(uri, consolidated=False)
    # NB: a zarr group's members have no order; a reader lists them by name
    assert set(written.data_vars) == {"total:t", "area:ha", "count"}
    assert written["area:ha"].dtype == numpy.int16
    assert written["total:t"].dims == ("scheme", "y", "x")
    assert list(written["scheme"].values) == ["c", "a", "b"]
    assert written.attrs["cornerstone-product-name"] == "example"
    assert storage.path_exists(uri=storage.get_zarr_complete_uri(path_to_zarr=uri))


def test_write_tile_as_zarr_writes_only_a_subset(
    tmp_path: pathlib.Path,
) -> None:
    uri = write_tile(
        format=export.Format.ZARR, tmp_path=tmp_path, variables=["count", "area:ha"]
    )
    written = xarray.open_zarr(uri, consolidated=False)
    assert set(written.data_vars) == {"count", "area:ha"}
    assert "scheme" not in written.dims


def test_write_tile_as_zarr_writes_over_an_earlier_export(
    tmp_path: pathlib.Path,
) -> None:
    write_tile(format=export.Format.ZARR, tmp_path=tmp_path)
    uri = write_tile(format=export.Format.ZARR, tmp_path=tmp_path, variables=["count"])
    assert list(xarray.open_zarr(uri, consolidated=False).data_vars) == ["count"]
    assert storage.path_exists(uri=storage.get_zarr_complete_uri(path_to_zarr=uri))


def test_write_tile_as_zarr_drops_the_marker_before_writing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = write_tile(format=export.Format.ZARR, tmp_path=tmp_path)

    def fail(**kwargs: object) -> None:
        raise RuntimeError("killed part-way")

    monkeypatch.setattr(storage, "write_dask_dataset_to_zarr", fail)
    with pytest.raises(RuntimeError, match="killed part-way"):
        write_tile(format=export.Format.ZARR, tmp_path=tmp_path)

    assert not storage.path_exists(uri=storage.get_zarr_complete_uri(path_to_zarr=uri))
