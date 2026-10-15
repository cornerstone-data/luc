import contextlib
import pathlib

import numpy
import pytest
import rasterio
import rasterio.enums
import rasterio.transform
import rasterio.warp
import shapely
import xarray
from rioxarray.rioxarray import affine_to_coords

from jdluc.geo import (
    clip_dset,
    convert_vector_to_flatgeobuf,
    downscale_darray,
    get_chunk_size,
    get_overlap_weights,
    get_overview_level,
    get_weighted_sums,
    validate_geotiff,
)


@pytest.mark.parametrize(
    ("written", "declared", "fails"),
    (
        ("uint8", "uint8", False),
        ("uint8", "uint16", True),
        ("float32", "float32", False),
        ("int16", "uint16", True),
    ),
)
def test_validate_geotiff_holds_a_dataset_to_its_declared_dtype(
    tmp_path: pathlib.Path, written: str, declared: str, fails: bool
) -> None:
    path_to_geotiff = str(tmp_path / "geotiff.tif")
    with rasterio.open(
        path_to_geotiff,
        "w",
        count=2,
        crs=4326,
        driver="GTiff",
        dtype=written,
        height=4,
        transform=rasterio.transform.from_origin(0, 10, 0.1, 0.1),
        width=4,
    ) as dataset:
        for band_idx in (1, 2):
            dataset.write(numpy.zeros((4, 4), dtype=written), band_idx)

    with pytest.raises(AssertionError) if fails else contextlib.nullcontext():
        validate_geotiff(dtype=declared, path_to_geotiff=path_to_geotiff)


@pytest.mark.parametrize(
    ("height", "width", "minimum_pixels", "expected"),
    (
        (1024, 1024, 1024, 0),
        (2, 2, 1, 1),
        (1024, 1024, 1, 10),
        (1 << 32, 1 << 24, 1 << 6, 18),
    ),
)
def test_get_overview_level(
    height: int, width: int, minimum_pixels: int, expected: int
) -> None:
    assert (
        get_overview_level(height=height, minimum_pixels=minimum_pixels, width=width)
        == expected
    )


@pytest.mark.parametrize(
    ("dtypes", "number_of_dimensions", "chunk_size"),
    (
        ([numpy.dtype("uint8")], 1, 1 << 32),
        ([numpy.dtype("uint8")], 2, 1 << 16),
        ([numpy.dtype("uint8")], 3, 1 << 10),
        ([numpy.dtype("uint8")], 4, 1 << 8),
        ([numpy.dtype("uint16")], 1, 1 << 31),
        ([numpy.dtype("uint16")], 2, 1 << 15),
        ([numpy.dtype("uint16")], 3, 1 << 10),
        ([numpy.dtype("uint16")], 4, 1 << 7),
        ([numpy.dtype("float32")], 1, 1 << 30),
        ([numpy.dtype("float32")], 2, 1 << 15),
        ([numpy.dtype("float32")], 3, 1 << 10),
        ([numpy.dtype("float32")], 4, 1 << 7),
        ([numpy.dtype("float32")] * 1, 2, 1 << 15),
        ([numpy.dtype("float32")] * 2, 2, 1 << 14),
        ([numpy.dtype("float32")] * 3, 2, 1 << 14),
        ([numpy.dtype("float32")] * 4, 2, 1 << 14),
        ([numpy.dtype("float32")] * 5, 2, 1 << 13),
        ([numpy.dtype("float32")] * 6, 2, 1 << 13),
        ([numpy.dtype("float32")] * 7, 2, 1 << 13),
        ([numpy.dtype("float32")] * 8, 2, 1 << 13),
        ([numpy.dtype("float32")] * 9, 2, 1 << 13),
        ([numpy.dtype("float32")] * 10, 2, 1 << 13),
    ),
)
def test_get_chunk_size(
    dtypes: list[numpy.dtype], number_of_dimensions: int, chunk_size: int
) -> None:
    assert (
        get_chunk_size(
            dtypes=dtypes,
            max_bytes_per_chunk=1 << 32,
            number_of_dimensions=number_of_dimensions,
        )
        == chunk_size
    )


@pytest.mark.parametrize(
    ("geometry", "expected"),
    (
        pytest.param(shapely.box(-5, -5, +5, +5), 9, id="much larger than dset"),
        pytest.param(shapely.box(-1.5, -1.5, +1.5, +1.5), 9, id="matching bounds"),
        pytest.param(shapely.box(-0.5, -0.5, +0.5, +0.5), 1, id="middle pixel"),
        pytest.param(shapely.box(-1.5, -1.5, +0.5, +0.5), 4, id="lower-left quadrant"),
        pytest.param(shapely.box(-1.5, -1.5, +0.5, +0.5), 4, id="lower-left quadrant"),
        pytest.param(shapely.box(+1.5, +1.5, +2.5, +2.5), 0, id="touch at corner"),
        pytest.param(shapely.box(+2.5, +2.5, +3.5, +3.5), 0, id="non-overlapping"),
    ),
)
def test_clip_dset(geometry: shapely.Polygon, expected: int) -> None:
    dset = (
        xarray.DataArray(
            coords={"y": [-1, 0, +1], "x": [-1, 0, +1]},
            data=numpy.ones(dtype=int, shape=(3, 3)),
            dims=("y", "x"),
        )
        .rio.write_crs(4326)
        .to_dataset(name="var")
    )
    result = clip_dset(dset=dset, geometry=geometry)
    assert int(result["var"].count()) == expected


@pytest.mark.parametrize(
    ("fine_start", "fine_stop", "fine_size", "coarse_cells", "weights"),
    (
        pytest.param(
            0,
            10,
            10,
            slice(0, 3),
            [
                [1, 1, 1, 1 / 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2 / 3, 1, 1, 2 / 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1 / 3, 1, 1, 1],
            ],
            id="the whole axis",
        ),
        pytest.param(
            3,
            7,
            10,
            slice(0, 3),
            [[1 / 3, 0, 0, 0], [2 / 3, 1, 1, 2 / 3], [0, 0, 0, 1 / 3]],
            id="fine cells cut by two coarse cell edges",
        ),
        pytest.param(
            4, 6, 10, slice(1, 2), [[1, 1]], id="fine cells inside one coarse cell"
        ),
        pytest.param(
            0, 4, 12, slice(0, 1), [[1] * 4], id="first coarse cell, ending on its edge"
        ),
        pytest.param(
            4,
            8,
            12,
            slice(1, 2),
            [[1] * 4],
            id="middle coarse cell, both ends on edges",
        ),
        pytest.param(11, 12, 12, slice(2, 3), [[1]], id="last fine cell"),
    ),
)
def test_get_overlap_weights(
    fine_start: int,
    fine_stop: int,
    fine_size: int,
    coarse_cells: slice,
    weights: list[list[float]],
) -> None:
    result = get_overlap_weights(
        coarse_size=3, fine_size=fine_size, fine_start=fine_start, fine_stop=fine_stop
    )
    assert result.coarse_cells == coarse_cells
    numpy.testing.assert_allclose(result.weights, weights)
    # Each fine cell's shares across the coarse cells add up to the whole cell
    numpy.testing.assert_allclose(result.weights.sum(axis=0), 1)


@pytest.mark.parametrize(
    ("block", "expected"),
    (
        pytest.param(
            [[numpy.nan, numpy.nan], [numpy.nan, numpy.nan]], [0, 0], id="all missing"
        ),
        pytest.param([[2, 4], [6, 8]], [10.5, 2.25], id="fractional weights"),
        pytest.param([[2, numpy.nan], [6, 8]], [6.5, 1.25], id="unequal valid counts"),
    ),
)
def test_get_weighted_sums(block: list[list[float]], expected: list[float]) -> None:
    values = numpy.asarray(block)
    actual = get_weighted_sums(
        block=values,
        column_weights=numpy.array([[0.5, 1.0]]),
        row_weights=numpy.array([[1.0, 0.5]]),
        valid=numpy.isfinite(values),
    )
    # Assert the weighted sum and count before either is divided.
    numpy.testing.assert_allclose(actual[:, 0, 0], expected)


@pytest.mark.parametrize(
    ("fine_size", "coarse_size"),
    (
        pytest.param(
            1_000,
            3,
            id="coarse cell edges cut through fine cells, as 30 m to 1/12° does",
        ),
        pytest.param(900, 3, id="coarse cell edges fall on fine cell edges"),
        pytest.param(2_051, 3, id="multiple chunks with short final chunks"),
    ),
)
def test_downscale_darray_matches_a_gdal_average_warp(
    coarse_size: int, fine_size: int
) -> None:
    generator = numpy.random.default_rng(seed=0)
    values = generator.random(dtype=numpy.float32, size=(fine_size, fine_size))
    values[generator.random(size=values.shape) < 0.3] = numpy.nan
    # Every fine cell inside the top-left coarse cell is missing
    values[: fine_size // 2, : fine_size // 2] = numpy.nan
    # Set the short last chunks apart, so a chunk that is dropped or put in the wrong place
    # changes the result (a no-op when the grid fits in one 2048-cell chunk)
    values[2_048:, :] = 1_000
    values[:, 2_048:] = 2_000
    fine_transform = rasterio.transform.from_origin(
        north=20, west=-90, xsize=10 / fine_size, ysize=10 / fine_size
    )
    coarse_transform = rasterio.transform.from_origin(
        north=20, west=-90, xsize=10 / coarse_size, ysize=10 / coarse_size
    )
    darray = xarray.DataArray(
        coords=affine_to_coords(
            affine=fine_transform, height=fine_size, width=fine_size
        ),
        data=values,
        dims=("y", "x"),
        name="var",
    ).rio.write_crs(4326)

    expected = numpy.full(
        dtype=numpy.float32, fill_value=numpy.nan, shape=(coarse_size, coarse_size)
    )
    rasterio.warp.reproject(
        destination=expected,
        dst_crs=4326,
        dst_nodata=numpy.nan,
        dst_transform=coarse_transform,
        resampling=rasterio.enums.Resampling.average,
        source=values,
        src_crs=4326,
        src_nodata=numpy.nan,
        src_transform=fine_transform,
    )
    result = downscale_darray(
        darray=darray,
        epsg=4326,
        height=coarse_size,
        resampling=rasterio.enums.Resampling.average,
        transform=coarse_transform.to_gdal(),
        width=coarse_size,
    )

    assert numpy.isnan(result.values[0, 0])
    numpy.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("nodata", [numpy.nan, -9999.0])
def test_downscale_darray_respects_source_nodata(nodata: float) -> None:
    values = numpy.array(
        [
            [nodata, nodata, 1, 3],
            [nodata, nodata, 5, 7],
            [2, nodata, 10, 20],
            [4, 6, 30, 40],
        ],
        dtype=numpy.float32,
    )
    source = (
        xarray.DataArray(values, dims=("y", "x"), name="var")
        .rio.write_crs(4326)
        .rio.write_transform(rasterio.transform.from_origin(0, 4, 1, 1))
        .rio.write_nodata(nodata)
    )
    result = downscale_darray(
        darray=source,
        epsg=4326,
        height=2,
        resampling=rasterio.enums.Resampling.average,
        transform=rasterio.transform.from_origin(0, 4, 2, 2).to_gdal(),
        width=2,
    )
    numpy.testing.assert_allclose(
        result.values, [[numpy.nan, 4.0], [4.0, 25.0]], equal_nan=True
    )


def get_dissolved_names(
    records: list[tuple[str, str, float]], tmp_path: pathlib.Path
) -> dict[str, str]:
    """`convert_vector_to_flatgeobuf` over one box per (id, name, side length)."""
    import geopandas

    path_to_vector, path_to_flatgeobuf = (
        str(tmp_path / "in.gpkg"),
        str(tmp_path / "out.fgb"),
    )
    geopandas.GeoDataFrame(
        crs=4326,
        data={
            "ISO_A3": [iso for iso, _, _ in records],
            "NAM_0": [name for _, name, _ in records],
        },
        geometry=[shapely.box(0, 0, side, side) for _, _, side in records],
    ).to_file(path_to_vector, driver="GPKG")
    convert_vector_to_flatgeobuf(
        id_column_names=("ISO_A3",),
        name_column_names=("NAM_0",),
        path_to_flatgeobuf=path_to_flatgeobuf,
        path_to_vector=path_to_vector,
    )
    gdf = geopandas.read_file(path_to_flatgeobuf)
    return dict(zip(gdf["id"], gdf["name"], strict=True))


def test_convert_vector_to_flatgeobuf_names_a_country_after_its_largest_part(
    tmp_path: pathlib.Path,
) -> None:
    # `dissolve` keeps the first row of each group, and the admin-0 layer lists Spain's
    # exclaves before Spain.  Ordering by size is what stops the country taking an exclave's
    # name; neither the first row nor the modal name would.
    assert get_dissolved_names(
        records=[
            ("ESP", "Ceuta (Sp.)", 1.0),
            ("ESP", "Melilla (Sp.)", 1.0),
            ("ESP", "Spain", 9.0),
        ],
        tmp_path=tmp_path,
    ) == {"ESP": "Spain"}


@pytest.mark.parametrize(
    ("stored", "expected"),
    (
        # The admin-0 layer stores `Türkiye` as the UTF-8 encoding of `TÃ¼rkiye`, so the file
        # is valid UTF-8 and no `encoding=` on the read undoes it
        ("TÃ¼rkiye", "Türkiye"),
        ("CuraÃ§ao (Neth.)", "Curaçao (Neth.)"),
        # ... while a name that was never mangled must not survive the Latin-1 round trip
        ("Spain", "Spain"),
        ("Andalucía", "Andalucía"),
        ("日本", "日本"),
    ),
)
def test_convert_vector_to_flatgeobuf_repairs_only_mojibake(
    expected: str, stored: str, tmp_path: pathlib.Path
) -> None:
    assert get_dissolved_names(records=[("AAA", stored, 1.0)], tmp_path=tmp_path) == {
        "AAA": expected
    }
