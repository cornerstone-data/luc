import pathlib

import numpy
import pytest
import shapely
import xarray

from jdluc.geo import (
    clip_dset,
    convert_vector_to_flatgeobuf,
    get_chunk_size,
    get_overview_level,
)


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
        get_overview_level(height=height, width=width, minimum_pixels=minimum_pixels)
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
            number_of_dimensions=number_of_dimensions,
            max_bytes_per_chunk=1 << 32,
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
            data=numpy.ones(shape=(3, 3), dtype=int),
            dims=("y", "x"),
        )
        .rio.write_crs(4326)
        .to_dataset(name="var")
    )
    result = clip_dset(dset=dset, geometry=geometry)
    assert int(result["var"].count()) == expected


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
        crs="EPSG:4326",
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
