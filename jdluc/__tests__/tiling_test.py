import pytest
import shapely

from jdluc.tiling import (
    GLOBAL_FOREST_WATCH_TILE_IDS,
    PARTITIONING_TO_IS_VALID_TILE_ID,
    Partitioning,
    get_box_for_tile_id,
    get_lat_lon_for_tile_id,
    get_tile_id_for_lat_lon,
    iter_ten_degree_tile_id_for_geometry,
)


@pytest.mark.parametrize(
    ("partitioning", "tile_id", "expected"),
    (
        (Partitioning.ONE_DEGREE_TILE, "00N_000E", True),
        (Partitioning.ONE_DEGREE_TILE, "01N_001E", True),
        (Partitioning.ONE_DEGREE_TILE, "1N_1E", False),
        (Partitioning.TEN_DEGREE_TILE, "00N_000E", True),
        (Partitioning.TEN_DEGREE_TILE, "10N_010E", True),
        (Partitioning.TEN_DEGREE_TILE, "05N_003E", False),
        (Partitioning.WHOLE_WORLD, "world", True),
        (Partitioning.WHOLE_WORLD, "CONUS", False),
        (Partitioning.XYZ_MERCATOR_TILE, "000X_000Y_00Z", True),
        (Partitioning.XYZ_MERCATOR_TILE, "001X_001Y_01Z", True),
        (Partitioning.XYZ_MERCATOR_TILE, "1X_2Y_3Z", False),
    ),
)
def test_partitioning_to_is_valid_tile_id(
    partitioning: Partitioning, tile_id: str, expected: bool
) -> None:
    assert PARTITIONING_TO_IS_VALID_TILE_ID[partitioning](tile_id) is expected


BOX_40N_080W = shapely.box(-80, 40, -70, 50)
PIXEL_45N_075W = shapely.Point(-75, +45).buffer(distance=1)
STREAK_SEVEN = shapely.LineString(((-75, +45), (-75, +35), (-85, +35))).buffer(
    distance=1
)


@pytest.mark.parametrize("lat", range(-90, +91, 10))
@pytest.mark.parametrize("lon", range(-180, +181, 10))
def test_lat_long_tile_id_roundtrip(lat: int, lon: int) -> None:
    assert get_lat_lon_for_tile_id(
        tile_id=get_tile_id_for_lat_lon(lat=lat, lon=lon)
    ) == (lat, lon)


@pytest.mark.parametrize(
    ("geometry", "expected"),
    (
        (BOX_40N_080W, ("50N_080W",)),
        (PIXEL_45N_075W, ("50N_080W",)),
        (STREAK_SEVEN, ("40N_090W", "40N_080W", "50N_080W")),
    ),
)
def test_iter_ten_degree_tile_id_for_geometry(
    geometry: shapely.Polygon | shapely.MultiPolygon, expected: tuple[str, ...]
) -> None:
    assert tuple(iter_ten_degree_tile_id_for_geometry(geometry=geometry)) == expected


def test_get_box_for_tile_id_uses_the_northern_lat() -> None:
    assert get_box_for_tile_id(tile_id="50N_080W").equals(BOX_40N_080W)


@pytest.mark.parametrize("tile_id", GLOBAL_FOREST_WATCH_TILE_IDS)
def test_get_box_for_tile_id_round_trips(tile_id: str) -> None:
    # These two functions must agree on the northern-lat convention, because
    # worldbank_jurisdictions.iter_jurisdiction_for_iso_3166_tile_id skips a province whose
    # geometry misses this box. Were the conventions to drift by one tile, a province could be
    # skipped for every tile it belongs to and disappear from the rollup entirely.
    assert tuple(
        iter_ten_degree_tile_id_for_geometry(
            geometry=get_box_for_tile_id(tile_id=tile_id)
        )
    ) == (tile_id,)
