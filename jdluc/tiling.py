import collections.abc
import dataclasses
import enum
import logging
import math
import re
import typing

import shapely

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class XY:
    x: int
    y: int

    def validated(self) -> typing.Self:
        assert self.x >= 0 and self.y >= 0
        return self


class TileResolution(XY, enum.Enum):
    # = number of (lon, lat) pixels per 10-degree tile
    GLAD = 40_000, 40_000
    MAPSPAM = 120, 120

    @property
    def pixels_per_degree(self) -> XY:
        assert self.x % 10 == self.y % 10 == 0
        return XY(x=self.x // 10, y=self.y // 10).validated()

    @property
    def degrees_per_pixel(self) -> tuple[float, float]:
        return 10 / self.x, 10 / self.y


class Partitioning(enum.StrEnum):
    @typing.override
    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")

    __repr__ = __str__

    ONE_DEGREE_TILE = enum.auto()
    TEN_DEGREE_TILE = enum.auto()
    WHOLE_WORLD = enum.auto()
    XYZ_MERCATOR_TILE = enum.auto()


PARTITIONING_TO_IS_VALID_TILE_ID: dict[
    Partitioning, collections.abc.Callable[[str], bool]
] = {
    Partitioning.ONE_DEGREE_TILE: (
        lambda tile_id: (
            re.compile(r"\d\d[NS]_\d\d\d[EW]").fullmatch(string=tile_id) is not None
        )
    ),
    Partitioning.TEN_DEGREE_TILE: (
        lambda tile_id: (
            re.compile(r"\d0[NS]_\d\d0[EW]").fullmatch(string=tile_id) is not None
        )
    ),
    Partitioning.WHOLE_WORLD: (lambda tile_id: tile_id == "world"),
    Partitioning.XYZ_MERCATOR_TILE: (
        lambda tile_id: (
            re.compile(r"\d\d\dX_\d\d\dY_\d\dZ").fullmatch(string=tile_id) is not None
        )
    ),
}
assert set(Partitioning) == set(PARTITIONING_TO_IS_VALID_TILE_ID), (
    f"{set(Partitioning)=:}; {set(PARTITIONING_TO_IS_VALID_TILE_ID)=:}"
)


def get_lat_lon_for_tile_id(tile_id: str) -> tuple[int, int]:
    lat_str, lon_str = tile_id.split("_")
    return (
        int(lat_str[:2]) * (+1 if lat_str[-1] == "N" else -1),
        int(lon_str[:3]) * (+1 if lon_str[-1] == "E" else -1),
    )


def get_tile_id_for_lat_lon(lat: int, lon: int) -> str:
    return f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}_{abs(lon):03d}{'E' if lon >= 0 else 'W'}"


def iter_ten_degree_tile_id_for_geometry(
    geometry: shapely.Polygon | shapely.MultiPolygon,
) -> collections.abc.Iterator[str]:
    def get_range_for_low_high(low: float, high: float) -> range:
        return range(
            math.floor(low / 10) * 10,
            math.ceil(high / 10) * 10,
            10,
        )

    min_lon, min_lat, max_lon, max_lat = geometry.bounds

    for lat in get_range_for_low_high(low=min_lat, high=max_lat):
        for lon in get_range_for_low_high(low=min_lon, high=max_lon):
            tile = shapely.box(xmin=lon, ymin=lat, xmax=lon + 10, ymax=lat + 10)
            if tile.intersects(other=geometry):
                # NB: convention is to use the northern lat for the id
                yield get_tile_id_for_lat_lon(lat=lat + 10, lon=lon)


GLOBAL_NATURE_WATCH_TILE_IDS: tuple[str, ...] = (
    "00N_000E",
    "00N_010E",
    "00N_020E",
    "00N_030E",
    "00N_040E",
    "00N_040W",
    "00N_050W",
    "00N_060W",
    "00N_070E",
    "00N_070W",
    "00N_080W",
    "00N_090E",
    "00N_090W",
    "00N_100E",
    "00N_100W",
    "00N_110E",
    "00N_120E",
    "00N_130E",
    "00N_140E",
    "00N_150E",
    "00N_160E",
    "10N_000E",
    "10N_010E",
    "10N_010W",
    "10N_020E",
    "10N_020W",
    "10N_030E",
    "10N_040E",
    "10N_050E",
    "10N_050W",
    "10N_060W",
    "10N_070E",
    "10N_070W",
    "10N_080E",
    "10N_080W",
    "10N_090E",
    "10N_090W",
    "10N_100E",
    "10N_100W",
    "10N_110E",
    "10N_120E",
    "10N_130E",
    "10S_010E",
    "10S_020E",
    "10S_030E",
    "10S_040E",
    "10S_040W",
    "10S_050E",
    "10S_050W",
    "10S_060W",
    "10S_070W",
    "10S_080W",
    "10S_110E",
    "10S_120E",
    "10S_130E",
    "10S_140E",
    "10S_150E",
    "10S_160E",
    "10S_170E",
    "10S_180W",
    "20N_000E",
    "20N_010E",
    "20N_010W",
    "20N_020E",
    "20N_020W",
    "20N_030E",
    "20N_040E",
    "20N_050E",
    "20N_060W",
    "20N_070E",
    "20N_070W",
    "20N_080E",
    "20N_080W",
    "20N_090E",
    "20N_090W",
    "20N_100E",
    "20N_100W",
    "20N_110E",
    "20N_110W",
    "20N_120E",
    "20N_120W",
    "20N_160W",
    "20S_010E",
    "20S_020E",
    "20S_030E",
    "20S_040E",
    "20S_050E",
    "20S_050W",
    "20S_060W",
    "20S_070W",
    "20S_080W",
    "20S_110E",
    "20S_120E",
    "20S_130E",
    "20S_140E",
    "20S_150E",
    "20S_160E",
    "30N_000E",
    "30N_010E",
    "30N_010W",
    "30N_020E",
    "30N_020W",
    "30N_030E",
    "30N_040E",
    "30N_050E",
    "30N_060E",
    "30N_070E",
    "30N_080E",
    "30N_080W",
    "30N_090E",
    "30N_090W",
    "30N_100E",
    "30N_100W",
    "30N_110E",
    "30N_110W",
    "30N_120E",
    "30N_120W",
    "30N_160W",
    "30N_170W",
    "30S_010E",
    "30S_020E",
    "30S_030E",
    "30S_060W",
    "30S_070W",
    "30S_080W",
    "30S_110E",
    "30S_120E",
    "30S_130E",
    "30S_140E",
    "30S_150E",
    "30S_170E",
    "40N_000E",
    "40N_010E",
    "40N_010W",
    "40N_020E",
    "40N_020W",
    "40N_030E",
    "40N_040E",
    "40N_050E",
    "40N_060E",
    "40N_070E",
    "40N_070W",
    "40N_080E",
    "40N_080W",
    "40N_090E",
    "40N_090W",
    "40N_100E",
    "40N_100W",
    "40N_110E",
    "40N_110W",
    "40N_120E",
    "40N_120W",
    "40N_130E",
    "40N_130W",
    "40N_140E",
    "40S_070W",
    "40S_080W",
    "40S_140E",
    "40S_160E",
    "40S_170E",
    "50N_000E",
    "50N_010E",
    "50N_010W",
    "50N_020E",
    "50N_030E",
    "50N_040E",
    "50N_050E",
    "50N_060E",
    "50N_060W",
    "50N_070E",
    "50N_070W",
    "50N_080E",
    "50N_080W",
    "50N_090E",
    "50N_090W",
    "50N_100E",
    "50N_100W",
    "50N_110E",
    "50N_110W",
    "50N_120E",
    "50N_120W",
    "50N_130E",
    "50N_130W",
    "50N_140E",
    "50N_150E",
    "50S_060W",
    "50S_070W",
    "50S_080W",
    "60N_000E",
    "60N_010E",
    "60N_010W",
    "60N_020E",
    "60N_020W",
    "60N_030E",
    "60N_040E",
    "60N_050E",
    "60N_060E",
    "60N_060W",
    "60N_070E",
    "60N_070W",
    "60N_080E",
    "60N_080W",
    "60N_090E",
    "60N_090W",
    "60N_100E",
    "60N_100W",
    "60N_110E",
    "60N_110W",
    "60N_120E",
    "60N_120W",
    "60N_130E",
    "60N_130W",
    "60N_140E",
    "60N_140W",
    "60N_150E",
    "60N_150W",
    "60N_160E",
    "60N_160W",
    "60N_170E",
    "60N_170W",
    "60N_180W",
    "70N_000E",
    "70N_010E",
    "70N_020E",
    "70N_020W",
    "70N_030E",
    "70N_030W",
    "70N_040E",
    "70N_050E",
    "70N_060E",
    "70N_070E",
    "70N_070W",
    "70N_080E",
    "70N_080W",
    "70N_090E",
    "70N_090W",
    "70N_100E",
    "70N_100W",
    "70N_110E",
    "70N_110W",
    "70N_120E",
    "70N_120W",
    "70N_130E",
    "70N_130W",
    "70N_140E",
    "70N_140W",
    "70N_150E",
    "70N_150W",
    "70N_160E",
    "70N_160W",
    "70N_170E",
    "70N_170W",
    "70N_180W",
    "80N_010E",
    "80N_020E",
    "80N_030E",
    "80N_050E",
    "80N_060E",
    "80N_070E",
    "80N_070W",
    "80N_080E",
    "80N_080W",
    "80N_090E",
    "80N_090W",
    "80N_100E",
    "80N_100W",
    "80N_110E",
    "80N_110W",
    "80N_120E",
    "80N_120W",
    "80N_130E",
    "80N_130W",
    "80N_140E",
    "80N_140W",
    "80N_150E",
    "80N_150W",
    "80N_160E",
    "80N_160W",
    "80N_170E",
    "80N_170W",
)
assert sorted(GLOBAL_NATURE_WATCH_TILE_IDS) == list(GLOBAL_NATURE_WATCH_TILE_IDS)
WHOLE_WORLD_TILE_ID = "world"


def get_box_for_tile_id(tile_id: str) -> shapely.Polygon:
    lat, lon = get_lat_lon_for_tile_id(tile_id=tile_id)

    return shapely.box(xmin=lon, ymin=lat - 10, xmax=lon + 10, ymax=lat)
