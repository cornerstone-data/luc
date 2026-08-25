import contextlib
import functools

import pytest

from jdluc import tiling
from jdluc.datasets.base import BandType
from jdluc.harmonize import Grid


@pytest.mark.parametrize(
    ("x", "y", "fails"),
    (
        (0, 0, False),
        (100, 100, False),
        (-1, -1, True),
    ),
)
def test_xy_validated(x: int, y: int, fails: bool) -> None:
    xy = tiling.XY(x=x, y=y)
    context = (
        functools.partial(pytest.raises, AssertionError)
        if fails
        else contextlib.nullcontext
    )
    with context():
        xy.validated()


@pytest.mark.parametrize(
    (
        "tile_id",
        "origin",
        "transform",
    ),
    (
        pytest.param(
            "00N_000W",
            tiling.XY(0, 0),
            (0, 1 / 4_000, 0, 0, 0, -1 / 4_000),
            id="origin at 0,0",
        ),
        pytest.param(
            "60N_060W",
            tiling.XY(-60, 60),
            (-60, 1 / 4_000, 0, 60, 0, -1 / 4_000),
            id="north-west of the meridian",
        ),
        pytest.param(
            "60S_060E",
            tiling.XY(60, -60),
            (60, 1 / 4_000, 0, -60, 0, -1 / 4_000),
            id="south-east of the meridian",
        ),
    ),
)
def test_grid_from_tile_id_resolution(
    tile_id: str,
    origin: tiling.XY,
    transform: tuple[float, float, float, float, float, float],
) -> None:
    result = Grid.from_tile_id_resolution(
        resolution=tiling.TileResolution.GLAD, tile_id=tile_id
    )
    assert result.origin == origin
    assert result.resolution == tiling.TileResolution.GLAD.value
    assert type(result.resolution) is tiling.XY
    assert result.transform == transform


def test_grid_get_offset_for_world() -> None:
    grid = Grid.from_tile_id_resolution(
        resolution=tiling.XY(10, 10), tile_id="60N_060W"
    )
    assert grid.get_offset_for_world(
        resolution=tiling.XY(360, 180), span=tiling.XY(360, 180)
    ) == tiling.XY(120, 30)


def test_grid_get_resolution_for_world() -> None:
    grid = Grid.from_tile_id_resolution(
        resolution=tiling.XY(10, 10), tile_id="60N_060W"
    )
    assert grid.get_resolution_for_world(
        resolution=tiling.XY(360, 180), span=tiling.XY(360, 180)
    ) == tiling.XY(10, 10)


@pytest.mark.parametrize(
    ("band_type", "resolution", "expected"),
    (
        (BandType.CATEGORICAL, tiling.XY(1, 1), "nearest"),
        (BandType.CATEGORICAL, tiling.XY(2, 2), "nearest"),
        (BandType.CATEGORICAL, tiling.XY(3, 3), "mode"),
        (BandType.INTENSIVE, tiling.XY(1, 1), "bilinear"),
        (BandType.INTENSIVE, tiling.XY(2, 2), "nearest"),
        (BandType.INTENSIVE, tiling.XY(3, 3), "average"),
        (BandType.EXTENSIVE, tiling.XY(2, 2), "nearest"),
    ),
)
def test_grid_get_resampling_for_band_type(
    band_type: BandType, resolution: tiling.XY, expected: str
) -> None:
    grid = Grid(origin=tiling.XY(0, 0), resolution=tiling.XY(2, 2))
    assert (
        grid.get_resampling_for_band_type(
            band_type=band_type,
            src_resolution=resolution,
            dest_resolution=grid.resolution,
        ).name
        == expected
    )


@pytest.mark.parametrize(
    ("src_resolution", "match"),
    (
        (tiling.XY(3, 3), "GDAL doesn't implement sum resampling"),
        (tiling.XY(1, 1), "GDAL doesn't implement distribution resampling"),
    ),
)
def test_grid_get_resampling_for_band_type_raises(
    src_resolution: tiling.XY, match: str
) -> None:
    grid = Grid(origin=tiling.XY(0, 0), resolution=tiling.XY(2, 2))
    with pytest.raises(NotImplementedError, match=match):
        assert grid.get_resampling_for_band_type(
            band_type=BandType.EXTENSIVE,
            src_resolution=src_resolution,
            dest_resolution=grid.resolution,
        )


def test_grid_get_resampling_for_band_type_upsamples_clipped_world() -> None:
    # A whole-world source that is large in absolute pixels but coarse per
    # degree: comparing its full resolution against the grid would pick
    # downsampling, but once clipped to the grid extent it is coarser than the
    # destination and must be upsampled.
    grid = Grid(origin=tiling.XY(0, 0), resolution=tiling.XY(36, 36))
    world_resolution = tiling.XY(360, 180)  # 1 px/degree, coarser than the grid
    src_resolution = grid.get_resolution_for_world(
        resolution=world_resolution, span=tiling.XY(360, 180)
    )
    # Clipped to the grid's 10x10-degree extent the source is only 10x10 px,
    # versus the 36x36 px destination, even though the full source is 360x180.
    assert src_resolution == tiling.XY(10, 10)
    assert world_resolution.x > grid.resolution.x  # the old, buggy comparison
    assert (
        grid.get_resampling_for_band_type(
            band_type=BandType.INTENSIVE,
            src_resolution=src_resolution,
            dest_resolution=grid.resolution,
        ).name
        == "bilinear"
    )
