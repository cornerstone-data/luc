import math

import geopandas
import numpy
import pytest
import rasterio.transform
import shapely

from jdluc.datasets import fdap
from jdluc.datasets.worldbank_jurisdictions import AdminLevel


def test_get_pixel_range_gives_each_degree_the_pixels_centred_in_it() -> None:
    pixel_ranges = [
        fdap.get_pixel_range(start_degrees=degrees, stop_degrees=degrees + 1)
        for degrees in range(-180, 180)
    ]
    # Read along the pixels, the degrees their centres fall in are every degree once, in order
    assert [
        math.floor((pixel + 0.5) * fdap.PIXEL_DEGREES)
        for pixels in pixel_ranges
        for pixel in (pixels.start - 1, pixels.start, pixels.stop - 1, pixels.stop)
    ] == [
        centre_degrees
        for degrees in range(-180, 180)
        for centre_degrees in (degrees - 1, degrees, degrees, degrees + 1)
    ]


@pytest.mark.parametrize("start_degrees", range(-180, 180))
def test_get_pixel_range_reads_each_source_inside_its_file(start_degrees: int) -> None:
    # A source file is 11133 pixels a side, from the pixel holding its west or south edge
    pixels = fdap.get_pixel_range(
        start_degrees=start_degrees, stop_degrees=start_degrees + 1
    )
    first_pixel = math.floor(start_degrees / fdap.PIXEL_DEGREES)
    assert first_pixel <= pixels.start
    assert pixels.stop <= first_pixel + 11_133


# Three countries side by side, one degree each: AAA has fitted thresholds, BBB is a producer
# without them, and CCC is not a producer. AAA's western half is an excluded province.
FDAP_NATIONAL = geopandas.GeoSeries(
    crs=4326,
    data=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1), shapely.box(2, 0, 3, 1)],
    index=["AAA", "BBB", "CCC"],
)
FDAP_PROVINCIAL = geopandas.GeoSeries(
    crs=4326, data=[shapely.box(0, 0, 0.5, 1)], index=["AAA001"]
)
FDAP_FLOOR = 0.5
# Four pixels per country, of which AAA's first two are in the excluded province
FDAP_TRANSFORM = rasterio.transform.from_origin(0, 1, 0.25, 1)
FDAP_SHAPE = (1, 12)
FDAP_ISO_3166_TO_UNEXCLUDED_PIXELS = {"AAA": slice(2, 4), "BBB": slice(4, 8)}


def get_fdap_mask(fill_probability: float, year: int) -> numpy.ndarray:
    return fdap.get_mask(
        admin_level_to_geometries={
            AdminLevel.NATIONAL: FDAP_NATIONAL,
            AdminLevel.PROVINCIAL: FDAP_PROVINCIAL,
            AdminLevel.DISTRICT: geopandas.GeoSeries(crs=4326),
        },
        excluded_admin_ids={
            AdminLevel.NATIONAL: set(),
            AdminLevel.PROVINCIAL: {"AAA001"},
            AdminLevel.DISTRICT: set(),
        },
        floor=FDAP_FLOOR,
        iso_3166_year_to_threshold={
            ("AAA", 2018): 0.2,
            ("AAA", 2020): 0.8,
            ("AAA", 2022): fdap.HIGHEST_THRESHOLD,
            ("AAA", 2024): 0.1,
        },
        probability=numpy.full(
            dtype="float32", fill_value=fill_probability, shape=FDAP_SHAPE
        ),
        producer_iso_3166s={"AAA", "BBB"},
        transform=FDAP_TRANSFORM,
        year=year,
    )


@pytest.mark.parametrize("year", fdap.YEARS)
def test_get_mask_leaves_out_exclusions_and_non_producers_even_at_1(
    year: int,
) -> None:
    assert get_fdap_mask(fill_probability=1.0, year=year).tolist() == [
        [0] * 2 + [1] * 6 + [0] * 4
    ]


@pytest.mark.parametrize(
    ("iso_3166", "year", "threshold"),
    [
        pytest.param("AAA", 2018, FDAP_FLOOR, id="fitted-below-the-floor"),
        pytest.param("AAA", 2020, 0.8, id="fitted-above-the-floor"),
        pytest.param(
            "AAA", 2022, fdap.HIGHEST_THRESHOLD, id="the-upper-end-of-the-sweep"
        ),
        pytest.param(
            "AAA", 2024, FDAP_FLOOR, id="the-lower-end-of-the-sweep-below-the-floor"
        ),
        pytest.param(
            "BBB", 2018, FDAP_FLOOR, id="a-producer-without-fitted-thresholds"
        ),
    ],
)
def test_get_mask_compares_strictly_with_the_threshold(
    iso_3166: str, year: int, threshold: float
) -> None:
    pixels = FDAP_ISO_3166_TO_UNEXCLUDED_PIXELS[iso_3166]
    at = numpy.float32(threshold)
    above = numpy.nextafter(at, numpy.float32(1))
    assert not get_fdap_mask(fill_probability=float(at), year=year)[:, pixels].any()
    assert get_fdap_mask(fill_probability=float(above), year=year)[:, pixels].all()


@pytest.mark.parametrize(
    ("masks", "expected"),
    [
        pytest.param((1, 0, 1, 1), (1, 1, 1, 1), id="a-one-year-gap-in-2020-fills"),
        pytest.param((0, 1, 0, 1), (0, 1, 1, 1), id="a-one-year-gap-in-2022-fills"),
        pytest.param((1, 0, 0, 1), (1, 0, 0, 1), id="a-two-year-gap-stays"),
        pytest.param((0, 1, 1, 1), (0, 1, 1, 1), id="2018-never-fills"),
        pytest.param((1, 1, 1, 0), (1, 1, 1, 0), id="2024-never-fills"),
    ],
)
def test_get_smoothed_masks_fills_one_year_gaps_in_the_inner_years(
    masks: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    smoothed = fdap.get_smoothed_masks(
        year_to_mask={
            year: numpy.full(dtype="uint8", fill_value=mask, shape=(1, 1))
            for year, mask in zip(fdap.YEARS, masks, strict=True)
        }
    )
    assert tuple(int(smoothed[year][0, 0]) for year in fdap.YEARS) == expected


@pytest.mark.parametrize(
    ("latitude", "longitude", "read"),
    [
        pytest.param(0, 0, True, id="a-cell-inside-a-producer"),
        pytest.param(0, 2, True, id="a-cell-touching-a-producer-at-its-edge"),
        pytest.param(0, 3, False, id="a-cell-touching-only-a-non-producer"),
        pytest.param(5, 5, False, id="a-cell-touching-no-country"),
    ],
)
def test_touches_a_producer(latitude: int, longitude: int, read: bool) -> None:
    assert (
        fdap.touches_a_producer(
            lat_lon=(latitude, longitude),
            national=FDAP_NATIONAL,
            producer_iso_3166s={"AAA", "BBB"},
        )
        is read
    )
