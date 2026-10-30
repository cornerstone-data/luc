"""Forest Data Partnership | Commodity probabilities, 2018-2024

license: CC BY 4.0

year: 2018, 2020, 2022, 2024

Google, for the Forest Data Partnership (2026). Commodity probabilities, release 2026a [Data set]. gs://earth-engine-public-requester-pays/forestdatapartnership/2026a -- the record ingested here, whose licence asks for the attribution "Produced by Google for the Forest Data Partnership"
TODO: add reference for the thresholds and exclusions

https://github.com/google/forest-data-partnership

# Methodology

- A multilayer perceptron probe, one per commodity, reads each 10 m pixel's Google AlphaEarth
  Foundations satellite embedding, a yearly summary of satellite imagery, and gives the
  probability that the pixel grows the commodity. It covers 24°S to 24°N
- The authors say the model maps too much of each tree crop on average, and that its results
  change a lot from year to year

We don't keep the probabilities. Ingest turns them into a binary mask for each year, in four steps:

1. Merge: the one-degree source files are pieced together into ten-degree tiles at their own 10 m
   resolution, without resampling. Neighbouring files overlap by one pixel, which is taken from
   the file it falls in
2. Threshold: a pixel is in the mask when its probability is above its country's threshold for
   that year, or above the commodity's floor if that is higher. A country without thresholds uses
   the floor if FAOSTAT says it grows the commodity, and is left out otherwise
3. Exclude: pixels in the excluded areas are left out
4. Smooth: a pixel missing from the 2020 or 2022 mask is added back when it is in the masks of both
   the year before and the year after. 2018 and 2024 are not smoothed
"""

import enum
import logging
import math
import typing

import geopandas
import google.auth
import numpy
import pandas
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.windows
import shapely

from jdluc import storage, tiling, utils
from jdluc.datasets import base, worldbank_jurisdictions

logger = logging.getLogger(__name__)

RELEASE = "2026a"
YEARS: typing.Final[tuple[int, ...]] = (2018, 2020, 2022, 2024)
# 10 m on the equator of the WGS 84 ellipsoid, whose semi-major axis is 6,378,137 m
PIXEL_DEGREES = 10 / (math.pi * 6_378_137 / 180)
# NB: The sweep for thresholds stopped here; a more precise value could be pursued but that would
# likely overfit within a region of the CDF that was quite steep
HIGHEST_THRESHOLD = 0.99
NO_DATA = None


class Commodity(enum.StrEnum):
    COCOA = enum.auto()
    COFFEE = enum.auto()


def get_pixel_range(start_degrees: int, stop_degrees: int) -> range:
    """Global indices of the pixels whose centres fall in [start, stop) degrees, on either axis."""
    return range(
        math.ceil(start_degrees / PIXEL_DEGREES - 0.5),
        math.ceil(stop_degrees / PIXEL_DEGREES - 0.5),
    )


def get_mask(
    admin_level_to_geometries: dict[
        worldbank_jurisdictions.AdminLevel, geopandas.GeoSeries
    ],
    excluded_admin_ids: dict[worldbank_jurisdictions.AdminLevel, set[str]],
    floor: float,
    iso_3166_year_to_threshold: dict[tuple[str, int], float],
    probability: numpy.ndarray,
    producer_iso_3166s: set[str],
    transform: rasterio.Affine,
    year: int,
) -> numpy.ndarray:
    """1 where `probability` is strictly greater than its pixel's threshold, else 0."""
    box = shapely.box(
        *rasterio.transform.array_bounds(*probability.shape, transform=transform)
    )
    national = admin_level_to_geometries[worldbank_jurisdictions.AdminLevel.NATIONAL]
    defaults = [
        (national[iso_3166], floor)
        for iso_3166 in sorted(producer_iso_3166s)
        if national[iso_3166].intersects(box)
    ]
    fitted = [
        (national[iso_3166], max(threshold, floor))
        for (iso_3166, the_year), threshold in sorted(
            iso_3166_year_to_threshold.items()
        )
        if the_year == year and national[iso_3166].intersects(box)
    ]
    exclusions = [
        (geometry, 1.0)
        for admin_level, admin_ids in sorted(excluded_admin_ids.items())
        for geometry in admin_level_to_geometries[admin_level][sorted(admin_ids)]
        if geometry.intersects(box)
    ]
    # Later shapes overwrite earlier ones, and nothing is greater than 1.0
    threshold = rasterio.features.rasterize(
        dtype="float32",
        fill=1,
        out_shape=probability.shape,
        shapes=defaults + fitted + exclusions,
        transform=transform,
    )
    return (probability > threshold).astype("uint8")


def get_smoothed_masks(
    year_to_mask: dict[int, numpy.ndarray],
) -> dict[int, numpy.ndarray]:
    """Each year's mask, with each inner year also 1 where the years either side are both 1."""
    years = sorted(year_to_mask)
    first_year, *_, last_year = years
    return {
        year: (
            year_to_mask[year]
            if year in {first_year, last_year}
            else year_to_mask[year]
            | (year_to_mask[years[idx - 1]] & year_to_mask[years[idx + 1]])
        )
        for idx, year in enumerate(years)
    }


@utils.threadsafe_cache
def get_billing_project() -> str:
    _, project = google.auth.default()
    assert project is not None
    return project


@utils.threadsafe_cache
def get_index() -> pandas.DataFrame:
    return pandas.read_parquet(
        path=f"gs://earth-engine-public-requester-pays/forestdatapartnership/{RELEASE:s}/fdap_index.parquet",
        storage_options={"requester_pays": get_billing_project()},
    )


def get_raster_dataset(
    commodity: Commodity,
    excluded_admin_ids: dict[worldbank_jurisdictions.AdminLevel, set[str]],
    floor: float,
    iso_3166_year_to_threshold: dict[tuple[str, int], float],
    producer_iso_3166s: set[str],
    version: str,
) -> base.RasterDataset:
    iso_3166s = {iso_3166 for iso_3166, _ in iso_3166_year_to_threshold}
    assert set(iso_3166_year_to_threshold) == {
        (iso_3166, year) for iso_3166 in iso_3166s for year in YEARS
    }
    assert iso_3166s <= producer_iso_3166s
    assert set(excluded_admin_ids) == set(worldbank_jurisdictions.AdminLevel)
    assert (
        not iso_3166s & excluded_admin_ids[worldbank_jurisdictions.AdminLevel.NATIONAL]
    )

    def save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
        admin_level_to_geometries = {
            admin_level: worldbank_jurisdictions.get_jurisdiction_for_admin_level(
                admin_level=admin_level
            ).geometry
            for admin_level in worldbank_jurisdictions.AdminLevel
        }
        max_latitude, min_longitude = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
        columns = get_pixel_range(
            start_degrees=min_longitude, stop_degrees=min_longitude + 10
        )
        rows = get_pixel_range(
            start_degrees=max_latitude - 10, stop_degrees=max_latitude
        )
        index = get_index()
        sources = index[
            (index["commodity"] == commodity)
            & (index["lng"] >= min_longitude)
            & (index["lng"] < min_longitude + 10)
            & (index["lat"] >= max_latitude - 10)
            & (index["lat"] < max_latitude)
        ]
        lat_lons = sorted(
            set(zip(sources["lat"].tolist(), sources["lng"].tolist(), strict=True))
        )
        logger.info(f"Thresholding {len(lat_lons):d} source cells onto {tile_id=:s}")
        with (
            rasterio.Env(
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_NUM_THREADS="ALL_CPUS",
                GS_USER_PROJECT=get_billing_project(),
            ),
            # Blocks no source writes are zeros when the file closes
            rasterio.open(
                BIGTIFF="YES",
                NUM_THREADS="ALL_CPUS",
                compress="deflate",
                count=len(YEARS),
                crs=4326,
                driver="GTiff",
                dtype="uint8",
                fp=local_path,
                height=len(rows),
                interleave="band",
                mode="w",
                nodata=NO_DATA,
                photometric="MINISBLACK",
                tiled=True,
                transform=rasterio.transform.from_origin(
                    columns.start * PIXEL_DEGREES,
                    rows.stop * PIXEL_DEGREES,
                    PIXEL_DEGREES,
                    PIXEL_DEGREES,
                ),
                width=len(columns),
            ) as dataset,
        ):
            for cell_idx, (latitude, longitude) in enumerate(lat_lons, start=1):
                logger.info(
                    f"Thresholding source cell {cell_idx:d} of {len(lat_lons):d} "
                    f"({latitude=:d}, {longitude=:d}) onto {tile_id=:s}"
                )
                source_columns = get_pixel_range(
                    start_degrees=longitude, stop_degrees=longitude + 1
                )
                source_rows = get_pixel_range(
                    start_degrees=latitude, stop_degrees=latitude + 1
                )
                window = rasterio.windows.Window(
                    col_off=source_columns.start - columns.start,
                    height=len(source_rows),
                    row_off=rows.stop - source_rows.stop,
                    width=len(source_columns),
                )
                year_to_mask: dict[int, numpy.ndarray] = {}
                for year in YEARS:
                    (uri,) = sources[
                        (sources["lat"] == latitude)
                        & (sources["lng"] == longitude)
                        & (sources["year"] == year)
                    ]["path"]
                    with rasterio.open(fp=storage.to_gdal_path(uri=uri)) as source:
                        left = round(source.transform.c / PIXEL_DEGREES)
                        top = round(source.transform.f / PIXEL_DEGREES)
                        # On the global grid, a whole number of pixels from (0, 0)
                        assert source.transform.almost_equals(
                            rasterio.transform.from_origin(
                                left * PIXEL_DEGREES,
                                top * PIXEL_DEGREES,
                                PIXEL_DEGREES,
                                PIXEL_DEGREES,
                            ),
                            precision=1e-12,
                        )
                        assert (
                            left <= source_columns.start
                            and source_columns.stop <= left + source.width
                        )
                        assert (
                            top - source.height <= source_rows.start
                            and source_rows.stop <= top
                        )
                        # Neighbouring sources share a column or row, which only one of them reads
                        probability = source.read(
                            1,
                            window=rasterio.windows.Window(
                                col_off=source_columns.start - left,
                                height=len(source_rows),
                                row_off=top - source_rows.stop,
                                width=len(source_columns),
                            ),
                        )
                    year_to_mask[year] = get_mask(
                        admin_level_to_geometries=admin_level_to_geometries,
                        excluded_admin_ids=excluded_admin_ids,
                        floor=floor,
                        iso_3166_year_to_threshold=iso_3166_year_to_threshold,
                        probability=probability,
                        producer_iso_3166s=producer_iso_3166s,
                        transform=rasterio.windows.transform(
                            transform=dataset.transform, window=window
                        ),
                        year=year,
                    )
                year_to_smoothed_mask = get_smoothed_masks(year_to_mask=year_to_mask)
                for band_idx, year in enumerate(YEARS, start=1):
                    dataset.write(year_to_smoothed_mask[year], band_idx, window=window)

    return base.RasterDataset(
        band_names=[f"{year=:d}" for year in YEARS],
        # Averaged rather than taken by mode, so each harmonized pixel keeps the share of it masked
        band_type=base.BandType.INTENSIVE,
        dtype="uint8",
        no_data=NO_DATA,
        partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
        product_name=f"{commodity:s}-{RELEASE:s}",
        save_tile_id_to_local_path=save_tile_id_to_local_path,
        source_name="fdap",
        version=version,
    )


# https://github.com/google/forest-data-partnership/tree/4d0fb3b8ec24f4788b9d46289cb19ad07f466ae0/models/model_2026a#model-selection-set
COCOA_PRECISION_RECALL_CROSSOVER = 0.54
COCOA_DATASET = get_raster_dataset(
    commodity=Commodity.COCOA,
    excluded_admin_ids={
        worldbank_jurisdictions.AdminLevel.NATIONAL: set(),
        worldbank_jurisdictions.AdminLevel.PROVINCIAL: {
            # Semi-arid, and agronomically unsuitable for cocoa
            "BRA002",  # Alagoas
            "BRA006",  # Ceará
            "BRA019",  # Paraíba
            "BRA021",  # Pernambuco
            "BRA022",  # Piauí
            "BRA024",  # Rio Grande do Norte
            # The detected signal is consistent with babaçu palm
            "BRA010",  # Maranhão
            # Young oil palm and rubber
            "IDN004",  # Riau
            "IDN005",  # Jambi
            "IDN009",  # Kepulauan Bangka Belitung
            "IDN022",  # Kalimantan Selatan
            "IDN024",  # Kalimantan Utara
            # No cocoa production
            "ECU020",  # Galápagos
        },
        worldbank_jurisdictions.AdminLevel.DISTRICT: set(),
    },
    floor=COCOA_PRECISION_RECALL_CROSSOVER,
    iso_3166_year_to_threshold={
        ("BRA", 2018): 0.923,
        ("BRA", 2020): 0.946,
        ("BRA", 2022): 0.976,
        ("BRA", 2024): 0.976,
        ("CIV", 2018): 0.972,
        ("CIV", 2020): 0.982,
        ("CIV", 2022): 0.969,
        ("CIV", 2024): 0.961,
        ("COL", 2018): 0.919,
        ("COL", 2020): 0.851,
        ("COL", 2022): 0.986,
        ("COL", 2024): 0.971,
        ("ECU", 2018): 0.862,
        ("ECU", 2020): 0.862,
        ("ECU", 2022): 0.862,
        ("ECU", 2024): 0.812,
        ("GHA", 2018): 0.972,
        ("GHA", 2020): HIGHEST_THRESHOLD,
        ("GHA", 2022): HIGHEST_THRESHOLD,
        ("GHA", 2024): HIGHEST_THRESHOLD,
        ("IDN", 2018): COCOA_PRECISION_RECALL_CROSSOVER,
        ("IDN", 2020): COCOA_PRECISION_RECALL_CROSSOVER,
        ("IDN", 2022): COCOA_PRECISION_RECALL_CROSSOVER,
        ("IDN", 2024): COCOA_PRECISION_RECALL_CROSSOVER,
        ("PER", 2018): 0.558,
        ("PER", 2020): 0.866,
        ("PER", 2022): 0.91,
        ("PER", 2024): 0.812,
    },
    # FAOSTAT's producers that devote at least 1% of their cropland to cocoa, plus the countries
    # with thresholds: `tools/fdap-producers.sql` with item = '661'
    producer_iso_3166s={
        "BRA",
        "CIV",
        "CMR",
        "COG",
        "COL",
        "CRI",
        "DMA",
        "DOM",
        "ECU",
        "FSM",
        "GHA",
        "GNQ",
        "GRD",
        "IDN",
        "LBR",
        "LCA",
        "NGA",
        "PAN",
        "PER",
        "PNG",
        "SLB",
        "SLE",
        "STP",
        "TTO",
        "VCT",
        "VEN",
        "VUT",
        "WSM",
    },
    version="v0",
)

# https://github.com/google/forest-data-partnership/tree/4d0fb3b8ec24f4788b9d46289cb19ad07f466ae0/models/model_2026a#model-selection-set
COFFEE_PRECISION_RECALL_CROSSOVER = 0.33
COFFEE_DATASET = get_raster_dataset(
    commodity=Commodity.COFFEE,
    excluded_admin_ids={
        worldbank_jurisdictions.AdminLevel.NATIONAL: set(),
        worldbank_jurisdictions.AdminLevel.PROVINCIAL: {
            # Likely confused with other cropland or natural grassland
            "BRA010",  # Maranhão
            "BRA011",  # Mato Grosso
            "BRA012",  # Mato Grosso do Sul
            "BRA018",  # Pará
            "BRA022",  # Piauí
            "BRA031",  # Tocantins
            # Confused with tea
            "IDN012",  # Jawa Barat
            # Confused with natural ecosystems
            "PER014",  # Lambayeque
        },
        worldbank_jurisdictions.AdminLevel.DISTRICT: set(),
    },
    floor=COFFEE_PRECISION_RECALL_CROSSOVER,
    iso_3166_year_to_threshold={
        ("BRA", 2018): 0.554,
        ("BRA", 2020): 0.658,
        ("BRA", 2022): 0.763,
        ("BRA", 2024): 0.593,
        ("CIV", 2018): 0.002,
        ("CIV", 2020): 0.005,
        ("CIV", 2022): 0.007,
        ("CIV", 2024): 0.008,
        ("COL", 2018): 0.285,
        ("COL", 2020): 0.196,
        ("COL", 2022): 0.845,
        ("COL", 2024): 0.674,
        ("ETH", 2018): 0.019,
        ("ETH", 2020): 0.187,
        ("ETH", 2022): 0.965,
        ("ETH", 2024): 0.705,
        ("GTM", 2018): 0.134,
        ("GTM", 2020): 0.083,
        ("GTM", 2022): 0.128,
        ("GTM", 2024): 0.073,
        ("HND", 2018): 0.189,
        ("HND", 2020): 0.205,
        ("HND", 2022): 0.524,
        ("HND", 2024): 0.202,
        ("IDN", 2018): 0.679,
        ("IDN", 2020): 0.337,
        ("IDN", 2022): 0.309,
        ("IDN", 2024): 0.752,
        ("IND", 2018): 0.055,
        ("IND", 2020): 0.191,
        ("IND", 2022): 0.316,
        ("IND", 2024): 0.131,
        ("NIC", 2018): 0.317,
        ("NIC", 2020): 0.172,
        ("NIC", 2022): 0.223,
        ("NIC", 2024): 0.124,
        ("PER", 2018): 0.024,
        ("PER", 2020): 0.081,
        ("PER", 2022): 0.337,
        ("PER", 2024): 0.063,
        ("SLV", 2018): 0.005,
        ("SLV", 2020): 0.002,
        ("SLV", 2022): 0.024,
        ("SLV", 2024): 0.006,
        ("UGA", 2018): 0.751,
        ("UGA", 2020): 0.741,
        ("UGA", 2022): 0.979,
        ("UGA", 2024): 0.956,
        ("VNM", 2018): 0.208,
        ("VNM", 2020): 0.049,
        ("VNM", 2022): 0.52,
        ("VNM", 2024): 0.296,
    },
    # FAOSTAT's producers that devote at least 1% of their cropland to coffee, plus the countries
    # with thresholds: `tools/fdap-producers.sql` with item = '656'
    producer_iso_3166s={
        "BRA",
        "CAF",
        "CIV",
        "CMR",
        "COG",
        "COL",
        "CRI",
        "DMA",
        "DOM",
        "ECU",
        "ETH",
        "GIN",
        "GNQ",
        "GTM",
        "HND",
        "IDN",
        "IND",
        "JAM",
        "KEN",
        "LAO",
        "MDG",
        "MEX",
        "NIC",
        "PAN",
        "PER",
        "PHL",
        "PNG",
        "PRI",
        "RWA",
        "SLV",
        "TGO",
        "TLS",
        "TTO",
        "TZA",
        "UGA",
        "VCT",
        "VEN",
        "VNM",
        "YEM",
    },
    version="v0",
)
