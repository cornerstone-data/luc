"""Global Land Analysis and Discovery | Global Land Cover and Land Use Change, 2000-2020

license: CC BY 4.0

year: 2000, 2005, 2010, 2015, 2020

Potapov P., Hansen M.C., Pickens A., Hernandez-Serna A., Tyukavina A., Turubanova S., Zalles V., Li X., Khan A., Stolle F., Harris N., Song X.-P., Baggett A., Kommareddy I., Kommareddy A. (2022) The global 2000-2020 land cover and land use change dataset derived from the Landsat archive: first results. Frontiers in Remote Sensing https://doi.org/10.3389/frsen.2022.856903

https://glad.umd.edu/dataset/GLCLUC2020

# Methodology

- Assemblage of five independent models with covariates from [GLAD ARD](https://glad.umd.edu/ard/home): 16-day composites of landsat from 1997 to present; 30m grid with 8 uint16 bands
    1. [Forest](https://doi.org/10.1016/j.rse.2020.112165): labels from GEDI RH95; random forest regressor; separate model for each 1° tile
    2. [Croplands](https://doi.org/10.1038/s43016-021-00429-z): labels manually curated from Google Earth Engine (bootstrapped in three phases); random forest classifier; across 1° tiles
    3. [Urban](https://doi.org/10.1088/1748-9326/ac46ec): labels from OpenStreetMap; U-Net CNN classifier; random 128x128 pixel patches
    4. [Water](https://doi.org/10.1016/j.rse.2020.111792): labels manually curated from Landsat (bootstrapped in five phases); random forest classifier; engineered bag of landsat spectral indices, SEM
    5. Snow: labels manually curated; random forest classifier; separate model for each geographical region
"""

import enum
import itertools
import logging
import os
import tempfile

import rasterio

from jdluc import tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


YEARS = (2000, 2005, 2010, 2015, 2020)
BAND_NAMES = [f"{year=:d}" for year in YEARS]


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        year_to_geotiff: dict[int, str] = {}
        year_path: str | None = None
        for year in YEARS:
            year_to_geotiff[year] = year_path = os.path.join(tmpdir, f"{year:d}.tif")
            utils.save_remote_url_to_local_path(
                local_path=year_path,
                params={},
                remote_url=f"https://storage.googleapis.com/earthenginepartners-hansen/GLCLU2000-2020/v2/{year:d}/{tile_id:s}.tif",
            )
        assert year_path is not None
        with rasterio.open(year_path) as dataset:
            profile = dataset.meta.copy()
        profile.update(count=len(year_to_geotiff))
        logger.info(f"Combining {YEARS=:} into separate bands for one GeoTIFF")
        with rasterio.open(fp=local_path, mode="w", **profile) as dataset:
            for idx, (_, year_path) in enumerate(
                sorted(year_to_geotiff.items()), start=1
            ):
                with rasterio.open(fp=year_path) as source:
                    dataset.write(source.read(1), idx)


DATASET = base.RasterDataset(
    band_names=BAND_NAMES,
    band_type=base.BandType.CATEGORICAL,
    dtype="uint8",
    no_data=(1 << 8) - 1,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="glcluc",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="glad",
    version="v0",
)


class LandClass(enum.IntEnum):
    BUILT_UP = enum.auto()
    CROPLAND = enum.auto()
    FOREST = enum.auto()
    GRASSLAND = enum.auto()
    OCEAN = enum.auto()
    SNOW_ICE = enum.auto()
    WATER = enum.auto()


def flatten_ranges(*ranges: range) -> list[int]:
    return list(itertools.chain.from_iterable(ranges))


LAND_CLASS_TO_VALUES: dict[LandClass, list[int]] = {
    # https://storage.googleapis.com/earthenginepartners-hansen/GLCLU2000-2020/legend.xlsx
    LandClass.BUILT_UP: [250],
    LandClass.CROPLAND: [244],
    LandClass.FOREST: (flatten_ranges(range(25, 49), range(125, 149))),
    # NB: although the POC considers 0 as bareground, we include it as grassland here
    LandClass.GRASSLAND: (flatten_ranges(range(25), range(100, 125))),
    LandClass.OCEAN: [254],
    LandClass.SNOW_ICE: [241],
    LandClass.WATER: flatten_ranges(range(200, 208)),
}
assert set(LandClass) == set(LAND_CLASS_TO_VALUES)
assert all(
    set(left).isdisjoint(set(right))
    for left, right in itertools.combinations(
        (values for _, values in LAND_CLASS_TO_VALUES.items()), r=2
    )
)
assert DATASET.no_data not in set(
    itertools.chain.from_iterable(LAND_CLASS_TO_VALUES.values())
)
