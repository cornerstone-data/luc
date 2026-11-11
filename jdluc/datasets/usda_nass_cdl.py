"""USDA National Agricultural Statistics Service | Cropland Data Layer

license: public domain

year: 2008, ..., 2025

United States Department of Agriculture (USDA) National Agricultural Statistics Service (NASS), 20260227, Cropland Data Layer: USDA NASS, USDA NASS Marketing and Information Services Office, Washington, D.C.

https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php

# Methodology

- Labels from "extensive agricultural ground truth"
- Random forest classifier
- Many remote sensing layers: HLSS, HLSL, NED(3DEP)
"""

import enum
import logging
import os
import tempfile
import zipfile

import rasterio
import rasterio.enums
import rasterio.transform
import rasterio.vrt
import rasterio.windows

from jdluc import tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


YEARS = tuple(range(2020, 2026))
BAND_NAMES = [f"{year=:d}" for year in YEARS]


# A shorter warp reads the source more times; a taller one holds more of the tile in memory
DESTINATION_BLOCK_ROWS = 2048


# All tiles share each year's (expensive) download, so cache it
@utils.threadsafe_cache
def get_path_to_geotiff_for_year(year: int) -> str:
    """The path to `year`'s source GeoTIFF, extracted from its zip to a local directory.

    The zip's external overviews stay behind, so a warp can only read full resolution.
    """
    # NB: deliberately not a TemporaryDirectory, so that the GeoTIFF outlives this call
    local_dir = tempfile.mkdtemp()
    path_to_zip = os.path.join(local_dir, "data.zip")
    utils.save_remote_url_to_local_path(
        local_path=path_to_zip,
        params={},
        remote_url=f"https://www.nass.usda.gov/sites/default/files/Research_and_Science/Cropland/Release/datasets/{year:d}_30m_cdls.zip",
    )
    logger.info(f"Extracting {year=:d}'s .tif from {path_to_zip=:s}")
    basename = f"{year:d}_30m_cdls.tif"
    with zipfile.ZipFile(file=path_to_zip) as zf:
        zf.extract(basename, local_dir)
    os.remove(path_to_zip)
    return os.path.join(local_dir, basename)


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    max_lat, min_lon = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
    transform = rasterio.transform.from_origin(
        min_lon, max_lat, *tiling.TileResolution.GLAD.degrees_per_pixel
    )
    with rasterio.open(
        BIGTIFF="YES",
        NUM_THREADS="ALL_CPUS",
        compress="deflate",
        count=len(YEARS),
        crs=4326,
        driver="GTiff",
        dtype=DATASET.dtype,
        fp=local_path,
        height=tiling.TileResolution.GLAD.y,
        interleave="band",
        mode="w",
        nodata=DATASET.no_data,
        tiled=True,
        transform=transform,
        width=tiling.TileResolution.GLAD.x,
    ) as dataset:
        for band_idx, year in enumerate(YEARS, start=1):
            path_to_geotiff = get_path_to_geotiff_for_year(year=year)
            logger.info(f"Warping {path_to_geotiff=:s} onto {tile_id=:s}")
            with (
                rasterio.open(fp=path_to_geotiff) as source,
                rasterio.vrt.WarpedVRT(
                    source,
                    crs=4326,
                    height=tiling.TileResolution.GLAD.y,
                    nodata=DATASET.no_data,
                    resampling=rasterio.enums.Resampling.nearest,
                    src_nodata=DATASET.no_data,
                    transform=transform,
                    width=tiling.TileResolution.GLAD.x,
                ) as warped,
            ):
                for top in range(
                    0, tiling.TileResolution.GLAD.y, DESTINATION_BLOCK_ROWS
                ):
                    window = rasterio.windows.Window(
                        col_off=0,
                        height=min(
                            DESTINATION_BLOCK_ROWS, tiling.TileResolution.GLAD.y - top
                        ),
                        row_off=top,
                        width=tiling.TileResolution.GLAD.x,
                    )
                    dataset.write(
                        warped.read(1, window=window), band_idx, window=window
                    )


DATASET = base.RasterDataset(
    band_names=BAND_NAMES,
    band_type=base.BandType.CATEGORICAL,
    dtype="uint8",
    no_data=0,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="cdl",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="usda-nass",
    # v1: reproject data so that when it doesn't cover a full tile it doesn't get stretched
    # v2: a band per year, 2020-2025, each warped straight from the source projection onto the tile
    version="v2",
)


@enum.unique
class CropClass(enum.IntEnum):
    # https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.7
    # NO DATA, BACKGROUND 0
    BACKGROUND = 0
    # CROPS 1-60
    CORN = 1
    COTTON = 2
    RICE = 3
    SORGHUM = 4
    SOYBEANS = 5
    SUNFLOWER = 6
    PEANUTS = 10
    TOBACCO = 11
    SWEET_CORN = 12
    POP_OR_ORN_CORN = 13
    MINT = 14
    BARLEY = 21
    DURUM_WHEAT = 22
    SPRING_WHEAT = 23
    WINTER_WHEAT = 24
    OTHER_SMALL_GRAINS = 25
    DBL_CROP_WINWHT_SOYBEANS = 26
    RYE = 27
    OATS = 28
    MILLET = 29
    SPELTZ = 30
    CANOLA = 31
    FLAXSEED = 32
    SAFFLOWER = 33
    RAPE_SEED = 34
    MUSTARD = 35
    ALFALFA = 36
    OTHER_HAY_NON_ALFALFA = 37
    CAMELINA = 38
    BUCKWHEAT = 39
    SUGARBEETS = 41
    DRY_BEANS = 42
    POTATOES = 43
    OTHER_CROPS = 44
    SUGARCANE = 45
    SWEET_POTATOES = 46
    MISC_VEGS_AND_FRUITS = 47
    WATERMELONS = 48
    ONIONS = 49
    CUCUMBERS = 50
    CHICK_PEAS = 51
    LENTILS = 52
    PEAS = 53
    TOMATOES = 54
    CANEBERRIES = 55
    HOPS = 56
    HERBS = 57
    CLOVER_WILDFLOWERS = 58
    SOD_GRASS_SEED = 59
    SWITCHGRASS = 60
    # NON-CROP 61-65
    FALLOW_IDLE_CROPLAND = 61
    PASTURE_GRASS = 62
    FOREST = 63
    SHRUBLAND = 64
    BARREN = 65
    # CROPS 66-80
    CHERRIES = 66
    PEACHES = 67
    APPLES = 68
    GRAPES = 69
    CHRISTMAS_TREES = 70
    OTHER_TREE_CROPS = 71
    CITRUS = 72
    PECANS = 74
    ALMONDS = 75
    WALNUTS = 76
    PEARS = 77
    # OTHER 81-109
    CLOUDS_NO_DATA = 81
    DEVELOPED = 82
    WATER = 83
    WETLANDS = 87
    NONAG_UNDEFINED = 88
    AQUACULTURE = 92
    # NLCD-DERIVED CLASSES 110-195
    OPEN_WATER = 111
    PERENNIAL_ICE_SNOW = 112
    DEVELOPED_OPEN_SPACE = 121
    DEVELOPED_LOW_INTENSITY = 122
    DEVELOPED_MED_INTENSITY = 123
    DEVELOPED_HIGH_INTENSITY = 124
    BARREN_NLCD = 131
    DECIDUOUS_FOREST = 141
    EVERGREEN_FOREST = 142
    MIXED_FOREST = 143
    SHRUBLAND_NLCD = 152
    GRASSLAND_PASTURE = 176
    WOODY_WETLANDS = 190
    HERBACEOUS_WETLANDS = 195
    # CROPS 195-255
    PISTACHIOS = 204
    TRITICALE = 205
    CARROTS = 206
    ASPARAGUS = 207
    GARLIC = 208
    CANTALOUPES = 209
    PRUNES = 210
    OLIVES = 211
    ORANGES = 212
    HONEYDEW_MELONS = 213
    BROCCOLI = 214
    AVOCADOS = 215
    PEPPERS = 216
    POMEGRANATES = 217
    NECTARINES = 218
    GREENS = 219
    PLUMS = 220
    STRAWBERRIES = 221
    SQUASH = 222
    APRICOTS = 223
    VETCH = 224
    DBL_CROP_WINWHT_CORN = 225
    DBL_CROP_OATS_CORN = 226
    LETTUCE = 227
    DBL_CROP_TRITICALE_CORN = 228
    PUMPKINS = 229
    DBL_CROP_LETTUCE_DURUM_WHT = 230
    DBL_CROP_LETTUCE_CANTALOUPE = 231
    DBL_CROP_LETTUCE_COTTON = 232
    DBL_CROP_LETTUCE_BARLEY = 233
    DBL_CROP_DURUM_WHT_SORGHUM = 234
    DBL_CROP_BARLEY_SORGHUM = 235
    DBL_CROP_WINWHT_SORGHUM = 236
    DBL_CROP_BARLEY_CORN = 237
    DBL_CROP_WINWHT_COTTON = 238
    DBL_CROP_SOYBEANS_COTTON = 239
    DBL_CROP_SOYBEANS_OATS = 240
    DBL_CROP_CORN_SOYBEANS = 241
    BLUEBERRIES = 242
    CABBAGE = 243
    CAULIFLOWER = 244
    CELERY = 245
    RADISHES = 246
    TURNIPS = 247
    EGGPLANTS = 248
    GOURDS = 249
    CRANBERRIES = 250
    DBL_CROP_BARLEY_SOYBEANS = 254
