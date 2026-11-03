"""Tang et al. | CROPGRIDS, crop area of 173 crops at 0.05 degrees, 2020

license: CC BY 4.0

year: 2020

Tang, F. H. M., Nguyen, T. H., Conchedda, G., Casse, L., Tubiello, F. N., and Maggi, F. (2024). CROPGRIDS: a global geo-referenced dataset of 173 crops. Scientific Data 11, 413. https://doi.org/10.1038/s41597-024-03247-7
Tang, F. H. M., Nguyen, T. H., Conchedda, G., Casse, L., Tubiello, F. N., and Maggi, F. (2024). CROPGRIDS, v1.08 [Data set]. figshare. https://doi.org/10.6084/m9.figshare.22491997.v9

# Methodology

- Combines 27 gridded crop maps (Monfreda, MapSPAM, GAEZ+, GEOGLAM, and single-crop maps of crops such as oil palm and rice), subnational statistics for 52 countries, and FAOSTAT's national totals for 2020
- Gives two areas per crop: harvested area, and crop area, the land the crop covers. Where an input map gives only harvested area, crop area is set equal to it for permanent crops; for temporary crops it is also limited by the cell's land area and by a 30 m cropland agreement map built from six satellite products

The figshare zip has one netCDF file per crop. Each file holds harvested area, crop area, a
data-quality score and the input dataset used, all float32 with -1 over the ocean. Only crop area
is kept: one band per crop, in hectares, on the source grid. The source rows run from south to
north, so they are flipped to run from north to south.
"""

import enum
import logging
import os
import tempfile
import zipfile

import numpy
import rasterio
import rasterio.transform
import xarray

from jdluc import tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


@enum.unique
class Crop(enum.StrEnum):
    ABACA = enum.auto()
    AGAVE = enum.auto()
    ALFALFA = enum.auto()
    ALMOND = enum.auto()
    ANISEETC = enum.auto()
    APPLE = enum.auto()
    APRICOT = enum.auto()
    ARECA = enum.auto()
    ARTICHOKE = enum.auto()
    ASPARAGUS = enum.auto()
    AVOCADO = enum.auto()
    BAMBARA = enum.auto()
    BANANA = enum.auto()
    BARLEY = enum.auto()
    BEAN = enum.auto()
    BEETFOR = enum.auto()
    BERRYNES = enum.auto()
    BLUEBERRY = enum.auto()
    BRAZIL = enum.auto()
    BROADBEAN = enum.auto()
    BUCKWHEAT = enum.auto()
    CABBAGE = enum.auto()
    CABBAGEFOR = enum.auto()
    CANARYSEED = enum.auto()
    CAROB = enum.auto()
    CARROT = enum.auto()
    CARROTFOR = enum.auto()
    CASHEW = enum.auto()
    CASHEWAPPLE = enum.auto()
    CASSAVA = enum.auto()
    CASTOR = enum.auto()
    CAULIFLOWER = enum.auto()
    CEREALNES = enum.auto()
    CHERRY = enum.auto()
    CHESTNUT = enum.auto()
    CHICKPEA = enum.auto()
    CHICORY = enum.auto()
    CHILLEETC = enum.auto()  # (sic) chillietc
    CINNAMON = enum.auto()
    CITRUSNES = enum.auto()
    CLOVE = enum.auto()
    CLOVER = enum.auto()
    COCOA = enum.auto()
    COCONUT = enum.auto()
    COFFEE = enum.auto()
    COTTON = enum.auto()
    COWPEA = enum.auto()
    CRANBERRY = enum.auto()
    CUCUMBERETC = enum.auto()
    CURRANT = enum.auto()
    DATE = enum.auto()
    EGGPLANT = enum.auto()
    FIBRENES = enum.auto()
    FIG = enum.auto()
    FLAX = enum.auto()
    FONIO = enum.auto()
    FORNES = enum.auto()
    FRUITNES = enum.auto()
    GARLIC = enum.auto()
    GINGER = enum.auto()
    GOOSEBERRY = enum.auto()
    GRAPE = enum.auto()
    GRAPEFRUITETC = enum.auto()
    GRASSNES = enum.auto()
    GREENBEAN = enum.auto()
    GREENBROADBEAN = enum.auto()
    GREENCORN = enum.auto()
    GREENONION = enum.auto()
    GREENPEA = enum.auto()
    GROUNDNUT = enum.auto()
    HAZELNUT = enum.auto()
    HEMP = enum.auto()
    HEMPSEED = enum.auto()
    HOP = enum.auto()
    JUTE = enum.auto()
    JUTELIKEFIBER = enum.auto()
    KAPOKFIBER = enum.auto()
    KAPOKSEED = enum.auto()
    KARITE = enum.auto()
    KIWI = enum.auto()
    KOLANUT = enum.auto()
    LEGUMENES = enum.auto()
    LEMONLIME = enum.auto()
    LENTIL = enum.auto()
    LETTUCE = enum.auto()
    LINSEED = enum.auto()
    LUPIN = enum.auto()
    MAIZE = enum.auto()
    MAIZEFOR = enum.auto()
    MANGO = enum.auto()
    MATE = enum.auto()
    MELONETC = enum.auto()
    MELONSEED = enum.auto()
    MILLET = enum.auto()
    MIXEDGRAIN = enum.auto()
    MIXEDGRASS = enum.auto()
    MUSHROOM = enum.auto()
    MUSTARD = enum.auto()
    NUTMEG = enum.auto()
    NUTNES = enum.auto()
    OATS = enum.auto()
    OILPALM = enum.auto()
    OILSEEDFOR = enum.auto()
    OILSEEDNES = enum.auto()
    OKRA = enum.auto()
    OLIVE = enum.auto()
    ONION = enum.auto()
    ORANGE = enum.auto()
    PAPAYA = enum.auto()
    PEA = enum.auto()
    PEACHETC = enum.auto()
    PEAR = enum.auto()
    PEPPER = enum.auto()
    PEPPERMINT = enum.auto()
    PERSIMMON = enum.auto()
    PIGEONPEA = enum.auto()
    PIMENTO = enum.auto()
    PINEAPPLE = enum.auto()
    PISTACHIO = enum.auto()
    PLANTAIN = enum.auto()
    PLUM = enum.auto()
    POPCORN = enum.auto()
    POPPY = enum.auto()
    POTATO = enum.auto()
    PULSENES = enum.auto()
    PUMPKINETC = enum.auto()
    PYRETHRUM = enum.auto()
    QUINCE = enum.auto()
    QUINOA = enum.auto()
    RAMIE = enum.auto()
    RAPESEED = enum.auto()
    RASBERRY = enum.auto()  # (sic) raspberry
    RICE = enum.auto()
    ROOTNES = enum.auto()
    RUBBER = enum.auto()
    RYE = enum.auto()
    RYEFOR = enum.auto()
    SAFFLOWER = enum.auto()
    SESAME = enum.auto()
    SISAL = enum.auto()
    SORGHUM = enum.auto()
    SORGHUMFOR = enum.auto()
    SOURCHERRY = enum.auto()
    SOYBEAN = enum.auto()
    SPICENES = enum.auto()
    SPINACH = enum.auto()
    STONEFRUITNES = enum.auto()
    STRAWBERRY = enum.auto()
    STRINGBEAN = enum.auto()
    SUGARBEET = enum.auto()
    SUGARCANE = enum.auto()
    SUGARNES = enum.auto()
    SUNFLOWER = enum.auto()
    SWEDEFOR = enum.auto()
    SWEETPOTATO = enum.auto()
    TANGETC = enum.auto()
    TARO = enum.auto()
    TEA = enum.auto()
    TOBACCO = enum.auto()
    TOMATO = enum.auto()
    TRITICALE = enum.auto()
    TROPICALNES = enum.auto()
    TUNG = enum.auto()
    TURNIPFOR = enum.auto()
    VANILLA = enum.auto()
    VEGETABLENES = enum.auto()
    VEGFOR = enum.auto()
    VETCH = enum.auto()
    WALNUT = enum.auto()
    WATERMELON = enum.auto()
    WHEAT = enum.auto()
    YAM = enum.auto()
    YAUTIA = enum.auto()


RESOLUTION = tiling.XY(x=7_200, y=3_600)
YEAR = 2020


def save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    assert tile_id == tiling.WHOLE_WORLD_TILE_ID
    with tempfile.TemporaryDirectory() as local_dir:
        path_to_zip = os.path.join(local_dir, "data.zip")
        utils.save_remote_url_to_local_path(
            local_path=path_to_zip,
            params={},
            remote_url="https://ndownloader.figshare.com/files/44950942",
        )
        with (
            zipfile.ZipFile(file=path_to_zip) as archive,
            rasterio.open(
                BIGTIFF="IF_SAFER",
                compress="deflate",
                count=len(Crop),
                crs=4326,
                driver="GTiff",
                dtype=DATASET.dtype,
                fp=local_path,
                height=RESOLUTION.y,
                interleave="band",
                mode="w",
                nodata=DATASET.no_data,
                tiled=True,
                transform=rasterio.transform.from_origin(
                    -180, 90, 360 / RESOLUTION.x, 180 / RESOLUTION.y
                ),
                width=RESOLUTION.x,
            ) as dataset,
        ):
            for band_idx, crop in enumerate(Crop, start=1):
                logger.info(f"Reading the crop area of {crop=:s}")
                path_to_netcdf = archive.extract(
                    member=f"CROPGRIDSv1.08_NC_maps/CROPGRIDSv1.08_{crop:s}.nc",
                    path=local_dir,
                )
                # NB: no masking, so the ocean stays -1
                with xarray.open_dataset(
                    filename_or_obj=path_to_netcdf, mask_and_scale=False
                ) as source:
                    # The source's rows run from south to north; a GeoTIFF's run from north to south
                    crop_area = source["croparea"].sortby(
                        ascending=False, variables="lat"
                    )
                    assert crop_area.dims == ("lat", "lon")
                    # Cell centers
                    assert numpy.allclose(
                        crop_area["lat"],
                        numpy.linspace(90, -90, endpoint=False, num=RESOLUTION.y)
                        - 180 / RESOLUTION.y / 2,
                    )
                    assert numpy.allclose(
                        crop_area["lon"],
                        numpy.linspace(-180, 180, endpoint=False, num=RESOLUTION.x)
                        + 360 / RESOLUTION.x / 2,
                    )
                    hectares = crop_area.values
                assert hectares.dtype == DATASET.dtype
                assert numpy.all((hectares >= 0) | (hectares == DATASET.no_data))
                dataset.write(hectares, band_idx)


DATASET = base.RasterDataset(
    band_names=[f"{crop:s}:ha" for crop in Crop],
    band_type=base.BandType.EXTENSIVE,
    dtype="float32",
    no_data=-1,
    partitioning=tiling.Partitioning.WHOLE_WORLD,
    product_name="cropgrids-crop-area",
    save_tile_id_to_local_path=save_tile_id_to_local_path,
    source_name="tang",
    version="v0",
)
