"""SoilGrids | Organic carbon stocks

license: CC BY 4.0

year: 2000-2010 (the years of covariates)

Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., and Rossiter, D.: SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty, SOIL, 7, 217-240, https://doi.org/10.5194/soil-7-217-2021, 2021.

https://gee-community-catalog.org/projects/isric/

# Methodology

- Labels from 196k soil profiles
- Quantile random forest regressors
- Climate, ecology, geology, LULC, SEM, landsat, MODIS, hydrography
"""

import urllib.parse

from jdluc import tiling, utils
from jdluc.datasets import base


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    max_lat, min_lon = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
    params_tuples = (
        ("COVERAGEID", "ocs_0-30cm_mean"),
        ("FORMAT", "image/tiff"),
        ("map", "/map/ocs.map"),
        ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
        ("REQUEST", "GetCoverage"),
        ("SERVICE", "WCS"),
        ("SUBSET", f"lat({max_lat - 10:d},{max_lat:d})"),
        ("SUBSET", f"long({min_lon:d},{min_lon + 10:d})"),
        ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
        ("VERSION", "2.0.1"),
    )
    return utils.save_remote_url_to_local_path(
        local_path=local_path,
        params=urllib.parse.urlencode(params_tuples, safe="/:(),"),
        remote_url="https://maps.isric.org/mapserv",
    )


DATASET = base.RasterDataset(
    band_names=["organic-soil-carbon-mg-per-ha"],
    band_type=base.BandType.INTENSIVE,
    dtype="int16",
    no_data=(1 << 15) - 1,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="organic-carbon-stocks",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="soilgrids",
    version="v0",
)
