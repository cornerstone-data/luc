"""Global Forest Watch | Aboveground Live Woody Biomass Density

license: CC BY 4.0

year: 2000

Harris, N. L. et al. (2021). Global maps of twenty-first century forest carbon fluxes. Nature Climate Change, 11, 234-240. DOI 10.1038/s41558-020-00976-6.

https://data.globalforestwatch.org/datasets/gfw::aboveground-live-woody-biomass-density

# Methodology

- Labels from 637k ground plots and 707k LiDAR waveforms along with regional allometric equations
- Random forest regressor
- Engineered bag of landsat spectral indices, SEM, climate aggregates as covariates
"""

import urllib.parse

from jdluc import tiling, utils
from jdluc.datasets import base


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    params_dict = {
        "grid": "10/40000",
        "pixel_meaning": "Mg_ha-1",
        # This is a public API key
        "x-api-key": "2d60cd88-8348-4c0f-a6d5-bd9adb585a8c",
    } | {"tile_id": tile_id}
    return utils.save_remote_url_to_local_path(
        local_path=local_path,
        params=urllib.parse.urlencode(params_dict, safe="/"),
        remote_url="https://data-api.globalforestwatch.org/dataset/whrc_aboveground_woody_biomass_stock_2000/v1.4/download/geotiff",
    )


DATASET = base.RasterDataset(
    band_names=["aboveground-biomass-mg-per-ha"],
    band_type=base.BandType.INTENSIVE,
    no_data=(1 << 16) - 1,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="harris-agb",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="gfw",
    version="v0",
)
