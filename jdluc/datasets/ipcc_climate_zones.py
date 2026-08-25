"""Intergovernmental Panel on Climate Change | Climate Zones

license: CC BY 4.0

year: ~static

Lewis, M. (2022). IPCC Climate Zones (from the 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories), version 0.1.0 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7303808
Calvo Buendia, E et al. (2019). 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories. IPCC, Switzerland. -- the decision tree the raster implements.

https://zenodo.org/records/7303808

# Methodology

- decision tree heuristically tuned into 12 classes
- elevation, temperature, precipitation, potential evapotransporation, frost occurrence
"""

import enum

from jdluc import tiling, utils
from jdluc.datasets import base


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    return utils.save_remote_url_to_local_path(
        local_path=local_path,
        params={},
        remote_url="https://zenodo.org/records/7303808/files/IPCC_Climate_Zones_ts_3.25.tif?download=1",
    )


DATASET = base.RasterDataset(
    band_names=["climate-zone"],
    band_type=base.BandType.CATEGORICAL,
    no_data=(1 << 8) - 1,
    partitioning=tiling.Partitioning.WHOLE_WORLD,
    product_name="climate-zones",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="ipcc",
    version="v0",
)


class Zone(enum.IntEnum):
    TROPICAL_MONTANE = enum.auto()
    TROPICAL_WET = enum.auto()
    TROPICAL_MOIST = enum.auto()
    TROPICAL_DRY = enum.auto()
    WARM_TEMPERATE_MOIST = enum.auto()
    WARM_TEMPERATE_DRY = enum.auto()
    COOL_TEMPERATE_MOIST = enum.auto()
    COOL_TEMPERATE_DRY = enum.auto()
    BOREAL_MOIST = enum.auto()
    BOREAL_DRY = enum.auto()
