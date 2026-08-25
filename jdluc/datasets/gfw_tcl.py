"""Global Forest Watch | Hansen Global Forest Change — tree-cover loss year

license: CC BY 4.0

year: 2000 baseline; annual gross loss 2001-2025

Hansen, M. C. et al. High-Resolution Global Maps of 21st-Century Forest Cover Change. Science 342, 850-853 (2013). Updated through 2025 (GFC-2025-v1.13).

https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/download.html

The `lossyear` band encodes the year of gross tree-cover loss: 0 = no loss, N in 1..25 = loss in calendar year 2000 + N (2001-2025).
"""

from jdluc import tiling, utils
from jdluc.datasets import base

LOSS_YEAR_OFFSET = 2000


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    utils.save_remote_url_to_local_path(
        local_path=local_path,
        params={},
        remote_url=(
            "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/"
            f"Hansen_GFC-2025-v1.13_lossyear_{tile_id:s}.tif"
        ),
    )


DATASET = base.RasterDataset(
    band_names=["lossyear"],
    band_type=base.BandType.CATEGORICAL,
    no_data=None,  # 0 = "no loss" is meaningful, not absent; there is no fill value
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="tree-cover-loss",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="gfw",
    # v1: include 2025
    version="v1",
)
