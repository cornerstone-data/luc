"""Global Forest Watch | Global Peatlands

license: CC BY 4.0

year: ~static

- Crezee, B. et al. Mapping peat thickness and carbon stocks of the central Congo Basin using field data. Nature Geoscience 15: 639-644 (2022). https://www.nature.com/articles/s41561-022-00966-7. Data downloaded from https://congopeat.net/maps/, using classes 4 and 5 only (peat classes).
- Gumbricht, T. et al. An expert system model for mapping tropical wetlands and peatlands reveals South America as the largest contributor. Global Change Biology 23, 3581-3599 (2017). https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.13689
- Hastie, A. et al. Risks to carbon storage from land-use change revealed by peat thickness maps of Peru. Nature Geoscience 15: 369-374 (2022). https://www.nature.com/articles/s41561-022-00923-4
- Miettinen, J., Shi, C. & Liew, S. C. Land cover distribution in the peatlands of Peninsular Malaysia, Sumatra and Borneo in 2015 with changes since 1990. Global Ecology and Conservation. 6, 67-78 (2016). https://www.sciencedirect.com/science/article/pii/S2351989415300470
- Xu et al. PEATMAP: Refining estimates of global peatland distribution based on a meta-analysis. CATENA 160: 134-140 (2018). https://www.sciencedirect.com/science/article/pii/S0341816217303004

https://data.globalforestwatch.org/datasets/gfw::global-peatlands
"""

import urllib.parse

from jdluc import tiling, utils
from jdluc.datasets import base


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    params_dict = {
        "grid": "10/40000",
        "pixel_meaning": "is",
        # This is a public API key
        "x-api-key": "2d60cd88-8348-4c0f-a6d5-bd9adb585a8c",
    } | {"tile_id": tile_id}
    return utils.save_remote_url_to_local_path(
        local_path=local_path,
        params=urllib.parse.urlencode(params_dict, safe="/"),
        remote_url="https://data-api.globalforestwatch.org/dataset/gfw_peatlands/v20230315/download/geotiff",
    )


DATASET = base.RasterDataset(
    band_names=["is-peatland"],
    band_type=base.BandType.CATEGORICAL,
    no_data=(1 << 8) - 1,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="global-peatlands",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="gfw",
    version="v0",
)
