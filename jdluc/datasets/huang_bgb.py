"""Huang et al. | A global map of root biomass across the world's forests

license: CC BY 4.0

year: ~2010 (the year of the main covariate AGB)

Huang, Y., Ciais, P., Santoro, M., et al. (2021). A global map of root biomass across the world's forests. Earth System Science Data, 13, 4263-4274. https://doi.org/10.5194/essd-13-4263-2021
Huang, Yuanyuan; Ciais, Phillipe; Santoro, Maurizio; Makowski, David; Chave, Jerome; Schepaschenko, Dmitry; et al. (2020). Supporting data and code for A global map of root biomass across the world's forests. figshare. Dataset. https://doi.org/10.6084/m9.figshare.12199637.v1

https://essd.copernicus.org/articles/13/4263/2021/

# Methodology

- Labels from 10k field measurements across 465 species and 10 biomes
- Random forest regressor
- 47 covariates: shoot biomass, height, soil properties, climate aggregates, water table properties
"""

import logging
import os
import tempfile
import zipfile

import numpy
import xarray

from jdluc import tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


# All tiles share one (expensive) download, so cache it
@utils.threadsafe_cache
def _get_dataarray() -> xarray.DataArray:
    with tempfile.TemporaryDirectory() as local_dir:
        path_to_zip = os.path.join(local_dir, "data.zip")
        utils.save_remote_url_to_local_path(
            local_path=path_to_zip,
            params={},
            remote_url="https://ndownloader.figshare.com/files/22432460",
        )
        logger.info(f"Extracting .nc from {path_to_zip=:s}")
        path_to_nc: str | None = None
        with zipfile.ZipFile(file=path_to_zip) as zf:
            for name in zf.namelist():
                if name.endswith("pergridarea_bgb.nc"):
                    path_to_nc = zf.extract(name, local_dir)
                    break
        assert path_to_nc is not None
        logger.info(f"Loading {path_to_nc=:s} into xarray")
        ds: xarray.Dataset = xarray.open_dataset(path_to_nc)
        return (
            ds.rio.set_spatial_dims(x_dim="LON", y_dim="LAT")
            .rio.write_crs("EPSG:4326")["AROOT"]
            .fillna(0)
            .rio.write_nodata(numpy.nan)
            .load()
        )


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    da = _get_dataarray()
    max_lat, min_lon = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
    tiled = (
        da
        # Clip to the tile
        .sel(LAT=slice(max_lat - 10, max_lat), LON=slice(min_lon, min_lon + 10))
        # We need to flipud since the array is provided with y increasing
        .isel(LAT=slice(None, None, -1))
    )
    logger.info(f"Saving xarray to {local_path=:s} for {tile_id=:s}")
    tiled.rio.to_raster(local_path, driver="GTiff", recalc_transform=True)


DATASET = base.RasterDataset(
    band_names=["belowground-biomass-mg-per-ha"],
    band_type=base.BandType.INTENSIVE,
    no_data=None,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="bgb",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="huang",
    version="v0",
)
