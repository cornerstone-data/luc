"""Descals et al. | Global oil palm extent and planting year, 1990-2021

license: CC BY 4.0

year: planting year per pixel, 1989-2022 (0 = no oil palm)

Descals, A., Gaveau, D. L. A., Wich, S., Szantoi, Z., and Meijaard, E. (2024). Global mapping of oil palm planting year from 1990 to 2021. Earth System Science Data 16, 5111-5129. https://doi.org/10.5194/essd-16-5111-2024
Descals, A. (2024). Global oil palm extent and planting year from 1990 to 2021, v1.2 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13379129 -- the record ingested here; 10.5281/zenodo.11034130 is the concept DOI, which resolves to whichever version is latest

https://zenodo.org/records/13379129

# Methodology

- A convolutional neural network over 2016-2021 Sentinel-1 composites maps oil palm at 10 m; the planting year comes from the Landsat time series at 30 m, where the clearing that precedes establishment shows up as a minimum in the normalized burn ratio
- Producer's and user's accuracy are 91.0% and 91.8% on industrial plantations but 71.4% and 72.4% on smallholders, and the planting year's RMSE against field data is 2.65 years
- The planting year dates the plantation standing in 2021, so a replant reads as the replant year rather than as the year the land was first cleared. It bounds when the current crop was established, not when conversion happened

Each uint16 pixel holds the planting year, 0 where there is no oil palm and 1989-2022 where there
is -- a year wider at each end than the advertised range, with 1989 saturating everything
established at or before it. 0 is a mapped class and not a gap, so there is no fill value.

The 609 published cells of 100 x 100 km sit on a lattice that divides neither the ten-degree
graticule nor a whole number of its own pixels, so a tile is resampled onto the GLAD grid rather
than cut from the cells -- nearest, since a planting year is categorical. Only 48 of the 280 tiles
overlap a cell at all, because the publisher maps one crop rather than the whole land surface.
"""

import logging
import os
import tempfile
import zipfile

import numpy
import rasterio
import rasterio.enums
import rasterio.transform
import rasterio.warp
import rasterio.windows
import shapely

from jdluc import tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


DESTINATION_BLOCK_ROWS = tiling.TileResolution.GLAD.pixels_per_degree.y // 2


@utils.threadsafe_cache
def get_path_to_archive() -> str:
    # NB: deliberately not a TemporaryDirectory, so that the archive outlives this call
    path_to_archive = os.path.join(tempfile.mkdtemp(), "data.zip")
    utils.save_remote_url_to_local_path(
        local_path=path_to_archive,
        params={},
        remote_url="https://zenodo.org/api/records/13379129/files/GlobalOilPalm_OP-YoP.zip/content",
    )
    return path_to_archive


@utils.threadsafe_cache
def get_member_to_box() -> dict[str, shapely.Polygon]:
    path_to_archive = get_path_to_archive()
    with zipfile.ZipFile(file=path_to_archive) as archive:
        members = sorted(archive.namelist())
    logger.info(f"Reading the extent of each of {len(members):d} source cells")
    member_to_box: dict[str, shapely.Polygon] = {}
    for member in members:
        with rasterio.open(fp=f"/vsizip/{path_to_archive:s}/{member:s}") as source:
            member_to_box[member] = shapely.box(
                xmax=source.bounds.right,
                xmin=source.bounds.left,
                ymax=source.bounds.top,
                ymin=source.bounds.bottom,
            )
    return member_to_box


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    max_latitude, min_longitude = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
    pixels_per_degree = tiling.TileResolution.GLAD.pixels_per_degree
    path_to_archive = get_path_to_archive()
    member_to_box = get_member_to_box()
    box = tiling.get_box_for_tile_id(tile_id=tile_id)
    members = sorted(
        member for member, cell in member_to_box.items() if cell.intersects(other=box)
    )
    assert members
    logger.info(f"Combining {len(members):d} source cells onto {tile_id=:s}")
    with rasterio.open(
        BIGTIFF="YES",
        NUM_THREADS="ALL_CPUS",
        compress="deflate",
        count=1,
        crs=rasterio.CRS.from_epsg(4326),
        driver="GTiff",
        dtype="uint16",
        fp=local_path,
        height=tiling.TileResolution.GLAD.y,
        mode="w",
        nodata=DATASET.no_data,
        tiled=True,
        transform=rasterio.transform.from_origin(
            min_longitude,
            max_latitude,
            *tiling.TileResolution.GLAD.degrees_per_pixel,
        ),
        width=tiling.TileResolution.GLAD.x,
    ) as dataset:
        for top in range(0, tiling.TileResolution.GLAD.y, DESTINATION_BLOCK_ROWS):
            height = min(DESTINATION_BLOCK_ROWS, tiling.TileResolution.GLAD.y - top)
            strip_max_latitude = max_latitude - top / pixels_per_degree.y
            strip_box = shapely.box(
                xmax=min_longitude + 10,
                xmin=min_longitude,
                ymax=strip_max_latitude,
                ymin=strip_max_latitude - height / pixels_per_degree.y,
            )
            strip_members = [
                member
                for member in members
                if member_to_box[member].intersects(other=strip_box)
            ]
            # NB: zeros = "no oil palm" which is semantically what we want for missing data
            strip = numpy.zeros(
                dtype="uint16", shape=(height, tiling.TileResolution.GLAD.x)
            )
            for member in strip_members:
                with rasterio.open(
                    fp=f"/vsizip/{path_to_archive:s}/{member:s}"
                ) as source:
                    rasterio.warp.reproject(
                        destination=strip,
                        dst_crs=4326,
                        # Map any missings to "no oil palm" (=0)
                        dst_nodata=0,
                        dst_transform=rasterio.transform.from_origin(
                            min_longitude,
                            strip_max_latitude,
                            *tiling.TileResolution.GLAD.degrees_per_pixel,
                        ),
                        # Keep what the cells already merged into this strip wrote
                        init_dest_nodata=False,
                        resampling=rasterio.enums.Resampling.nearest,
                        source=rasterio.band(source, 1),
                        # Cells overhang each other, so mask the zeros rather than let one cell's
                        # "no oil palm" erase its neighbour's planting year along the seam
                        src_nodata=0,
                    )
            dataset.write(
                strip,
                1,
                window=rasterio.windows.Window(
                    col_off=0,
                    height=height,
                    row_off=top,
                    width=tiling.TileResolution.GLAD.x,
                ),
            )


DATASET = base.RasterDataset(
    band_names=["planting-year"],
    band_type=base.BandType.CATEGORICAL,
    dtype="uint16",
    # NB: convention is that any areas not covered lack oil palm (=0) which we enforce; there is no NODATA
    no_data=None,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="oil-palm",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="descals",
    version="v0",
)
