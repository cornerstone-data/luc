"""Global Pasture Watch | Annual grassland class and extent, 2000-2024

license: CC BY 4.0

year: 2000, 2001, ..., 2024

Parente, L., Sloat, L., Mesquita, V., Consoli, D., Stanimirova, R., Hengl, T., Bonannella, C., Teles, N., Wheeler, I., Hunter, M., Ehrmann, S., Ferreira, L., Mattos, A. P., Oliveira, B., Meyer, C., Şahin, M., Witjes, M., Fritz, S., Malek, Z., and Stolle, F. (2024). Annual 30-m maps of global grassland class and extent (2000-2022) based on spatiotemporal Machine Learning. Scientific Data 11, 1303. https://doi.org/10.1038/s41597-024-04139-6 -- describes v1
Parente, L. et al. (2025). Global Pasture Watch - Annual grassland class and extent maps at 30-m spatial resolution (2000-2024), v2-beta [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13890400 -- the concept DOI, resolving to the latest of the several records this release is split across

https://github.com/wri/global-pasture-watch

NB: Global Pasture Watch, not SEDAC's Gridded Population of the World, which shares the abbreviation.

# Methodology

- Random Forest over 197 features from Landsat GLAD ARD, MODIS, terrain and accessibility, trained
  on 2.3M very-high-resolution image interpretations, then a temporal median filter (`med.filt`) and
  a balanced probability threshold (`bthr`) that equalizes precision and recall
- Five-fold spatially blocked cross-validation gives F1 0.64 for cultivated and 0.76 for
  natural/semi-natural. Both are v1 figures: open shrubland is new in v2-beta and unvalidated
- Extent is any land cover at least 30% low vegetation under 3 m, with at most 50% tree canopy, 70%
  other woody vegetation and 50% active cropland -- that last clause makes it overlap a cropland map
  by construction, not only through mapping error

Each band is one year of the dominant class against a no-data of 255: 0 other land cover,
1 cultivated grassland, 2 natural/semi-natural, 3 open shrubland; extent is the union of the three
non-zero classes, and 0 is not resolved further.

The mosaic is one global COG per year whose pixel edges fall on whole degrees at 4000 pixels per
degree -- the GLAD grid -- so a ten-degree tile is a windowed read at an integer offset.
"""

import enum
import logging

import rasterio
import rasterio.transform
import rasterio.windows

from jdluc import tiling
from jdluc.datasets import base

logger = logging.getLogger(__name__)


YEARS = tuple(range(2000, 2025))
BAND_NAMES = [f"{year=:d}" for year in YEARS]


class Grassland(enum.IntEnum):
    OTHER = 0
    CULTIVATED = 1
    NATURAL = 2
    OPEN_SHRUBLAND = 3


# The mosaic's north-west corner sits half a pixel outside (-179, 76), but its pixel edges still fall
# on whole degrees -- two pixels in from that corner on each axis
SOURCE_ORIGIN = tiling.XY(x=-179, y=76)
SOURCE_ORIGIN_OFFSET = tiling.XY(x=2, y=2)


# The sources are tiled at 2048; a shorter read fetches whole blocks and discards most of them
SOURCE_BLOCK_ROWS = 2048


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    max_latitude, min_longitude = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
    pixels_per_degree = tiling.TileResolution.GLAD.pixels_per_degree
    # NB: not validated(), because an edge tile's offset is legitimately negative
    source_offset = tiling.XY(
        x=(min_longitude - SOURCE_ORIGIN.x) * pixels_per_degree.x
        + SOURCE_ORIGIN_OFFSET.x,
        y=(SOURCE_ORIGIN.y - max_latitude) * pixels_per_degree.y
        + SOURCE_ORIGIN_OFFSET.y,
    )
    with (
        rasterio.Env(
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        ),
        rasterio.open(
            BIGTIFF="YES",
            NUM_THREADS="ALL_CPUS",
            compress="deflate",
            count=len(YEARS),
            crs=rasterio.CRS.from_epsg(4326),
            driver="GTiff",
            dtype="uint8",
            fp=local_path,
            height=tiling.TileResolution.GLAD.y,
            interleave="band",
            mode="w",
            nodata=DATASET.no_data,
            tiled=True,
            transform=rasterio.transform.from_origin(
                min_longitude,
                max_latitude,
                *tiling.TileResolution.GLAD.degrees_per_pixel,
            ),
            width=tiling.TileResolution.GLAD.x,
        ) as dataset,
    ):
        for band_idx, year in enumerate(YEARS, start=1):
            logger.info(f"Windowing {year=:d} onto {tile_id=:s}")
            url = (
                "https://s3.eu-central-1.wasabisys.com/arco/"
                f"gpw_grassland_rf.med.filt.bthr_c_30m_{year:d}0101_{year:d}1231_go_epsg.4326_v2.tif"
            )
            with rasterio.open(fp=url) as source:
                # Iterate over blocks in the source TIF to bound RAM usage
                for top in range(0, tiling.TileResolution.GLAD.y, SOURCE_BLOCK_ROWS):
                    height = min(SOURCE_BLOCK_ROWS, tiling.TileResolution.GLAD.y - top)
                    dataset.write(
                        source.read(
                            1,
                            boundless=True,
                            fill_value=DATASET.no_data,
                            window=rasterio.windows.Window(
                                col_off=source_offset.x,
                                height=height,
                                row_off=source_offset.y + top,
                                width=tiling.TileResolution.GLAD.x,
                            ),
                        ),
                        band_idx,
                        window=rasterio.windows.Window(
                            col_off=0,
                            height=height,
                            row_off=top,
                            width=tiling.TileResolution.GLAD.x,
                        ),
                    )


DATASET = base.RasterDataset(
    band_names=BAND_NAMES,
    band_type=base.BandType.CATEGORICAL,
    dtype="uint8",
    no_data=(1 << 8) - 1,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="grassland",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="gpw",
    version="v0",
)
assert DATASET.no_data not in set(Grassland)
