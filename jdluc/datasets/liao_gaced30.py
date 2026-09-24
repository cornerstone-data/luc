"""Liao et al. | Global 30-m Annual Cropland Extent Dynamics (GACED30), 2000-2024

license: CC BY 4.0

year: 2000, 2001, ..., 2024

Liao, Y., Chen, S., Bai, Y., Wang, J., and Gong, P. (2026). Global 30-m annual cropland extent dynamics (2000-2024): A consistent baseline of structural evolution and regional disparities. Earth System Science Data Discussions [preprint]. https://doi.org/10.5194/essd-2025-838
Chen, S., Liao, Y., Bai, Y., Wang, J., and Gong, P. (2026). Global 30-m Annual Cropland Extent Dynamics (GACED30), v1 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18199675 -- the raster ingested here, whose creator order differs from the paper's

https://zenodo.org/records/18199675

# Methodology

- Covariates from the gap-free SDC30 30 m Landsat time series
- CatBoost classifier, with a spectral-semantic sample alignment step to reconcile disagreeing cropland sample sets, then spatiotemporal post-processing
- A rule-based step separates active fallow from natural bareland
- Aligned to the FAO Cropland definition: permanent woody crops, active fallow and agricultural structures are in, temporary meadows and pastures are out. GLAD GLCLUC excludes perennial woody crops, so the two disagree by definition as well as by mapping error
- Reported overall accuracy 96.5%, and R^2 0.95 against FAO national cropland area

Every band is one year, valued 0 for non-cropland and 10 for cropland against a no-data of 255. The
product is binary: it separates cropland from everything else, not one crop from another.
"""

import enum
import logging
import os
import shutil
import tempfile
import urllib.parse
import zipfile

import numpy
import rasterio
import rasterio.enums
import rasterio.transform
import rasterio.warp

from jdluc import tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


YEARS = tuple(range(2000, 2025))
BAND_NAMES = [f"{year=:d}" for year in YEARS]


class Cropland(enum.IntEnum):
    NOT_CROPLAND = 0
    CROPLAND = 10


TILE_DEGREES = 3
ARCHIVE_DEGREES = 6
SOURCE_PIXELS = 11_132
PIXELS_PER_DEGREE = tiling.TileResolution.GLAD.pixels_per_degree
# A source tile spans 3.0000137 degrees, not 3, so it reaches one destination pixel further
TILE = tiling.XY(
    x=TILE_DEGREES * PIXELS_PER_DEGREE.x + 1,
    y=TILE_DEGREES * PIXELS_PER_DEGREE.y + 1,
).validated()


def get_transform(max_lat: float, min_lon: float) -> rasterio.Affine:
    return rasterio.transform.from_origin(
        min_lon, max_lat, *tiling.TileResolution.GLAD.degrees_per_pixel
    )


# Whole archives, cached for the process. Fetching only a tile's own bytes is worse: 3 does not
# divide 10, so each source tile serves two to four destination tiles and would be re-fetched.
@utils.threadsafe_cache
def get_path_to_archive_for_low_longitude(low_longitude: int) -> str:
    # The published names sign both bounds, as in lon_-006_to_+000.zip
    name = urllib.parse.quote(
        f"lon_{low_longitude:+04d}_to_{low_longitude + ARCHIVE_DEGREES:+04d}.zip"
    )
    # NB: deliberately not a TemporaryDirectory, so that the archive outlives this call
    path_to_archive = os.path.join(tempfile.mkdtemp(), name)
    utils.save_remote_url_to_local_path(
        local_path=path_to_archive,
        params={},
        remote_url=f"https://zenodo.org/api/records/18199675/files/{name:s}/content",
    )
    return path_to_archive


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    max_latitude, min_longitude = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        lat_lon_to_path: dict[tuple[int, int], str] = {}
        for longitude in range(
            min_longitude // TILE_DEGREES * TILE_DEGREES,
            min_longitude + 10,
            TILE_DEGREES,
        ):
            with zipfile.ZipFile(
                file=get_path_to_archive_for_low_longitude(
                    low_longitude=longitude // ARCHIVE_DEGREES * ARCHIVE_DEGREES
                )
            ) as archive:
                names = set(archive.namelist())
                for latitude in range(
                    (max_latitude - 10) // TILE_DEGREES * TILE_DEGREES,
                    max_latitude,
                    TILE_DEGREES,
                ):
                    if (
                        name := f"Crop_{longitude:.1f}_{latitude:.1f}.tif"
                    ) not in names:
                        # Cells that are entirely ocean are absent, not published as no-data
                        logger.info(f"Skipping {name=:s}, which is not published")
                        continue
                    path = os.path.join(tmpdir, name)
                    logger.info(f"Extracting {name=:s} to {path=:s}")
                    with archive.open(name=name) as source, open(path, "wb") as fp:
                        shutil.copyfileobj(source, fp)
                    lat_lon_to_path[(latitude, longitude)] = path
        assert lat_lon_to_path, f"{tile_id=:s} is covered by no published source tile"

        logger.info(
            f"Combining {len(lat_lon_to_path):d} source tiles into {len(YEARS):d} bands"
        )
        with rasterio.open(
            BIGTIFF="YES",
            compress="deflate",
            count=len(YEARS),
            crs=4326,
            driver="GTiff",
            dtype="uint8",
            fp=local_path,
            height=tiling.TileResolution.GLAD.y,
            interleave="band",
            mode="w",
            nodata=DATASET.no_data,
            tiled=True,
            transform=get_transform(max_lat=max_latitude, min_lon=min_longitude),
            width=tiling.TileResolution.GLAD.x,
        ) as dataset:
            for band_idx, year in enumerate(YEARS, start=1):
                logger.info(f"Resampling {year=:d} onto {tile_id=:s}")
                # Uncovered pixels must read as no-data, not as 0 (= non-cropland): only 118 of
                # the 280 tiles are covered by all sixteen of their source tiles
                band = numpy.full(
                    dtype="uint8",
                    fill_value=DATASET.no_data,
                    shape=(tiling.TileResolution.GLAD.y, tiling.TileResolution.GLAD.x),
                )
                for (latitude, longitude), path in sorted(lat_lon_to_path.items()):
                    # Each source tile is anchored on whole degrees at its north-west corner,
                    # so its offset onto the 4000-pixel-per-degree destination is exact
                    left = (longitude - min_longitude) * PIXELS_PER_DEGREE.x
                    top = (max_latitude - latitude - TILE_DEGREES) * PIXELS_PER_DEGREE.y
                    col_offset, row_offset = max(0, left), max(0, top)
                    width = (
                        min(
                            tiling.TileResolution.GLAD.x,
                            left + TILE.x,
                        )
                        - col_offset
                    )
                    height = (
                        min(tiling.TileResolution.GLAD.y, top + TILE.y) - row_offset
                    )

                    # reproject cannot write into a strided view, so fill a patch and copy it in
                    patch = numpy.full(
                        dtype="uint8", fill_value=DATASET.no_data, shape=(height, width)
                    )
                    with rasterio.open(fp=path) as source:
                        # The offsets above are hardcoded to this grid, so pin it
                        assert source.width == source.height == SOURCE_PIXELS
                        assert (source.bounds.left, source.bounds.top) == (
                            longitude,
                            latitude + TILE_DEGREES,
                        )
                        assert source.descriptions[band_idx - 1] == f"Crop_{year:d}"
                        rasterio.warp.reproject(
                            destination=patch,
                            dst_crs=source.crs,
                            dst_nodata=DATASET.no_data,
                            dst_transform=get_transform(
                                max_lat=max_latitude - row_offset / PIXELS_PER_DEGREE.y,
                                min_lon=min_longitude
                                + col_offset / PIXELS_PER_DEGREE.x,
                            ),
                            resampling=rasterio.enums.Resampling.nearest,
                            source=rasterio.band(source, band_idx),
                            src_nodata=DATASET.no_data,
                        )
                    # Source tiles overlap by their overhang, so copy only where this one has
                    # data: a plain assignment would let its no-data erase a neighbour's cropland
                    numpy.copyto(
                        casting="no",
                        dst=band[
                            row_offset : row_offset + height,
                            col_offset : col_offset + width,
                        ],
                        src=patch,
                        where=patch != DATASET.no_data,
                    )
                dataset.write(band, band_idx)


DATASET = base.RasterDataset(
    band_names=BAND_NAMES,
    band_type=base.BandType.CATEGORICAL,
    dtype="uint8",
    no_data=(1 << 8) - 1,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="gaced30",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="liao",
    version="v0",
)
assert DATASET.no_data not in set(Cropland)
