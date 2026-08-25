"""Place one ingested source tile per dataset onto a common grid and harmonize them.

For a single ten-degree tile, each raster dataset's tile becomes a per-band GDAL VRT on the
shared grid, returned as an xarray.Dataset with one variable per source band. Whole-world
datasets are windowed onto the same grid. The result is cached.

The tile set comes from the positional ISO 3166 alpha-3 codes -- or, with `--backfill`,
from `tiling.GLOBAL_FOREST_WATCH_TILE_IDS`.

Example invocations:
  uv run python jdluc/harmonize.py USA
  uv run python jdluc/harmonize.py --backfill
"""

import argparse
import collections.abc
import dataclasses
import logging
import tempfile
import typing

import numpy
import rasterio
import rasterio.enums
import rasterio.errors
import rioxarray
import xarray

from jdluc import config, geo, ingest, storage, tiling
from jdluc.datasets import NAME_TO_CLS, DatasetName, base, worldbank_jurisdictions

logger = logging.getLogger(__name__)


RIO_TO_GDAL_DTYPE = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


@dataclasses.dataclass
class Tile:
    band_type: base.BandType
    dtype: str
    no_data: float | int | None
    resolution: tiling.XY
    uri: str

    @classmethod
    def from_dataset_tile_id(
        cls, dataset: base.RasterDataset, root: str, tile_id: str
    ) -> typing.Self:
        uri = storage.join_uri(root=root, prefix=dataset.get_prefix(tile_id=tile_id))
        with rasterio.open(fp=uri) as ds:
            rio_dtype = next(iter(ds.dtypes))
            return cls(
                band_type=dataset.band_type,
                dtype=RIO_TO_GDAL_DTYPE[rio_dtype],
                no_data=ds.nodata,
                resolution=tiling.XY(x=ds.width, y=ds.height).validated(),
                uri=uri,
            )

    @property
    def gdal_path(self) -> str:
        return storage.to_gdal_path(uri=self.uri)


def iter_vrt_band_header(
    band_name: str, dtype: str, no_data: int | float | None
) -> collections.abc.Iterator[str]:
    yield f'  <VRTRasterBand dataType="{dtype:s}" band="1">'
    yield f"    <Description>{band_name:s}</Description>"
    if no_data is not None:
        yield f"    <NoDataValue>{no_data}</NoDataValue>"


def iter_vrt_band_content(
    band_idx: int,
    dest_resolution: tiling.XY,
    path_to_tile: str,
    resampling: rasterio.enums.Resampling,
    src_offset: tiling.XY,
    src_resolution: tiling.XY,
) -> collections.abc.Iterator[str]:
    yield f'    <SimpleSource resampling="{resampling.name:}">'
    yield f'      <SourceFilename relativeToVRT="0">{path_to_tile:s}</SourceFilename>'
    yield f"      <SourceBand>{band_idx:d}</SourceBand>"
    yield f'      <SrcRect xOff="{src_offset.x:d}" yOff="{src_offset.y:d}" xSize="{src_resolution.x:d}" ySize="{src_resolution.y:d}"/>'
    # NB: a VRT covers exactly one tile, so the destination rect always starts at the origin
    yield f'      <DstRect xOff="0" yOff="0" xSize="{dest_resolution.x:d}" ySize="{dest_resolution.y:d}"/>'
    yield "    </SimpleSource>"


@dataclasses.dataclass
class Grid:
    origin: tiling.XY
    resolution: tiling.XY

    @property
    def epsg(self) -> int:
        return 4326

    @classmethod
    def from_tile_id_resolution(
        cls, tile_id: str, resolution: tiling.XY
    ) -> typing.Self:
        lat, lon = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
        return cls(
            origin=tiling.XY(x=lon, y=lat),
            # NB: sanitize the provided resolution to the class we want
            resolution=tiling.XY(x=resolution.x, y=resolution.y),
        )

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.origin.x,
            10 / self.resolution.x,
            0,
            self.origin.y,
            0,
            -10 / self.resolution.y,
        )

    @property
    def iter_preamble(self) -> collections.abc.Iterator[str]:
        yield f'<VRTDataset rasterXSize="{self.resolution.x:d}" rasterYSize="{self.resolution.y:d}">'
        yield f"  <SRS>EPSG:{self.epsg:d}</SRS>"
        yield f"  <GeoTransform>{', '.join(map(str, self.transform))}</GeoTransform>"

    def get_offset_for_world(self, resolution: tiling.XY, span: tiling.XY) -> tiling.XY:
        pixels_per_degree = tiling.XY(
            x=resolution.x // span.x,
            y=resolution.y // span.y,
        ).validated()
        return tiling.XY(
            x=(self.origin.x + span.x // 2) * pixels_per_degree.x,
            y=(span.y // 2 - self.origin.y) * pixels_per_degree.y,
        ).validated()

    def get_resolution_for_world(
        self, resolution: tiling.XY, span: tiling.XY
    ) -> tiling.XY:
        return tiling.XY(
            x=resolution.x * 10 // span.x,
            y=resolution.y * 10 // span.y,
        ).validated()

    @staticmethod
    def get_resampling_for_band_type(
        band_type: base.BandType,
        dest_resolution: tiling.XY,
        src_resolution: tiling.XY,
    ) -> rasterio.enums.Resampling:
        if src_resolution == dest_resolution:
            return rasterio.enums.Resampling.nearest
        elif (
            src_resolution.x > dest_resolution.x
            and src_resolution.y > dest_resolution.y
        ):
            match band_type:
                case base.BandType.CATEGORICAL:
                    return rasterio.enums.Resampling.mode
                case base.BandType.EXTENSIVE:
                    raise NotImplementedError("GDAL doesn't implement sum resampling")
                case base.BandType.INTENSIVE:
                    return rasterio.enums.Resampling.average
                case _:
                    raise ValueError(band_type)
        else:
            match band_type:
                case base.BandType.CATEGORICAL:
                    return rasterio.enums.Resampling.nearest
                case base.BandType.EXTENSIVE:
                    raise NotImplementedError(
                        "GDAL doesn't implement distribution resampling"
                    )
                case base.BandType.INTENSIVE:
                    return rasterio.enums.Resampling.bilinear
                case _:
                    raise ValueError(band_type)


def get_vrt_for_dataset_band_tile_id(
    band_idx: int,
    band_name: str,
    dataset: base.RasterDataset,
    grid: Grid,
    ignore_missing_tiles: bool,
    root: str,
    tile_id: str,
) -> str:
    lines = list(grid.iter_preamble)

    logger.info(f"Processing {dataset=} and {band_name=:s}")
    if dataset.partitioning == tiling.Partitioning.TEN_DEGREE_TILE:
        try:
            tile = Tile.from_dataset_tile_id(
                root=root,
                dataset=dataset,
                tile_id=tile_id,
            )
        except rasterio.errors.RasterioIOError:
            if ignore_missing_tiles:
                logger.warning(
                    f"{tile_id=:s} is missing for {dataset=} but due to {ignore_missing_tiles=} we are emitting an all-no-data band"
                )
                # Yield an empty tile band
                lines.extend(
                    iter_vrt_band_header(
                        band_name=band_name, dtype="Float32", no_data=dataset.no_data
                    )
                )
            else:
                raise
        else:
            lines.extend(
                iter_vrt_band_header(
                    band_name=band_name,
                    dtype=tile.dtype,
                    no_data=tile.no_data,
                )
            )
            lines.extend(
                iter_vrt_band_content(
                    band_idx=band_idx,
                    dest_resolution=grid.resolution,
                    path_to_tile=tile.gdal_path,
                    resampling=grid.get_resampling_for_band_type(
                        band_type=tile.band_type,
                        dest_resolution=grid.resolution,
                        src_resolution=tile.resolution,
                    ),
                    src_offset=tiling.XY(x=0, y=0),
                    src_resolution=tile.resolution,
                )
            )
    elif dataset.partitioning == tiling.Partitioning.WHOLE_WORLD:
        tile = Tile.from_dataset_tile_id(
            root=root,
            dataset=dataset,
            tile_id=tiling.WHOLE_WORLD_TILE_ID,
        )
        lines.extend(
            iter_vrt_band_header(
                band_name=band_name,
                dtype=tile.dtype,
                no_data=tile.no_data,
            )
        )
        world_span = tiling.XY(x=360, y=180)
        src_offset = grid.get_offset_for_world(
            resolution=tile.resolution, span=world_span
        )
        src_resolution = grid.get_resolution_for_world(
            resolution=tile.resolution, span=world_span
        )
        lines.extend(
            iter_vrt_band_content(
                band_idx=band_idx,
                dest_resolution=grid.resolution,
                path_to_tile=tile.gdal_path,
                resampling=grid.get_resampling_for_band_type(
                    band_type=tile.band_type,
                    src_resolution=src_resolution,
                    dest_resolution=grid.resolution,
                ),
                src_offset=src_offset,
                src_resolution=src_resolution,
            )
        )
    else:
        raise ValueError(dataset.partitioning)
    lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".vrt") as fp:
        logger.debug(f"Writing VRT to {fp.name=:s}")
        fp.writelines(line + "\n" for line in lines)
        path_to_vrt = fp.name
    return path_to_vrt


def get_dset_for_output(path_to_vrts: collections.abc.Sequence[str]) -> xarray.Dataset:
    darrays: list[xarray.DataArray] = []
    chunk_size = geo.get_chunk_size(dtypes=[numpy.dtype("float32")] * len(path_to_vrts))
    for path_to_vrt in path_to_vrts:
        logger.debug(f"Opening {path_to_vrt=:s} with {chunk_size=:d}")
        darray = rioxarray.open_rasterio(
            filename=path_to_vrt,
            chunks=chunk_size,
            # Remove the serialization lock because this is read-only
            lock=False,
        )
        assert isinstance(darray, xarray.DataArray)
        darray = darray.isel(band=0, drop=True)
        darrays.append(
            geo.unify_dtype_and_no_data(
                darray=darray.rename(darray.attrs.pop("long_name"))
            )
        )

    return xarray.Dataset({darray.name: darray for darray in darrays})


@storage.cache_to_zarr(version=0, ignored_args=["ignore_missing_tiles", "skip_ingest"])
def workflow(
    dataset_names: tuple[DatasetName, ...],
    ignore_missing_tiles: bool,
    skip_ingest: bool,
    tile_id: str,
    tile_resolution: tiling.XY,
) -> xarray.Dataset:
    logger.info(
        f"Running the harmonize workflow for {dataset_names=:} and {tile_id=:s}"
    )
    datasets = list(map(NAME_TO_CLS.__getitem__, dataset_names))
    cfg = config.Config.from_dot_env()

    if not skip_ingest:
        for dataset in datasets:
            # NB: this doesn't take advantage of ingest's concurrency, so
            # consider running ingest over the AOI beforehand
            ingest.workflow(
                concurrency=1,
                dataset=dataset,
                overwrite=False,
                root=cfg.ingest_root,
                tile_ids=(tile_id,),
            )

    logger.info(f"Constructing common grid for {tile_resolution=:}")
    grid = Grid.from_tile_id_resolution(tile_id=tile_id, resolution=tile_resolution)
    path_to_vrts = [
        get_vrt_for_dataset_band_tile_id(
            band_idx=band_idx,
            band_name=band_name,
            root=cfg.ingest_root,
            dataset=dataset,
            grid=grid,
            ignore_missing_tiles=ignore_missing_tiles,
            tile_id=tile_id,
        )
        for dataset in datasets
        if isinstance(dataset, base.RasterDataset)
        for band_idx, band_name in enumerate(
            dataset.fully_qualified_band_names, start=1
        )
    ]
    return get_dset_for_output(path_to_vrts=path_to_vrts)


LUC_AND_EMISSIONS_DATASET_NAMES = (
    DatasetName.GFW_GLOBAL_PEATLANDS,
    DatasetName.GFW_HARRIS_AGB,
    DatasetName.GLAD_GLCLUC,
    DatasetName.HUANG_BGB,
    DatasetName.IPCC_CLIMATE_ZONES,
    DatasetName.SOILGRIDS_OCS,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "iso_3166s",
        help="cover exactly the tiles these countries' boundaries touch",
        nargs=argparse.ZERO_OR_MORE,
        type=worldbank_jurisdictions.iso_3166_str,
    )
    parser.add_argument("--backfill", action="store_true", help="cover all GFW tiles")
    parser.add_argument(
        "--grid-name",
        choices=sorted(e.name for e in tiling.TileResolution),
        default=tiling.TileResolution.GLAD.name,
        type=str,
    )
    parser.add_argument("--ignore-missing-tiles", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()
    assert bool(args.iso_3166s) ^ bool(args.backfill), (
        "pass either one-or-more iso_3166s or --backfill"
    )

    for tile_id in sorted(
        tiling.GLOBAL_FOREST_WATCH_TILE_IDS
        if args.backfill
        else worldbank_jurisdictions.get_ten_degree_tile_ids_for_iso_3166s(
            iso_3166s=args.iso_3166s
        )
    ):
        workflow(
            dataset_names=LUC_AND_EMISSIONS_DATASET_NAMES,
            ignore_missing_tiles=args.ignore_missing_tiles,
            skip_ingest=args.skip_ingest,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution[str(args.grid_name)],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
