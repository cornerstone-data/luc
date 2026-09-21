import collections.abc
import dataclasses
import enum
import logging
import os
import tempfile
import typing

import numpy
import pandas

from jdluc import geo, storage, tiling, utils

logger = logging.getLogger(__name__)


class AssetType(enum.StrEnum):
    @typing.override
    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")

    RASTER = enum.auto()
    VECTOR = enum.auto()
    TABULAR = enum.auto()


class BandType(enum.IntEnum):
    CATEGORICAL = enum.auto()
    EXTENSIVE = enum.auto()
    INTENSIVE = enum.auto()


SaveTileIdToLocalPathType = collections.abc.Callable[[str, str], None]


@dataclasses.dataclass
class RasterDataset:
    band_names: list[str]
    band_type: BandType
    dtype: str
    no_data: float | int | None
    partitioning: tiling.Partitioning
    product_name: str
    source_name: str
    version: str
    save_tile_id_to_local_path: SaveTileIdToLocalPathType = dataclasses.field(
        repr=False
    )

    def __post_init__(self) -> None:
        # Ensure the dtype is valid and canonical
        assert numpy.dtype(self.dtype).name == self.dtype

    def get_prefix(self, tile_id: str) -> str:
        return os.path.join(
            str(AssetType.RASTER),
            self.source_name,
            self.product_name,
            self.version,
            str(self.partitioning),
            f"{tile_id:s}.tif",
        )

    def ingest_a_tile(self, overwrite: bool, root: str, tile_id: str) -> str:
        # Stage a geotiff to a tmpdir, clean up and convert to COG, then write to a templated path
        uri = storage.join_uri(prefix=self.get_prefix(tile_id=tile_id), root=root)
        if overwrite or not storage.path_exists(uri=uri):
            with tempfile.TemporaryDirectory() as tmpdir:
                path_to_geotiff = os.path.join(tmpdir, "geotiff.tif")
                self.save_tile_id_to_local_path(path_to_geotiff, tile_id)
                geo.validate_geotiff(dtype=self.dtype, path_to_geotiff=path_to_geotiff)
                geo.set_band_names_for_geotiff(
                    band_names=self.band_names, path_to_geotiff=path_to_geotiff
                )
                path_to_cog = os.path.join(tmpdir, "cog.tif")
                geo.convert_geotiff_to_cog(
                    metadata={
                        "watershed-data-version": self.version,
                        "watershed-processing-time": utils.get_utc_timestamp(),
                        "watershed-processing-version": utils.get_git_version(),
                        "watershed-product-name": self.product_name,
                        "watershed-remote-url": utils.get_git_remote_url(),
                        "watershed-source-name": self.source_name,
                    },
                    no_data=self.no_data,
                    path_to_cog=path_to_cog,
                    path_to_geotiff=path_to_geotiff,
                )
                storage.put_file(local_path=path_to_cog, uri=uri)
        else:
            logger.info(
                f"Skipping writing of {tile_id=:s} because {uri=:s} exists and {overwrite=:}"
            )
        return uri

    @property
    def fully_qualified_band_names(self) -> list[str]:
        return [
            ":".join((self.source_name, self.product_name, band_name))
            for band_name in self.band_names
        ]

    @property
    def fully_qualified_band_name(self) -> str:
        (ret,) = self.fully_qualified_band_names
        return ret


@dataclasses.dataclass
class VectorDataset:
    id_column_names: tuple[str, ...]
    name_column_names: tuple[str, ...]
    product_name: str
    save_tile_id_to_local_path: SaveTileIdToLocalPathType
    source_name: str
    version: str

    @property
    def partitioning(self) -> tiling.Partitioning:
        return tiling.Partitioning.WHOLE_WORLD

    def get_prefix(self, tile_id: str) -> str:
        return os.path.join(
            str(AssetType.VECTOR),
            self.source_name,
            self.product_name,
            self.version,
            str(self.partitioning),
            f"{tile_id:s}.fgb",
        )

    def ingest_a_tile(self, overwrite: bool, root: str, tile_id: str) -> str:
        # Stage a vector file to a tmpdir, clean up and convert to flatgeobuf, then write to a templated path
        uri = storage.join_uri(prefix=self.get_prefix(tile_id=tile_id), root=root)
        if overwrite or not storage.path_exists(uri=uri):
            with tempfile.TemporaryDirectory() as tmpdir:
                path_to_vector = os.path.join(tmpdir, "vector.dat")
                self.save_tile_id_to_local_path(path_to_vector, tile_id)
                path_to_flatgeobuf = os.path.join(tmpdir, "vector.fgb")
                geo.convert_vector_to_flatgeobuf(
                    id_column_names=self.id_column_names,
                    name_column_names=self.name_column_names,
                    path_to_flatgeobuf=path_to_flatgeobuf,
                    path_to_vector=path_to_vector,
                )
                storage.put_file(local_path=path_to_flatgeobuf, uri=uri)
        else:
            logger.warning(
                f"Skipping writing of {tile_id=:s} because {uri=:s} exists and {overwrite=:}"
            )
        return uri


GetRecordsForTileIdType = typing.Callable[[str], list[dict[str, str | float]]]


@dataclasses.dataclass
class TabularDataset:
    get_records_for_tile_id: GetRecordsForTileIdType
    idx_column_names: list[str]
    product_name: str
    source_name: str
    version: str

    @property
    def partitioning(self) -> tiling.Partitioning:
        return tiling.Partitioning.WHOLE_WORLD

    def get_prefix(self, tile_id: str) -> str:
        return os.path.join(
            str(AssetType.TABULAR),
            self.source_name,
            self.product_name,
            self.version,
            str(self.partitioning),
            f"{tile_id:s}.parquet",
        )

    def ingest_a_tile(self, overwrite: bool, root: str, tile_id: str) -> str:
        uri = storage.join_uri(prefix=self.get_prefix(tile_id=tile_id), root=root)
        if overwrite or not storage.path_exists(uri=uri):
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "data.parquet")
                records = self.get_records_for_tile_id(tile_id)
                df = (
                    pandas.DataFrame.from_records(data=records)
                    .set_index(self.idx_column_names)
                    .sort_index()
                )
                df.to_parquet(path=local_path)
                storage.put_file(local_path=local_path, uri=uri)
        else:
            logger.warning(
                f"Skipping writing of {tile_id=:s} because {uri=:s} exists and {overwrite=:}"
            )
        return uri
