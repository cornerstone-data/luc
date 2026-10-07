"""World Bank | Official Boundaries

license: CC BY 4.0

year: 2026

https://datacatalog.worldbank.org/search/dataset/0038272/world-bank-official-boundaries
"""

import collections.abc
import dataclasses
import enum
import functools
import logging
import os

import geopandas
import pandas
import shapely

from jdluc import config, storage, tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


class AdminLevel(enum.IntEnum):
    NATIONAL = 0
    PROVINCIAL = 1
    DISTRICT = 2


# Taiwan is its own country here, following CEDA's convention. The World Bank layers record it
# inconsistently: admin 1 gives Taiwan a unit of its own, `TWN001`, but files it under China, and
# admin 0 has no Taiwan at all. This fence separates Taiwan from China across the strait.
TAIWAN_FENCE = shapely.Polygon(
    shell=[(119.0, 21.5), (122.5, 21.5), (122.5, 25.9), (121.0, 25.9), (119.0, 24.0)]
)


def with_taiwan_carved_out_of_china(
    admin_0: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame:
    """The World Bank admin-0 layer with Taiwan as its own row, subtracted from CHN's polygon."""
    is_china = admin_0["ISO_A3"] == "CHN"
    china = admin_0[is_china]
    (china_geometry,) = china.geometry
    assert not china_geometry.intersects(TAIWAN_FENCE.exterior)

    china_without_taiwan = china.set_geometry(
        col=china.geometry.difference(TAIWAN_FENCE)
    )
    taiwan = geopandas.GeoDataFrame(
        crs=admin_0.crs,
        data={"ISO_A3": ["TWN"], "NAM_0": ["Taiwan"]},
        geometry=[china_geometry.intersection(TAIWAN_FENCE)],
    )
    return pandas.concat(
        [admin_0[~is_china], china_without_taiwan, taiwan], ignore_index=True
    )


def with_taiwan_province_separated(
    admin_1: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame:
    """The World Bank admin-1 layer with `TWN001` under TWN, so its name reads "Taiwan | Taiwan Sheng"."""
    is_taiwan = admin_1["ADM1CD_c"].str.startswith("TWN")
    assert is_taiwan.sum() == 1
    return admin_1.assign(
        ISO_A3=admin_1["ISO_A3"].where(cond=~is_taiwan, other="TWN"),
        NAM_0=admin_1["NAM_0"].where(cond=~is_taiwan, other="Taiwan"),
    )


def save_tile_id_to_local_path_for_admin_level(
    admin_level: AdminLevel,
) -> base.SaveTileIdToLocalPathType:
    remote_url = (
        "https://datacatalogfiles.worldbank.org/ddh-published/0038272/2/DR0095370/"
        "World Bank Official Boundaries (GeoPackage)/World Bank Official Boundaries - "
        f"Admin {admin_level.value:d}.gpkg"
    )

    def inner(local_path: str, tile_id: str) -> None:
        match admin_level:
            case AdminLevel.NATIONAL:
                path_to_raw = os.path.join(os.path.dirname(local_path), "raw.gpkg")
                utils.save_remote_url_to_local_path(
                    local_path=path_to_raw,
                    params={},
                    remote_url=remote_url,
                )
                with_taiwan_carved_out_of_china(
                    admin_0=geopandas.read_file(filename=path_to_raw)
                ).to_file(driver="GPKG", filename=local_path)
            case AdminLevel.PROVINCIAL:
                path_to_raw = os.path.join(os.path.dirname(local_path), "raw.gpkg")
                utils.save_remote_url_to_local_path(
                    local_path=path_to_raw,
                    params={},
                    remote_url=remote_url,
                )
                with_taiwan_province_separated(
                    admin_1=geopandas.read_file(filename=path_to_raw)
                ).to_file(driver="GPKG", filename=local_path)
            case AdminLevel.DISTRICT:
                utils.save_remote_url_to_local_path(
                    local_path=local_path,
                    params={},
                    remote_url=remote_url,
                )

    return inner


ADMIN_0_DATASET = base.VectorDataset(
    id_column_names=("ISO_A3",),
    name_column_names=("NAM_0",),
    product_name="admin-0",
    save_tile_id_to_local_path=save_tile_id_to_local_path_for_admin_level(
        admin_level=AdminLevel.NATIONAL
    ),
    source_name="world-bank",
    # v1: TWN carved out of CHN
    version="v1",
)

ADMIN_1_DATASET = base.VectorDataset(
    id_column_names=("ADM1CD_c",),
    name_column_names=("NAM_0", "NAM_1"),
    product_name="admin-1",
    save_tile_id_to_local_path=save_tile_id_to_local_path_for_admin_level(
        admin_level=AdminLevel.PROVINCIAL
    ),
    source_name="world-bank",
    # v1: TWN001 under TWN rather than CHN
    version="v1",
)

ADMIN_2_DATASET = base.VectorDataset(
    id_column_names=("ADM2CD_c",),
    name_column_names=("NAM_0", "NAM_1", "NAM_2"),
    product_name="admin-2",
    save_tile_id_to_local_path=save_tile_id_to_local_path_for_admin_level(
        admin_level=AdminLevel.DISTRICT
    ),
    source_name="world-bank",
    version="v0",
)


ADMIN_LEVEL_TO_DATASET = {
    AdminLevel.NATIONAL: ADMIN_0_DATASET,
    AdminLevel.PROVINCIAL: ADMIN_1_DATASET,
    AdminLevel.DISTRICT: ADMIN_2_DATASET,
}
assert set(AdminLevel) == set(ADMIN_LEVEL_TO_DATASET)


@utils.threadsafe_cache
def get_jurisdiction_for_admin_level(admin_level: AdminLevel) -> geopandas.GeoDataFrame:
    dataset = ADMIN_LEVEL_TO_DATASET[admin_level]
    path_to_fgb = storage.join_uri(
        prefix=dataset.get_prefix(tile_id="world"),
        root=config.Config.from_dot_env().ingest_root,
    )
    logger.info(f"Loading {admin_level=:} geometrys from {path_to_fgb=:s}")
    return geopandas.read_file(filename=path_to_fgb).set_index(keys="id")


@functools.cache
def get_ten_degree_tile_ids_for_admin_id(
    admin_id: str, admin_level: AdminLevel
) -> set[str]:
    gdf = get_jurisdiction_for_admin_level(admin_level=admin_level)
    geometry = gdf.loc[admin_id].geometry
    assert isinstance(geometry, shapely.Polygon | shapely.MultiPolygon)
    return set(
        tiling.iter_ten_degree_tile_id_for_geometry(geometry=geometry)
    ).intersection(tiling.GLOBAL_NATURE_WATCH_TILE_IDS)


def get_ten_degree_tile_ids_for_iso_3166s(
    iso_3166s: collections.abc.Iterable[str],
) -> set[str]:
    return {
        tile_id
        for iso_3166 in iso_3166s
        for tile_id in get_ten_degree_tile_ids_for_admin_id(
            admin_id=iso_3166,
            admin_level=AdminLevel.NATIONAL,
        )
    }


def iso_3166_str(s: str) -> str:
    assert len(s) == 3, f"iso_3166 code {s=:s} must be three characters"
    return s


# Countries the pipeline produces nothing for. Each does intersect a published tile, so no
# geometric predicate replaces this list; MapSPAM coverage is the real criterion and reading it
# needs the ingested rasters.
UNPRODUCTIVE_ISO_3166S = {"BLM", "GIB", "GRL", "MAF", "MCO", "NRU", "TUV", "VAT"}


def get_all_iso_3166s() -> set[str]:
    """Every country the pipeline can produce an emissions factor for.

    Two halves: the country touches a ten-degree tile the sources publish, and it grows something.
    The first is geometry and the second is `UNPRODUCTIVE_ISO_3166S`; see there for why the second
    is a list rather than a predicate.
    """
    return {
        iso_3166_str(s=iso_3166)
        for iso_3166 in get_jurisdiction_for_admin_level(
            admin_level=AdminLevel.NATIONAL
        ).index
        if iso_3166 not in UNPRODUCTIVE_ISO_3166S
        and get_ten_degree_tile_ids_for_admin_id(
            admin_id=iso_3166, admin_level=AdminLevel.NATIONAL
        )
    }


@dataclasses.dataclass
class Jurisdiction:
    level: str
    id: str
    name: str
    geometry: shapely.Geometry


def iter_jurisdiction_for_iso_3166_tile_id(
    admin_level: AdminLevel, iso_3166: str, tile_id: str
) -> collections.abc.Generator[Jurisdiction]:
    gdf = get_jurisdiction_for_admin_level(admin_level=admin_level)
    the_iso_3166 = gdf[gdf.index.str.startswith(iso_3166)].sort_index()
    tile = tiling.get_box_for_tile_id(tile_id=tile_id)
    for admin_id, row in the_iso_3166.iterrows():
        if row["geometry"].intersects(other=tile):
            yield Jurisdiction(
                geometry=row["geometry"],
                id=str(admin_id),
                level=admin_level.name,
                name=row["name"],
            )
