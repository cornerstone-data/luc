import enum

from jdluc.datasets import (
    base,
    faostat_production,
    gfw_global_peatlands,
    gfw_harris_agb,
    gfw_tcl,
    glad_glcluc,
    huang_bgb,
    ifpri_mapspam,
    ipcc_climate_zones,
    soilgrids_ocs,
    usda_nass_cdl,
    usda_nass_quickstats,
    worldbank_jurisdictions,
)


@enum.unique
class DatasetName(enum.StrEnum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name

    FAOSTAT_PRODUCTION = enum.auto()
    GFW_GLOBAL_PEATLANDS = enum.auto()
    GFW_HARRIS_AGB = enum.auto()
    GFW_TREE_COVER_LOSS = enum.auto()
    GLAD_GLCLUC = enum.auto()
    HUANG_BGB = enum.auto()
    IFPRI_MAPSPAM_PHYSICAL_AREA_2000 = enum.auto()
    IFPRI_MAPSPAM_PHYSICAL_AREA_2005 = enum.auto()
    IFPRI_MAPSPAM_PHYSICAL_AREA_2010 = enum.auto()
    IFPRI_MAPSPAM_PHYSICAL_AREA_2020 = enum.auto()
    IFPRI_MAPSPAM_PRODUCTION_2000 = enum.auto()
    IFPRI_MAPSPAM_PRODUCTION_2005 = enum.auto()
    IFPRI_MAPSPAM_PRODUCTION_2010 = enum.auto()
    IFPRI_MAPSPAM_PRODUCTION_2020 = enum.auto()
    IPCC_CLIMATE_ZONES = enum.auto()
    SOILGRIDS_OCS = enum.auto()
    USDA_NASS_CDL = enum.auto()
    USDA_NASS_QUICKSTATS = enum.auto()
    WORLD_BANK_ADMIN_0 = enum.auto()
    WORLD_BANK_ADMIN_1 = enum.auto()
    WORLD_BANK_ADMIN_2 = enum.auto()


NAME_TO_CLS: dict[
    DatasetName, base.RasterDataset | base.TabularDataset | base.VectorDataset
] = {
    DatasetName.FAOSTAT_PRODUCTION: faostat_production.DATASET,
    DatasetName.GFW_GLOBAL_PEATLANDS: gfw_global_peatlands.DATASET,
    DatasetName.GFW_HARRIS_AGB: gfw_harris_agb.DATASET,
    DatasetName.GFW_TREE_COVER_LOSS: gfw_tcl.DATASET,
    DatasetName.GLAD_GLCLUC: glad_glcluc.DATASET,
    DatasetName.HUANG_BGB: huang_bgb.DATASET,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2000: ifpri_mapspam.PHYSICAL_AREA_2000,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2005: ifpri_mapspam.PHYSICAL_AREA_2005,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2010: ifpri_mapspam.PHYSICAL_AREA_2010,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2020: ifpri_mapspam.PHYSICAL_AREA_2020,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2000: ifpri_mapspam.PRODUCTION_2000,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2005: ifpri_mapspam.PRODUCTION_2005,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2010: ifpri_mapspam.PRODUCTION_2010,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2020: ifpri_mapspam.PRODUCTION_2020,
    DatasetName.IPCC_CLIMATE_ZONES: ipcc_climate_zones.DATASET,
    DatasetName.SOILGRIDS_OCS: soilgrids_ocs.DATASET,
    DatasetName.USDA_NASS_CDL: usda_nass_cdl.DATASET,
    DatasetName.USDA_NASS_QUICKSTATS: usda_nass_quickstats.DATASET,
    DatasetName.WORLD_BANK_ADMIN_0: worldbank_jurisdictions.ADMIN_0_DATASET,
    DatasetName.WORLD_BANK_ADMIN_1: worldbank_jurisdictions.ADMIN_1_DATASET,
    DatasetName.WORLD_BANK_ADMIN_2: worldbank_jurisdictions.ADMIN_2_DATASET,
}
assert set(DatasetName) == set(NAME_TO_CLS), (
    f"{set(DatasetName)=:}; {set(NAME_TO_CLS)=:}"
)
