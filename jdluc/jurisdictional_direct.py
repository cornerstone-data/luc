import collections
import collections.abc
import enum
import logging

import pandas
import xarray
from rioxarray.exceptions import NoDataInBounds

from jdluc import emit, geo, harmonize, storage, tiling, utils
from jdluc.datasets import (
    DatasetName,
    gnw_global_peatlands,
    usda_nass_cdl,
    worldbank_jurisdictions,
)

logger = logging.getLogger(__name__)


@enum.unique
class Crop(enum.Enum):
    BARLEY = (usda_nass_cdl.CropClass.BARLEY,)
    BEAN = (usda_nass_cdl.CropClass.DRY_BEANS,)
    COTTON = (usda_nass_cdl.CropClass.COTTON,)
    MAIZE = (usda_nass_cdl.CropClass.CORN,)
    POTATO = (usda_nass_cdl.CropClass.POTATOES,)
    RICE = (usda_nass_cdl.CropClass.RICE,)
    SORGHUM = (usda_nass_cdl.CropClass.SORGHUM,)
    SOYBEAN = (usda_nass_cdl.CropClass.SOYBEANS,)
    SUGARBEET = (usda_nass_cdl.CropClass.SUGARBEETS,)
    SUGARCANE = (usda_nass_cdl.CropClass.SUGARCANE,)
    WHEAT = (
        usda_nass_cdl.CropClass.DURUM_WHEAT,
        usda_nass_cdl.CropClass.SPRING_WHEAT,
        usda_nass_cdl.CropClass.WINTER_WHEAT,
    )


DATASET_NAMES = (DatasetName.USDA_NASS_CDL,)


EMISSIONS_AND_CROP_VARIABLE_NAMES = [
    # Harmonized
    gnw_global_peatlands.DATASET.fully_qualified_band_name,
    # Emissions
    "cropland-peatland-occupation:tco2e-per-ha",
    "dropped-emissions:tco2e-per-ha",
    "emissions-per-hectare:tco2e-per-ha",
    "hectares-per-pixel:ha",
    "pastureland-peatland-occupation:tco2e-per-ha",
    *(f"{component!s}-emissions:tco2e-per-ha" for component in emit.EmissionComponent),
    # Crop
    usda_nass_cdl.DATASET.fully_qualified_band_name,
]


def get_commodity_name_to_totals(
    component_to_per_hectare: dict[emit.EmissionComponent, xarray.DataArray],
    crop_class: xarray.DataArray,
    crops: tuple[Crop, ...],
    dropped_per_hectare: xarray.DataArray,
    emissions_per_hectare: xarray.DataArray,
    hectares_per_pixel: xarray.DataArray,
    is_peatland: xarray.DataArray,
    peatland_occupation_per_hectare: xarray.DataArray,
) -> dict[str, dict[str, float]]:

    peatland_occupation_emissions = peatland_occupation_per_hectare * hectares_per_pixel
    emissions_mt = emissions_per_hectare * hectares_per_pixel
    component_to_emissions = {
        component: per_hectare * hectares_per_pixel
        for component, per_hectare in component_to_per_hectare.items()
    }

    crop_to_totals: dict[Crop | emit.NonCommodity, dict[str, xarray.DataArray]] = (
        collections.defaultdict(dict)
    )

    for crop in crops:
        crop_mask = crop_class.isin([value.value for value in crop.value])
        crop_to_totals[crop]["commodity_hectares"] = commodity_hectares = (
            hectares_per_pixel.where(crop_mask)
        )
        crop_to_totals[crop]["peatland_commodity_hectares"] = commodity_hectares.where(
            is_peatland
        )
        crop_to_totals[crop]["peatland_occupation_emissions_mt"] = (
            peatland_occupation_emissions.where(crop_mask)
        )
        crop_to_totals[crop]["emissions_mt"] = emissions_mt.where(crop_mask)
        for component, emissions in component_to_emissions.items():
            crop_to_totals[crop][component.column] = emissions.where(crop_mask)

    crop_to_totals[emit.NonCommodity.DROPPED]["emissions_mt"] = (
        dropped_per_hectare * hectares_per_pixel
    )

    return utils.get_sum_totals(enum_to_name_to_darray=crop_to_totals)


SCHEMA = {
    "admin_id": str,
    "admin_level": str,
    "commodity_hectares": float,
    "commodity_name": str,
    "emissions_mt": float,
    "forest_emissions_mt": float,
    "grassland_emissions_mt": float,
    "jurisdiction_name": str,
    "peatland_commodity_hectares": float,
    "peatland_conversion_emissions_mt": float,
    "peatland_occupation_emissions_mt": float,
}


@storage.cache_to_parquet(version=1)
def workflow(
    commodity_names: tuple[str, ...],
    iso_3166: str,
    tile_id: str,
) -> pandas.DataFrame:
    crops = tuple(Crop[commodity_name] for commodity_name in commodity_names)
    assert iso_3166 == "USA", "JD only supports USA today"

    logger.info(f"Computing emissions for {crops=:} and {tile_id=:s}")
    merged = geo.exact_merge(
        harmonize.workflow(
            dataset_names=harmonize.Stack.LUC_AND_EMISSIONS.value,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.GLAD,
        ),
        # NB: this should call the harmonize workflow with identical args and hit
        # the cache from the preceding call
        emit.workflow(tile_id=tile_id),
        # CDL
        harmonize.workflow(
            dataset_names=DATASET_NAMES,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.GLAD,
        ),
    )
    merged = merged.assign(
        {
            f"{component!s}-emissions:tco2e-per-ha": emit.get_linear_discounted_total(
                span_to_value={
                    span: component_to_emissions[component]
                    for span, component_to_emissions in emit.get_span_to_component_to_emissions(
                        dset=merged
                    ).items()
                }
            )
            for component in emit.EmissionComponent
        }
    )

    def it() -> collections.abc.Iterator[dict[str, float | str]]:
        for (
            jurisdiction
        ) in worldbank_jurisdictions.iter_jurisdiction_for_iso_3166_tile_id(
            admin_level=worldbank_jurisdictions.AdminLevel.PROVINCIAL,
            iso_3166=iso_3166,
            tile_id=tile_id,
        ):
            logger.info(f"Clipping to provincial geometry for {jurisdiction.id=:s}")
            try:
                clipped = geo.clip_dset(
                    dset=merged[EMISSIONS_AND_CROP_VARIABLE_NAMES],
                    geometry=jurisdiction.geometry,
                )
            except NoDataInBounds as exc:
                logger.warning(repr(exc))
            else:
                logger.info(
                    f"Populating emissions for {jurisdiction.id=!s}/{len(crops)=:d} crops"
                )
                commodity_name_to_totals = get_commodity_name_to_totals(
                    component_to_per_hectare={
                        component: clipped[f"{component!s}-emissions:tco2e-per-ha"]
                        for component in emit.EmissionComponent
                    },
                    crop_class=clipped[usda_nass_cdl.DATASET.fully_qualified_band_name],
                    crops=crops,
                    dropped_per_hectare=clipped["dropped-emissions:tco2e-per-ha"],
                    # NB: this leg allocates to CDL crops alone, so peat drained under
                    # pasture has no row to land on, and leaves the total along with it
                    emissions_per_hectare=clipped["emissions-per-hectare:tco2e-per-ha"]
                    - clipped["pastureland-peatland-occupation:tco2e-per-ha"],
                    hectares_per_pixel=clipped["hectares-per-pixel:ha"],
                    # NB: no data has to read as not peat rather than truth-testing to peat
                    is_peatland=clipped[
                        gnw_global_peatlands.DATASET.fully_qualified_band_name
                    ]
                    == 1,
                    peatland_occupation_per_hectare=clipped[
                        "cropland-peatland-occupation:tco2e-per-ha"
                    ],
                )
                for commodity_name, totals in commodity_name_to_totals.items():
                    yield totals | {
                        "admin_id": jurisdiction.id,
                        "admin_level": jurisdiction.level,
                        "commodity_name": commodity_name,
                        "jurisdiction_name": jurisdiction.name,
                    }

    if data := list(it()):
        assert set(data[0]) == set(SCHEMA)
    return (
        pandas.DataFrame.from_records(columns=list(SCHEMA), data=data)
        .astype(SCHEMA)
        .set_index(["admin_level", "admin_id", "commodity_name"])
    )
