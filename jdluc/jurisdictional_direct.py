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
    gfw_global_peatlands,
    glad_glcluc,
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


GLAD_AND_CROP_VARIABLE_NAMES = [
    # Harmonized
    gfw_global_peatlands.DATASET.fully_qualified_band_name,
    # Emissions
    "emissions-per-hectare:tco2e-per-ha",
    "forest-emissions:tco2e-per-ha",
    "hectares-per-pixel:ha",
    f"land-class:{max(glad_glcluc.YEARS):d}",
    "peatland-conversion-emissions:tco2e-per-ha",
    "peatland-occupation:tco2e-per-ha",
    # Crop
    usda_nass_cdl.DATASET.fully_qualified_band_name,
]


def get_forest_and_peatland_conversion_per_hectare(
    dset: xarray.Dataset,
) -> dict[str, xarray.DataArray]:
    forest_class = glad_glcluc.LandClass.FOREST.value
    is_peat = dset[gfw_global_peatlands.DATASET.fully_qualified_band_name]
    span_to_forest: dict[emit.SpanType, xarray.DataArray] = {}
    span_to_peatland_conversion: dict[emit.SpanType, xarray.DataArray] = {}
    for before, after in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT:
        before_class = dset[f"land-class:{before:d}"]
        vegetation = dset[f"vegetation-emissions:tco2e-per-ha:{before:d}-{after:d}"]
        soil = dset[f"soil-emissions:tco2e-per-ha:{before:d}-{after:d}"]
        span_to_forest[(before, after)] = vegetation.where(
            before_class == forest_class, other=0
        ) + soil.where((before_class == forest_class) & (is_peat != 1), other=0)
        span_to_peatland_conversion[(before, after)] = soil.where(is_peat == 1, other=0)
    return {
        "forest-emissions:tco2e-per-ha": emit.get_linear_discounted_total(
            span_to_value=span_to_forest
        ),
        "peatland-conversion-emissions:tco2e-per-ha": emit.get_linear_discounted_total(
            span_to_value=span_to_peatland_conversion
        ),
    }


def get_crop_name_to_totals(
    crop_class: xarray.DataArray,
    crops: tuple[Crop, ...],
    emissions_per_hectare: xarray.DataArray,
    forest_emissions_per_hectare: xarray.DataArray,
    glad_class: xarray.DataArray,
    hectares_per_pixel: xarray.DataArray,
    is_peatland: xarray.DataArray,
    peatland_conversion_per_hectare: xarray.DataArray,
    peatland_occupation_per_hectare: xarray.DataArray,
    skip_glad_crop_filter: bool,
) -> dict[str, dict[str, float]]:
    glad_crop_mask = (
        xarray.ones_like(glad_class)
        if skip_glad_crop_filter
        else (glad_class == glad_glcluc.LandClass.CROPLAND.value)
    )
    crop_class = crop_class.where(glad_crop_mask)

    peatland_occupation_emissions = peatland_occupation_per_hectare * hectares_per_pixel
    emissions_mt = emissions_per_hectare * hectares_per_pixel
    forest_emissions = forest_emissions_per_hectare * hectares_per_pixel
    peatland_conversion_emissions = peatland_conversion_per_hectare * hectares_per_pixel

    crop_to_totals: dict[Crop, dict[str, xarray.DataArray]] = collections.defaultdict(
        dict
    )

    for crop in crops:
        crop_mask = crop_class.isin([value.value for value in crop.value])
        crop_to_totals[crop]["crop_hectares"] = crop_hectares = (
            hectares_per_pixel.where(crop_mask)
        )
        crop_to_totals[crop]["peatland_crop_hectares"] = crop_hectares.where(
            is_peatland == 1
        )
        crop_to_totals[crop]["peatland_occupation_emissions_mt"] = (
            peatland_occupation_emissions.where(crop_mask)
        )
        crop_to_totals[crop]["emissions_mt"] = emissions_mt.where(crop_mask)
        crop_to_totals[crop]["forest_emissions_mt"] = forest_emissions.where(crop_mask)
        crop_to_totals[crop]["peatland_conversion_emissions_mt"] = (
            peatland_conversion_emissions.where(crop_mask)
        )

    return utils.get_sum_totals(enum_to_name_to_darray=crop_to_totals)


SCHEMA = {
    "admin_id": str,
    "admin_level": str,
    "crop_hectares": float,
    "crop_name": str,
    "emissions_mt": float,
    "forest_emissions_mt": float,
    "jurisdiction_name": str,
    "peatland_conversion_emissions_mt": float,
    "peatland_crop_hectares": float,
    "peatland_occupation_emissions_mt": float,
}


@storage.cache_to_parquet(version=1)
def workflow(
    crop_names: tuple[str, ...],
    iso_3166: str,
    skip_glad_crop_filter: bool,
    tile_id: str,
) -> pandas.DataFrame:
    crops = tuple(Crop[crop_name] for crop_name in crop_names)
    assert iso_3166 == "USA", "JD only supports USA today"

    logger.info(f"Computing emissions for {crops=:} and {tile_id=:s}")
    merged = geo.exact_merge(
        harmonize.workflow(
            dataset_names=harmonize.LUC_AND_EMISSIONS_DATASET_NAMES,
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
    merged = merged.assign(get_forest_and_peatland_conversion_per_hectare(dset=merged))

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
                    dset=merged[GLAD_AND_CROP_VARIABLE_NAMES],
                    geometry=jurisdiction.geometry,
                )
            except NoDataInBounds as exc:
                logger.warning(repr(exc))
            else:
                logger.info(
                    f"Populating emissions for {jurisdiction.id=!s}/{len(crops)=:d} crops"
                )
                crop_name_to_totals = get_crop_name_to_totals(
                    crop_class=clipped[usda_nass_cdl.DATASET.fully_qualified_band_name],
                    crops=crops,
                    emissions_per_hectare=clipped["emissions-per-hectare:tco2e-per-ha"],
                    forest_emissions_per_hectare=clipped[
                        "forest-emissions:tco2e-per-ha"
                    ],
                    glad_class=clipped[f"land-class:{max(glad_glcluc.YEARS):d}"],
                    hectares_per_pixel=clipped["hectares-per-pixel:ha"],
                    is_peatland=clipped[
                        gfw_global_peatlands.DATASET.fully_qualified_band_name
                    ],
                    peatland_conversion_per_hectare=clipped[
                        "peatland-conversion-emissions:tco2e-per-ha"
                    ],
                    peatland_occupation_per_hectare=clipped[
                        "peatland-occupation:tco2e-per-ha"
                    ],
                    skip_glad_crop_filter=skip_glad_crop_filter,
                )
                for crop_name, totals in crop_name_to_totals.items():
                    yield totals | {
                        "admin_id": jurisdiction.id,
                        "admin_level": jurisdiction.level,
                        "crop_name": crop_name,
                        "jurisdiction_name": jurisdiction.name,
                    }

    if data := list(it()):
        assert set(data[0]) == set(SCHEMA)
    return (
        pandas.DataFrame.from_records(columns=list(SCHEMA), data=data)
        .astype(SCHEMA)
        .set_index(["admin_level", "admin_id", "crop_name"])
    )
