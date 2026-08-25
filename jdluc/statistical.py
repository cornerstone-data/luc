import collections
import collections.abc
import enum
import logging

import pandas
import rasterio.enums
import xarray

from jdluc import emit, geo, harmonize, storage, tiling, utils
from jdluc.datasets import (
    DatasetName,
    gfw_global_peatlands,
    glad_glcluc,
    ifpri_mapspam,
    worldbank_jurisdictions,
)

logger = logging.getLogger(__name__)


@enum.unique
class Crop(enum.StrEnum):
    BARLEY = ifpri_mapspam.Crop2000.BARL.name
    BEAN = ifpri_mapspam.Crop2000.BEAN.name
    CASSAVA = ifpri_mapspam.Crop2000.CASS.name
    COTTON = ifpri_mapspam.Crop2000.COTT.name
    GROUNDNUT = ifpri_mapspam.Crop2000.GROU.name
    MAIZE = ifpri_mapspam.Crop2000.MAIZ.name
    POTATO = ifpri_mapspam.Crop2000.POTA.name
    RICE = ifpri_mapspam.Crop2000.RICE.name
    SORGHUM = ifpri_mapspam.Crop2000.SORG.name
    SOYBEAN = ifpri_mapspam.Crop2000.SOYB.name
    SUGARBEET = ifpri_mapspam.Crop2000.SUGB.name
    SUGARCANE = ifpri_mapspam.Crop2000.SUGC.name
    WHEAT = ifpri_mapspam.Crop2000.WHEA.name
    # Crops which need to be decomposed for 2000
    BANANA = ifpri_mapspam.Crop2005.BANA.name
    PLANTAIN = ifpri_mapspam.Crop2005.PLNT.name
    ARABICA_COFFEE = ifpri_mapspam.Crop2005.ACOF.name
    ROBUSTA_COFFEE = ifpri_mapspam.Crop2005.RCOF.name
    PEARL_MILLET = ifpri_mapspam.Crop2005.PMIL.name
    SMALL_MILLET = ifpri_mapspam.Crop2005.SMIL.name
    COCONUT = ifpri_mapspam.Crop2005.CNUT.name
    OILPALM = ifpri_mapspam.Crop2005.OILP.name
    SUNFLOWER = ifpri_mapspam.Crop2005.SUNF.name
    RAPESEED = ifpri_mapspam.Crop2005.RAPE.name
    SESAME_SEED = ifpri_mapspam.Crop2005.SESA.name
    OTHER_OILCROPS = ifpri_mapspam.Crop2005.OOIL.name
    CHICKPEA = ifpri_mapspam.Crop2005.CHIC.name
    COWPEA = ifpri_mapspam.Crop2005.COWP.name
    PIGEONPEA = ifpri_mapspam.Crop2005.PIGE.name
    LENTIL = ifpri_mapspam.Crop2005.LENT.name
    OTHER_PULSES = ifpri_mapspam.Crop2005.OPUL.name
    SWEET_POTATO = ifpri_mapspam.Crop2005.SWPO.name
    YAM = ifpri_mapspam.Crop2005.YAMS.name


assert {e.value for e in Crop} == ifpri_mapspam.RECOVERABLE_CROP_NAMES


DATASET_NAMES = (
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2000,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2005,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2010,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2020,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2000,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2005,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2010,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2020,
)


CROPLAND_EMISSIONS_VARIABLE_NAMES = [
    "emissions:tco2e-per-ha:2000-2005",
    "emissions:tco2e-per-ha:2005-2010",
    "emissions:tco2e-per-ha:2010-2015",
    "emissions:tco2e-per-ha:2015-2020",
    "peatland-occupation:tco2e-per-ha",
]
GLAD_VARIABLE_NAMES = [
    "gfw:global-peatlands:is-peatland",
    *CROPLAND_EMISSIONS_VARIABLE_NAMES,
]


@storage.cache_to_zarr(version=0)
def get_downscaled_luc_emissions(
    skip_glad_crop_filter: bool, tile_id: str
) -> xarray.Dataset:
    logger.info("Computing emissions on the GLAD grid")
    glad_emissions = geo.exact_merge(
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
    )

    logger.info("Splitting forest / peatland-conversion per span")
    forest_class = glad_glcluc.LandClass.FOREST.value
    is_peat = glad_emissions[gfw_global_peatlands.DATASET.fully_qualified_band_name]
    source_variable_names: list[str] = []
    for before, after in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT:
        before_class = glad_emissions[f"land-class:{before:d}"]
        vegetation = glad_emissions[
            f"vegetation-emissions:tco2e-per-ha:{before:d}-{after:d}"
        ]
        soil = glad_emissions[f"soil-emissions:tco2e-per-ha:{before:d}-{after:d}"]
        source_to_darray = {
            "forest": vegetation.where(before_class == forest_class, other=0)
            + soil.where((before_class == forest_class) & (is_peat != 1), other=0),
            "peatland_conversion": soil.where(is_peat == 1, other=0),
        }
        for source, darray in source_to_darray.items():
            name = f"{source}:tco2e-per-ha:{before:d}-{after:d}"
            glad_emissions[name] = darray
            source_variable_names.append(name)

    if not skip_glad_crop_filter:
        logger.info("Masking emissions to current cropland")
        is_currently_cropland = (
            glad_emissions[f"land-class:{max(glad_glcluc.YEARS):d}"]
            == glad_glcluc.LandClass.CROPLAND.value
        )
        for variable_name in (
            *CROPLAND_EMISSIONS_VARIABLE_NAMES,
            *source_variable_names,
        ):
            glad_emissions[variable_name] = glad_emissions[variable_name].where(
                is_currently_cropland, other=0
            )

    logger.info("Downsampling emissions from the GLAD grid to the MAPSPAM grid")
    grid = harmonize.Grid.from_tile_id_resolution(
        resolution=tiling.TileResolution.MAPSPAM, tile_id=tile_id
    )
    return xarray.Dataset(
        {
            variable_name: geo.downscale_darray(
                darray=glad_emissions[variable_name],
                epsg=grid.epsg,
                height=grid.resolution.y,
                # NB: without an equal-area CRS, this presents a small error
                resampling=rasterio.enums.Resampling.average,
                transform=grid.transform,
                width=grid.resolution.x,
            )
            for variable_name in (*GLAD_VARIABLE_NAMES, *source_variable_names)
        }
    )


def get_crop_to_share(
    after: int, before: int, crops: tuple[Crop, ...], dset: xarray.Dataset
) -> dict[Crop, xarray.DataArray]:
    def get_expansion(canonical_crop_name: str) -> xarray.DataArray:
        return (
            ifpri_mapspam.get_canonical_quantity(
                canonical_crop_name=canonical_crop_name,
                dset=dset,
                quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                year=after,
            )
            - ifpri_mapspam.get_canonical_quantity(
                canonical_crop_name=canonical_crop_name,
                dset=dset,
                quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                year=before,
            )
        ).clip(min=0)

    attributed_expansion = sum(
        get_expansion(canonical_crop_name=name)
        for name in sorted(ifpri_mapspam.RECOVERABLE_CROP_NAMES)
    )

    # Drop any crops which are being newly tracked so they aren't interpreted as an expansion from zero
    before_names = ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[before]
    after_names = ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[after]
    if (before, after) in ifpri_mapspam.SPANS_WITH_COMPARABLE_CROP_NAMES:
        before_names = after_names = before_names & after_names

    def get_unattributed(year: int, names: set[str]) -> xarray.DataArray:
        # NB: no canonical lookup is required -- these names are already `year`'s own, and
        # nothing divides by them, so the raw band is the whole answer
        ret = sum(
            ifpri_mapspam.get_reported_quantity(
                dset=dset,
                quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                reported_crop_name=name,
                year=year,
            )
            for name in sorted(names)
        )
        assert isinstance(ret, xarray.DataArray)
        return ret

    unattributed_expansion = (
        get_unattributed(year=after, names=after_names)
        - get_unattributed(year=before, names=before_names)
    ).clip(min=0)
    total_expansion = attributed_expansion + unattributed_expansion
    total_expansion = total_expansion.where(total_expansion > 0)
    return {
        # Share is zero when there is no expansion at all
        crop: (get_expansion(canonical_crop_name=crop.value) / total_expansion).fillna(
            0
        )
        for crop in crops
    }


GLAD_TO_MAPSPAM_SPAN = {
    (2000, 2005): (2000, 2005),
    (2005, 2010): (2005, 2010),
    # NB: because MAPSPAM has no 2015 snapshot, we assume crop expansion is constant between
    # 2010 and 2020 -- and then use that one crop expansion for both spans
    (2010, 2015): (2010, 2020),
    (2015, 2020): (2010, 2020),
}
assert set(GLAD_TO_MAPSPAM_SPAN) == set(emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT)


def get_crop_name_to_totals(
    dset: xarray.Dataset,
    crop_to_span_to_share: dict[Crop, dict[emit.SpanType, xarray.DataArray]],
    occupation_shares: dict[Crop, xarray.DataArray],
) -> dict[str, dict[str, float]]:
    peatland_fraction = dset[gfw_global_peatlands.DATASET.fully_qualified_band_name]
    hectares = emit.get_hectares_per_pixel(darray=peatland_fraction)
    peatland_occupation = dset["peatland-occupation:tco2e-per-ha"] * hectares

    crop_to_totals: dict[Crop, dict[str, xarray.DataArray]] = collections.defaultdict(
        dict
    )
    weight_total = sum(emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT.values())
    for crop, span_to_share in crop_to_span_to_share.items():
        conversion = emit.get_linear_discounted_total(
            span_to_value={
                (before, after): dset[f"emissions:tco2e-per-ha:{before:d}-{after:d}"]
                * hectares
                * span_to_share[GLAD_TO_MAPSPAM_SPAN[(before, after)]]
                for (
                    before,
                    after,
                ) in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT
            }
        )
        peatland = peatland_occupation * occupation_shares[crop]
        crop_hectares = (
            emit.get_linear_discounted_total(
                span_to_value={
                    glad_span: (
                        ifpri_mapspam.get_canonical_quantity(
                            canonical_crop_name=crop.value,
                            dset=dset,
                            quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                            year=before,
                        )
                        + ifpri_mapspam.get_canonical_quantity(
                            canonical_crop_name=crop.value,
                            dset=dset,
                            quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                            year=after,
                        )
                    )
                    / 2
                    for glad_span, (before, after) in GLAD_TO_MAPSPAM_SPAN.items()
                }
            )
            / weight_total
        )
        crop_to_totals[crop]["crop_hectares"] = crop_hectares
        for source in ("forest", "peatland_conversion"):
            crop_to_totals[crop][f"{source:s}_emissions_mt"] = (
                emit.get_linear_discounted_total(
                    span_to_value={
                        (before, after): dset[
                            f"{source:s}:tco2e-per-ha:{before:d}-{after:d}"
                        ]
                        * hectares
                        * span_to_share[GLAD_TO_MAPSPAM_SPAN[(before, after)]]
                        for (before, after) in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT
                    }
                )
            )
        crop_to_totals[crop]["peatland_crop_hectares"] = (
            crop_hectares * peatland_fraction
        )
        crop_to_totals[crop]["peatland_occupation_emissions_mt"] = peatland
        crop_to_totals[crop]["emissions_mt"] = conversion + peatland
        crop_to_totals[crop]["production_mt"] = (
            emit.get_linear_discounted_total(
                span_to_value={
                    glad_span: (
                        ifpri_mapspam.get_canonical_quantity(
                            canonical_crop_name=crop.value,
                            dset=dset,
                            quantity=ifpri_mapspam.Quantity.PRODUCTION,
                            year=before,
                        )
                        + ifpri_mapspam.get_canonical_quantity(
                            canonical_crop_name=crop.value,
                            dset=dset,
                            quantity=ifpri_mapspam.Quantity.PRODUCTION,
                            year=after,
                        )
                    )
                    / 2
                    for glad_span, (before, after) in GLAD_TO_MAPSPAM_SPAN.items()
                }
            )
            / weight_total
        )
    return utils.get_sum_totals(enum_to_name_to_darray=crop_to_totals)


def get_crop_to_area_share(
    crops: tuple[Crop, ...], dset: xarray.Dataset, year: int
) -> dict[Crop, xarray.DataArray]:
    # NB: the denominator walks `year`'s own taxonomy, so every name here is already reported
    total_area = sum(
        ifpri_mapspam.get_reported_quantity(
            dset=dset,
            quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
            reported_crop_name=e.name,
            year=year,
        )
        for e in ifpri_mapspam.YEAR_TO_CROP_CLS[year]
    )
    assert isinstance(total_area, xarray.DataArray)
    total_area = total_area.where(total_area > 0)
    return {
        # Share is zero when no crop occupies the cell at all
        crop: (
            ifpri_mapspam.get_canonical_quantity(
                canonical_crop_name=crop.value,
                dset=dset,
                quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                year=year,
            )
            / total_area
        ).fillna(0)
        for crop in crops
    }


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
    "production_mt": float,
}


@storage.cache_to_parquet(version=1)
def workflow(
    crop_names: tuple[str, ...],
    iso_3166: str,
    skip_glad_crop_filter: bool,
    tile_id: str,
) -> pandas.DataFrame:
    from rioxarray.exceptions import NoDataInBounds

    crops = tuple(Crop[crop_name] for crop_name in crop_names)
    logger.info(f"Computing emissions for {crops=:} and {tile_id=:s}")
    merged = geo.exact_merge(
        # NB: this is deferred because it is expensive and would like to cache it
        get_downscaled_luc_emissions(
            skip_glad_crop_filter=skip_glad_crop_filter, tile_id=tile_id
        ),
        # MAPSPAM
        harmonize.workflow(
            dataset_names=DATASET_NAMES,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.MAPSPAM,
        ),
    )

    def it() -> collections.abc.Iterator[dict[str, float | str]]:
        for (
            jurisdiction
        ) in worldbank_jurisdictions.iter_jurisdiction_for_iso_3166_tile_id(
            admin_level=worldbank_jurisdictions.AdminLevel.PROVINCIAL,
            iso_3166=iso_3166,
            tile_id=tile_id,
        ):
            logger.info(f"Clipping to provincial geometry from {jurisdiction.id=:s}")
            try:
                clipped = geo.clip_dset(dset=merged, geometry=jurisdiction.geometry)
            except NoDataInBounds as exc:
                logger.warning(repr(exc))
            else:
                span_to_crop_to_share = {
                    (before, after): get_crop_to_share(
                        after=after,
                        before=before,
                        crops=crops,
                        dset=clipped,
                    )
                    for (before, after) in dict.fromkeys(GLAD_TO_MAPSPAM_SPAN.values())
                }
                crop_to_span_to_share = {
                    crop: {
                        span: crop_to_share[crop]
                        for span, crop_to_share in span_to_crop_to_share.items()
                    }
                    for crop in crops
                }

                logger.info(
                    f"Populating emissions for {jurisdiction.id=!s}/{len(crops)=:d} crops"
                )
                crop_name_to_totals = get_crop_name_to_totals(
                    crop_to_span_to_share=crop_to_span_to_share,
                    dset=clipped,
                    # NB: unlike conversion emissions, peatland occupation is a land-management flux on land
                    # that is drained *now* -- it has no relationship to expansion, and expansion is zero on
                    # the long-established peat cropland that dominates this pool. So allocate it by each
                    # crop's share of area occupied -- matching the approach for the jurisdictional-direct leg.
                    occupation_shares=get_crop_to_area_share(
                        crops=crops,
                        dset=clipped,
                        year=max(ifpri_mapspam.YEARS),
                    ),
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
