import collections
import collections.abc
import enum
import logging

import numpy
import pandas
import rasterio.enums
import xarray
from rioxarray.exceptions import NoDataInBounds

from jdluc import emit, geo, harmonize, storage, tiling, utils
from jdluc.datasets import (
    faostat,
    gpw_grassland,
    gpw_livestock,
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


# `Crop` stays exactly MapSPAM's recoverable crops, because everything that divides by one looks
# its name up there
type Commodity = Crop | gpw_livestock.Livestock


EMIT_VARIABLE_NAMES = [
    f"{emit.CROPLAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha",
    f"{emit.PASTURELAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha",
]


@storage.cache_to_zarr(version=1)
def get_downscaled_luc_emissions(assessment_year: int, tile_id: str) -> xarray.Dataset:
    logger.info("Computing emissions on the GLAD grid")
    luc_and_emissions = geo.exact_merge(
        harmonize.workflow(
            dataset_names=harmonize.Stack.LAND_CONVERSION_AND_EMISSIONS.value.dataset_names,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=harmonize.Stack.LAND_CONVERSION_AND_EMISSIONS.value.tile_resolution,
        ),
        # NB: this should call the harmonize workflow with identical args and hit
        # the cache from the preceding call
        emit.workflow(assessment_year=assessment_year, tile_id=tile_id),
    )

    logger.info("Discounting emissions by component and splitting them by MapSPAM span")
    conversion_year = luc_and_emissions["conversion-year"]
    component_to_emissions = {
        component: emit.get_linear_discounted(
            assessment_year=assessment_year,
            conversion_year=conversion_year,
            darray=emissions,
        )
        for component, emissions in emit.get_component_to_emissions(
            dset=luc_and_emissions
        ).items()
    }
    derived_variable_names: list[str] = []
    # NB: nothing departs in the lookback's opening year, so every other year can date a conversion
    _, *conversion_years = emit.get_lookback_years_range(
        assessment_year=assessment_year
    )
    for before, after in MAPSPAM_SPANS:
        # A conversion takes the shares of the span holding its year, open below and closed above.
        # MapSPAM has no 2015 snapshot, so conversions from 2011 to 2020 all take the span from 2010
        # to 2020, and it has none after 2020, so every later conversion takes that span too
        in_span = conversion_year.isin(
            [
                year
                for year in conversion_years
                if before < min(year, LAST_SNAPSHOT_YEAR) <= after
            ]
        )
        # NB: `emit` charges only converted pixels, and every conversion ends in cropland or
        # pasture, so each component's two bands add up to the whole component
        for component in emit.EmissionComponent:
            for land_class, conversions in (
                (emit.CROPLAND, emit.TO_CROPLAND),
                (emit.PASTURELAND, emit.TO_PASTURE),
            ):
                name = f"{land_class:s}-{component!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"
                luc_and_emissions[name] = component_to_emissions[component].where(
                    in_span & luc_and_emissions["conversion"].isin(conversions),
                    other=0,
                )
                derived_variable_names.append(name)

    logger.info("Discounting the source carbon no destination claimed")
    name = "dropped-emissions:tco2e-per-ha"
    luc_and_emissions[name] = emit.get_linear_discounted(
        assessment_year=assessment_year,
        conversion_year=conversion_year,
        darray=luc_and_emissions[name],
    )
    derived_variable_names.append(name)

    logger.info("Measuring pasture extent at the last MAPSPAM snapshot")
    name = f"pasture:fraction:{LAST_SNAPSHOT_YEAR:d}"
    # NB: a mask here and an averaging downscale below, so this arrives as a cell fraction
    luc_and_emissions[name] = (
        luc_and_emissions[
            gpw_grassland.DATASET.fully_qualified_band_names[
                gpw_grassland.YEARS.index(LAST_SNAPSHOT_YEAR)
            ]
        ]
        == gpw_grassland.Grassland.CULTIVATED
    ).astype(numpy.float32)
    derived_variable_names.append(name)

    logger.info("Downsampling emissions from the GLAD grid to the MAPSPAM grid")
    grid = harmonize.Grid.from_tile_id_resolution(
        resolution=tiling.TileResolution.MAPSPAM, tile_id=tile_id
    )
    return geo.downscale_dset(
        dset=luc_and_emissions[[*EMIT_VARIABLE_NAMES, *derived_variable_names]],
        epsg=grid.epsg,
        height=grid.resolution.y,
        # NB: without an equal-area CRS, this presents a small error
        resampling=rasterio.enums.Resampling.average,
        transform=grid.transform,
        width=grid.resolution.x,
    )


def get_crop_hectares(crop: Crop, dset: xarray.Dataset, year: int) -> xarray.DataArray:
    return ifpri_mapspam.get_canonical_quantity(
        canonical_crop_name=crop.value,
        dset=dset,
        quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
        year=year,
    )


def get_crop_expansion(
    after: int, before: int, crop: Crop, dset: xarray.Dataset
) -> xarray.DataArray:
    return (
        get_crop_hectares(crop=crop, dset=dset, year=after)
        - get_crop_hectares(crop=crop, dset=dset, year=before)
    ).clip(min=0)


def get_cropland_expansion(
    after: int, before: int, dset: xarray.Dataset
) -> xarray.DataArray:
    """Every crop's expansion in a cell over a span, whether or not `Crop` claims it."""
    attributed_expansion = sum(
        get_crop_expansion(after=after, before=before, crop=crop, dset=dset)
        for crop in sorted(Crop)
    )

    def get_unattributed(year: int) -> xarray.DataArray:
        # NB: no canonical lookup is required -- these names are already `year`'s own, and
        # nothing divides by them, so the raw band is the whole answer. They include the crops
        # 2020 names first, whose area 2005 and 2010 held in a group
        # (`ifpri_mapspam.NEWLY_NAMED_CROP_TO_EARLIER_GROUP_NAME`)
        return sum(
            (
                ifpri_mapspam.get_reported_quantity(
                    dset=dset,
                    quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                    reported_crop_name=name,
                    year=year,
                )
                for name in sorted(ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[year])
            ),
            start=xarray.DataArray(numpy.float32(0)),
        )

    unattributed_expansion = (
        get_unattributed(year=after) - get_unattributed(year=before)
    ).clip(min=0)
    return attributed_expansion + unattributed_expansion


def get_crop_to_expansion_share(
    after: int,
    before: int,
    crops: tuple[Crop, ...],
    dset: xarray.Dataset,
) -> dict[Crop, xarray.DataArray]:
    """Each crop's share of a cell's crop expansion over a span, zero where no crop expanded."""
    cropland_expansion = get_cropland_expansion(after=after, before=before, dset=dset)
    cropland_expansion = cropland_expansion.where(cropland_expansion > 0)
    return {
        crop: (
            get_crop_expansion(after=after, before=before, crop=crop, dset=dset)
            / cropland_expansion
        ).fillna(0)
        for crop in crops
    }


MAPSPAM_SPANS = ((2000, 2005), (2005, 2010), (2010, 2020))
MAPSPAM_SNAPSHOT_YEARS = tuple(
    sorted({year for span in MAPSPAM_SPANS for year in span})
)
assert set(MAPSPAM_SNAPSHOT_YEARS).issubset(gpw_livestock.YEARS)
LAST_SNAPSHOT_YEAR = max(MAPSPAM_SNAPSHOT_YEARS)
HERD_HEADS_COLUMN = gpw_livestock.get_herd_heads_column(year=LAST_SNAPSHOT_YEAR)


def get_commodity_name_to_totals(
    commodities: tuple[Commodity, ...],
    commodity_to_share: dict[Commodity, xarray.DataArray],
    dset: xarray.Dataset,
    span_to_crop_to_share: dict[tuple[int, int], dict[Crop, xarray.DataArray]],
) -> dict[str, dict[str, float]]:
    """Each commodity's emissions and area and a herd's heads, summed over a jurisdiction's cells.

    `commodity_to_share` holds each crop's 2020 area share, which divides cropland peat occupation,
    and each grazer's 2020 grazing share, which divides every span's pasture conversions, pasture
    peat occupation and pasture area. A crop's conversions take its expansion share of each span.

    Beside the commodities sit `DROPPED` and `UNATTRIBUTED` (see `emit.NonCommodity`). Here
    `UNATTRIBUTED` is what the shares leave of the charge: conversion and occupation by the
    MapSPAM crops no row models, and everything converted to cropland where no crop expanded.
    """
    hectares = dset["hectares-per-cell:ha"]
    cropland_occupation = (
        dset[f"{emit.CROPLAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha"] * hectares
    )
    pastureland_occupation = (
        dset[f"{emit.PASTURELAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha"]
        * hectares
    )

    commodity_to_totals: dict[
        Commodity | emit.NonCommodity, dict[str, xarray.DataArray]
    ] = collections.defaultdict(dict)
    for commodity in commodities:
        totals = commodity_to_totals[commodity]
        if isinstance(commodity, gpw_livestock.Livestock):
            land_class = emit.PASTURELAND
            span_to_share = dict.fromkeys(MAPSPAM_SPANS, commodity_to_share[commodity])
        else:
            land_class = emit.CROPLAND
            span_to_share = {
                span: span_to_crop_to_share[span][commodity] for span in MAPSPAM_SPANS
            }
        component_to_emissions = {
            component: sum(
                (
                    dset[
                        f"{land_class:s}-{component!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"
                    ]
                    * hectares
                    * span_to_share[(before, after)]
                    for before, after in MAPSPAM_SPANS
                ),
                start=xarray.zeros_like(hectares),
            )
            for component in emit.EmissionComponent
        }
        totals |= {
            component.column: component_to_emissions[component]
            for component in emit.EmissionComponent
        }

        # A livestock commodity takes its grazing share of the pasture and of the pasture's peat
        # occupation, and reports its herd; a crop takes its area share of the cropland's peat
        # occupation
        if isinstance(commodity, gpw_livestock.Livestock):
            occupation = pastureland_occupation * commodity_to_share[commodity]
            totals["commodity_hectares"] = gpw_livestock.get_livestock_hectares(
                dset=dset,
                grazing_share=commodity_to_share[commodity],
                year=LAST_SNAPSHOT_YEAR,
            )
            if commodity in gpw_livestock.LIVESTOCK_TO_SPECIES:
                totals[HERD_HEADS_COLUMN] = gpw_livestock.get_herd_heads(
                    dset=dset,
                    species=gpw_livestock.LIVESTOCK_TO_SPECIES[commodity],
                    year=LAST_SNAPSHOT_YEAR,
                )
        else:
            occupation = cropland_occupation * commodity_to_share[commodity]
            totals["commodity_hectares"] = get_crop_hectares(
                crop=commodity, dset=dset, year=LAST_SNAPSHOT_YEAR
            )
        # NB: occupation is a flat annual rate over the area drained, so dividing it back out
        # recovers that area on `emit`'s 30 m intersection
        totals["peatland_commodity_hectares"] = (
            occupation / emit.PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA
        )
        totals["peatland_occupation_emissions_mt"] = occupation
        totals["emissions_mt"] = sum(component_to_emissions.values(), start=occupation)

    commodity_to_totals[emit.NonCommodity.DROPPED]["emissions_mt"] = (
        dset["dropped-emissions:tco2e-per-ha"] * hectares
    )

    # The whole charge each component carries, as a commodity holding every share would take it
    component_to_charged = {
        component: sum(
            (
                dset[
                    f"{land_class:s}-{component!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"
                ]
                * hectares
                for land_class in (emit.CROPLAND, emit.PASTURELAND)
                for before, after in MAPSPAM_SPANS
            ),
            start=xarray.zeros_like(hectares),
        )
        for component in emit.EmissionComponent
    }
    unattributed = commodity_to_totals[emit.NonCommodity.UNATTRIBUTED]
    # NB: clipped at zero, since float32 shares that sum to one can round a hair above it
    unattributed |= {
        component.column: (
            component_to_charged[component]
            - sum(
                commodity_to_totals[commodity][component.column]
                for commodity in commodities
            )
        ).clip(min=0)
        for component in emit.EmissionComponent
    }
    unattributed["peatland_occupation_emissions_mt"] = (
        cropland_occupation
        + pastureland_occupation
        - sum(
            commodity_to_totals[commodity]["peatland_occupation_emissions_mt"]
            for commodity in commodities
        )
    ).clip(min=0)
    unattributed["emissions_mt"] = sum(
        (unattributed[component.column] for component in emit.EmissionComponent),
        start=unattributed["peatland_occupation_emissions_mt"],
    )

    # NB: the residual pasture row names no herd, so it carries no heads and `trace` declines to
    # publish an emission factor for it rather than dividing by zero
    return {
        name: {HERD_HEADS_COLUMN: numpy.nan} | totals
        for name, totals in utils.get_sum_totals(
            enum_to_name_to_darray=commodity_to_totals
        ).items()
    }


def get_crop_to_area_share(
    crops: tuple[Crop, ...], dset: xarray.Dataset, year: int
) -> dict[Crop, xarray.DataArray]:
    # NB: the denominator walks `year`'s own taxonomy, so every name here is already reported
    total_area = sum(
        (
            ifpri_mapspam.get_reported_quantity(
                dset=dset,
                quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                reported_crop_name=e.name,
                year=year,
            )
            for e in ifpri_mapspam.YEAR_TO_CROP_CLS[year]
        ),
        start=xarray.DataArray(numpy.float32(0)),
    )
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
    "commodity_hectares": float,
    "commodity_name": str,
    "emissions_mt": float,
    "forest_emissions_mt": float,
    "grassland_emissions_mt": float,
    "jurisdiction_name": str,
    "peatland_commodity_hectares": float,
    "peatland_conversion_emissions_mt": float,
    "peatland_occupation_emissions_mt": float,
    HERD_HEADS_COLUMN: float,
}


@storage.cache_to_parquet(version=1)
def workflow(
    assessment_year: int,
    commodity_names: tuple[str, ...],
    iso_3166: str,
    tile_id: str,
) -> pandas.DataFrame:
    commodities: tuple[Commodity, ...] = tuple(
        Crop[name] if name in Crop.__members__ else gpw_livestock.Livestock[name]
        for name in commodity_names
    )
    crops = tuple(commodity for commodity in commodities if isinstance(commodity, Crop))
    logger.info(
        f"Computing emissions for {commodities=:}, {assessment_year=:d} and {tile_id=:s}"
    )
    dairy_share = faostat.get_dairy_share(
        iso_3166=iso_3166,
        livestock=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
        years=emit.get_production_years_range(assessment_year=assessment_year),
    )
    logger.info(f"Milking cows are {dairy_share:.1%} of {iso_3166:s}'s cattle")
    merged = geo.exact_merge(
        get_downscaled_luc_emissions(assessment_year=assessment_year, tile_id=tile_id),
        harmonize.workflow(
            dataset_names=harmonize.Stack.STATISTICAL_ALLOCATION.value.dataset_names,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=harmonize.Stack.STATISTICAL_ALLOCATION.value.tile_resolution,
        ),
    )
    # Each grazer's livestock units per head, as FAOSTAT counts them for this country
    merged = merged.assign(
        gpw_livestock.get_band_name_to_livestock_units(
            dset=merged,
            species_to_units_per_head={
                gpw_livestock.Species[species.name]: units_per_head
                for species, units_per_head in faostat.get_species_to_livestock_units_per_head(
                    iso_3166=iso_3166,
                    livestock=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
                    livestock_units=faostat.load(
                        dataset=faostat.LIVESTOCK_PATTERNS_DATASET
                    ),
                ).items()
            },
        )
    )
    # NB: on the whole tile, since a clip one cell wide has no spacing to measure a cell by
    merged["hectares-per-cell:ha"] = emit.get_hectares_per_pixel(
        darray=merged["dropped-emissions:tco2e-per-ha"]
    )
    # NB: small enough to hold in memory, which speeds up the many reads for each jurisdiction
    merged = merged.load()

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
                    (before, after): get_crop_to_expansion_share(
                        after=after,
                        before=before,
                        crops=crops,
                        dset=clipped,
                    )
                    for (before, after) in MAPSPAM_SPANS
                }

                logger.info(
                    f"Populating emissions for {jurisdiction.id=!s} over "
                    f"{len(commodities):d} commodities"
                )
                commodity_name_to_totals = get_commodity_name_to_totals(
                    commodities=commodities,
                    # NB: peat occupation is a flux on land drained now, unrelated to expansion, so
                    # crops divide it by area share
                    commodity_to_share=get_crop_to_area_share(
                        crops=crops,
                        dset=clipped,
                        year=LAST_SNAPSHOT_YEAR,
                    )
                    | gpw_livestock.get_livestock_to_grazing_share(
                        dairy_share=dairy_share,
                        dset=clipped,
                        year=LAST_SNAPSHOT_YEAR,
                    ),
                    dset=clipped,
                    span_to_crop_to_share=span_to_crop_to_share,
                )
                for commodity_name, totals in commodity_name_to_totals.items():
                    yield totals | {
                        "admin_id": jurisdiction.id,
                        "admin_level": jurisdiction.level,
                        "commodity_name": commodity_name,
                        "jurisdiction_name": jurisdiction.name,
                    }

    if data := list(it()):
        first_record, *_ = data
        assert set(first_record) == set(SCHEMA)
    return (
        pandas.DataFrame.from_records(columns=list(SCHEMA), data=data)
        .astype(SCHEMA)
        .set_index(["admin_level", "admin_id", "commodity_name"])
    )
