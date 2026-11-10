import collections
import collections.abc
import enum
import logging
import math

import numpy
import pandas
import rasterio.enums
import xarray

from jdluc import emit, geo, harmonize, storage, tiling, utils
from jdluc.datasets import (
    DatasetName,
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


@enum.unique
class Livestock(enum.StrEnum):
    """The commodities dividing pasture's share of a cell; PASTURE keeps what the others don't."""

    BEEF_CATTLE = enum.auto()
    PASTURE = enum.auto()


LIVESTOCK_TO_SPECIES = {
    Livestock.BEEF_CATTLE: (gpw_livestock.Species.CATTLE,),
}
assert set(LIVESTOCK_TO_SPECIES) == set(Livestock) - {Livestock.PASTURE}
# No grazer may be named by two commodities, or its livestock units would count toward both shares
assert sum(len(species) for species in LIVESTOCK_TO_SPECIES.values()) == len(
    set(LIVESTOCK_TO_SPECIES.values())
)
# `trace` prices a commodity's herd at one species' carcass weight per head
assert all(len(species) == 1 for species in LIVESTOCK_TO_SPECIES.values())

assert {e.name for e in faostat.Species} == {e.name for e in gpw_livestock.Species}


# `Crop` stays exactly MapSPAM's recoverable crops, because everything that divides by one looks
# its name up there
type Commodity = Crop | Livestock


DATASET_NAMES = (
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2000,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2005,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2010,
    DatasetName.IFPRI_MAPSPAM_PHYSICAL_AREA_2020,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2000,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2005,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2010,
    DatasetName.IFPRI_MAPSPAM_PRODUCTION_2020,
    DatasetName.GPW_LIVESTOCK,
)


EMIT_VARIABLE_NAMES = [
    f"{emit.CROPLAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha",
    f"{emit.PASTURELAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha",
]

# Components divided by the pooled expansion share. The peat pulse is not one of them: it is stored
# as two bands, peat drained for cropland and peat drained for pasture, each with its own share.
POOLED_COMPONENTS = (emit.EmissionComponent.FOREST, emit.EmissionComponent.GRASSLAND)
assert set(POOLED_COMPONENTS) == set(emit.EmissionComponent) - {
    emit.EmissionComponent.PEATLAND_CONVERSION
}


@storage.cache_to_zarr(version=0)
def get_downscaled_luc_emissions(assessment_year: int, tile_id: str) -> xarray.Dataset:
    logger.info("Computing emissions on the GLAD grid")
    luc_and_emissions = geo.exact_merge(
        harmonize.workflow(
            dataset_names=harmonize.Stack.LUC_AND_EMISSIONS.value,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.GLAD,
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
        in_span = conversion_year.isin(
            [
                year
                for year in conversion_years
                if get_mapspam_span(year=year) == (before, after)
            ]
        )
        for component in POOLED_COMPONENTS:
            name = f"{component!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"
            luc_and_emissions[name] = component_to_emissions[component].where(
                in_span, other=0
            )
            derived_variable_names.append(name)
        # NB: `emit` puts soil emissions only on converted pixels, and every conversion ends in
        # cropland or pasture, so the two bands add up to the whole pulse
        for land_class, conversions in (
            (emit.CROPLAND, emit.TO_CROPLAND),
            (emit.PASTURELAND, emit.TO_PASTURE),
        ):
            name = f"{land_class:s}-{emit.EmissionComponent.PEATLAND_CONVERSION!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"
            luc_and_emissions[name] = component_to_emissions[
                emit.EmissionComponent.PEATLAND_CONVERSION
            ].where(
                in_span & luc_and_emissions["conversion"].isin(conversions), other=0
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

    logger.info("Measuring pasture extent at each MAPSPAM snapshot")
    for year in MAPSPAM_SNAPSHOT_YEARS:
        name = f"pasture:fraction:{year:d}"
        # NB: a mask here and an averaging downscale below, so this arrives as a cell fraction
        luc_and_emissions[name] = (
            luc_and_emissions[
                gpw_grassland.DATASET.fully_qualified_band_names[
                    gpw_grassland.YEARS.index(year)
                ]
            ]
            == gpw_grassland.Grassland.CULTIVATED
        ).astype(numpy.float32)
        derived_variable_names.append(name)

    logger.info("Downsampling emissions from the GLAD grid to the MAPSPAM grid")
    grid = harmonize.Grid.from_tile_id_resolution(
        resolution=tiling.TileResolution.MAPSPAM, tile_id=tile_id
    )
    return xarray.Dataset(
        {
            variable_name: geo.downscale_darray(
                darray=luc_and_emissions[variable_name],
                epsg=grid.epsg,
                height=grid.resolution.y,
                # NB: without an equal-area CRS, this presents a small error
                resampling=rasterio.enums.Resampling.average,
                transform=grid.transform,
                width=grid.resolution.x,
            )
            for variable_name in (*EMIT_VARIABLE_NAMES, *derived_variable_names)
        }
    )


def get_band_name_to_livestock_units(
    dset: xarray.Dataset, species_to_units_per_head: dict[gpw_livestock.Species, float]
) -> dict[str, xarray.DataArray]:
    """Each grazer's density in livestock units/ha, for assigning beside its density in heads.

    Livestock units weigh how much one head of a species grazes, so they divide pasture between
    grazers; production counts heads, and reads the heads bands.
    """
    return {
        f"{species:s}:livestock-units-per-ha:{year:d}": dset[
            gpw_livestock.get_band_name(species=species, year=year)
        ]
        * species_to_units_per_head[species]
        for species, year in gpw_livestock.SPECIES_YEARS
    }


def get_livestock_to_grazing_share(
    dset: xarray.Dataset, year: int
) -> dict[Livestock, xarray.DataArray]:
    """Each commodity's share of a cell's pasture: its grazers' livestock units over all of them.

    Each livestock unit in a cell is taken to graze an equal slice of that cell's pasture. Only the
    ratios between species' units per head matter, since a share divides each by the cell's total.
    """
    species_to_units = {
        species: dset[f"{species:s}:livestock-units-per-ha:{year:d}"]
        for species in gpw_livestock.Species
    }
    zero = xarray.DataArray(numpy.float32(0))
    total = sum(species_to_units.values(), start=zero)
    total = total.where(total > 0)
    shares = {
        # Share is zero where no grazer is mapped at all
        livestock: (
            sum((species_to_units[one] for one in species), start=zero) / total
        ).fillna(0)
        for livestock, species in LIVESTOCK_TO_SPECIES.items()
    }
    return shares | {Livestock.PASTURE: 1 - sum(shares.values(), start=zero)}


def get_commodity_hectares(
    commodity: Commodity, dset: xarray.Dataset, year: int
) -> xarray.DataArray:
    if isinstance(commodity, Livestock):
        pasture_fraction = dset[f"pasture:fraction:{year:d}"]
        return (
            pasture_fraction
            * emit.get_hectares_per_pixel(darray=pasture_fraction)
            * get_livestock_to_grazing_share(dset=dset, year=year)[commodity]
        )
    else:
        return ifpri_mapspam.get_canonical_quantity(
            canonical_crop_name=commodity.value,
            dset=dset,
            quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
            year=year,
        )


def get_crop_expansion(
    after: int, before: int, crop: Crop, dset: xarray.Dataset
) -> xarray.DataArray:
    return (
        get_commodity_hectares(commodity=crop, dset=dset, year=after)
        - get_commodity_hectares(commodity=crop, dset=dset, year=before)
    ).clip(min=0)


def get_cropland_expansion(
    after: int, before: int, dset: xarray.Dataset
) -> xarray.DataArray:
    """Every crop's expansion in a cell over a span, whether or not `Crop` claims it."""
    attributed_expansion = sum(
        get_crop_expansion(after=after, before=before, crop=crop, dset=dset)
        for crop in sorted(Crop)
    )

    # Drop any crops which are being newly tracked so they aren't interpreted as an expansion from zero
    before_names = ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[before]
    after_names = ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[after]
    if (before, after) in ifpri_mapspam.SPANS_WITH_COMPARABLE_CROP_NAMES:
        before_names = after_names = before_names & after_names

    def get_unattributed(names: set[str], year: int) -> xarray.DataArray:
        # NB: no canonical lookup is required -- these names are already `year`'s own, and
        # nothing divides by them, so the raw band is the whole answer
        return sum(
            (
                ifpri_mapspam.get_reported_quantity(
                    dset=dset,
                    quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
                    reported_crop_name=name,
                    year=year,
                )
                for name in sorted(names)
            ),
            start=xarray.DataArray(numpy.float32(0)),
        )

    unattributed_expansion = (
        get_unattributed(names=after_names, year=after)
        - get_unattributed(names=before_names, year=before)
    ).clip(min=0)
    return attributed_expansion + unattributed_expansion


def get_commodity_to_share(
    after: int, before: int, crops: tuple[Crop, ...], dset: xarray.Dataset
) -> dict[Commodity, xarray.DataArray]:
    """Each commodity's share of everything a cell emitted in a span, whatever each pixel became."""
    # Pastureland expands alongside the crops, measured net and clipped exactly as a crop's is:
    # GPW is annual, and its gross gain would charge pasture for churn MapSPAM cannot see
    pasture_fraction_after = dset[f"pasture:fraction:{after:d}"]
    pasture_fraction_before = dset[f"pasture:fraction:{before:d}"]
    pasture_expansion = (
        pasture_fraction_after
        * emit.get_hectares_per_pixel(darray=pasture_fraction_after)
        - pasture_fraction_before
        * emit.get_hectares_per_pixel(darray=pasture_fraction_before)
    ).clip(min=0)
    total_expansion = (
        get_cropland_expansion(after=after, before=before, dset=dset)
        + pasture_expansion
    )
    total_expansion = total_expansion.where(total_expansion > 0)
    # Share is zero when there is no expansion at all
    pasture_share = (pasture_expansion / total_expansion).fillna(0)
    return {
        crop: (
            get_crop_expansion(after=after, before=before, crop=crop, dset=dset)
            / total_expansion
        ).fillna(0)
        for crop in crops
    } | {
        # NB: pasture's share is split by who grazes at the end of the span. Measuring each
        # livestock commodity's own expansion instead would count one grazer replacing another as
        # new pasture.
        livestock: pasture_share * grazing_share
        for livestock, grazing_share in get_livestock_to_grazing_share(
            dset=dset, year=after
        ).items()
    }


def get_commodity_to_peatland_conversion_share(
    after: int, before: int, crops: tuple[Crop, ...], dset: xarray.Dataset
) -> dict[Commodity, xarray.DataArray]:
    """Each commodity's share of the pulse on peat drained for its own land class.

    A crop takes its share of crop expansion alone, so pasture expanding beside it takes none of
    the pulse on peat drained for cropland, and where no crop expanded that pulse goes to nobody.
    The grazers divide the pulse on peat drained for pasture by grazing share at the end of the
    span, which always sums to one.
    """
    cropland_expansion = get_cropland_expansion(after=after, before=before, dset=dset)
    cropland_expansion = cropland_expansion.where(cropland_expansion > 0)
    return {
        crop: (
            get_crop_expansion(after=after, before=before, crop=crop, dset=dset)
            / cropland_expansion
        ).fillna(0)
        for crop in crops
    } | get_livestock_to_grazing_share(dset=dset, year=after)


MAPSPAM_SPANS = ((2000, 2005), (2005, 2010), (2010, 2020))
MAPSPAM_SNAPSHOT_YEARS = tuple(
    sorted({year for span in MAPSPAM_SPANS for year in span})
)
assert set(MAPSPAM_SNAPSHOT_YEARS).issubset(gpw_livestock.YEARS)


def get_mapspam_span(year: int) -> tuple[int, int]:
    """The MapSPAM span whose shares a conversion in `year` takes: the one holding it, open below
    and closed above.

    MapSPAM has no 2015 snapshot, so every conversion from 2011 to 2020 takes the one expansion
    between 2010 and 2020, and it has none after 2020, so every later conversion takes it too.
    """
    *_, last_snapshot_year = MAPSPAM_SNAPSHOT_YEARS
    (span,) = (
        (before, after)
        for before, after in MAPSPAM_SPANS
        if before < min(year, last_snapshot_year) <= after
    )
    return span


def get_snapshot_year_to_weight(assessment_year: int) -> dict[int, float]:
    """Each snapshot's weight in the means of a crop's production and a commodity's area: every
    conversion year's discount weight, split evenly between the two snapshots of its MapSPAM span.
    A year after the last snapshot gives its whole weight to that snapshot instead.
    """
    *_, last_snapshot_year = MAPSPAM_SNAPSHOT_YEARS
    year_to_weight = emit.get_year_to_linear_discount_weight(
        assessment_year=assessment_year
    )
    year_to_snapshot_years = {
        year: (last_snapshot_year,)
        if year > last_snapshot_year
        else get_mapspam_span(year=year)
        for year in year_to_weight
    }
    snapshot_year_to_weight = {
        snapshot_year: sum(
            weight / len(year_to_snapshot_years[year])
            for year, weight in year_to_weight.items()
            if snapshot_year in year_to_snapshot_years[year]
        )
        for snapshot_year in MAPSPAM_SNAPSHOT_YEARS
    }
    assert math.isclose(sum(snapshot_year_to_weight.values()), 1)
    return snapshot_year_to_weight


def get_herd_heads_column(year: int) -> str:
    return f"herd_heads_{year:d}"


HERD_HEADS_COLUMNS = tuple(
    get_herd_heads_column(year=year) for year in MAPSPAM_SNAPSHOT_YEARS
)


def get_commodity_name_to_totals(
    commodities: tuple[Commodity, ...],
    dset: xarray.Dataset,
    occupation_shares: dict[Commodity, xarray.DataArray],
    snapshot_year_to_weight: dict[int, float],
    span_to_commodity_to_peatland_conversion_share: dict[
        tuple[int, int], dict[Commodity, xarray.DataArray]
    ],
    span_to_commodity_to_share: dict[
        tuple[int, int], dict[Commodity, xarray.DataArray]
    ],
) -> dict[str, dict[str, float]]:
    """Each commodity's emissions and area, a crop's production and a herd's heads, summed over a
    jurisdiction's cells.

    Both share mappings are keyed by MapSPAM span, and each divides that span's discounted
    emissions bands.

    Beside the commodities sit `DROPPED` and `UNATTRIBUTED` (see `emit.NonCommodity`). Here
    `UNATTRIBUTED` is what the shares leave of the charge: conversion and occupation by the
    MapSPAM crops no row models, the forest and grassland charge where neither cropland nor
    pasture expanded, and the cropland peat pulse where no crop expanded.
    """
    cropland_occupation_per_hectare = dset[
        f"{emit.CROPLAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha"
    ]
    hectares = emit.get_hectares_per_pixel(darray=cropland_occupation_per_hectare)
    cropland_occupation = cropland_occupation_per_hectare * hectares
    pastureland_occupation = (
        dset[f"{emit.PASTURELAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha"]
        * hectares
    )

    commodity_to_totals: dict[
        Commodity | emit.NonCommodity, dict[str, xarray.DataArray]
    ] = collections.defaultdict(dict)
    for commodity in commodities:
        totals = commodity_to_totals[commodity]
        land_class = (
            emit.PASTURELAND if isinstance(commodity, Livestock) else emit.CROPLAND
        )
        component_to_emissions = {
            component: sum(
                (
                    dset[f"{component!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"]
                    * hectares
                    * span_to_commodity_to_share[(before, after)][commodity]
                    for before, after in MAPSPAM_SPANS
                ),
                start=xarray.zeros_like(hectares),
            )
            for component in POOLED_COMPONENTS
        } | {
            emit.EmissionComponent.PEATLAND_CONVERSION: sum(
                (
                    dset[
                        f"{land_class:s}-{emit.EmissionComponent.PEATLAND_CONVERSION!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"
                    ]
                    * hectares
                    * span_to_commodity_to_peatland_conversion_share[(before, after)][
                        commodity
                    ]
                    for before, after in MAPSPAM_SPANS
                ),
                start=xarray.zeros_like(hectares),
            )
        }
        totals |= {
            component.column: component_to_emissions[component]
            for component in emit.EmissionComponent
        }

        # Peat occupation and production: a livestock commodity takes its grazing share of the
        # pasture peat band and reports its herd; a crop takes its area share of the cropland peat
        # band and reports MapSPAM's production
        if isinstance(commodity, Livestock):
            occupation = pastureland_occupation * occupation_shares[commodity]
            if commodity in LIVESTOCK_TO_SPECIES:
                # NB: GPW's density is heads per hectare of the whole cell, so times the cell's
                # hectares it is the cell's entire herd, all of it grazing the cell's pasture
                # however little of the cell that is. Scaling by the pasture fraction too would
                # drop part of the herd. Cells with no mapped pasture -- feedlots, natural
                # rangeland -- produce nothing. Counting production where the herd stands, as a
                # crop's is counted where it grows, keeps it on the same animals the grazing split
                # weighs.
                (species,) = LIVESTOCK_TO_SPECIES[commodity]
                totals |= {
                    get_herd_heads_column(year=year): dset[
                        gpw_livestock.get_band_name(species=species, year=year)
                    ].where(dset[f"pasture:fraction:{year:d}"] > 0, other=0)
                    * hectares
                    for year in MAPSPAM_SNAPSHOT_YEARS
                }
        else:
            occupation = cropland_occupation * occupation_shares[commodity]
            # NB: a level, so a mean over the snapshots under the weights emissions are summed with
            totals["production_mt"] = sum(
                (
                    snapshot_year_to_weight[year]
                    * ifpri_mapspam.get_canonical_quantity(
                        canonical_crop_name=commodity.value,
                        dset=dset,
                        quantity=ifpri_mapspam.Quantity.PRODUCTION,
                        year=year,
                    )
                    for year in MAPSPAM_SNAPSHOT_YEARS
                ),
                start=xarray.DataArray(numpy.float32(0)),
            )

        totals["commodity_hectares"] = sum(
            (
                snapshot_year_to_weight[year]
                * get_commodity_hectares(commodity=commodity, dset=dset, year=year)
                for year in MAPSPAM_SNAPSHOT_YEARS
            ),
            start=xarray.DataArray(numpy.float32(0)),
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
                dset[f"{component!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"]
                * hectares
                for before, after in MAPSPAM_SPANS
            ),
            start=xarray.zeros_like(hectares),
        )
        for component in POOLED_COMPONENTS
    } | {
        emit.EmissionComponent.PEATLAND_CONVERSION: sum(
            (
                dset[
                    f"{land_class:s}-{emit.EmissionComponent.PEATLAND_CONVERSION!s}-emissions:tco2e-per-ha:{before:d}-{after:d}"
                ]
                * hectares
                for land_class in (emit.CROPLAND, emit.PASTURELAND)
                for before, after in MAPSPAM_SPANS
            ),
            start=xarray.zeros_like(hectares),
        )
    }
    unattributed = commodity_to_totals[emit.NonCommodity.UNATTRIBUTED]
    unattributed |= {
        component.column: component_to_charged[component]
        - sum(
            commodity_to_totals[commodity][component.column]
            for commodity in commodities
        )
        for component in emit.EmissionComponent
    }
    unattributed["peatland_occupation_emissions_mt"] = (
        cropland_occupation
        + pastureland_occupation
        - sum(
            commodity_to_totals[commodity]["peatland_occupation_emissions_mt"]
            for commodity in commodities
        )
    )
    unattributed["emissions_mt"] = sum(
        (unattributed[component.column] for component in emit.EmissionComponent),
        start=unattributed["peatland_occupation_emissions_mt"],
    )

    # NB: the residual pasture row names no herd, so it carries no production and `trace` declines
    # to publish an emission factor for it rather than dividing by zero
    return {
        name: {"production_mt": numpy.nan}
        | dict.fromkeys(HERD_HEADS_COLUMNS, numpy.nan)
        | totals
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
    "production_mt": float,
} | dict.fromkeys(HERD_HEADS_COLUMNS, float)


@storage.cache_to_parquet(version=0)
def workflow(
    assessment_year: int,
    commodity_names: tuple[str, ...],
    iso_3166: str,
    tile_id: str,
) -> pandas.DataFrame:
    from rioxarray.exceptions import NoDataInBounds

    commodities: tuple[Commodity, ...] = tuple(
        Crop[name] if name in Crop.__members__ else Livestock[name]
        for name in commodity_names
    )
    crops = tuple(commodity for commodity in commodities if isinstance(commodity, Crop))
    logger.info(
        f"Computing emissions for {commodities=:}, {assessment_year=:d} and {tile_id=:s}"
    )
    snapshot_year_to_weight = get_snapshot_year_to_weight(
        assessment_year=assessment_year
    )
    merged = geo.exact_merge(
        # NB: this is deferred because it is expensive and would like to cache it
        get_downscaled_luc_emissions(assessment_year=assessment_year, tile_id=tile_id),
        # MAPSPAM
        harmonize.workflow(
            dataset_names=DATASET_NAMES,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.MAPSPAM,
        ),
    )

    # Each grazer's livestock units per head, as FAOSTAT counts them for this country
    merged = merged.assign(
        get_band_name_to_livestock_units(
            dset=merged,
            species_to_units_per_head={
                gpw_livestock.Species[species.name]: units_per_head
                for species, units_per_head in faostat.get_species_to_livestock_units_per_head(
                    iso_3166=iso_3166,
                    livestock_units=faostat.load(
                        dataset=faostat.LIVESTOCK_PATTERNS_DATASET
                    ),
                    stocks=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
                ).items()
            },
        )
    )
    # NB: usually we avoid dumping to local mmemory, but since this xarray is small there is no
    # OOM risk.  Doing so speeds up the many downstream reads for each jurisdiction
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
                span_to_commodity_to_share = {
                    (before, after): get_commodity_to_share(
                        after=after,
                        before=before,
                        crops=crops,
                        dset=clipped,
                    )
                    for (before, after) in MAPSPAM_SPANS
                }
                span_to_commodity_to_peatland_conversion_share = {
                    (before, after): get_commodity_to_peatland_conversion_share(
                        after=after,
                        before=before,
                        crops=crops,
                        dset=clipped,
                    )
                    for (before, after) in span_to_commodity_to_share
                }

                logger.info(
                    f"Populating emissions for {jurisdiction.id=!s} over "
                    f"{len(commodities):d} commodities"
                )
                commodity_name_to_totals = get_commodity_name_to_totals(
                    commodities=commodities,
                    dset=clipped,
                    # NB: unlike conversion emissions, peatland occupation is a land-management flux on land
                    # that is drained *now* -- it has no relationship to expansion, and expansion is zero on
                    # the long-established peat cropland that dominates this pool. So allocate it by each
                    # crop's share of area occupied -- matching the approach for the jurisdictional-direct leg.
                    # The pasture peat band is divided by grazing share instead.
                    occupation_shares=get_crop_to_area_share(
                        crops=crops,
                        dset=clipped,
                        year=max(ifpri_mapspam.YEARS),
                    )
                    | get_livestock_to_grazing_share(
                        dset=clipped, year=max(ifpri_mapspam.YEARS)
                    ),
                    snapshot_year_to_weight=snapshot_year_to_weight,
                    span_to_commodity_to_peatland_conversion_share=span_to_commodity_to_peatland_conversion_share,
                    span_to_commodity_to_share=span_to_commodity_to_share,
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
