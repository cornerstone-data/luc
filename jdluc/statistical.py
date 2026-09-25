import collections
import collections.abc
import enum
import logging

import numpy
import pandas
import rasterio.enums
import xarray

from jdluc import emit, geo, harmonize, storage, tiling, utils
from jdluc.datasets import (
    DatasetName,
    faostat_production,
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
    """The commodities dividing the pasture pool, PASTURE keeping what no other one's grazers do."""

    BEEF_CATTLE = enum.auto()
    PASTURE = enum.auto()


LIVESTOCK_TO_SPECIES = {
    Livestock.BEEF_CATTLE: frozenset({gpw_livestock.Species.CATTLE}),
}
assert set(LIVESTOCK_TO_SPECIES) == set(Livestock) - {Livestock.PASTURE}
# No grazer may be named by two commodities, or its units would be shared out twice
assert sum(len(species) for species in LIVESTOCK_TO_SPECIES.values()) == len(
    frozenset().union(*LIVESTOCK_TO_SPECIES.values())
)

# FAO's livestock units per head, a head's grazing equivalent: the South America row of the regional
# coefficients FAOSTAT Livestock Patterns applies (FAO 2011, after Chilonda and Otte 2006), applied
# everywhere. That row leaves buffalo blank, so it takes the 0.70 most other regions give it. Only
# the ratios between species matter, since a share divides each by the cell's total, and FAOSTAT
# holds the coefficients fixed over time.
SPECIES_TO_LIVESTOCK_UNITS_PER_HEAD = {
    gpw_livestock.Species.BUFFALO: 0.70,
    gpw_livestock.Species.CATTLE: 0.70,
    gpw_livestock.Species.GOAT: 0.10,
    gpw_livestock.Species.HORSE: 0.65,
    gpw_livestock.Species.SHEEP: 0.10,
}
assert set(SPECIES_TO_LIVESTOCK_UNITS_PER_HEAD) == set(gpw_livestock.Species)
assert {e.name for e in faostat_production.Species} == {
    e.name for e in gpw_livestock.Species
}


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
    "cropland-peatland-occupation:tco2e-per-ha",
    "dropped-emissions:tco2e-per-ha",
    "emissions:tco2e-per-ha:2000-2005",
    "emissions:tco2e-per-ha:2005-2010",
    "emissions:tco2e-per-ha:2010-2015",
    "emissions:tco2e-per-ha:2015-2020",
    "pastureland-peatland-occupation:tco2e-per-ha",
]


@storage.cache_to_zarr(version=0)
def get_downscaled_luc_emissions(tile_id: str) -> xarray.Dataset:
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
        emit.workflow(tile_id=tile_id),
    )

    logger.info("Splitting emissions by component per span")
    derived_variable_names: list[str] = []
    for (before, after), component_to_darray in emit.get_span_to_component_to_emissions(
        dset=luc_and_emissions
    ).items():
        for component, darray in component_to_darray.items():
            name = f"{component!s}:tco2e-per-ha:{before:d}-{after:d}"
            luc_and_emissions[name] = darray
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


def get_pasture_hectares(dset: xarray.Dataset, year: int) -> xarray.DataArray:
    darray = dset[f"pasture:fraction:{year:d}"]
    return darray * emit.get_hectares_per_pixel(darray=darray)


def get_livestock_to_grazing_share(
    dset: xarray.Dataset, year: int
) -> dict[Livestock, xarray.DataArray]:
    """Each commodity's share of a cell's pasture: its grazers' livestock units over all of them.

    Each livestock unit in a cell is taken to graze an equal slice of that cell's pasture.
    """
    species_to_units = {
        species: dset[gpw_livestock.get_band_name(species=species, year=year)]
        * units_per_head
        for species, units_per_head in SPECIES_TO_LIVESTOCK_UNITS_PER_HEAD.items()
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
        return (
            get_pasture_hectares(dset=dset, year=year)
            * get_livestock_to_grazing_share(dset=dset, year=year)[commodity]
        )
    else:
        return ifpri_mapspam.get_canonical_quantity(
            canonical_crop_name=commodity.value,
            dset=dset,
            quantity=ifpri_mapspam.Quantity.PHYSICAL_AREA,
            year=year,
        )


def get_commodity_to_share(
    after: int, before: int, crops: tuple[Crop, ...], dset: xarray.Dataset
) -> dict[Commodity, xarray.DataArray]:
    """Each commodity's share of all a cell emitted in a span, whatever each pixel's destination."""

    def get_expansion(crop: Crop) -> xarray.DataArray:
        return (
            get_commodity_hectares(commodity=crop, dset=dset, year=after)
            - get_commodity_hectares(commodity=crop, dset=dset, year=before)
        ).clip(min=0)

    attributed_expansion = sum(get_expansion(crop=crop) for crop in sorted(Crop))

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
    # Pastureland expands alongside the crops, measured net and clipped exactly as a crop's is:
    # GPW is annual, and its gross gain would charge pasture for churn MapSPAM cannot see
    pasture_expansion = (
        get_pasture_hectares(dset=dset, year=after)
        - get_pasture_hectares(dset=dset, year=before)
    ).clip(min=0)
    total_expansion = attributed_expansion + unattributed_expansion + pasture_expansion
    total_expansion = total_expansion.where(total_expansion > 0)
    # Share is zero when there is no expansion at all
    pasture_share = (pasture_expansion / total_expansion).fillna(0)
    return {
        crop: (get_expansion(crop=crop) / total_expansion).fillna(0) for crop in crops
    } | {
        # NB: split by who grazes the pasture at the end of the span, since clipping each livestock
        # commodity's own expansion would read a shift between grazers as new pasture
        livestock: pasture_share * grazing_share
        for livestock, grazing_share in get_livestock_to_grazing_share(
            dset=dset, year=after
        ).items()
    }


SPAN_TO_MAPSPAM_SPAN = {
    (2000, 2005): (2000, 2005),
    (2005, 2010): (2005, 2010),
    # NB: because MAPSPAM has no 2015 snapshot, we assume crop expansion is constant between
    # 2010 and 2020 -- and then use that one crop expansion for both spans
    (2010, 2015): (2010, 2020),
    (2015, 2020): (2010, 2020),
}
assert set(SPAN_TO_MAPSPAM_SPAN) == set(emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT)
MAPSPAM_SNAPSHOT_YEARS = tuple(
    sorted({year for span in SPAN_TO_MAPSPAM_SPAN.values() for year in span})
)
assert set(MAPSPAM_SNAPSHOT_YEARS).issubset(gpw_livestock.YEARS)


def get_discounted_snapshot_mean(
    year_to_value: dict[int, xarray.DataArray],
) -> xarray.DataArray:
    """A level -- area, production -- as a mean over the windows emissions are summed on."""
    return emit.get_linear_discounted_total(
        span_to_value={
            span: (year_to_value[before] + year_to_value[after]) / 2
            for span, (before, after) in SPAN_TO_MAPSPAM_SPAN.items()
        }
    ) / sum(emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT.values())


def get_commodity_name_to_totals(
    commodity_to_span_to_share: dict[Commodity, dict[emit.SpanType, xarray.DataArray]],
    dset: xarray.Dataset,
    occupation_shares: dict[Commodity, xarray.DataArray],
    species_to_year_to_kg_per_head: dict[gpw_livestock.Species, dict[int, float]],
) -> dict[str, dict[str, float]]:
    cropland_occupation_per_hectare = dset["cropland-peatland-occupation:tco2e-per-ha"]
    hectares = emit.get_hectares_per_pixel(darray=cropland_occupation_per_hectare)
    cropland_occupation = cropland_occupation_per_hectare * hectares
    pastureland_occupation = (
        dset["pastureland-peatland-occupation:tco2e-per-ha"] * hectares
    )

    commodity_to_totals: dict[
        Commodity | emit.NonCommodity, dict[str, xarray.DataArray]
    ] = collections.defaultdict(dict)
    for commodity, span_to_share in commodity_to_span_to_share.items():
        totals = commodity_to_totals[commodity]
        component_to_emissions = {
            component: emit.get_linear_discounted_total(
                span_to_value={
                    (before, after): dset[
                        f"{component!s}:tco2e-per-ha:{before:d}-{after:d}"
                    ]
                    * hectares
                    * span_to_share[SPAN_TO_MAPSPAM_SPAN[(before, after)]]
                    for (before, after) in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT
                }
            )
            for component in ("emissions", *emit.EmissionComponent)
        }
        totals |= {
            component.column: component_to_emissions[component]
            for component in emit.EmissionComponent
        }

        # A livestock commodity takes its grazing share of the pasture band and its herd's carcass
        # weight, while a crop takes its area share of the cropland band and MapSPAM's production
        if isinstance(commodity, Livestock):
            occupation = pastureland_occupation * occupation_shares[commodity]
            if commodity in LIVESTOCK_TO_SPECIES:
                # NB: a cell's herd grazes that cell's pasture, so its heads count wherever the
                # cell holds any rather than weighted by the fraction, which would discount them
                # twice. Herds in cells without pasture -- feedlots, rangeland -- earn nothing.
                # Production sits where the herd grazes, as a crop's sits where it grows, so it
                # follows the same standing herd the grazing split does.
                year_to_kg_per_ha = {
                    year: sum(
                        (
                            dset[gpw_livestock.get_band_name(species=one, year=year)]
                            * species_to_year_to_kg_per_head[one][year]
                            for one in sorted(LIVESTOCK_TO_SPECIES[commodity])
                        ),
                        start=xarray.DataArray(numpy.float32(0)),
                    ).where(dset[f"pasture:fraction:{year:d}"] > 0, other=0)
                    for year in MAPSPAM_SNAPSHOT_YEARS
                }
                totals["production_mt"] = (
                    get_discounted_snapshot_mean(year_to_value=year_to_kg_per_ha)
                    * hectares
                    / faostat_production.KG_PER_TONNE
                )
        else:
            occupation = cropland_occupation * occupation_shares[commodity]
            totals["production_mt"] = get_discounted_snapshot_mean(
                year_to_value={
                    year: ifpri_mapspam.get_canonical_quantity(
                        canonical_crop_name=commodity.value,
                        dset=dset,
                        quantity=ifpri_mapspam.Quantity.PRODUCTION,
                        year=year,
                    )
                    for year in MAPSPAM_SNAPSHOT_YEARS
                },
            )

        totals["commodity_hectares"] = get_discounted_snapshot_mean(
            year_to_value={
                year: get_commodity_hectares(commodity=commodity, dset=dset, year=year)
                for year in MAPSPAM_SNAPSHOT_YEARS
            },
        )
        # NB: occupation is a flat annual rate over the area drained, so dividing it back out
        # recovers that area on `emit`'s 30 m intersection
        totals["peatland_commodity_hectares"] = (
            occupation / emit.PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA
        )
        totals["peatland_occupation_emissions_mt"] = occupation
        totals["emissions_mt"] = component_to_emissions["emissions"] + occupation

    commodity_to_totals[emit.NonCommodity.DROPPED]["emissions_mt"] = (
        dset["dropped-emissions:tco2e-per-ha"] * hectares
    )

    # NB: the residual pasture row names no herd, so it carries no production and `trace` declines
    # to publish an emission factor for it rather than dividing by zero
    return {
        name: {"production_mt": numpy.nan} | totals
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
}


@storage.cache_to_parquet(version=0)
def workflow(
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
    logger.info(f"Computing emissions for {commodities=:} and {tile_id=:s}")
    merged = geo.exact_merge(
        # NB: this is deferred because it is expensive and would like to cache it
        get_downscaled_luc_emissions(tile_id=tile_id),
        # MAPSPAM
        harmonize.workflow(
            dataset_names=DATASET_NAMES,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.MAPSPAM,
        ),
    )

    # National carcass weight per standing head. A year FAOSTAT reports no meat for yields none,
    # as `attribute` would read a missing production anyway.
    livestock = faostat_production.load(
        dataset=faostat_production.LIVESTOCK_DATASET
    ).reset_index()
    livestock = livestock[
        (livestock["admin_id"] == iso_3166)
        & livestock["year"].isin(MAPSPAM_SNAPSHOT_YEARS)
    ]
    kg_per_head = (livestock["production_kg"] / livestock["stocks_head"]).fillna(0)
    species_to_year_to_kg_per_head = {
        species: dict.fromkeys(MAPSPAM_SNAPSHOT_YEARS, 0.0)
        for species in gpw_livestock.Species
    }
    for name, year, value in zip(
        livestock["commodity_name"].astype(str),
        livestock["year"].astype(int),
        kg_per_head.astype(float),
        strict=True,
    ):
        species_to_year_to_kg_per_head[gpw_livestock.Species[name]][year] = value

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
                    for (before, after) in dict.fromkeys(SPAN_TO_MAPSPAM_SPAN.values())
                }
                commodity_to_span_to_share = {
                    commodity: {
                        span: commodity_to_share[commodity]
                        for span, commodity_to_share in span_to_commodity_to_share.items()
                    }
                    for commodity in commodities
                }

                logger.info(
                    f"Populating emissions for {jurisdiction.id=!s} over "
                    f"{len(commodities):d} commodities"
                )
                commodity_name_to_totals = get_commodity_name_to_totals(
                    commodity_to_span_to_share=commodity_to_span_to_share,
                    dset=clipped,
                    # NB: unlike conversion emissions, peatland occupation is a land-management flux on land
                    # that is drained *now* -- it has no relationship to expansion, and expansion is zero on
                    # the long-established peat cropland that dominates this pool. So allocate it by each
                    # crop's share of area occupied -- matching the approach for the jurisdictional-direct leg.
                    # Pasture's divides by grazing share instead.
                    occupation_shares=get_crop_to_area_share(
                        crops=crops,
                        dset=clipped,
                        year=max(ifpri_mapspam.YEARS),
                    )
                    | get_livestock_to_grazing_share(
                        dset=clipped, year=max(ifpri_mapspam.YEARS)
                    ),
                    species_to_year_to_kg_per_head=species_to_year_to_kg_per_head,
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
