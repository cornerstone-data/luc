"""Compute per-pixel land-conversion emissions from the harmonized dataset.

Resolves which of five conversions each pixel underwent and when, releases the carbon pools that
conversion carries, charges them to the five-year span holding that year, adds ongoing
peatland-occupation emissions split by destination, and applies the GHGP 20-year linear discount.

Returns a cached xarray.Dataset: the conversion, the year its source class ended and the datasets
that claimed its destination; vegetation, soil and total emissions per span; the two occupation
bands; the discounted per-hectare total; the source carbon no destination claimed; and a
hectares-per-pixel band for downstream area-scaling.

Source carbon that no destination claimed is charged to nobody and reported on its own as
`dropped-emissions`.

The tile set comes from the positional ISO 3166 alpha-3 codes -- or, with `--backfill`,
from `tiling.GLOBAL_NATURE_WATCH_TILE_IDS`.

Example invocations:
  uv run python jdluc/emit.py USA
  uv run python jdluc/emit.py --backfill
"""

import argparse
import collections.abc
import dataclasses
import enum
import logging
import math
import typing

import numpy
import xarray

from jdluc import geo, harmonize, storage, tiling, utils
from jdluc.datasets import (
    descals_oil_palm,
    gnw_global_peatlands,
    gnw_harris_agb,
    gnw_tcl,
    gpw_grassland,
    huang_bgb,
    ipcc_climate_zones,
    liao_gaced30,
    soilgrids_ocs,
    worldbank_jurisdictions,
)

logger = logging.getLogger(__name__)


# IPCC 2006, Vol 4, Ch 4, §4.5 (living woody biomass).
CARBON_PER_BIOMASS_LIVE_WOOD: float = 0.47
# Global forest mean root-to-shoot ratio from Huang et al. (2021), Earth System Science Data 13:4263-4274
ROOT_TO_SHOOT_RATIO = 0.25


def get_belowground_carbon(
    aboveground_biomass: xarray.DataArray, belowground_biomass: xarray.DataArray
) -> xarray.DataArray:
    return (
        (
            # Default to BGB when it is provided
            belowground_biomass.where(belowground_biomass > 0, other=0)
            # Fallback to AGB * R2S when it isn't
            + ROOT_TO_SHOOT_RATIO
            * aboveground_biomass.where(belowground_biomass == 0, other=0)
        )
        * CARBON_PER_BIOMASS_LIVE_WOOD
    ).rename("tcarbon-per-ha")


# CDM AR-TOOL-12 dead wood and litter factors, expressed as fractions of above-ground biomass
CLIMATE_ZONE_TO_DEAD_ORGANIC_MATTER_PARAMETERS: dict[
    ipcc_climate_zones.Zone, tuple[float, float]
] = {
    ipcc_climate_zones.Zone.TROPICAL_WET: (0.06, 0.01),
    ipcc_climate_zones.Zone.TROPICAL_MOIST: (0.01, 0.01),
    ipcc_climate_zones.Zone.TROPICAL_DRY: (0.02, 0.04),
    ipcc_climate_zones.Zone.TROPICAL_MONTANE: (0.07, 0.01),
    ipcc_climate_zones.Zone.WARM_TEMPERATE_MOIST: (0.08, 0.04),
    ipcc_climate_zones.Zone.WARM_TEMPERATE_DRY: (0.08, 0.04),
    ipcc_climate_zones.Zone.COOL_TEMPERATE_MOIST: (0.08, 0.04),
    ipcc_climate_zones.Zone.COOL_TEMPERATE_DRY: (0.08, 0.04),
    ipcc_climate_zones.Zone.BOREAL_MOIST: (0.08, 0.04),
    ipcc_climate_zones.Zone.BOREAL_DRY: (0.08, 0.04),
}
assert set(ipcc_climate_zones.Zone) == set(
    CLIMATE_ZONE_TO_DEAD_ORGANIC_MATTER_PARAMETERS
)

# CDM AR-TOOL-12 (dead wood and litter pools).
CARBON_PER_BIOMASS_DEAD_WOOD = 0.50
CARBON_PER_BIOMASS_LITTER = 0.37


def get_dead_organic_matter_carbon(
    aboveground_biomass: xarray.DataArray, climate_zones: xarray.DataArray
) -> xarray.DataArray:
    ret = sum(
        aboveground_biomass.where(climate_zones == climate_zone.value, other=0)
        * (
            CARBON_PER_BIOMASS_DEAD_WOOD * dead_wood_fraction
            + CARBON_PER_BIOMASS_LITTER * litter_fraction
        )
        for climate_zone, (
            dead_wood_fraction,
            litter_fraction,
        ) in CLIMATE_ZONE_TO_DEAD_ORGANIC_MATTER_PARAMETERS.items()
    )
    return typing.cast(xarray.DataArray, ret).rename("tcarbon-per-ha")


# Houghton/BLUE total vegetation carbon density for grassland / shrubland pixels
CLIMATE_ZONE_TO_GRASSLAND_TCARBON_PER_HA: dict[ipcc_climate_zones.Zone, float] = {
    ipcc_climate_zones.Zone.TROPICAL_WET: 18.0,
    ipcc_climate_zones.Zone.TROPICAL_MOIST: 18.0,
    ipcc_climate_zones.Zone.TROPICAL_DRY: 7.0,
    ipcc_climate_zones.Zone.TROPICAL_MONTANE: 7.0,
    ipcc_climate_zones.Zone.WARM_TEMPERATE_MOIST: 7.0,
    ipcc_climate_zones.Zone.WARM_TEMPERATE_DRY: 5.0,
    ipcc_climate_zones.Zone.COOL_TEMPERATE_MOIST: 7.0,
    ipcc_climate_zones.Zone.COOL_TEMPERATE_DRY: 5.0,
    ipcc_climate_zones.Zone.BOREAL_MOIST: 6.0,
    ipcc_climate_zones.Zone.BOREAL_DRY: 3.0,
}
assert set(ipcc_climate_zones.Zone) == set(CLIMATE_ZONE_TO_GRASSLAND_TCARBON_PER_HA)


def get_grassland_carbon(climate_zones: xarray.DataArray) -> xarray.DataArray:
    lookup = numpy.zeros(256, dtype=numpy.float32)
    for (
        climate_zone,
        tcarbon_per_ha,
    ) in CLIMATE_ZONE_TO_GRASSLAND_TCARBON_PER_HA.items():
        lookup[climate_zone.value] = tcarbon_per_ha
    return xarray.apply_ufunc(
        lookup.__getitem__,
        climate_zones.fillna(0).astype(numpy.uint8),
        dask="parallelized",
        output_dtypes=[numpy.float32],
    ).rename("tcarbon-per-ha")


CO2E_PER_CARBON = 44 / 12
# IPCC 2019 Vol 4 Table 5.5
CLIMATE_ZONE_TO_SOC_RETENTION_FRACTION: dict[ipcc_climate_zones.Zone, float] = {
    ipcc_climate_zones.Zone.TROPICAL_WET: 0.83,
    ipcc_climate_zones.Zone.TROPICAL_MOIST: 0.83,
    ipcc_climate_zones.Zone.TROPICAL_DRY: 0.92,
    # Tropical montane factors are approximated as the mean of WARM_TEMPERATE_MOIST and TROPICAL_MOIST_WET
    ipcc_climate_zones.Zone.TROPICAL_MONTANE: 0.76,
    ipcc_climate_zones.Zone.WARM_TEMPERATE_MOIST: 0.69,
    ipcc_climate_zones.Zone.WARM_TEMPERATE_DRY: 0.76,
    # Cool Temperate and Boreal share rows in Table 5.5.
    ipcc_climate_zones.Zone.COOL_TEMPERATE_MOIST: 0.70,
    ipcc_climate_zones.Zone.COOL_TEMPERATE_DRY: 0.77,
    ipcc_climate_zones.Zone.BOREAL_MOIST: 0.70,
    ipcc_climate_zones.Zone.BOREAL_DRY: 0.77,
}


def get_mineral_soil_emissions(
    climate_zones: xarray.DataArray,
    soil_organic_carbon: xarray.DataArray,
) -> xarray.DataArray:
    lookup = numpy.zeros(256, dtype=numpy.float32)
    for (
        climate_zone,
        soc_retention_fraction,
    ) in CLIMATE_ZONE_TO_SOC_RETENTION_FRACTION.items():
        lookup[climate_zone.value] = (1 - soc_retention_fraction) * CO2E_PER_CARBON
    return (
        xarray.apply_ufunc(
            lookup.__getitem__,
            climate_zones.fillna(0).astype(numpy.uint8),
            dask="parallelized",
            output_dtypes=[numpy.float32],
        )
        # NB: SoilGrids has genuine gaps (water, rock, ice) that harmonize turns into NaN.
        # Treat a missing stock as zero rather than letting NaN reach emissions-per-hectare,
        # where it would silently discard the pixel's vegetation emissions too.
        * soil_organic_carbon.fillna(0)
    ).rename("tco2e-per-ha")


SpanType = tuple[int, int]
SPAN_TO_LINEAR_DISCOUNT_WEIGHT: dict[SpanType, float] = {
    (2000, 2005): 0.0125,
    (2005, 2010): 0.0375,
    (2010, 2015): 0.0625,
    (2015, 2020): 0.0875,
}
assert all((after - before) == 5 for (before, after) in SPAN_TO_LINEAR_DISCOUNT_WEIGHT)
assert math.isclose(sum(SPAN_TO_LINEAR_DISCOUNT_WEIGHT.values()), 0.2)
LOOKBACK_YEARS = 20
assert (
    max(after for _, after in SPAN_TO_LINEAR_DISCOUNT_WEIGHT)
    - min(before for before, _ in SPAN_TO_LINEAR_DISCOUNT_WEIGHT)
    == LOOKBACK_YEARS
)


ASSESSMENT_YEAR = 2020
LOOKBACK_YEARS_RANGE = tuple(
    range(ASSESSMENT_YEAR - LOOKBACK_YEARS, ASSESSMENT_YEAR + 1)
)


class Conversion(enum.IntEnum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> int:
        # Count from zero, so NONE is zero: an absent or unwritten part of the band then reads as
        # no conversion rather than as no member at all
        return count

    # NB: These are mutually exclusive, which is what lets a masked sum stand in for a
    # per-pixel choice wherever one is needed below
    NONE = enum.auto()
    FOREST_TO_CROPLAND = enum.auto()
    FOREST_TO_PASTURE = enum.auto()
    RANGELAND_TO_CROPLAND = enum.auto()
    RANGELAND_TO_PASTURE = enum.auto()
    PASTURE_TO_CROPLAND = enum.auto()
    # NB: Peatland is independent and may coincide


FROM_FOREST = (Conversion.FOREST_TO_CROPLAND, Conversion.FOREST_TO_PASTURE)
FROM_GRASSLAND = (
    Conversion.PASTURE_TO_CROPLAND,
    Conversion.RANGELAND_TO_CROPLAND,
    Conversion.RANGELAND_TO_PASTURE,
)
assert set(FROM_FOREST) ^ set(FROM_GRASSLAND) == frozenset(Conversion) - {
    Conversion.NONE
}


class DestinationDataset(enum.IntFlag):
    """Which datasets (the can superpose) claim a pixel's land class at the assessment year."""

    DESCALS_OIL_PALM = enum.auto()
    LIAO_GACED30 = enum.auto()
    GPW_GRASSLAND = enum.auto()


DESTINATION_DATASET_TO_PREDICATE: dict[
    DestinationDataset, collections.abc.Callable[[xarray.Dataset], xarray.DataArray]
] = {
    DestinationDataset.DESCALS_OIL_PALM: lambda dset: (
        # NB: zero is "no oil palm"
        (
            (planting_year := dset[descals_oil_palm.DATASET.fully_qualified_band_name])
            > 0
        )
        & (planting_year <= ASSESSMENT_YEAR)
    ),
    DestinationDataset.LIAO_GACED30: lambda dset: (
        dset[
            liao_gaced30.DATASET.fully_qualified_band_names[
                liao_gaced30.YEARS.index(ASSESSMENT_YEAR)
            ]
        ]
        == liao_gaced30.Cropland.CROPLAND
    ),
    DestinationDataset.GPW_GRASSLAND: lambda dset: (
        dset[
            gpw_grassland.DATASET.fully_qualified_band_names[
                gpw_grassland.YEARS.index(ASSESSMENT_YEAR)
            ]
        ]
        == gpw_grassland.Grassland.CULTIVATED
    ),
}
assert set(DestinationDataset) == set(DESTINATION_DATASET_TO_PREDICATE)

TO_CROPLAND = DestinationDataset.DESCALS_OIL_PALM | DestinationDataset.LIAO_GACED30
TO_PASTURE = DestinationDataset.GPW_GRASSLAND
# Each member belongs to one group, and the cropland members come first -- so a cropland bit
# outranks a pasture one, which is what lets a pixel be tested against the two groups in turn
# rather than resolved to a single member first
assert tuple(DestinationDataset) == (*TO_CROPLAND, *TO_PASTURE)


def get_last_departure_year(is_source: xarray.DataArray) -> xarray.DataArray:
    """The last year a pixel left the class it held the year before, or 0 where it never did."""
    assert "year" in is_source.coords
    assert tuple(is_source.year.values) == LOOKBACK_YEARS_RANGE
    departed = is_source.shift(fill_value=False, year=1) & ~is_source
    return departed.year.where(departed).max(dim="year").fillna(0)


@dataclasses.dataclass
class ConversionRecord:
    """One pixel's conversion, the year its source class ended, and the halves behind both.

    The halves outlive the conversion they resolve into, so each is carried rather than derived
    again: the source masks drive the carbon no destination claimed, and the destination masks
    each charge their own peatland-occupation band.
    """

    conversion: xarray.DataArray
    destination_dataset: xarray.DataArray
    from_forest: xarray.DataArray
    from_grassland: xarray.DataArray
    to_cropland: xarray.DataArray
    to_pasture: xarray.DataArray
    year: xarray.DataArray


def get_conversion_record(dset: xarray.Dataset) -> ConversionRecord:
    """Which conversion fired on each pixel, and the year its source class ended."""
    grassland = xarray.concat(
        [
            dset[band_name]
            for year, band_name in zip(
                gpw_grassland.YEARS,
                gpw_grassland.DATASET.fully_qualified_band_names,
                strict=True,
            )
            if year in LOOKBACK_YEARS_RANGE
        ],
        dim=xarray.DataArray(list(LOOKBACK_YEARS_RANGE), dims="year", name="year"),
    )

    rangeland_departure_year = get_last_departure_year(
        is_source=grassland.isin(
            (
                gpw_grassland.Grassland.NATURAL,
                gpw_grassland.Grassland.OPEN_SHRUBLAND,
            )
        )
    )
    pasture_departure_year = get_last_departure_year(
        is_source=grassland == gpw_grassland.Grassland.CULTIVATED
    )
    year_of_loss = (
        gnw_tcl.LOSS_YEAR_OFFSET + dset[gnw_tcl.DATASET.fully_qualified_band_name]
    )
    from_forest = (
        # NB: Zero means no loss rather than a loss in the base year
        (year_of_loss > gnw_tcl.LOSS_YEAR_OFFSET)
        # NB: the same bound as the line above while the lookback opens at the offset, so no test
        # can tell them apart; they stop coinciding the moment the assessment year moves
        & (year_of_loss > LOOKBACK_YEARS_RANGE[0])
        & (year_of_loss <= ASSESSMENT_YEAR)
    )
    from_rangeland = ~from_forest & (rangeland_departure_year > pasture_departure_year)
    from_pasture = ~from_forest & (pasture_departure_year > rangeland_departure_year)

    # NB: destinations are flags, so we can superpose them to track intersections
    destination_dataset = sum(
        predicate(dset) * numpy.uint8(destination_dataset)
        for destination_dataset, predicate in DESTINATION_DATASET_TO_PREDICATE.items()
    )
    assert isinstance(destination_dataset, xarray.DataArray)
    destination_dataset = destination_dataset.astype(numpy.uint8)
    # NB: a cropland bit outranks a pasture one, so testing the groups in turn is the whole of
    # the priority order
    to_cropland = (destination_dataset & TO_CROPLAND).astype(bool)
    to_pasture = ~to_cropland & (destination_dataset & TO_PASTURE).astype(bool)

    conversion_to_mask = {
        Conversion.FOREST_TO_CROPLAND: from_forest & to_cropland,
        Conversion.FOREST_TO_PASTURE: from_forest & to_pasture,
        Conversion.RANGELAND_TO_CROPLAND: from_rangeland & to_cropland,
        Conversion.RANGELAND_TO_PASTURE: from_rangeland & to_pasture,
        Conversion.PASTURE_TO_CROPLAND: from_pasture & to_cropland,
    }
    conversion = sum(
        # NB: the cast keeps this a byte; the enum would widen it to int64
        mask * numpy.uint8(conversion_class)
        # NB: at most one mask holds a pixel, so the sum is that pixel's one member
        for conversion_class, mask in conversion_to_mask.items()
    )
    assert isinstance(conversion, xarray.DataArray)

    return ConversionRecord(
        conversion=conversion.rename(None),
        destination_dataset=destination_dataset.rename(None),
        from_forest=from_forest,
        from_grassland=from_rangeland | from_pasture,
        to_cropland=to_cropland,
        to_pasture=to_pasture,
        # NB: a forest source is dated by its loss, a grassland one by its last departure
        # NB: an unobserved pixel arrives as NaN, fails every from_forest test, and so takes a
        # departure year instead -- so the cast never coalesces to zero
        year=year_of_loss.where(
            from_forest,
            other=numpy.maximum(rangeland_departure_year, pasture_departure_year),
        )
        .astype(numpy.uint16)
        .rename(None),
    )


PEATLAND_EMISSIONS_PULSE_TCO2E_PER_HA = 621


@dataclasses.dataclass
class ConversionEmissions:
    soil: xarray.DataArray
    vegetation: xarray.DataArray


def get_conversion_emissions(
    climate_zones: xarray.DataArray,
    conversion_record: ConversionRecord,
    forest_carbon: xarray.DataArray,
    grassland_carbon: xarray.DataArray,
    is_peatland: xarray.DataArray,
    soil_organic_carbon: xarray.DataArray,
) -> ConversionEmissions:
    """One model per conversion, each naming the pools it releases.

    The whole source stock goes, since a conversion has no destination stock to subtract, and a
    pixel the table does not name emits nothing -- there is no default left to fire on it. A
    source whose destination no layer resolves, or a destination with no source, is one of those
    pixels: its carbon is real and is reported by `get_dropped_emissions`, but it is charged to
    nobody, so nothing downstream can divide it among crops.
    """
    forest_biomass = CO2E_PER_CARBON * forest_carbon
    grassland_biomass = CO2E_PER_CARBON * grassland_carbon
    # NB: peat replaces the mineral term rather than adding to it
    cropland_soil = get_mineral_soil_emissions(
        climate_zones=climate_zones, soil_organic_carbon=soil_organic_carbon
    ).where(~is_peatland, other=PEATLAND_EMISSIONS_PULSE_TCO2E_PER_HA)
    # NB: assume a land-use factor of 1.0 and a management factor of 1.0 for mineral
    pasture_soil = PEATLAND_EMISSIONS_PULSE_TCO2E_PER_HA * is_peatland.astype(
        numpy.float32
    )

    fired = conversion_record.conversion != Conversion.NONE
    return ConversionEmissions(
        # NB: soil only depends on the destination
        soil=(
            cropland_soil.where(conversion_record.to_cropland, other=0)
            + pasture_soil.where(conversion_record.to_pasture, other=0)
        )
        .where(fired, other=0)
        .rename("tco2e-per-ha"),
        # NB: vegetation only depends on the source
        vegetation=(
            forest_biomass.where(conversion_record.from_forest, other=0)
            + grassland_biomass.where(conversion_record.from_grassland, other=0)
        )
        .where(fired, other=0)
        .rename("tco2e-per-ha"),
    )


def get_dropped_emissions(
    forest_carbon: xarray.DataArray,
    from_forest: xarray.DataArray,
    from_grassland: xarray.DataArray,
    grassland_carbon: xarray.DataArray,
    has_destination: xarray.DataArray,
) -> xarray.DataArray:
    return (
        CO2E_PER_CARBON
        * (
            forest_carbon.where(from_forest, other=0)
            + grassland_carbon.where(from_grassland, other=0)
        ).where(~has_destination, other=0)
    ).rename("tco2e-per-ha")


PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA = 37.3


def get_peatland_occupation_emissions(
    destination: xarray.DataArray, is_peatland: xarray.DataArray
) -> xarray.DataArray:
    return (
        PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA
        # NB: ensure no nan's are created which would clobber other nonzero emissions when combined
        * (destination & is_peatland).astype(numpy.float32)
    ).rename("tco2e-per-ha")


def get_span_to_charge(
    conversion_year: xarray.DataArray, darray: xarray.DataArray
) -> dict[SpanType, xarray.DataArray]:
    return {
        (before, after): darray.where(
            (conversion_year > before) & (conversion_year <= after), other=0
        )
        for before, after in SPAN_TO_LINEAR_DISCOUNT_WEIGHT
    }


def get_linear_discounted_total(
    span_to_value: dict[SpanType, xarray.DataArray],
) -> xarray.DataArray:
    """`span_to_value` reduced by the linear discount each span carries.

    Quantity-neutral: a flux -- emissions -- takes the weighted sum as it stands, while a level
    -- area, production -- divides the result by the weights' total, making the same reduction
    a weighted mean over the same windows.
    """
    ret = sum(
        value * SPAN_TO_LINEAR_DISCOUNT_WEIGHT[span]
        for span, value in span_to_value.items()
    )
    return typing.cast(xarray.DataArray, ret)


def get_hectares_per_pixel(darray: xarray.DataArray) -> xarray.DataArray:
    # https://en.wikipedia.org/wiki/Earth%27s_circumference
    equator_meters_per_longitude_degrees = 40_075_017 / 360
    meters_per_latitude_degrees = 40_007_863 / 360

    # NB: coords are 1-D and tiny, so reducing them to scalar spacings is cheap/greedy
    delta_longitude_degrees = abs(float(darray.x.diff("x").mean()))
    delta_latitude_degrees = abs(float(darray.y.diff("y").mean()))

    return (
        (
            # hectares at equator
            (delta_latitude_degrees * meters_per_latitude_degrees)
            * (delta_longitude_degrees * equator_meters_per_longitude_degrees)
            / 10_000
            # projection to latitude
            * numpy.cos(darray.y * numpy.pi / 180)
            # broadcast across x while inheriting darray's aligned 2-D chunking
            * xarray.ones_like(darray)
        )
        .astype(numpy.float32)
        .rename("ha")
    )


@enum.unique
class NonCommodity(enum.StrEnum):
    DROPPED = "DROPPED"


@enum.unique
class EmissionComponent(enum.StrEnum):
    FOREST = enum.auto()
    GRASSLAND = enum.auto()
    PEATLAND_CONVERSION = enum.auto()

    @typing.override
    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")

    @property
    def column(self) -> str:
        return f"{self.name.lower():s}_emissions_mt"


def get_span_to_component_to_emissions(
    dset: xarray.Dataset,
) -> dict[SpanType, dict[EmissionComponent, xarray.DataArray]]:
    """Split each span's emissions into the three components that claim them.

    A span's `emissions` is vegetation plus soil, and every unit of both is claimed exactly once:
    peat takes all the soil it sits under whatever the source class, and the source class takes
    the soil that is left. So the three sum back to the span's own total, which is what lets
    `emissions_mt` check them rather than restate them.

    There is no residual, because `get_conversion_emissions` charges only where a conversion
    fired and every conversion names a source class. Carbon that reaches no conversion is real,
    and is reported by `get_dropped_emissions` under that one name rather than under two.
    """
    conversion = dset["conversion"]
    is_peat = dset[gnw_global_peatlands.DATASET.fully_qualified_band_name] == 1
    from_forest = conversion.isin(FROM_FOREST)
    from_grassland = conversion.isin(FROM_GRASSLAND)
    span_to_component_to_emissions: dict[
        SpanType, dict[EmissionComponent, xarray.DataArray]
    ] = {}
    for before, after in SPAN_TO_LINEAR_DISCOUNT_WEIGHT:
        vegetation = dset[f"vegetation-emissions:tco2e-per-ha:{before:d}-{after:d}"]
        soil = dset[f"soil-emissions:tco2e-per-ha:{before:d}-{after:d}"]
        span_to_component_to_emissions[(before, after)] = {
            EmissionComponent.FOREST: vegetation.where(from_forest, other=0)
            + soil.where(from_forest & ~is_peat, other=0),
            EmissionComponent.GRASSLAND: vegetation.where(from_grassland, other=0)
            + soil.where(from_grassland & ~is_peat, other=0),
            EmissionComponent.PEATLAND_CONVERSION: soil.where(is_peat, other=0),
        }
    assert all(
        set(component_to_emissions) == set(EmissionComponent)
        for component_to_emissions in span_to_component_to_emissions.values()
    )
    return span_to_component_to_emissions


def get_dset_for_output(name_to_darray: dict[str, xarray.DataArray]) -> xarray.Dataset:
    chunk_size = geo.get_chunk_size(
        dtypes=[numpy.dtype("float32")] * len(name_to_darray)
    )

    def merge_name_units(name: str, units: typing.Hashable | None) -> str:
        prefix, _, suffix = name.partition(":")
        words = filter(bool, (prefix, units, suffix))
        return ":".join(map(str, words))

    return xarray.Dataset(
        {
            merge_name_units(name=name, units=darray.name): geo.unify_dtype_and_no_data(
                darray=darray
            ).chunk(chunks=chunk_size)
            for name, darray in name_to_darray.items()
        }
    )


@storage.cache_to_zarr(version=1)
def workflow(tile_id: str) -> xarray.Dataset:
    logger.info(f"Running the land conversion and emissions worflow for {tile_id=:s}")
    dset = geo.exact_merge(
        harmonize.workflow(
            dataset_names=harmonize.Stack.LUC_AND_EMISSIONS.value,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.GLAD,
        ),
        harmonize.workflow(
            dataset_names=harmonize.Stack.CROP_SUPPLEMENT.value,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.GLAD,
        ),
    )

    logger.info("Resolving which conversion each pixel underwent, and when")
    conversion_record: ConversionRecord = get_conversion_record(dset=dset)

    logger.info("Quantifying the carbon each source class holds")
    aboveground_biomass = dset[gnw_harris_agb.DATASET.fully_qualified_band_name]
    aboveground_carbon = (aboveground_biomass * CARBON_PER_BIOMASS_LIVE_WOOD).rename(
        "tcarbon-per-ha"
    )
    climate_zones = dset[ipcc_climate_zones.DATASET.fully_qualified_band_name]
    forest_carbon = (
        aboveground_carbon
        + get_belowground_carbon(
            aboveground_biomass=aboveground_biomass,
            belowground_biomass=dset[huang_bgb.DATASET.fully_qualified_band_name],
        )
        + get_dead_organic_matter_carbon(
            aboveground_biomass=aboveground_biomass, climate_zones=climate_zones
        )
    ).rename("tcarbon-per-ha")
    grassland_carbon = get_grassland_carbon(climate_zones=climate_zones)

    logger.info("Releasing the pools each conversion carries")
    is_peatland = dset[gnw_global_peatlands.DATASET.fully_qualified_band_name] == 1
    conversion_emissions: ConversionEmissions = get_conversion_emissions(
        climate_zones=climate_zones,
        conversion_record=conversion_record,
        forest_carbon=forest_carbon,
        grassland_carbon=grassland_carbon,
        is_peatland=is_peatland,
        soil_organic_carbon=dset[soilgrids_ocs.DATASET.fully_qualified_band_name],
    )
    span_to_vegetation_emissions = get_span_to_charge(
        conversion_year=conversion_record.year, darray=conversion_emissions.vegetation
    )
    span_to_soil_emissions = get_span_to_charge(
        conversion_year=conversion_record.year, darray=conversion_emissions.soil
    )

    logger.info("Summing emissions from vegetation and soil")
    span_to_emissions: dict[SpanType, xarray.DataArray] = {
        span: (
            span_to_vegetation_emissions[span] + span_to_soil_emissions[span]
        ).rename("tco2e-per-ha")
        for span in span_to_vegetation_emissions
    }

    logger.info("Quantifying and adding peatland occupation emissions")
    # NB: split by destination on the 30 m grid so each allocation leg can take the peat drained
    # under its own land class whole, rather than assuming peat is spread evenly across a cell
    cropland_occupation_emissions = get_peatland_occupation_emissions(
        destination=conversion_record.to_cropland, is_peatland=is_peatland
    )
    pastureland_occupation_emissions = get_peatland_occupation_emissions(
        destination=conversion_record.to_pasture, is_peatland=is_peatland
    )
    emissions_per_hectare: xarray.DataArray = (
        get_linear_discounted_total(span_to_value=span_to_emissions)
        + cropland_occupation_emissions
        + pastureland_occupation_emissions
    )

    logger.info("Quantifying the source carbon no destination claimed")
    dropped_emissions = get_linear_discounted_total(
        span_to_value=get_span_to_charge(
            conversion_year=conversion_record.year,
            darray=get_dropped_emissions(
                forest_carbon=forest_carbon,
                from_forest=conversion_record.from_forest,
                from_grassland=conversion_record.from_grassland,
                grassland_carbon=grassland_carbon,
                has_destination=conversion_record.to_cropland
                | conversion_record.to_pasture,
            ),
        )
    ).rename("tco2e-per-ha")

    return get_dset_for_output(
        name_to_darray={
            "conversion": conversion_record.conversion,
            "conversion-year": conversion_record.year,
            "destination-dataset": conversion_record.destination_dataset,
        }
        | {
            f"emissions:{before:d}-{after:d}": darray
            for (before, after), darray in span_to_emissions.items()
        }
        | {
            f"soil-emissions:{before:d}-{after:d}": darray
            for (before, after), darray in span_to_soil_emissions.items()
        }
        | {
            f"vegetation-emissions:{before:d}-{after:d}": darray
            for (before, after), darray in span_to_vegetation_emissions.items()
        }
        | {
            "cropland-peatland-occupation": cropland_occupation_emissions,
            # NB: charged to nobody, so it is reported beside the total rather than inside it
            "dropped-emissions": dropped_emissions,
            "emissions-per-hectare": emissions_per_hectare,
            "hectares-per-pixel": get_hectares_per_pixel(darray=emissions_per_hectare),
            "pastureland-peatland-occupation": pastureland_occupation_emissions,
        }
    )


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "iso_3166s",
        help="cover exactly the tiles these countries' boundaries touch",
        nargs=argparse.ZERO_OR_MORE,
        type=worldbank_jurisdictions.iso_3166_str,
    )
    parser.add_argument("--backfill", action="store_true", help="cover all GNW tiles")
    parser.add_argument(
        "--modulus",
        default=1,
        type=int,
        help="split the tiles into this many disjoint shards",
    )
    parser.add_argument(
        "--residues",
        action="append",
        type=int,
        help="build only these shards, each in [0, --modulus); repeatable, all by default",
    )
    args = parser.parse_args()
    assert bool(args.iso_3166s) ^ bool(args.backfill), (
        "pass either one-or-more iso_3166s or --backfill"
    )

    for tile_id in utils.iter_sharded(
        modulus=int(args.modulus),
        residues=args.residues,
        values=(
            tiling.GLOBAL_NATURE_WATCH_TILE_IDS
            if args.backfill
            else worldbank_jurisdictions.get_ten_degree_tile_ids_for_iso_3166s(
                iso_3166s=args.iso_3166s
            )
        ),
    ):
        workflow(tile_id=tile_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
