"""Crop-class emissions layer: the emissions IF a given crop class occupies each pixel.

Reads `emit`'s Crop-Independent Emissions (CIE) bands (see `docs/crop_independent_emissions.md`)
and applies what CIE deliberately leaves out:

- the crop-class land-use factor on mineral soil (F_LU, by climate zone),
- the crop-class emission factor for occupied peat (by peat climate regime),
- the linear discount that spreads each conversion's emissions over the lookback window.

There is no destination gate: unlike `emit.derive_from_cie`, nothing here asks what land cover
the pixel actually became. Every pixel answers "what if this crop class were here?".

The layer is lazy and never stored. What gets materialised is a product: a named choice of
layer fields, crop classes and discountings, written per tile by `export.write_tile` straight
from `emit`'s cached output. A debugging export is just another product, with more of the
intermediate fields.

The file reads top to bottom as:

1. Factor tables  -- the numbers, with their sources
2. Calculations   -- one function per field of the layer, arithmetic only
3. Products       -- each a choice of layer fields, crop classes and discountings
4. Plumbing       -- reading CIE bands, table lookups, and the call to `export`

Example invocations:
  uv run python -m jdluc.derive HND
  uv run python -m jdluc.derive HND --product=perennial-debug --format=zarr
  uv run python -m jdluc.derive --backfill --modulus=8 --residues=0
"""

import argparse
import collections.abc
import dataclasses
import enum
import logging
import math

import numpy
import numpy.typing
import xarray

from jdluc import emit, export, tiling, utils
from jdluc.datasets import worldbank_jurisdictions
from jdluc.datasets.ipcc_climate_zones import Zone

logger = logging.getLogger(__name__)

# ============================================================================================
# 1. Factor tables
#
# Mineral soil: the stock change factor for land use, F_LU, from IPCC 2019 Refinement Vol 4
# Ch 5 Table 5.5 (cropland) and Ch 6 Table 6.2 (grassland). Drained peat: the IPCC 2013
# Wetlands Supplement (Ch 2), by the peat climate regime each IPCC climate zone falls in.
# ============================================================================================

REFERENCE_YEAR = emit.ASSESSMENT_YEAR


class CropClass(enum.StrEnum):
    ANNUAL = "annual"
    PERENNIAL = "perennial"
    PADDY_RICE = "paddy-rice"
    PASTURE = "pasture"


# F_LU: the fraction of reference soil organic carbon retained once the crop class occupies the
# land. Capped at 1.0: the layer does not model SOC gain, so paddy rice (1.35) and tropical
# perennials (1.01) lose nothing. Pasture, as nominally managed grassland, is 1.0 everywhere.
#
# Annual crops take `emit`'s long-term-cultivated retention fractions, so that the two agree,
# including on its approximation for tropical montane, which Table 5.5 has no row for. Tropical
# montane perennials take the tropical value.
CROP_CLASS_TO_ZONE_TO_FLU: dict[CropClass, dict[Zone, float]] = {
    CropClass.ANNUAL: dict(emit.CLIMATE_ZONE_TO_SOC_RETENTION_FRACTION),
    CropClass.PERENNIAL: {
        Zone.TROPICAL_MONTANE: 1.00,
        Zone.TROPICAL_WET: 1.00,
        Zone.TROPICAL_MOIST: 1.00,
        Zone.TROPICAL_DRY: 1.00,
        Zone.WARM_TEMPERATE_MOIST: 0.72,
        Zone.WARM_TEMPERATE_DRY: 0.72,
        Zone.COOL_TEMPERATE_MOIST: 0.72,
        Zone.COOL_TEMPERATE_DRY: 0.72,
        Zone.BOREAL_MOIST: 0.72,
        Zone.BOREAL_DRY: 0.72,
    },
    CropClass.PADDY_RICE: dict.fromkeys(Zone, 1.0),
    CropClass.PASTURE: dict.fromkeys(Zone, 1.0),
}


class PeatRegime(enum.StrEnum):
    TROPICAL = "tropical"
    TEMPERATE = "temperate"
    BOREAL = "boreal"


ZONE_TO_PEAT_REGIME: dict[Zone, PeatRegime] = {
    Zone.TROPICAL_MONTANE: PeatRegime.TROPICAL,
    Zone.TROPICAL_WET: PeatRegime.TROPICAL,
    Zone.TROPICAL_MOIST: PeatRegime.TROPICAL,
    Zone.TROPICAL_DRY: PeatRegime.TROPICAL,
    Zone.WARM_TEMPERATE_MOIST: PeatRegime.TEMPERATE,
    Zone.WARM_TEMPERATE_DRY: PeatRegime.TEMPERATE,
    Zone.COOL_TEMPERATE_MOIST: PeatRegime.TEMPERATE,
    Zone.COOL_TEMPERATE_DRY: PeatRegime.TEMPERATE,
    Zone.BOREAL_MOIST: PeatRegime.BOREAL,
    Zone.BOREAL_DRY: PeatRegime.BOREAL,
}

# Annual emissions from drained peat occupied by the crop class, tCO2e/ha/yr: on-site CO2,
# dissolved organic carbon, CH4 and N2O together. Annual crops take cropland, perennials
# plantation, and pasture deep-drained grassland. Paddy rice has no category of its own, so
# takes cropland.
# Where the climate zone is missing, the CIE value (`emit.PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA`)
# stands.
CROP_CLASS_TO_PEAT_REGIME_TO_OCCUPATION_EF: dict[CropClass, dict[PeatRegime, float]] = {
    CropClass.ANNUAL: {
        PeatRegime.TROPICAL: 56.92,
        PeatRegime.TEMPERATE: 38.18,
        PeatRegime.BOREAL: 37.48,
    },
    CropClass.PERENNIAL: {
        PeatRegime.TROPICAL: 60.76,
        PeatRegime.TEMPERATE: 12.25,
        PeatRegime.BOREAL: 1.88,
    },
    CropClass.PADDY_RICE: {
        PeatRegime.TROPICAL: 56.92,
        PeatRegime.TEMPERATE: 38.18,
        PeatRegime.BOREAL: 37.48,
    },
    CropClass.PASTURE: {
        PeatRegime.TROPICAL: 42.32,
        PeatRegime.TEMPERATE: 29.87,
        PeatRegime.BOREAL: 27.82,
    },
}

# The 621 tCO2e/ha peat transformation pulse is crop-generic: every class takes it whole.
PEAT_TRANSFORMATION_FACTOR = 1.0

# Discounting: the fraction of a conversion's emissions charged in REFERENCE_YEAR, by the year
# the conversion happened. A scheme is a 1-D array over the assessment window -- dimension
# `conversion_year`, LOOKBACK_YEARS (20) long, REFERENCE_YEAR - 19 .. REFERENCE_YEAR -- that
# sums to 1. Conversions outside the window are charged nothing.
LOOKBACK_YEARS = emit.LOOKBACK_YEARS
CONVERSION_YEAR = "conversion_year"
# The dimensions the layer adds beyond the pixel grid (see section 2)
CROP_CLASS = "crop_class"
DISCOUNTING = "discounting"
ASSESSMENT_WINDOW = numpy.arange(
    REFERENCE_YEAR - LOOKBACK_YEARS + 1, REFERENCE_YEAR + 1
)


def get_discount_weights(weights: numpy.typing.ArrayLike) -> xarray.DataArray:
    """`weights`, one per year of the assessment window, as a (conversion_year: 20) array."""
    return xarray.DataArray(
        numpy.asarray(weights, dtype=numpy.float64),
        coords={CONVERSION_YEAR: ASSESSMENT_WINDOW},
        dims=(CONVERSION_YEAR,),
    )


# Linear, the year-by-year form of `emit`'s span weights: rising in equal steps from 0.0025
# (REFERENCE_YEAR - 19) to 0.0975 (REFERENCE_YEAR).
LINEAR_DISCOUNT_WEIGHTS = get_discount_weights(
    numpy.linspace(
        start=1 / LOOKBACK_YEARS**2,
        stop=(2 * LOOKBACK_YEARS - 1) / LOOKBACK_YEARS**2,
        num=LOOKBACK_YEARS,
    )
)

# Equal: 1/20 for every year of the window.
EQUAL_DISCOUNT_WEIGHTS = get_discount_weights(
    numpy.full(LOOKBACK_YEARS, 1 / LOOKBACK_YEARS)
)

# Five-year spans, as `emit` discounts to line up with MAPSPAM's 5-year snapshots: every year
# in a span (before, after] takes the span's weight, which is the linear weights' mean over it.
FIVE_YEAR_SPAN_DISCOUNT_WEIGHTS = get_discount_weights(
    [
        next(
            weight
            for (before, after), weight in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT.items()
            if before < year <= after
        )
        for year in ASSESSMENT_WINDOW
    ]
)

DISCOUNTING_TO_WEIGHTS: dict[str, xarray.DataArray] = {
    "linear": LINEAR_DISCOUNT_WEIGHTS,
    "equal": EQUAL_DISCOUNT_WEIGHTS,
    "five-year-spans": FIVE_YEAR_SPAN_DISCOUNT_WEIGHTS,
}
DEFAULT_DISCOUNTING = "linear"

assert all(
    weights.sizes == {CONVERSION_YEAR: LOOKBACK_YEARS}
    and math.isclose(float(weights.sum()), 1)
    for weights in DISCOUNTING_TO_WEIGHTS.values()
)
assert numpy.allclose(
    LINEAR_DISCOUNT_WEIGHTS.values.reshape(-1, 5).mean(axis=1),
    FIVE_YEAR_SPAN_DISCOUNT_WEIGHTS.values.reshape(-1, 5).mean(axis=1),
)

assert all(
    set(zone_to_flu) == set(Zone) for zone_to_flu in CROP_CLASS_TO_ZONE_TO_FLU.values()
)
assert all(
    0 <= flu <= 1
    for zone_to_flu in CROP_CLASS_TO_ZONE_TO_FLU.values()
    for flu in zone_to_flu.values()
)
assert set(ZONE_TO_PEAT_REGIME) == set(Zone)
assert all(
    set(regime_to_ef) == set(PeatRegime)
    for regime_to_ef in CROP_CLASS_TO_PEAT_REGIME_TO_OCCUPATION_EF.values()
)
assert (
    set(CROP_CLASS_TO_ZONE_TO_FLU)
    == set(CROP_CLASS_TO_PEAT_REGIME_TO_OCCUPATION_EF)
    == set(CropClass)
)


# ============================================================================================
# 2. Calculations
#
# Every function is arithmetic on named arrays. Units: tCO2e per hectare, except where named
# per year.
#
# Dimensions. CIE bands are on the pixel grid, (y, x). Two inputs add a dimension:
#
# - the factors (F_LU and the peat occupation EF) are (y, x, crop_class);
# - the discount weights are (conversion_year: 20, discounting), one column per scheme.
#
# Arithmetic broadcasts by dimension name, so a field carries `crop_class` exactly when it
# depends on a factor, and `discounting` exactly when it is discounted -- nothing here needs to
# say which. `crop_class_layer` notes the dimensions of each step; `get_layer` logs their sizes.
# ============================================================================================


def is_conversion(conversion_source: xarray.DataArray) -> xarray.DataArray:
    """Whether CIE records a conversion event (forest, natural grassland or pasture lost).

    Which event a pixel carries, where sources conflict, is resolved upstream in `emit`.
    """
    return conversion_source != emit.ConversionSource.NONE


def discount_weight(
    conversion_year: xarray.DataArray, discount_weights: xarray.DataArray
) -> xarray.DataArray:
    """The fraction of a pixel's conversion emissions charged in the reference year.

    (y, x) conversion years and (conversion_year: 20, ...) weights -> (y, x, ...) weights,
    where `...` is whatever else the weights carry (e.g. `discounting`); 0 outside the window.
    """
    weight = sum(
        xarray.where(
            conversion_year == year,
            discount_weights.sel({CONVERSION_YEAR: year}, drop=True),
            0.0,
        )
        for year in discount_weights[CONVERSION_YEAR].values.tolist()
    )
    assert isinstance(weight, xarray.DataArray)
    return weight


def vegetation_emissions_amortized(
    vegetation_emissions_undiscounted: xarray.DataArray,
    is_conversion: xarray.DataArray,
    discount_weight: xarray.DataArray,
) -> xarray.DataArray:
    """Vegetation carbon lost at conversion, discounted."""
    return (
        xarray.where(is_conversion, vegetation_emissions_undiscounted, 0.0)
        * discount_weight
    )


def mineral_soil_emissions_amortized(
    mineral_soil_carbon_at_risk: xarray.DataArray,
    flu: xarray.DataArray,
    is_conversion: xarray.DataArray,
    discount_weight: xarray.DataArray,
) -> xarray.DataArray:
    """Mineral soil carbon lost to the crop class's land use, discounted.

    CIE's carbon at risk is the full stock (zero on peat); the crop class keeps the fraction F_LU.
    """
    soil_loss = mineral_soil_carbon_at_risk * (1 - flu)
    return xarray.where(is_conversion, soil_loss, 0.0) * discount_weight


def peat_transformation_emissions_amortized(
    peat_transformation_emissions_undiscounted: xarray.DataArray,
    is_conversion: xarray.DataArray,
    discount_weight: xarray.DataArray,
) -> xarray.DataArray:
    """The one-off pulse from draining peat at conversion, discounted."""
    pulse = peat_transformation_emissions_undiscounted * PEAT_TRANSFORMATION_FACTOR
    return xarray.where(is_conversion, pulse, 0.0) * discount_weight


def is_peat(peat_occupation_emissions: xarray.DataArray) -> xarray.DataArray:
    """Whether the pixel is on peat. CIE puts a nonzero occupation potential on all peat."""
    return peat_occupation_emissions > 0


def peat_occupation_emissions_per_year(
    is_peat: xarray.DataArray, peat_occupation_ef: xarray.DataArray
) -> xarray.DataArray:
    """Annual emissions from the crop class occupying drained peat, whether or not converted."""
    return xarray.where(is_peat, peat_occupation_ef, 0.0)


def total_emissions_amortized(
    vegetation_emissions_amortized: xarray.DataArray,
    mineral_soil_emissions_amortized: xarray.DataArray,
    peat_transformation_emissions_amortized: xarray.DataArray,
    peat_occupation_emissions_per_year: xarray.DataArray,
) -> xarray.DataArray:
    """Everything charged to the crop class on this pixel in the reference year."""
    return (
        vegetation_emissions_amortized
        + mineral_soil_emissions_amortized
        + peat_transformation_emissions_amortized
        + peat_occupation_emissions_per_year
    )


@dataclasses.dataclass(frozen=True)
class CIE:
    """The `emit` CIE bands that this layer reads (see `read_cie`)."""

    conversion_source: xarray.DataArray
    conversion_year: xarray.DataArray
    vegetation_emissions_undiscounted: xarray.DataArray
    mineral_soil_carbon_at_risk: xarray.DataArray
    peat_transformation_emissions_undiscounted: xarray.DataArray
    peat_occupation_emissions: xarray.DataArray
    climate_zone: xarray.DataArray
    hectares_per_pixel: xarray.DataArray


@dataclasses.dataclass(frozen=True)
class Factors:
    """The factors, per pixel and per crop class: (y, x, crop_class). See `lookup_factors`."""

    flu: xarray.DataArray
    peat_occupation_ef: xarray.DataArray


@dataclasses.dataclass(frozen=True)
class CropClassLayer:
    """The crop-class layer, one field per quantity.

    The fields that depend on a factor carry the `crop_class` dimension; the rest do not.
    """

    conversion_source: xarray.DataArray
    conversion_year: xarray.DataArray
    climate_zone: xarray.DataArray
    discount_weight: xarray.DataArray
    flu: xarray.DataArray
    peat_occupation_ef: xarray.DataArray
    mineral_soil_emissions_amortized: xarray.DataArray
    peat_transformation_emissions_amortized: xarray.DataArray
    vegetation_emissions_amortized: xarray.DataArray
    peat_occupation_emissions_per_year: xarray.DataArray
    is_peat: xarray.DataArray
    total_emissions_amortized: xarray.DataArray
    hectares_per_pixel: xarray.DataArray


FIELD_TO_UNITS: dict[str, str | None] = {
    "conversion_source": None,
    "conversion_year": None,
    "climate_zone": None,
    "discount_weight": None,
    "flu": None,
    "peat_occupation_ef": "tco2e-per-ha-per-year",
    "mineral_soil_emissions_amortized": "tco2e-per-ha",
    "peat_transformation_emissions_amortized": "tco2e-per-ha",
    "vegetation_emissions_amortized": "tco2e-per-ha",
    "peat_occupation_emissions_per_year": "tco2e-per-ha-per-year",
    "is_peat": None,
    # NB: adds the amortized terms to a year's peat occupation, as `emit` does
    "total_emissions_amortized": "tco2e-per-ha",
    "hectares_per_pixel": "ha",
}
assert set(FIELD_TO_UNITS) == {
    field.name for field in dataclasses.fields(CropClassLayer)
}


def crop_class_layer(
    cie: CIE, factors: Factors, discount_weights: xarray.DataArray
) -> CropClassLayer:
    """The layer, computed from the CIE bands, for every crop class the factors carry.

    `discount_weights` are the discounting schemes, (conversion_year: 20, discounting).
    """
    # (y, x): one value per pixel
    converted = is_conversion(cie.conversion_source)
    on_peat = is_peat(cie.peat_occupation_emissions)

    # (y, x, discounting): discounted, but the same for every crop class
    weight = discount_weight(cie.conversion_year, discount_weights)
    vegetation = vegetation_emissions_amortized(
        cie.vegetation_emissions_undiscounted, converted, weight
    )
    peat_transformation = peat_transformation_emissions_amortized(
        cie.peat_transformation_emissions_undiscounted, converted, weight
    )

    # (y, x, discounting, crop_class): discounted, and takes a factor
    mineral_soil = mineral_soil_emissions_amortized(
        cie.mineral_soil_carbon_at_risk, factors.flu, converted, weight
    )

    # (y, x, crop_class): takes a factor, but is a yearly rate, so is not discounted
    peat_occupation = peat_occupation_emissions_per_year(
        on_peat, factors.peat_occupation_ef
    )

    return CropClassLayer(
        conversion_source=cie.conversion_source,
        conversion_year=cie.conversion_year,
        climate_zone=cie.climate_zone,
        discount_weight=weight,
        flu=factors.flu,
        peat_occupation_ef=factors.peat_occupation_ef,
        mineral_soil_emissions_amortized=mineral_soil,
        peat_transformation_emissions_amortized=peat_transformation,
        vegetation_emissions_amortized=vegetation,
        peat_occupation_emissions_per_year=peat_occupation,
        is_peat=on_peat,
        # (y, x, discounting, crop_class): the union of its terms' dimensions
        total_emissions_amortized=total_emissions_amortized(
            vegetation, mineral_soil, peat_transformation, peat_occupation
        ),
        hectares_per_pixel=cie.hectares_per_pixel,
    )


# ============================================================================================
# 3. Products
#
# A product is written per tile by `export.write_tile`: layer fields, for the crop classes and
# discounting schemes it names. In a COG each field is one band per combination of the values
# it varies by, named `<variable>:<crop class>:<discounting>`; a zarr keeps them as dimensions.
# Any field of `CropClassLayer` is on the menu. Defining a new product means adding one here.
# ============================================================================================

MENU: tuple[str, ...] = tuple(
    field.name for field in dataclasses.fields(CropClassLayer)
)
VARIABLE_PREFIX = "ccl-"


def get_variable_name(field: str) -> str:
    """The variable a layer field is written as: `ccl-<field>[:<units>]`."""
    words = [VARIABLE_PREFIX + field.replace("_", "-"), FIELD_TO_UNITS[field]]
    return ":".join(word for word in words if word)


@dataclasses.dataclass(frozen=True)
class Product:
    name: str
    fields: tuple[str, ...]
    crop_classes: tuple[CropClass, ...] = tuple(CropClass)
    discountings: tuple[str, ...] = (DEFAULT_DISCOUNTING,)
    format: export.Format = export.Format.COG

    def __post_init__(self) -> None:
        off_menu = [field for field in self.fields if field not in MENU]
        assert not off_menu, f"{self.name!r}: {off_menu} are not on the menu"
        unknown = set(self.discountings) - set(DISCOUNTING_TO_WEIGHTS)
        assert not unknown, f"{self.name!r}: unknown discountings {unknown}"
        assert self.crop_classes and self.discountings, f"{self.name!r} selects nothing"

    @property
    def variables(self) -> list[str]:
        return [get_variable_name(field) for field in self.fields]


CROP_CLASS_EMISSIONS = Product(
    name="crop-class-emissions",
    fields=("total_emissions_amortized", "hectares_per_pixel"),
)

PERENNIAL_DEBUG = Product(
    name="perennial-debug",
    fields=MENU,
    crop_classes=(CropClass.PERENNIAL,),
    format=export.Format.ZARR,
)

PERENNIAL_DISCOUNTING_COMPARISON = Product(
    name="perennial-discounting-comparison",
    fields=(
        "vegetation_emissions_amortized",
        "mineral_soil_emissions_amortized",
        "peat_transformation_emissions_amortized",
        # NB: a yearly rate, not discounted, so one band whatever the discounting
        "peat_occupation_emissions_per_year",
        "total_emissions_amortized",
        "hectares_per_pixel",
    ),
    crop_classes=(CropClass.PERENNIAL,),
    discountings=("equal", "linear"),
)

NAME_TO_PRODUCT: dict[str, Product] = {
    product.name: product
    for product in (
        CROP_CLASS_EMISSIONS,
        PERENNIAL_DEBUG,
        PERENNIAL_DISCOUNTING_COMPARISON,
    )
}


# ============================================================================================
# 4. Plumbing
# ============================================================================================


def read_cie(dset: xarray.Dataset) -> CIE:
    """The CIE bands of an `emit` output dataset, by field name."""

    def cie(name: str) -> xarray.DataArray:
        return dset[f"cie-{name:s}"]

    return CIE(
        # NB: no conversion is NONE (0); NaN no-data reads as no conversion too
        conversion_source=cie("conversion-source").fillna(0),
        conversion_year=cie("conversion-year").fillna(0),
        vegetation_emissions_undiscounted=cie(
            "vegetation-emissions-undiscounted:tco2e-per-ha"
        ),
        mineral_soil_carbon_at_risk=cie("mineral-soil-carbon-at-risk:tco2e-per-ha"),
        peat_transformation_emissions_undiscounted=cie(
            "peat-transformation-emissions-undiscounted:tco2e-per-ha"
        ),
        peat_occupation_emissions=cie(
            "peat-occupation-emissions:tco2e-per-ha-per-year"
        ),
        climate_zone=cie("climate-zone"),
        hectares_per_pixel=cie("hectares-per-pixel:ha"),
    )


def lookup_by_zone(
    climate_zone: xarray.DataArray, zone_to_value: dict[Zone, float], default: float
) -> xarray.DataArray:
    """`zone_to_value` per pixel, as a lazy blockwise lookup; `default` where the zone is missing."""
    lookup = numpy.full(256, default, dtype=numpy.float32)
    for zone, value in zone_to_value.items():
        lookup[zone.value] = value
    return xarray.apply_ufunc(
        lookup.__getitem__,
        climate_zone.fillna(0).astype(numpy.uint8),
        dask="parallelized",
        output_dtypes=[numpy.float32],
    )


def lookup_factors(
    climate_zone: xarray.DataArray, crop_classes: collections.abc.Sequence[CropClass]
) -> Factors:
    """Per-pixel factors from the tables, for `crop_classes` only, along `crop_class`.

    Each class is its own lookup, so its own dask chunk along `crop_class`: selecting one
    class computes only that class. A missing zone takes F_LU = 1 (no mineral soil loss, as in
    `emit`) and the CIE peat occupation value.
    """

    def by_class(lookup_one: collections.abc.Callable[[CropClass], xarray.DataArray]):
        return (
            xarray.concat(
                [lookup_one(crop_class) for crop_class in crop_classes], dim=CROP_CLASS
            )
            .assign_coords(
                {CROP_CLASS: [str(crop_class) for crop_class in crop_classes]}
            )
            .transpose(..., CROP_CLASS)
        )  # (y, x, crop_class), as the fields they reach

    def flu(crop_class: CropClass) -> xarray.DataArray:
        return lookup_by_zone(
            climate_zone=climate_zone,
            zone_to_value=CROP_CLASS_TO_ZONE_TO_FLU[crop_class],
            default=1.0,
        )

    def peat_occupation_ef(crop_class: CropClass) -> xarray.DataArray:
        regime_to_ef = CROP_CLASS_TO_PEAT_REGIME_TO_OCCUPATION_EF[crop_class]
        return lookup_by_zone(
            climate_zone=climate_zone,
            zone_to_value={
                zone: regime_to_ef[regime]
                for zone, regime in ZONE_TO_PEAT_REGIME.items()
            },
            default=emit.PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA,
        )

    return Factors(flu=by_class(flu), peat_occupation_ef=by_class(peat_occupation_ef))


def get_discount_weights_by_scheme(
    discountings: collections.abc.Sequence[str],
) -> xarray.DataArray:
    """The named schemes side by side: (conversion_year: 20, discounting)."""
    return (
        xarray.concat(
            [DISCOUNTING_TO_WEIGHTS[discounting] for discounting in discountings],
            dim=DISCOUNTING,
        )
        .assign_coords({DISCOUNTING: list(discountings)})
        .transpose(CONVERSION_YEAR, ...)
    )


def to_dataset(layer: CropClassLayer) -> xarray.Dataset:
    """Every field of `layer` as a variable, `crop_class` and `discounting` kept as dims.

    The extra dims lead, crop class first, so a COG band is named for its crop class, then its
    discounting.
    """

    def to_variable(darray: xarray.DataArray) -> xarray.DataArray:
        leading = [dim for dim in (CROP_CLASS, DISCOUNTING) if dim in darray.dims]
        return darray.transpose(*leading, ...).rename(None)

    return xarray.Dataset(
        {get_variable_name(field): to_variable(getattr(layer, field)) for field in MENU}
    )


def get_layer(
    dset: xarray.Dataset,
    crop_classes: collections.abc.Sequence[CropClass] = tuple(CropClass),
    discountings: collections.abc.Sequence[str] = (DEFAULT_DISCOUNTING,),
) -> CropClassLayer:
    """The layer for `crop_classes` and `discountings`, from an `emit` output dataset."""
    cie = read_cie(dset)
    layer = crop_class_layer(
        cie=cie,
        factors=lookup_factors(cie.climate_zone, crop_classes),
        discount_weights=get_discount_weights_by_scheme(discountings),
    )
    log_layer_shapes(layer=layer)
    return layer


def log_layer_shapes(layer: CropClassLayer) -> None:
    """At DEBUG, each field's dimensions, sizes and chunking: what a computation will touch."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for field in MENU:
        darray = getattr(layer, field)
        chunks = (
            {dim: len(sizes) for dim, sizes in darray.chunksizes.items()}
            if darray.chunks is not None
            else "in memory"
        )
        logger.debug(f"{field:s}: sizes {dict(darray.sizes)}, chunk count {chunks}")


def derive(
    dset: xarray.Dataset,
    crop_classes: collections.abc.Sequence[CropClass] = tuple(CropClass),
    discountings: collections.abc.Sequence[str] = (DEFAULT_DISCOUNTING,),
) -> xarray.Dataset:
    """Every field of the layer for `crop_classes` and `discountings`, as a lazy dataset."""
    return to_dataset(get_layer(dset, crop_classes, discountings))


def export_workflow(
    tile_id: str,
    product_names: collections.abc.Sequence[str] = (CROP_CLASS_EMISSIONS.name,),
    format: export.Format | None = None,
    output_root: str | None = None,
    version: str = "",
) -> list[str]:
    """Write each named product for one tile, from `emit`'s cached output; return the URIs.

    `format` overrides each product's own. Only the crop classes and discountings a product
    names are computed.
    """
    dset = emit.workflow(tile_id=tile_id)
    return [
        export.write_tile(
            dset=derive(
                dset,
                crop_classes=product.crop_classes,
                discountings=product.discountings,
            ),
            format=format or product.format,
            name=product.name,
            output_root=output_root,
            source_name="jdluc-derive",
            tile_id=tile_id,
            variables=product.variables,
            version=version,
        )
        for product in map(NAME_TO_PRODUCT.__getitem__, product_names)
    ]


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "iso_3166s",
        help="cover exactly the tiles these countries' boundaries touch",
        nargs=argparse.ZERO_OR_MORE,
        type=worldbank_jurisdictions.iso_3166_str,
    )
    parser.add_argument("--backfill", action="store_true", help="cover all GNW tiles")
    parser.add_argument(
        "--product",
        action="append",
        choices=sorted(NAME_TO_PRODUCT),
        dest="products",
        help=f"write this product; repeatable, {CROP_CLASS_EMISSIONS.name:s} by default",
    )
    parser.add_argument(
        "--format",
        choices=list(export.Format),
        type=export.Format,
        help="override each product's own format",
    )
    parser.add_argument(
        "--version",
        default="",
        help="write under {EXPORT_ROOT}/{version}/, e.g. a run id or a release label",
    )
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
        help="write only these shards, each in [0, --modulus); repeatable, all by default",
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
        export_workflow(
            tile_id=tile_id,
            product_names=args.products or (CROP_CLASS_EMISSIONS.name,),
            format=args.format,
            version=args.version,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
