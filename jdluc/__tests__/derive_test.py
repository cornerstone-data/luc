"""Tests for the crop-class emissions layer (`jdluc.derive`).

Read top to bottom as:

1. Factor tables    -- completeness and the values the science pins down
2. Calculations     -- one hand-computed case per function, on a few pixels
3. Regression       -- the layer agrees with `emit.derive_from_cie` wherever the two should
4. Products         -- each writes its chosen fields, classes and discountings
5. Laziness         -- nothing is computed until asked
"""

import collections.abc
import dataclasses
import math
import pathlib

import dask.array
import numpy
import pytest
import xarray

from jdluc import derive, emit, export
from jdluc.__tests__.emit_test import get_harmonized_dset_for_every_case
from jdluc.datasets.ipcc_climate_zones import Zone
from jdluc.derive import CIE, CropClass, CropClassLayer, Factors, PeatRegime
from jdluc.emit import ConversionSource


def get_darray_for_data(
    data: collections.abc.Sequence[collections.abc.Sequence[float | bool]],
) -> xarray.DataArray:
    arr = numpy.array(data)
    y, x = arr.shape
    return xarray.DataArray(
        coords={"y": range(y), "x": range(x)}, data=arr, dims=("y", "x")
    )


TROPICAL_ZONES = (
    Zone.TROPICAL_MONTANE,
    Zone.TROPICAL_WET,
    Zone.TROPICAL_MOIST,
    Zone.TROPICAL_DRY,
)


# ============================================================================================
# 1. Factor tables
# ============================================================================================


def test_reference_year_is_2020() -> None:
    assert derive.REFERENCE_YEAR == 2020


@pytest.mark.parametrize("crop_class", list(CropClass))
def test_every_crop_class_has_a_flu_for_every_zone(crop_class: CropClass) -> None:
    assert set(derive.CROP_CLASS_TO_ZONE_TO_FLU[crop_class]) == set(Zone)


@pytest.mark.parametrize("crop_class", list(CropClass))
def test_every_crop_class_has_an_occupation_ef_for_every_peat_regime(
    crop_class: CropClass,
) -> None:
    assert set(derive.CROP_CLASS_TO_PEAT_REGIME_TO_OCCUPATION_EF[crop_class]) == set(
        PeatRegime
    )


def test_every_zone_has_a_peat_regime() -> None:
    assert set(derive.ZONE_TO_PEAT_REGIME) == set(Zone)
    assert {derive.ZONE_TO_PEAT_REGIME[zone] for zone in TROPICAL_ZONES} == {
        PeatRegime.TROPICAL
    }
    assert derive.ZONE_TO_PEAT_REGIME[Zone.BOREAL_MOIST] == PeatRegime.BOREAL
    assert derive.ZONE_TO_PEAT_REGIME[Zone.BOREAL_DRY] == PeatRegime.BOREAL


@pytest.mark.parametrize("crop_class", list(CropClass))
def test_flu_is_a_fraction(crop_class: CropClass) -> None:
    # The layer does not model soil carbon gain, so F_LU is capped at 1
    assert all(
        0 <= flu <= 1 for flu in derive.CROP_CLASS_TO_ZONE_TO_FLU[crop_class].values()
    )


@pytest.mark.parametrize("crop_class", (CropClass.PADDY_RICE, CropClass.PASTURE))
def test_paddy_rice_and_pasture_lose_no_mineral_soil(crop_class: CropClass) -> None:
    assert all(
        flu == 1.0 for flu in derive.CROP_CLASS_TO_ZONE_TO_FLU[crop_class].values()
    )


@pytest.mark.parametrize("zone", TROPICAL_ZONES)
def test_tropical_perennials_lose_no_mineral_soil(zone: Zone) -> None:
    assert derive.CROP_CLASS_TO_ZONE_TO_FLU[CropClass.PERENNIAL][zone] == 1.0


@pytest.mark.parametrize("zone", list(Zone))
def test_annual_flu_matches_emit_cropland_retention(zone: Zone) -> None:
    assert (
        derive.CROP_CLASS_TO_ZONE_TO_FLU[CropClass.ANNUAL][zone]
        == emit.CLIMATE_ZONE_TO_SOC_RETENTION_FRACTION[zone]
    )


@pytest.mark.parametrize("crop_class", list(CropClass))
def test_occupation_efs_are_positive(crop_class: CropClass) -> None:
    assert all(
        ef > 0
        for ef in derive.CROP_CLASS_TO_PEAT_REGIME_TO_OCCUPATION_EF[crop_class].values()
    )


def test_peat_transformation_is_taken_whole() -> None:
    assert derive.PEAT_TRANSFORMATION_FACTOR == 1.0


@pytest.mark.parametrize("discounting", list(derive.DISCOUNTING_TO_WEIGHTS))
def test_every_discounting_spans_the_window_and_sums_to_one(discounting: str) -> None:
    weights = derive.DISCOUNTING_TO_WEIGHTS[discounting]
    assert weights.dims == (derive.CONVERSION_YEAR,)
    assert list(weights[derive.CONVERSION_YEAR].values) == list(range(2001, 2021))
    assert math.isclose(float(weights.sum()), 1)


def test_discount_weights_rise_in_equal_steps() -> None:
    weights = derive.LINEAR_DISCOUNT_WEIGHTS.values
    numpy.testing.assert_allclose(weights[[0, -1]], [0.0025, 0.0975])
    numpy.testing.assert_allclose(numpy.diff(weights), 0.005)


def test_equal_discount_weights_are_one_twentieth() -> None:
    numpy.testing.assert_allclose(derive.EQUAL_DISCOUNT_WEIGHTS.values, 0.05)


@pytest.mark.parametrize("span", list(emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT))
def test_five_year_span_weights_are_emits_and_the_linear_mean(
    span: tuple[int, int],
) -> None:
    """Linear and five-year-span weights differ per year, but agree on average over a span."""
    before, after = span
    years = slice(before + 1, after)
    numpy.testing.assert_allclose(
        derive.FIVE_YEAR_SPAN_DISCOUNT_WEIGHTS.sel(
            {derive.CONVERSION_YEAR: years}
        ).values,
        emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT[span],
    )
    numpy.testing.assert_allclose(
        float(
            derive.LINEAR_DISCOUNT_WEIGHTS.sel({derive.CONVERSION_YEAR: years}).mean()
        ),
        emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT[span],
    )


# ============================================================================================
# 2. Calculations
# ============================================================================================

SOURCES = [
    [
        ConversionSource.NONE,
        ConversionSource.FOREST,
        ConversionSource.NATURAL_GRASSLAND,
        ConversionSource.PASTURE,
    ]
]


def test_is_conversion() -> None:
    # Any recorded source counts, pasture included: a pasture source is a pasture loss event
    result = derive.is_conversion(conversion_source=get_darray_for_data(SOURCES))
    numpy.testing.assert_array_equal(result.data, [[False, True, True, True]])


@pytest.mark.parametrize(
    ("conversion_year", "weight"),
    (
        pytest.param(0, 0.0, id="no conversion"),
        pytest.param(1999, 0.0, id="before the window"),
        # NB: the window is the 20 years ending in the reference year, so 2000 falls outside it
        pytest.param(2000, 0.0, id="2000"),
        pytest.param(2001, 0.0025, id="2001"),
        pytest.param(2005, 0.0225, id="2005"),
        pytest.param(2006, 0.0275, id="2006"),
        pytest.param(2010, 0.0475, id="2010"),
        pytest.param(2011, 0.0525, id="2011"),
        pytest.param(2015, 0.0725, id="2015"),
        pytest.param(2016, 0.0775, id="2016"),
        pytest.param(2020, 0.0975, id="2020"),
        pytest.param(2021, 0.0, id="after the reference year"),
    ),
)
def test_discount_weight_with_linear_weights(
    conversion_year: int, weight: float
) -> None:
    result = derive.discount_weight(
        get_darray_for_data([[conversion_year]]), derive.LINEAR_DISCOUNT_WEIGHTS
    )
    numpy.testing.assert_allclose(result.data, [[weight]])


@pytest.mark.parametrize("conversion_year", [0, *range(1999, 2022)])
def test_discount_weight_with_five_year_spans_is_emits(conversion_year: int) -> None:
    year = get_darray_for_data([[conversion_year]])
    expected = emit.get_linear_discounted_total(
        emit.get_span_to_charge(
            conversion_year=year, darray=xarray.ones_like(year, dtype=float)
        )
    )
    numpy.testing.assert_allclose(
        derive.discount_weight(year, derive.FIVE_YEAR_SPAN_DISCOUNT_WEIGHTS).data,
        expected.data,
    )


def test_vegetation_emissions_amortized() -> None:
    result = derive.vegetation_emissions_amortized(
        vegetation_emissions_undiscounted=get_darray_for_data([[100.0, 100.0, 100.0]]),
        is_conversion=get_darray_for_data([[True, True, False]]),
        discount_weight=get_darray_for_data([[0.0875, 0.0125, 0.0875]]),
    )
    # 100 x 0.0875; 100 x 0.0125; not converted
    numpy.testing.assert_allclose(result.data, [[8.75, 1.25, 0.0]])


def test_mineral_soil_emissions_amortized() -> None:
    result = derive.mineral_soil_emissions_amortized(
        mineral_soil_carbon_at_risk=get_darray_for_data(
            [[200.0, 200.0, 200.0, 0.0, 200.0]]
        ),
        flu=get_darray_for_data([[0.83, 1.0, 0.69, 0.83, 0.83]]),
        is_conversion=get_darray_for_data([[True, True, True, True, False]]),
        discount_weight=get_darray_for_data([[0.0875] * 5]),
    )
    numpy.testing.assert_allclose(
        result.data,
        [
            [
                200 * (1 - 0.83) * 0.0875,  # 2.975
                0.0,  # F_LU 1 keeps all the soil
                200 * (1 - 0.69) * 0.0875,  # 5.425
                0.0,  # on peat CIE puts no mineral carbon at risk
                0.0,  # not converted
            ]
        ],
    )


def test_peat_transformation_emissions_amortized() -> None:
    result = derive.peat_transformation_emissions_amortized(
        peat_transformation_emissions_undiscounted=get_darray_for_data(
            [[621.0, 621.0, 0.0]]
        ),
        is_conversion=get_darray_for_data([[True, False, True]]),
        discount_weight=get_darray_for_data([[0.0625, 0.0625, 0.0625]]),
    )
    # 621 x 0.0625; not converted; off peat
    numpy.testing.assert_allclose(result.data, [[38.8125, 0.0, 0.0]])


def test_is_peat() -> None:
    result = derive.is_peat(get_darray_for_data([[0.0, 37.3]]))
    numpy.testing.assert_array_equal(result.data, [[False, True]])


def test_peat_occupation_emissions_per_year() -> None:
    result = derive.peat_occupation_emissions_per_year(
        is_peat=get_darray_for_data([[True, False]]),
        peat_occupation_ef=get_darray_for_data([[56.92, 56.92]]),
    )
    numpy.testing.assert_allclose(result.data, [[56.92, 0.0]])


def test_total_emissions_amortized() -> None:
    result = derive.total_emissions_amortized(
        vegetation_emissions_amortized=get_darray_for_data([[1.0, 0.0]]),
        mineral_soil_emissions_amortized=get_darray_for_data([[10.0, 0.0]]),
        peat_transformation_emissions_amortized=get_darray_for_data([[100.0, 0.0]]),
        peat_occupation_emissions_per_year=get_darray_for_data([[1000.0, 0.0]]),
    )
    numpy.testing.assert_allclose(result.data, [[1111.0, 0.0]])


# Four pixels, through the whole layer by hand:
#   0. forest lost in 2018 on mineral soil (tropical moist)
#   1. forest lost in 2003 on peat (tropical moist)
#   2. never converted, but on peat (tropical moist): occupation is charged regardless
#   3. pasture that left pasture in 2012, on mineral soil (warm temperate moist)
def get_cie_for_four_pixels() -> CIE:
    return CIE(
        conversion_source=get_darray_for_data(
            [
                [
                    ConversionSource.FOREST,
                    ConversionSource.FOREST,
                    ConversionSource.NONE,
                    ConversionSource.PASTURE,
                ]
            ]
        ),
        conversion_year=get_darray_for_data([[2018, 2003, 0, 2012]]),
        vegetation_emissions_undiscounted=get_darray_for_data(
            [[400.0, 400.0, 0.0, 25.0]]
        ),
        mineral_soil_carbon_at_risk=get_darray_for_data([[200.0, 0.0, 0.0, 300.0]]),
        peat_transformation_emissions_undiscounted=get_darray_for_data(
            [[0.0, 621.0, 0.0, 0.0]]
        ),
        peat_occupation_emissions=get_darray_for_data([[0.0, 37.3, 37.3, 0.0]]),
        climate_zone=get_darray_for_data(
            [
                [
                    Zone.TROPICAL_MOIST,
                    Zone.TROPICAL_MOIST,
                    Zone.TROPICAL_MOIST,
                    Zone.WARM_TEMPERATE_MOIST,
                ]
            ]
        ),
        hectares_per_pixel=get_darray_for_data([[0.09, 0.09, 0.09, 0.08]]),
    )


def get_annual_factors_for_four_pixels() -> Factors:
    return Factors(
        flu=get_darray_for_data([[0.83, 0.83, 0.83, 0.69]]),
        peat_occupation_ef=get_darray_for_data([[56.92, 56.92, 56.92, 38.18]]),
    )


def test_crop_class_layer_for_annual_crops() -> None:
    layer = derive.crop_class_layer(
        cie=get_cie_for_four_pixels(),
        factors=get_annual_factors_for_four_pixels(),
        discount_weights=derive.LINEAR_DISCOUNT_WEIGHTS,
    )
    numpy.testing.assert_allclose(
        layer.discount_weight.data, [[0.0875, 0.0125, 0.0, 0.0575]]
    )
    numpy.testing.assert_allclose(
        layer.vegetation_emissions_amortized.data,
        [[400 * 0.0875, 400 * 0.0125, 0.0, 25 * 0.0575]],  # 35, 5, 0, 1.4375
    )
    numpy.testing.assert_allclose(
        layer.mineral_soil_emissions_amortized.data,
        [[200 * 0.17 * 0.0875, 0.0, 0.0, 300 * 0.31 * 0.0575]],  # 2.975, 0, 0, 5.3475
    )
    numpy.testing.assert_allclose(
        layer.peat_transformation_emissions_amortized.data,
        [[0.0, 621 * 0.0125, 0.0, 0.0]],  # 7.7625
    )
    numpy.testing.assert_array_equal(layer.is_peat.data, [[False, True, True, False]])
    # NB: not gated on conversion: the never-converted peat pixel is charged too
    numpy.testing.assert_allclose(
        layer.peat_occupation_emissions_per_year.data, [[0.0, 56.92, 56.92, 0.0]]
    )
    numpy.testing.assert_allclose(
        layer.total_emissions_amortized.data,
        [[35 + 2.975, 5 + 7.7625 + 56.92, 56.92, 1.4375 + 5.3475]],
    )
    numpy.testing.assert_allclose(
        layer.hectares_per_pixel.data, [[0.09, 0.09, 0.09, 0.08]]
    )


def test_crop_class_layer_for_pasture_loses_no_mineral_soil() -> None:
    layer = derive.crop_class_layer(
        cie=get_cie_for_four_pixels(),
        factors=Factors(
            flu=get_darray_for_data([[1.0, 1.0, 1.0, 1.0]]),
            peat_occupation_ef=get_darray_for_data([[42.32, 42.32, 42.32, 29.87]]),
        ),
        discount_weights=derive.LINEAR_DISCOUNT_WEIGHTS,
    )
    # Pixel 3 lost pasture in 2012: its vegetation is charged, like any other class's
    numpy.testing.assert_allclose(
        layer.vegetation_emissions_amortized.data, [[35.0, 5.0, 0.0, 1.4375]]
    )
    numpy.testing.assert_allclose(
        layer.mineral_soil_emissions_amortized.data, [[0.0, 0.0, 0.0, 0.0]]
    )
    numpy.testing.assert_allclose(
        layer.total_emissions_amortized.data,
        [[35.0, 5 + 7.7625 + 42.32, 42.32, 1.4375]],
    )


def test_crop_class_layer_total_is_the_sum_of_its_terms() -> None:
    layer = derive.crop_class_layer(
        cie=get_cie_for_four_pixels(),
        factors=get_annual_factors_for_four_pixels(),
        discount_weights=derive.LINEAR_DISCOUNT_WEIGHTS,
    )
    numpy.testing.assert_allclose(
        layer.total_emissions_amortized.data,
        (
            layer.vegetation_emissions_amortized
            + layer.mineral_soil_emissions_amortized
            + layer.peat_transformation_emissions_amortized
            + layer.peat_occupation_emissions_per_year
        ).data,
    )


# ============================================================================================
# 3. Regression against `emit.derive_from_cie`, end to end through `derive.derive`
#
# `derive_from_cie` gates on the observed destination, with the annual soil factor for cropland
# and a flat 37.3 peat occupation. With the layer on `emit`'s five-year-span discounting,
# everything but peat occupation must agree wherever the destination matches the crop class.
# ============================================================================================


def get_band(
    dset: xarray.Dataset,
    field: str,
    crop_class: str | None = None,
    discounting: str | None = None,
) -> numpy.ndarray:
    """`field`'s (y, x) values, at `crop_class` and/or `discounting` if it carries them."""
    darray = dset[derive.get_variable_name(field)]
    point = {derive.CROP_CLASS: crop_class, derive.DISCOUNTING: discounting}
    return darray.sel({dim: point[dim] for dim in darray.dims if dim in point}).data


@dataclasses.dataclass(frozen=True)
class Comparison:
    """`derive.derive` and `emit.derive_from_cie` on the same `emit` output, computed."""

    crop_class: CropClass
    discounting: str
    output: xarray.Dataset
    legacy: xarray.Dataset
    layer: xarray.Dataset

    def band(self, field: str) -> numpy.ndarray:
        """The layer's `field`, at this comparison's crop class and discounting."""
        return get_band(self.layer, field, self.crop_class, self.discounting)

    @property
    def to_cropland(self) -> numpy.ndarray:
        destination = (
            self.output["cie-destination-dataset"].fillna(0).data.astype(numpy.uint8)
        )
        return (destination & emit.CROPLAND_DATASETS).astype(bool)

    @property
    def to_pasture(self) -> numpy.ndarray:
        destination = (
            self.output["cie-destination-dataset"].fillna(0).data.astype(numpy.uint8)
        )
        return ~self.to_cropland & (destination & emit.PASTURE_DATASETS).astype(bool)

    @property
    def climate_zone(self) -> numpy.ndarray:
        return self.output["cie-climate-zone"].data

    @property
    def source(self) -> numpy.ndarray:
        return self.output["cie-conversion-source"].data


def get_comparison(crop_class: CropClass) -> Comparison:
    output = emit.get_output_dset(dset=get_harmonized_dset_for_every_case()).compute()
    return Comparison(
        output=output,
        legacy=emit.derive_from_cie(dset=output).compute(),
        crop_class=crop_class,
        discounting="five-year-spans",
        layer=derive.derive(
            dset=output, crop_classes=(crop_class,), discountings=("five-year-spans",)
        ).compute(),
    )


def test_derive_writes_every_field_with_the_requested_classes_and_discountings() -> (
    None
):
    output = emit.get_output_dset(dset=get_harmonized_dset_for_every_case())
    layer = derive.derive(
        dset=output,
        crop_classes=(CropClass.ANNUAL, CropClass.PERENNIAL),
        discountings=("equal", "linear"),
    )
    assert list(layer) == [derive.get_variable_name(field) for field in derive.MENU]
    assert list(layer[derive.CROP_CLASS].values) == ["annual", "perennial"]
    assert list(layer[derive.DISCOUNTING].values) == ["equal", "linear"]


def test_each_field_carries_the_dimensions_of_what_it_depends_on() -> None:
    by_class = {"flu", "peat_occupation_ef", "peat_occupation_emissions_per_year"}
    discounted = {
        "discount_weight",
        "vegetation_emissions_amortized",
        "peat_transformation_emissions_amortized",
    }
    both = {"mineral_soil_emissions_amortized", "total_emissions_amortized"}
    layer = derive.derive(
        dset=emit.get_output_dset(dset=get_harmonized_dset_for_every_case())
    )
    for field in derive.MENU:
        dims = set(layer[derive.get_variable_name(field)].dims) - {"y", "x"}
        if field in by_class:
            assert dims == {derive.CROP_CLASS}, field
        elif field in discounted:
            assert dims == {derive.DISCOUNTING}, field
        elif field in both:
            assert dims == {derive.CROP_CLASS, derive.DISCOUNTING}, field
        else:
            assert dims == set(), field


def test_variable_names_carry_their_units() -> None:
    assert (
        derive.get_variable_name("total_emissions_amortized")
        == "ccl-total-emissions-amortized:tco2e-per-ha"
    )
    assert (
        derive.get_variable_name("peat_occupation_emissions_per_year")
        == "ccl-peat-occupation-emissions-per-year:tco2e-per-ha-per-year"
    )
    assert derive.get_variable_name("flu") == "ccl-flu"


def test_cog_band_names_append_the_crop_class_then_the_discounting() -> None:
    layer = derive.derive(
        dset=emit.get_output_dset(dset=get_harmonized_dset_for_every_case()),
        crop_classes=(CropClass.PADDY_RICE,),
        discountings=("equal",),
    )
    cog = export.to_cog_dataset(
        dset=layer[[derive.get_variable_name("total_emissions_amortized")]]
    )
    assert list(cog) == ["ccl-total-emissions-amortized:tco2e-per-ha:paddy-rice:equal"]


def test_annual_agrees_with_derive_from_cie_on_cropland() -> None:
    comparison = get_comparison(crop_class=CropClass.ANNUAL)
    mask = comparison.to_cropland
    assert mask.any()

    layer_without_occupation = comparison.band(
        "total_emissions_amortized"
    ) - comparison.band("peat_occupation_emissions_per_year")
    legacy_without_occupation = (
        comparison.legacy["emissions-per-hectare:tco2e-per-ha"].data
        - comparison.legacy["cropland-peatland-occupation:tco2e-per-ha"].data
    )
    # The comparison is not vacuous: some cropland pixels lose mineral soil
    assert (comparison.band("mineral_soil_emissions_amortized")[mask] > 0).any()
    numpy.testing.assert_allclose(
        layer_without_occupation[mask],
        legacy_without_occupation[mask],
        atol=1e-4,
        rtol=1e-6,
    )


def test_pasture_agrees_with_derive_from_cie_on_pasture() -> None:
    comparison = get_comparison(crop_class=CropClass.PASTURE)
    # NB: `derive_from_cie` charges nothing for pasture lost to pasture; the layer charges it,
    # as it does any pasture loss event (see the next test)
    mask = comparison.to_pasture & (comparison.source != ConversionSource.PASTURE)
    assert mask.any()

    layer_without_occupation = comparison.band(
        "total_emissions_amortized"
    ) - comparison.band("peat_occupation_emissions_per_year")
    legacy_without_occupation = (
        comparison.legacy["emissions-per-hectare:tco2e-per-ha"].data
        - comparison.legacy["pastureland-peatland-occupation:tco2e-per-ha"].data
    )
    # The comparison is not vacuous: some pasture pixels are charged for their conversion
    assert (layer_without_occupation[mask] > 0).any()
    # Pasture keeps all its mineral soil, in both
    numpy.testing.assert_array_equal(
        comparison.band("mineral_soil_emissions_amortized")[mask], 0.0
    )
    numpy.testing.assert_allclose(
        layer_without_occupation[mask],
        legacy_without_occupation[mask],
        atol=1e-4,
        rtol=1e-6,
    )


def test_pasture_charges_pasture_lost_to_pasture() -> None:
    comparison = get_comparison(crop_class=CropClass.PASTURE)
    mask = comparison.to_pasture & (comparison.source == ConversionSource.PASTURE)
    assert mask.any()
    assert (comparison.band("vegetation_emissions_amortized")[mask] > 0).any()


@pytest.mark.parametrize("crop_class", list(CropClass))
def test_vegetation_where_no_destination_is_derive_from_cies_dropped_emissions(
    crop_class: CropClass,
) -> None:
    comparison = get_comparison(crop_class=crop_class)
    mask = ~comparison.to_cropland & ~comparison.to_pasture
    assert (comparison.legacy["dropped-emissions:tco2e-per-ha"].data[mask] > 0).any()
    numpy.testing.assert_allclose(
        comparison.band("vegetation_emissions_amortized")[mask],
        comparison.legacy["dropped-emissions:tco2e-per-ha"].data[mask],
        atol=1e-4,
        rtol=1e-6,
    )


def test_read_cie_reads_missing_source_and_year_as_no_conversion() -> None:
    output = emit.get_output_dset(dset=get_harmonized_dset_for_every_case())
    output["cie-conversion-source"] = xarray.full_like(
        output["cie-conversion-source"], numpy.nan
    )
    output["cie-conversion-year"] = xarray.full_like(
        output["cie-conversion-year"], numpy.nan
    )
    cie = derive.read_cie(output)
    assert not derive.is_conversion(cie.conversion_source).any()
    for weights in derive.DISCOUNTING_TO_WEIGHTS.values():
        assert (derive.discount_weight(cie.conversion_year, weights) == 0).all()


# ============================================================================================
# 4. Products
# ============================================================================================


def cog_band(dset: xarray.Dataset, field: str, *values: str) -> numpy.ndarray:
    """A product's COG band for `field` at `values` (crop class, then discounting)."""
    return dset[":".join((derive.get_variable_name(field), *values))].data


def get_product_layer(product: derive.Product) -> xarray.Dataset:
    output = emit.get_output_dset(dset=get_harmonized_dset_for_every_case())
    return derive.derive(
        dset=output,
        crop_classes=product.crop_classes,
        discountings=product.discountings,
    )[product.variables]


def to_cog_dataset(product: derive.Product) -> xarray.Dataset:
    return export.to_cog_dataset(dset=get_product_layer(product)).compute()


def test_a_product_only_takes_fields_from_the_menu() -> None:
    with pytest.raises(AssertionError, match="not on the menu"):
        derive.Product(name="bad", fields=("total_emisions",))


def test_a_product_only_takes_known_discountings() -> None:
    with pytest.raises(AssertionError, match="unknown discountings"):
        derive.Product(
            name="bad", fields=("discount_weight",), discountings=("lineal",)
        )


def test_a_product_selects_something() -> None:
    with pytest.raises(AssertionError, match="selects nothing"):
        derive.Product(name="bad", fields=("flu",), crop_classes=())


@pytest.mark.parametrize("product", list(derive.NAME_TO_PRODUCT.values()))
def test_a_product_is_built_for_only_the_classes_and_discountings_it_names(
    product: derive.Product,
) -> None:
    layer = get_product_layer(product)
    assert list(layer) == product.variables
    if derive.CROP_CLASS in layer.dims:
        assert list(layer[derive.CROP_CLASS].values) == list(
            map(str, product.crop_classes)
        )
    if derive.DISCOUNTING in layer.dims:
        assert list(layer[derive.DISCOUNTING].values) == list(product.discountings)


def test_crop_class_emissions_carries_each_classs_total() -> None:
    output = emit.get_output_dset(dset=get_harmonized_dset_for_every_case()).compute()
    layer = derive.derive(dset=output).compute()
    result = to_cog_dataset(derive.CROP_CLASS_EMISSIONS)
    assert list(result) == [
        *(
            f"ccl-total-emissions-amortized:tco2e-per-ha:{crop_class:s}:linear"
            for crop_class in CropClass
        ),
        "ccl-hectares-per-pixel:ha",
    ]
    for crop_class in CropClass:
        # NB: the COG is float32; the layer keeps the precision of its inputs
        numpy.testing.assert_allclose(
            cog_band(result, "total_emissions_amortized", crop_class, "linear"),
            get_band(layer, "total_emissions_amortized", crop_class, "linear"),
            rtol=1e-6,
        )
    # The classes really differ: perennials keep their soil carbon in the tropics
    assert not numpy.array_equal(
        cog_band(result, "total_emissions_amortized", CropClass.ANNUAL, "linear"),
        cog_band(result, "total_emissions_amortized", CropClass.PERENNIAL, "linear"),
    )


def test_perennial_discounting_comparison_discounts_each_term_both_ways() -> None:
    output = emit.get_output_dset(dset=get_harmonized_dset_for_every_case()).compute()
    result = to_cog_dataset(derive.PERENNIAL_DISCOUNTING_COMPARISON)
    cie = derive.read_cie(output)
    converted = derive.is_conversion(cie.conversion_source).data
    vegetation = numpy.where(converted, cie.vegetation_emissions_undiscounted.data, 0)
    # Equal discounting charges 1/20 of every conversion in the window, whatever its year
    in_window = (cie.conversion_year.data > 2000) & (cie.conversion_year.data <= 2020)
    numpy.testing.assert_allclose(
        cog_band(result, "vegetation_emissions_amortized", "equal"),
        numpy.where(in_window, vegetation * 0.05, 0),
        rtol=1e-6,
    )
    assert not numpy.allclose(
        cog_band(result, "vegetation_emissions_amortized", "equal"),
        cog_band(result, "vegetation_emissions_amortized", "linear"),
    )
    # Peat occupation is one band, and each discounting's total is its own terms' sum
    occupation = cog_band(result, "peat_occupation_emissions_per_year", "perennial")
    assert [
        name for name in map(str, result) if "peat-occupation-emissions" in name
    ] == ["ccl-peat-occupation-emissions-per-year:tco2e-per-ha-per-year:perennial"]
    for discounting in ("equal", "linear"):
        numpy.testing.assert_allclose(
            cog_band(result, "total_emissions_amortized", "perennial", discounting),
            cog_band(result, "vegetation_emissions_amortized", discounting)
            + cog_band(
                result, "mineral_soil_emissions_amortized", "perennial", discounting
            )
            + cog_band(result, "peat_transformation_emissions_amortized", discounting)
            + occupation,
            rtol=1e-5,
        )


def test_export_workflow_writes_each_product_in_its_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    import rasterio

    output = emit.get_output_dset(dset=get_harmonized_dset_for_every_case())
    monkeypatch.setattr(emit, "workflow", lambda tile_id: output)
    cog, zarr = derive.CROP_CLASS_EMISSIONS, derive.PERENNIAL_DEBUG
    assert (cog.format, zarr.format) == (export.Format.COG, export.Format.ZARR)
    uris = derive.export_workflow(
        tile_id="00N_000E",
        product_names=[cog.name, zarr.name],
        output_root=str(tmp_path),
    )
    assert uris == [
        str(tmp_path / cog.name / "00N_000E.tif"),
        str(tmp_path / zarr.name / "00N_000E.zarr"),
    ]
    with rasterio.open(uris[0]) as dataset:
        assert list(dataset.descriptions) == list(to_cog_dataset(cog))
    written = xarray.open_zarr(uris[1], consolidated=False)
    assert set(written) == set(zarr.variables)
    total = written[derive.get_variable_name("total_emissions_amortized")]
    assert set(total.dims) == {derive.DISCOUNTING, derive.CROP_CLASS, "y", "x"}
    assert list(total[derive.CROP_CLASS].values) == ["perennial"]
    assert written.attrs["cornerstone-product-name"] == zarr.name


# ============================================================================================
# 5. Laziness
# ============================================================================================


def chunked(darray: xarray.DataArray) -> xarray.DataArray:
    return darray.chunk({"x": 2})


def test_crop_class_layer_stays_lazy() -> None:
    cie = get_cie_for_four_pixels()
    factors = get_annual_factors_for_four_pixels()
    layer = derive.crop_class_layer(
        cie=CIE(
            **{
                field.name: chunked(getattr(cie, field.name))
                for field in dataclasses.fields(CIE)
            }
        ),
        factors=Factors(
            **{
                field.name: chunked(getattr(factors, field.name))
                for field in dataclasses.fields(Factors)
            }
        ),
        discount_weights=derive.LINEAR_DISCOUNT_WEIGHTS,
    )
    for field in dataclasses.fields(CropClassLayer):
        assert isinstance(getattr(layer, field.name).data, dask.array.Array), field.name


def test_lookup_factors_builds_one_chunk_per_requested_crop_class() -> None:
    climate_zone = chunked(get_darray_for_data([[Zone.TROPICAL_WET, Zone.BOREAL_DRY]]))
    factors = derive.lookup_factors(
        climate_zone, crop_classes=(CropClass.PERENNIAL, CropClass.PASTURE)
    )
    for factor in (factors.flu, factors.peat_occupation_ef):
        assert factor.dims == ("y", "x", derive.CROP_CLASS)
        assert list(factor[derive.CROP_CLASS].values) == ["perennial", "pasture"]
        assert factor.chunksizes[derive.CROP_CLASS] == (1, 1)
    numpy.testing.assert_allclose(
        factors.flu.compute().data,
        # (y, x, crop_class): tropical wet then boreal dry; perennial then pasture
        [[[1.0, 1.0], [0.72, 1.0]]],
    )
