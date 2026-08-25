import collections
import collections.abc

import numpy
import pytest
import xarray

from jdluc.datasets.glad_glcluc import LandClass
from jdluc.datasets.ipcc_climate_zones import Zone
from jdluc.emit import (
    CARBON_PER_BIOMASS_LIVE_WOOD,
    CO2E_PER_CARBON,
    SPAN_TO_LINEAR_DISCOUNT_WEIGHT,
    get_belowground_carbon,
    get_dead_organic_matter_carbon,
    get_grassland_carbon,
    get_hectares_per_pixel,
    get_land_class,
    get_linear_discounted_total,
    get_mineral_soil_emissions,
    get_peatland_occupation_emissions,
    get_soil_emissions,
    get_vegetation_carbon,
    get_vegetation_emissions,
)


def get_darray_for_data(
    data: collections.abc.Sequence[collections.abc.Sequence[int | float]],
) -> xarray.DataArray:
    arr = numpy.array(data)
    y, x = arr.shape
    return xarray.DataArray(
        coords={
            "y": range(y),
            "x": range(x),
        },
        data=arr,
        dims=("y", "x"),
    )


@pytest.mark.parametrize(
    ("data", "expected"),
    (
        ([[0, 100]] * 2, {LandClass.GRASSLAND.value: 4}),
        ([[25, 125]] * 2, {LandClass.FOREST.value: 4}),
        ([[200, 207]] * 2, {LandClass.WATER.value: 4}),
        (
            [[241, 244, 250, 254]],
            {
                LandClass.BUILT_UP.value: 1,
                LandClass.CROPLAND.value: 1,
                LandClass.SNOW_ICE.value: 1,
                LandClass.OCEAN.value: 1,
            },
        ),
        ([[-1, 255, 1 << 10, numpy.nan]], {0: 4}),
    ),
)
def test_get_land_class(data: list[list[int]], expected: dict[int, int]) -> None:
    result = get_land_class(glad_class=get_darray_for_data(data=data))
    assert result.name is None
    assert collections.Counter(result.data.ravel()) == expected


def test_get_belowground_carbon() -> None:
    agb = [[0, 1, 2]]
    bgb = [[2, 1, 0]]
    result = get_belowground_carbon(
        aboveground_biomass=get_darray_for_data(agb),
        belowground_biomass=get_darray_for_data(bgb),
    )
    assert result.name == "tcarbon-per-ha"
    assert numpy.array_equal(
        result.data.ravel(), numpy.array([2, 1, 0.5]) * CARBON_PER_BIOMASS_LIVE_WOOD
    )


def test_get_dead_organic_matter_carbon() -> None:
    agb = [[0, 1]] * 4
    zones = [
        [Zone.TROPICAL_WET.value] * 2,
        [Zone.BOREAL_DRY.value] * 2,
        [-1] * 2,
        [numpy.nan] * 2,
    ]
    result = get_dead_organic_matter_carbon(
        aboveground_biomass=get_darray_for_data(data=agb),
        climate_zones=get_darray_for_data(data=zones),  # type: ignore
    )
    assert result.name == "tcarbon-per-ha"
    assert numpy.array_equal(
        result,
        numpy.array(
            [
                [
                    0,
                    0.5 * 0.06 + 0.37 * 0.01,
                ],
                [
                    0,
                    0.5 * 0.08 + 0.37 * 0.04,
                ],
                [0, 0],
                [0, 0],
            ]
        ),
    )


def test_get_grassland_carbon() -> None:
    data = [[Zone.TROPICAL_WET.value, Zone.BOREAL_DRY.value, -1, numpy.nan]]
    result = get_grassland_carbon(climate_zones=get_darray_for_data(data=data))
    assert result.name == "tcarbon-per-ha"
    assert numpy.array_equal(result.data, [[18, 3, 0, 0]])


def test_get_vegetation_carbon() -> None:
    agb = [[0, 1], [2, 3], [4, 5]]
    bgb = [[1, 1], [0, 0], [0, 0]]
    dom = [[1, 1]] * 3
    grassland = [[1, 1]] * 3
    classes = [
        [LandClass.FOREST.value, LandClass.GRASSLAND.value],
        [LandClass.FOREST.value, LandClass.GRASSLAND.value],
        [LandClass.CROPLAND.value, LandClass.BUILT_UP.value],
    ]
    result = get_vegetation_carbon(
        aboveground_carbon=get_darray_for_data(data=agb),
        belowground_carbon=get_darray_for_data(data=bgb),
        dead_organic_carbon=get_darray_for_data(data=dom),
        grassland_carbon=get_darray_for_data(data=grassland),
        land_class=get_darray_for_data(data=classes),
    )
    assert result.name == "tcarbon-per-ha"
    assert numpy.array_equal(result.data, numpy.array([[2, 1], [3, 1], [0, 0]]))


def test_get_vegetation_emissions() -> None:
    after = [[0, 1, 2, numpy.nan]]
    before = [[numpy.nan, 2, 1, 0]]
    result = get_vegetation_emissions(
        after=get_darray_for_data(data=after), before=get_darray_for_data(data=before)
    )
    assert result.name == "tco2e-per-ha"
    assert numpy.array_equal(
        result.data, [[numpy.nan, CO2E_PER_CARBON, 0, numpy.nan]], equal_nan=True
    )


def test_get_mineral_soil_emissions() -> None:
    soc = [[0] * 4, [1] * 4, [2] * 4]
    zones = [[Zone.BOREAL_DRY.value, Zone.TROPICAL_WET.value, -1, numpy.nan]] * 3
    result = get_mineral_soil_emissions(
        climate_zones=get_darray_for_data(data=zones),
        soil_organic_carbon=get_darray_for_data(data=soc),
    )
    assert result.name == "tco2e-per-ha"
    numpy.testing.assert_allclose(
        result.data * 3,
        numpy.array([[0, 0, 0, 0], [2.53, 1.87, 0, 0], [5.06, 3.74, 0, 0]]),
    )


def test_get_soil_emissions() -> None:
    after = [
        [
            LandClass.FOREST.value,
            LandClass.GRASSLAND.value,
            LandClass.CROPLAND.value,
            LandClass.CROPLAND.value,
        ]
    ] * 4
    before = [
        [LandClass.FOREST.value] * 4,
        [LandClass.GRASSLAND.value] * 4,
        [LandClass.CROPLAND.value] * 4,
        [LandClass.CROPLAND.value] * 4,
    ]
    zones = [[Zone.TROPICAL_WET.value] * 4] * 4
    is_peatland = [[0, 0, 0, 1]] * 4
    soc = [[1] * 4] * 4
    result = get_soil_emissions(
        after=get_darray_for_data(data=after),
        before=get_darray_for_data(data=before),
        climate_zones=get_darray_for_data(data=zones),
        is_peatland=get_darray_for_data(data=is_peatland),
        soil_organic_carbon=get_darray_for_data(data=soc),
    )
    assert result.name == "tco2e-per-ha"
    numpy.testing.assert_allclose(
        result.data,
        numpy.array(
            [
                [0, 0, 1.87 / 3, 621],
                [0, 0, 1.87 / 3, 621],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
    )


def test_get_peatland_occupation_emissions() -> None:
    is_peatland = [[1] * 7 + [numpy.nan, 0]]
    land_class = [
        [
            LandClass.BUILT_UP.value,
            LandClass.CROPLAND.value,
            LandClass.FOREST.value,
            LandClass.GRASSLAND.value,
            LandClass.OCEAN.value,
            LandClass.SNOW_ICE.value,
            LandClass.WATER.value,
            # drained, but the peat mask is no-data / not peat
            LandClass.CROPLAND.value,
            LandClass.CROPLAND.value,
        ]
    ]
    result = get_peatland_occupation_emissions(
        is_peatland=get_darray_for_data(data=is_peatland),
        latest_land_class=get_darray_for_data(data=land_class),
    )
    assert result.name == "tco2e-per-ha"
    # a no-data peat mask must read as "not peat" rather than propagating NaN into the total
    numpy.testing.assert_allclose(result.data, [[37.3, 37.3, 0, 0, 0, 0, 0, 0, 0]])


def test_get_soil_emissions_tolerates_missing_soil_carbon() -> None:
    # SoilGrids has genuine gaps over water, rock and ice; a missing stock must not poison
    # the pixel, which would silently discard its vegetation emissions too
    result = get_soil_emissions(
        after=get_darray_for_data(data=[[LandClass.CROPLAND.value] * 2]),
        before=get_darray_for_data(data=[[LandClass.FOREST.value] * 2]),
        climate_zones=get_darray_for_data(data=[[Zone.TROPICAL_WET.value] * 2]),
        is_peatland=get_darray_for_data(data=[[0, 1]]),
        soil_organic_carbon=get_darray_for_data(data=[[numpy.nan, numpy.nan]]),
    )
    assert not numpy.isnan(result.data).any()
    # mineral pixel falls back to zero stock; the peat pulse does not depend on SoilGrids
    numpy.testing.assert_allclose(result.data, [[0, 621]])


def test_get_linear_discounted_total() -> None:
    result = get_linear_discounted_total(
        span_to_value={
            (2000, 2005): get_darray_for_data(data=[[1]]),
            (2005, 2010): get_darray_for_data(data=[[2]]),
            (2010, 2015): get_darray_for_data(data=[[3]]),
            (2015, 2020): get_darray_for_data(data=[[4]]),
        }
    )
    assert numpy.array_equal(result.data, [[0.625]])


def test_get_hectares_per_pixel() -> None:
    result = get_hectares_per_pixel(darray=get_darray_for_data(data=[[0] * 2] * 2))
    assert result.name == "ha"
    numpy.testing.assert_allclose(
        result.data,
        numpy.array(
            [[1237126.38106379, 1237126.38106379], [1236937.9607238, 1236937.9607238]]
        ),
    )


def test_get_linear_discounted_total_imposes_no_quantity() -> None:
    # Area and production run through this too, so a name of its own would be a lie for two of
    # the three callers.  The caller names the result.
    result = get_linear_discounted_total(
        span_to_value=dict.fromkeys(
            SPAN_TO_LINEAR_DISCOUNT_WEIGHT, get_darray_for_data(data=[[1]])
        )
    )
    assert result.name is None


def test_get_linear_discounted_total_over_the_weight_total_is_a_mean() -> None:
    # How `statistical` puts area and production on the emissions' window: the same reduction,
    # normalised.  A quantity that never moves must survive it unchanged.
    result = get_linear_discounted_total(
        span_to_value=dict.fromkeys(
            SPAN_TO_LINEAR_DISCOUNT_WEIGHT, get_darray_for_data(data=[[7]])
        )
    ) / sum(SPAN_TO_LINEAR_DISCOUNT_WEIGHT.values())
    numpy.testing.assert_allclose(result.data, [[7]])
