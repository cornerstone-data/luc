import collections.abc

import numpy
import xarray

from jdluc import emit, statistical
from jdluc.datasets import usda_nass_quickstats
from jdluc.jurisdictional_direct import Crop, get_commodity_name_to_totals


def get_darray_for_data(
    data: collections.abc.Sequence[collections.abc.Sequence[float]],
) -> xarray.DataArray:
    arr = numpy.array(data)
    y, x = arr.shape
    return xarray.DataArray(
        coords={"y": range(y), "x": range(x)}, data=arr, dims=("y", "x")
    )


def test_get_commodity_name_to_totals() -> None:
    result = get_commodity_name_to_totals(
        # A distinct array per component, so a column reading the wrong one is visible
        component_to_per_hectare={
            emit.EmissionComponent.FOREST: get_darray_for_data(
                [[2, 3, 99], [99, 5, 99]]
            ),
            emit.EmissionComponent.GRASSLAND: get_darray_for_data(
                [[1, 1, 99], [99, 1, 99]]
            ),
            emit.EmissionComponent.PEATLAND_CONVERSION: get_darray_for_data(
                [[1, 2, 99], [99, 3, 99]]
            ),
        },
        # crop pixels: (0,0), (0,1), (1,1); of those, peatland at (0,0) and (1,1).
        crop_class=get_darray_for_data([[1, 1, 0], [0, 1, 0]]),
        crops=(Crop.MAIZE,),
        dropped_per_hectare=get_darray_for_data([[1, 0, 1], [0, 1, 0]]),
        # 99s sit on non-crop pixels — they must be excluded by the mask.
        emissions_per_hectare=get_darray_for_data([[1, 2, 99], [99, 4, 99]]),
        hectares_per_pixel=get_darray_for_data([[10, 10, 10], [10, 10, 10]]),
        is_peatland=get_darray_for_data([[True, False, False], [False, True, False]]),
        peatland_occupation_per_hectare=get_darray_for_data([[5, 5, 5], [5, 5, 5]]),
    )
    assert set(result) == {
        Crop.MAIZE.name,
        emit.NonCommodity.DROPPED.name,
    }
    # Summed over the whole clip rather than masked, so the non-crop pixel at (0,2) counts
    assert result[emit.NonCommodity.DROPPED.name] == {
        "emissions_mt": 30
    }  # (1+1+1) x 10
    assert result[Crop.MAIZE.name] == {
        "commodity_hectares": 30,  # 3 crop pixels x 10 ha
        "emissions_mt": 70,  # (1+2+4) x 10; non-crop 99s excluded
        "forest_emissions_mt": 100,  # (2+3+5) x 10; non-crop 99s excluded
        "grassland_emissions_mt": 30,  # (1+1+1) x 10; non-crop 99s excluded
        "peatland_commodity_hectares": 20,  # only (0,0) and (1,1) are crop AND peatland
        "peatland_conversion_emissions_mt": 60,  # (1+2+3) x 10; non-crop 99s excluded
        "peatland_occupation_emissions_mt": 150,  # 3 crop pixels x (5 x 10)
    }


def test_get_commodity_name_to_totals_with_no_crop_pixels() -> None:
    result = get_commodity_name_to_totals(
        component_to_per_hectare=dict.fromkeys(
            emit.EmissionComponent, get_darray_for_data([[1, 2], [3, 4]])
        ),
        crop_class=get_darray_for_data([[0, 0], [0, 0]]),
        crops=(Crop.SOYBEAN, Crop.WHEAT),
        dropped_per_hectare=get_darray_for_data([[0, 0, 0], [0, 0, 0]]),
        emissions_per_hectare=get_darray_for_data([[1, 2], [3, 4]]),
        hectares_per_pixel=get_darray_for_data([[10, 10], [10, 10]]),
        is_peatland=get_darray_for_data([[True, True], [True, True]]),
        peatland_occupation_per_hectare=get_darray_for_data([[5, 5], [5, 5]]),
    )
    assert result[emit.NonCommodity.DROPPED.name] == {"emissions_mt": 0}
    zeros = {
        "commodity_hectares": 0,
        "emissions_mt": 0,
        "forest_emissions_mt": 0,
        "grassland_emissions_mt": 0,
        "peatland_commodity_hectares": 0,
        "peatland_conversion_emissions_mt": 0,
        "peatland_occupation_emissions_mt": 0,
    }
    assert {name: result[name] for name in (Crop.SOYBEAN.name, Crop.WHEAT.name)} == {
        Crop.SOYBEAN.name: zeros,
        Crop.WHEAT.name: zeros,
    }


def test_every_crop_has_a_nass_series() -> None:
    """`jdluc.datasets` cannot import the pipeline, so `CropSeries` repeats this crop list and
    nothing else ties the two together. A name in one and not the other is silent rather than loud:
    `trace` joins the yield in on the left, so the crop still reaches the artifact, carrying
    hectares and emissions against a NaN emissions factor.
    """
    assert {crop.name for crop in Crop} == {
        crop_series.name for crop_series in usda_nass_quickstats.CropSeries
    }


def test_every_crop_is_also_a_statistical_crop() -> None:
    """The sLUC-against-jdLUC head-to-head is the one comparison with no external arbiter, and it
    is per (country, crop). A jdLUC crop the statistical leg does not model has nothing to be
    compared against, and drops out of that comparison without failing it.
    """
    assert {crop.name for crop in Crop} <= {crop.name for crop in statistical.Crop}
