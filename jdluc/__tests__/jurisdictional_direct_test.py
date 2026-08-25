import collections.abc

import numpy
import pytest
import xarray

from jdluc import statistical
from jdluc.datasets import usda_nass_quickstats
from jdluc.jurisdictional_direct import Crop, get_crop_name_to_totals


def get_darray_for_data(
    data: collections.abc.Sequence[collections.abc.Sequence[float]],
) -> xarray.DataArray:
    arr = numpy.array(data)
    y, x = arr.shape
    return xarray.DataArray(
        coords={"y": range(y), "x": range(x)}, data=arr, dims=("y", "x")
    )


def test_get_crop_name_to_totals() -> None:
    result = get_crop_name_to_totals(
        crops=(Crop.MAIZE,),
        # crop pixels: (0,0), (0,1), (1,1); of those, peatland at (0,0) and (1,1).
        crop_class=get_darray_for_data([[1, 1, 0], [0, 1, 0]]),
        # 99s sit on non-crop pixels — they must be excluded by the mask.
        emissions_per_hectare=get_darray_for_data([[1, 2, 99], [99, 4, 99]]),
        forest_emissions_per_hectare=get_darray_for_data([[2, 3, 99], [99, 5, 99]]),
        glad_class=get_darray_for_data([[2] * 2] * 2),
        hectares_per_pixel=get_darray_for_data([[10, 10, 10], [10, 10, 10]]),
        is_peatland=get_darray_for_data([[1, 0, 0], [0, 1, 0]]),
        peatland_conversion_per_hectare=get_darray_for_data([[1, 2, 99], [99, 3, 99]]),
        peatland_occupation_per_hectare=get_darray_for_data([[5, 5, 5], [5, 5, 5]]),
        skip_glad_crop_filter=False,
    )
    assert set(result) == {
        Crop.MAIZE.name,
    }
    assert result[Crop.MAIZE.name] == {
        "crop_hectares": 30,  # 3 crop pixels x 10 ha
        "emissions_mt": 70,  # (1+2+4) x 10; non-crop 99s excluded
        "forest_emissions_mt": 100,  # (2+3+5) x 10; non-crop 99s excluded
        "peatland_crop_hectares": 20,  # only (0,0) and (1,1) are crop AND peatland
        "peatland_conversion_emissions_mt": 60,  # (1+2+3) x 10; non-crop 99s excluded
        "peatland_occupation_emissions_mt": 150,  # 3 crop pixels x (5 x 10)
    }


def test_get_crop_name_to_totals_masked_out_by_glad() -> None:
    result = get_crop_name_to_totals(
        crop_class=get_darray_for_data([[1, 1], [1, 1]]),
        crops=(Crop.MAIZE,),
        emissions_per_hectare=get_darray_for_data([[1, 2], [3, 4]]),
        forest_emissions_per_hectare=get_darray_for_data([[1, 2], [3, 4]]),
        glad_class=get_darray_for_data([[0] * 2] * 2),
        hectares_per_pixel=get_darray_for_data([[10, 10], [10, 10]]),
        is_peatland=get_darray_for_data([[1, 1], [1, 1]]),
        peatland_conversion_per_hectare=get_darray_for_data([[1, 2], [3, 4]]),
        peatland_occupation_per_hectare=get_darray_for_data([[5, 5], [5, 5]]),
        skip_glad_crop_filter=False,
    )
    assert set(result) == {
        Crop.MAIZE.name,
    }
    assert result[Crop.MAIZE.name] == {
        "crop_hectares": 0,
        "emissions_mt": 0,
        "forest_emissions_mt": 0,
        "peatland_crop_hectares": 0,
        "peatland_conversion_emissions_mt": 0,
        "peatland_occupation_emissions_mt": 0,
    }


@pytest.mark.parametrize("glad_value", (0, 1, 2))
@pytest.mark.parametrize("skip_glad_crop_filter", (False, True))
def test_get_crop_name_to_totals_with_no_crop_pixels(
    glad_value: int, skip_glad_crop_filter: bool
) -> None:
    result = get_crop_name_to_totals(
        crop_class=get_darray_for_data([[0, 0], [0, 0]]),
        crops=(Crop.SOYBEAN, Crop.WHEAT),
        emissions_per_hectare=get_darray_for_data([[1, 2], [3, 4]]),
        forest_emissions_per_hectare=get_darray_for_data([[1, 2], [3, 4]]),
        glad_class=get_darray_for_data([[glad_value] * 2] * 2),
        hectares_per_pixel=get_darray_for_data([[10, 10], [10, 10]]),
        is_peatland=get_darray_for_data([[1, 1], [1, 1]]),
        peatland_conversion_per_hectare=get_darray_for_data([[1, 2], [3, 4]]),
        peatland_occupation_per_hectare=get_darray_for_data([[5, 5], [5, 5]]),
        skip_glad_crop_filter=skip_glad_crop_filter,
    )
    assert set(result) == {Crop.SOYBEAN.name, Crop.WHEAT.name}
    assert result[Crop.SOYBEAN.name] == {
        "crop_hectares": 0,
        "emissions_mt": 0,
        "forest_emissions_mt": 0,
        "peatland_crop_hectares": 0,
        "peatland_conversion_emissions_mt": 0,
        "peatland_occupation_emissions_mt": 0,
    }
    assert result[Crop.WHEAT.name] == {
        "crop_hectares": 0,
        "emissions_mt": 0,
        "forest_emissions_mt": 0,
        "peatland_crop_hectares": 0,
        "peatland_conversion_emissions_mt": 0,
        "peatland_occupation_emissions_mt": 0,
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
