import math

import numpy
import pytest
import xarray

from jdluc.datasets.gnw_global_peatlands import DATASET as PEATLANDS
from jdluc.datasets.gpw_grassland import DATASET as GRASSLAND
from jdluc.datasets.gpw_grassland import YEARS as GRASSLAND_YEARS
from jdluc.datasets.gpw_grassland import Grassland
from jdluc.datasets.tang_cropgrids import YEAR as CROPGRIDS_YEAR
from jdluc.emit import (
    CROPLAND,
    PASTURELAND,
    SPAN_TO_LINEAR_DISCOUNT_WEIGHT,
    Conversion,
    EmissionComponent,
)
from jdluc.jurisdictional_direct_proxy import get_name_to_band

VEGETATION = 10.0
SOIL = 100.0
TOTAL_WEIGHT = sum(SPAN_TO_LINEAR_DISCOUNT_WEIGHT.values())
LAND_CLASS_COMPONENT_NAMES = [
    f"{land_class:s}-{component!s}-emissions"
    for land_class in (CROPLAND, PASTURELAND)
    for component in EmissionComponent
]


def get_dset_for_pixel(
    conversion: Conversion, grassland: float, is_peat: bool
) -> xarray.Dataset:
    """One pixel, charged the same emissions in every span."""
    values = {
        "conversion": float(conversion),
        f"{CROPLAND:s}-peatland-occupation:tco2e-per-ha": 0.0,
        "dropped-emissions:tco2e-per-ha": 0.0,
        GRASSLAND.fully_qualified_band_names[
            GRASSLAND_YEARS.index(CROPGRIDS_YEAR)
        ]: grassland,
        f"{PASTURELAND:s}-peatland-occupation:tco2e-per-ha": 0.0,
        PEATLANDS.fully_qualified_band_name: float(is_peat),
    } | {
        name: value
        for before, after in SPAN_TO_LINEAR_DISCOUNT_WEIGHT
        for name, value in (
            (f"soil-emissions:tco2e-per-ha:{before:d}-{after:d}", SOIL),
            (f"vegetation-emissions:tco2e-per-ha:{before:d}-{after:d}", VEGETATION),
        )
    }
    return xarray.Dataset(
        {
            name: xarray.DataArray(
                coords={"y": [0], "x": [0]},
                data=numpy.array([[value]]),
                dims=("y", "x"),
            )
            for name, value in values.items()
        }
    )


@pytest.mark.parametrize(
    ("conversion", "is_peat", "name_to_expected"),
    (
        pytest.param(
            Conversion.FOREST_TO_CROPLAND,
            False,
            {f"{CROPLAND:s}-forest-emissions": VEGETATION + SOIL},
            id="forest-to-cropland-charges-cropland",
        ),
        pytest.param(
            Conversion.FOREST_TO_PASTURE,
            False,
            {f"{PASTURELAND:s}-forest-emissions": VEGETATION + SOIL},
            id="forest-to-pasture-charges-pastureland",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_CROPLAND,
            True,
            {
                f"{CROPLAND:s}-grassland-emissions": VEGETATION,
                f"{CROPLAND:s}-peatland-conversion-emissions": SOIL,
            },
            id="peat-soil-stays-with-the-cropland-it-became",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_PASTURE,
            True,
            {
                f"{PASTURELAND:s}-grassland-emissions": VEGETATION,
                f"{PASTURELAND:s}-peatland-conversion-emissions": SOIL,
            },
            id="peat-soil-stays-with-the-pasture-it-became",
        ),
    ),
)
def test_get_name_to_band_splits_each_component_by_land_class(
    conversion: Conversion, is_peat: bool, name_to_expected: dict[str, float]
) -> None:
    name_to_band = get_name_to_band(
        dset=get_dset_for_pixel(
            conversion=conversion, grassland=float(Grassland.OTHER), is_peat=is_peat
        )
    )
    # Every other land class and component is charged nothing
    expected = dict.fromkeys(LAND_CLASS_COMPONENT_NAMES, 0.0) | name_to_expected
    for name, value in expected.items():
        assert math.isclose(
            name_to_band[f"{name:s}:tco2e-per-ha"].item(), value * TOTAL_WEIGHT
        ), name


@pytest.mark.parametrize(
    ("conversion", "is_peat"),
    [
        pytest.param(
            conversion,
            is_peat,
            id=f"{conversion.name.lower().replace('_', '-'):s}-{'on' if is_peat else 'off'}-peat",
        )
        for conversion in Conversion
        for is_peat in (False, True)
    ],
)
def test_get_name_to_band_charges_every_conversion_exactly_once(
    conversion: Conversion, is_peat: bool
) -> None:
    name_to_band = get_name_to_band(
        dset=get_dset_for_pixel(
            conversion=conversion, grassland=float(Grassland.OTHER), is_peat=is_peat
        )
    )
    charged = sum(
        name_to_band[f"{name:s}:tco2e-per-ha"].item()
        for name in LAND_CLASS_COMPONENT_NAMES
    )
    assert math.isclose(
        charged,
        0.0 if conversion == Conversion.NONE else (VEGETATION + SOIL) * TOTAL_WEIGHT,
    )


@pytest.mark.parametrize(
    ("grassland", "expected"),
    (
        pytest.param(
            float(Grassland.CULTIVATED), 1.0, id="cultivated-grassland-is-pasture"
        ),
        pytest.param(float(Grassland.NATURAL), 0.0, id="natural-grassland-is-not"),
        pytest.param(numpy.nan, 0.0, id="no-observation-reads-as-not-pasture"),
    ),
)
def test_get_name_to_band_pasture_fraction(grassland: float, expected: float) -> None:
    name_to_band = get_name_to_band(
        dset=get_dset_for_pixel(
            conversion=Conversion.NONE, grassland=grassland, is_peat=False
        )
    )
    assert name_to_band[f"pasture:fraction:{CROPGRIDS_YEAR:d}"].item() == expected
