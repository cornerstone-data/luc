import numpy
import pytest
import xarray

from jdluc.datasets.gnw_global_peatlands import DATASET as PEATLANDS
from jdluc.datasets.gpw_grassland import DATASET as GRASSLAND
from jdluc.datasets.gpw_grassland import YEARS as GRASSLAND_YEARS
from jdluc.datasets.gpw_grassland import Grassland
from jdluc.datasets.gpw_livestock import Species, get_band_name
from jdluc.datasets.tang_cropgrids import DATASET as CROPGRIDS
from jdluc.datasets.tang_cropgrids import YEAR as CROPGRIDS_YEAR
from jdluc.datasets.tang_cropgrids import Crop
from jdluc.emit import (
    CROPLAND,
    DEFAULT_ASSESSMENT_YEAR,
    PASTURELAND,
    PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA,
    Conversion,
    EmissionComponent,
    NonCommodity,
)
from jdluc.jurisdictional_direct_proxy import (
    HERD_HEADS_COLUMN,
    get_commodity_name_to_totals,
    get_crop_to_area_share,
    get_name_to_band,
)
from jdluc.statistical import Livestock

# A conversion in 2013 takes a factor of 0.0625, so 160 and 1600 discount to 10 and 100
CONVERSION_YEAR = 2013
VEGETATION = 160.0
SOIL = 1600.0
LAND_CLASS_COMPONENT_NAMES = [
    f"{land_class:s}-{component!s}-emissions:tco2e-per-ha"
    for land_class in (CROPLAND, PASTURELAND)
    for component in EmissionComponent
]


def get_darray_for_cells(data: list[list[float]]) -> xarray.DataArray:
    arr = numpy.array(data, dtype=numpy.float64)
    y, x = arr.shape
    return xarray.DataArray(
        coords={"y": range(y), "x": range(x)}, data=arr, dims=("y", "x")
    )


def get_dset_for_pixel(
    conversion: Conversion, grassland: float, is_peat: bool
) -> xarray.Dataset:
    """One pixel, converted in `CONVERSION_YEAR`."""
    values = {
        "conversion": float(conversion),
        "conversion-year": float(CONVERSION_YEAR),
        f"{CROPLAND:s}-peatland-occupation:tco2e-per-ha": 0.0,
        "dropped-emissions:tco2e-per-ha": 0.0,
        GRASSLAND.fully_qualified_band_names[
            GRASSLAND_YEARS.index(CROPGRIDS_YEAR)
        ]: grassland,
        f"{PASTURELAND:s}-peatland-occupation:tco2e-per-ha": 0.0,
        PEATLANDS.fully_qualified_band_name: float(is_peat),
        "soil-emissions:tco2e-per-ha": SOIL,
        "vegetation-emissions:tco2e-per-ha": VEGETATION,
    }
    return xarray.Dataset(
        {name: get_darray_for_cells(data=[[value]]) for name, value in values.items()}
    )


def get_dset_for_crop_areas(
    crop_to_hectares: dict[Crop, list[list[float]]],
) -> xarray.Dataset:
    """Every CROPGRIDS crop band, zero unless given."""
    zeros = numpy.zeros_like(next(iter(crop_to_hectares.values()))).tolist()
    crop_to_data = dict.fromkeys(Crop, zeros) | crop_to_hectares
    return xarray.Dataset(
        {
            band_name: get_darray_for_cells(data=crop_to_data[crop])
            for crop, band_name in zip(
                Crop, CROPGRIDS.fully_qualified_band_names, strict=True
            )
        }
    )


@pytest.mark.parametrize(
    ("conversion", "is_peat", "name_to_expected"),
    (
        pytest.param(
            Conversion.FOREST_TO_CROPLAND,
            False,
            {f"{CROPLAND:s}-forest-emissions:tco2e-per-ha": 110.0},
            id="forest-to-cropland-charges-cropland",
        ),
        pytest.param(
            Conversion.FOREST_TO_PASTURE,
            False,
            {f"{PASTURELAND:s}-forest-emissions:tco2e-per-ha": 110.0},
            id="forest-to-pasture-charges-pastureland",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_CROPLAND,
            True,
            {
                f"{CROPLAND:s}-grassland-emissions:tco2e-per-ha": 10.0,
                f"{CROPLAND:s}-peatland-conversion-emissions:tco2e-per-ha": 100.0,
            },
            id="peat-soil-stays-with-the-cropland-it-became",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_PASTURE,
            True,
            {
                f"{PASTURELAND:s}-grassland-emissions:tco2e-per-ha": 10.0,
                f"{PASTURELAND:s}-peatland-conversion-emissions:tco2e-per-ha": 100.0,
            },
            id="peat-soil-stays-with-the-pasture-it-became",
        ),
    ),
)
def test_get_name_to_band_splits_each_component_by_land_class(
    conversion: Conversion, is_peat: bool, name_to_expected: dict[str, float]
) -> None:
    name_to_band = get_name_to_band(
        assessment_year=DEFAULT_ASSESSMENT_YEAR,
        dset=get_dset_for_pixel(
            conversion=conversion, grassland=float(Grassland.OTHER), is_peat=is_peat
        ),
    )
    # Every other land class and component is charged nothing
    assert {
        name: name_to_band[name].item() for name in LAND_CLASS_COMPONENT_NAMES
    } == pytest.approx(
        dict.fromkeys(LAND_CLASS_COMPONENT_NAMES, 0.0) | name_to_expected
    )


@pytest.mark.parametrize("conversion", Conversion)
@pytest.mark.parametrize("is_peat", (False, True))
def test_get_name_to_band_charges_every_conversion_exactly_once(
    conversion: Conversion, is_peat: bool
) -> None:
    name_to_band = get_name_to_band(
        assessment_year=DEFAULT_ASSESSMENT_YEAR,
        dset=get_dset_for_pixel(
            conversion=conversion, grassland=float(Grassland.OTHER), is_peat=is_peat
        ),
    )
    charged = sum(name_to_band[name].item() for name in LAND_CLASS_COMPONENT_NAMES)
    assert charged == pytest.approx(0.0 if conversion == Conversion.NONE else 110.0)


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
        assessment_year=DEFAULT_ASSESSMENT_YEAR,
        dset=get_dset_for_pixel(
            conversion=Conversion.NONE, grassland=grassland, is_peat=False
        ),
    )
    assert name_to_band[f"pasture:fraction:{CROPGRIDS_YEAR:d}"].item() == expected


def test_get_crop_to_area_share_divides_by_every_crop_cropgrids_maps() -> None:
    result = get_crop_to_area_share(
        crops=(Crop.COCOA,),
        dset=get_dset_for_crop_areas(
            crop_to_hectares={
                # Cocoa beside coffee, a cell with no crop at all, and the ocean
                Crop.COCOA: [[1.0, 0.0, numpy.nan]],
                Crop.COFFEE: [[3.0, 0.0, numpy.nan]],
            }
        ),
    )
    assert set(result) == {Crop.COCOA}
    assert result[Crop.COCOA].values.tolist() == [[0.25, 0.0, 0.0]]


def test_get_commodity_name_to_totals() -> None:
    """Two cells of 2 and 3 hectares, one row as a province one cell tall clips to.

    Cocoa grows on the first cell and no crop on the second; half of each is pasture, and each
    holds 10 cattle a hectare. Every band charges both cells alike, and each land class and
    component charges its own amount, so a total reading the wrong band is visible. Cropland's peat
    occupation drains the whole cell and pasture's half of it.
    """
    per_hectare = {
        f"{CROPLAND:s}-forest-emissions:tco2e-per-ha": 100.0,
        f"{CROPLAND:s}-grassland-emissions:tco2e-per-ha": 20.0,
        f"{CROPLAND:s}-peatland-conversion-emissions:tco2e-per-ha": 3.0,
        f"{CROPLAND:s}-peatland-occupation:tco2e-per-ha": PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA,
        "dropped-emissions:tco2e-per-ha": 9.0,
        get_band_name(species=Species.CATTLE, year=CROPGRIDS_YEAR): 10.0,
        f"pasture:fraction:{CROPGRIDS_YEAR:d}": 0.5,
        f"{PASTURELAND:s}-forest-emissions:tco2e-per-ha": 4000.0,
        f"{PASTURELAND:s}-grassland-emissions:tco2e-per-ha": 500.0,
        f"{PASTURELAND:s}-peatland-conversion-emissions:tco2e-per-ha": 60.0,
        f"{PASTURELAND:s}-peatland-occupation:tco2e-per-ha": PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA
        / 2,
    }
    dset = get_dset_for_crop_areas(crop_to_hectares={Crop.COCOA: [[50.0, 0.0]]}).assign(
        {
            name: get_darray_for_cells(data=[[value, value]])
            for name, value in per_hectare.items()
        }
        | {"hectares-per-cell:ha": get_darray_for_cells(data=[[2.0, 3.0]])}
    )
    result = get_commodity_name_to_totals(
        commodities=(Crop.COCOA, Livestock.BEEF_CATTLE),
        commodity_to_share={
            Crop.COCOA: get_darray_for_cells(data=[[1.0, 0.0]]),
            # PASTURE's quarter is not asked for
            Livestock.BEEF_CATTLE: get_darray_for_cells(data=[[0.75, 0.75]]),
        },
        dset=dset,
    )
    assert result == {
        # Cropland's charge on the first cell's 2 ha, and CROPGRIDS' area rather than a share of it
        Crop.COCOA.name: pytest.approx(
            {
                "commodity_hectares": 50.0,
                "emissions_mt": 320.6,
                HERD_HEADS_COLUMN: numpy.nan,
                "forest_emissions_mt": 200.0,
                "grassland_emissions_mt": 40.0,
                "peatland_commodity_hectares": 2.0,
                "peatland_conversion_emissions_mt": 6.0,
                "peatland_occupation_emissions_mt": 74.6,
            },
            nan_ok=True,
        ),
        # Pasture's charge on three quarters of both cells' 5 ha, and of their 2.5 ha of pasture
        Livestock.BEEF_CATTLE.name: pytest.approx(
            {
                "commodity_hectares": 1.875,
                "emissions_mt": 17169.9375,
                # Both cells' whole herd, since both have pasture
                HERD_HEADS_COLUMN: 50.0,
                "forest_emissions_mt": 15000.0,
                "grassland_emissions_mt": 1875.0,
                "peatland_commodity_hectares": 1.875,
                "peatland_conversion_emissions_mt": 225.0,
                "peatland_occupation_emissions_mt": 69.9375,
            }
        ),
        NonCommodity.DROPPED.name: pytest.approx(
            {"emissions_mt": 45.0, HERD_HEADS_COLUMN: numpy.nan}, nan_ok=True
        ),
        # Cropland's charge on the second cell's 3 ha, plus the quarter of pasture's no row takes
        NonCommodity.UNATTRIBUTED.name: pytest.approx(
            {
                "emissions_mt": 6204.2125,
                "forest_emissions_mt": 5300.0,
                HERD_HEADS_COLUMN: numpy.nan,
                "grassland_emissions_mt": 685.0,
                "peatland_conversion_emissions_mt": 84.0,
                "peatland_occupation_emissions_mt": 135.2125,
            },
            nan_ok=True,
        ),
    }
