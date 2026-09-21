import collections
import collections.abc

import numpy
import pytest
import xarray

from jdluc.datasets.descals_oil_palm import DATASET as OIL_PALM
from jdluc.datasets.gnw_global_peatlands import DATASET as PEATLANDS
from jdluc.datasets.gnw_tcl import DATASET as TREE_COVER_LOSS
from jdluc.datasets.gnw_tcl import LOSS_YEAR_OFFSET
from jdluc.datasets.gpw_grassland import DATASET as GRASSLAND
from jdluc.datasets.gpw_grassland import YEARS as GRASSLAND_YEARS
from jdluc.datasets.gpw_grassland import Grassland
from jdluc.datasets.ipcc_climate_zones import Zone
from jdluc.datasets.liao_gaced30 import DATASET as CROPLAND
from jdluc.datasets.liao_gaced30 import YEARS as CROPLAND_YEARS
from jdluc.datasets.liao_gaced30 import Cropland
from jdluc.emit import (
    ASSESSMENT_YEAR,
    CARBON_PER_BIOMASS_LIVE_WOOD,
    CLIMATE_ZONE_TO_SOC_RETENTION_FRACTION,
    CO2E_PER_CARBON,
    FROM_FOREST,
    FROM_GRASSLAND,
    LOOKBACK_YEARS_RANGE,
    PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA,
    PEATLAND_EMISSIONS_PULSE_TCO2E_PER_HA,
    SPAN_TO_LINEAR_DISCOUNT_WEIGHT,
    Conversion,
    ConversionEmissions,
    ConversionRecord,
    DestinationDataset,
    EmissionComponent,
    SpanType,
    get_belowground_carbon,
    get_conversion_emissions,
    get_conversion_record,
    get_dead_organic_matter_carbon,
    get_dropped_emissions,
    get_dset_for_output,
    get_grassland_carbon,
    get_hectares_per_pixel,
    get_last_departure_year,
    get_linear_discounted_total,
    get_mineral_soil_emissions,
    get_peatland_occupation_emissions,
    get_span_to_charge,
    get_span_to_component_to_emissions,
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


FOREST_TCARBON_PER_HA = 100.0
GRASSLAND_TCARBON_PER_HA = 10.0
SOIL_ORGANIC_CARBON = 1000.0
MINERAL_SOIL_TCO2E_PER_HA = (
    (1 - CLIMATE_ZONE_TO_SOC_RETENTION_FRACTION[Zone.TROPICAL_DRY])
    * SOIL_ORGANIC_CARBON
    * CO2E_PER_CARBON
)


def get_record_for(conversion: Conversion, width: int = 1) -> ConversionRecord:
    """A record holding one conversion, with its halves read off the axis tuples.

    Those tuples are the oracle here: `get_conversion_record` derives the same masks from the
    layers, and a disagreement between the two shows up as a pool charged to the wrong source.
    """

    to_cropland = (
        Conversion.FOREST_TO_CROPLAND,
        Conversion.PASTURE_TO_CROPLAND,
        Conversion.RANGELAND_TO_CROPLAND,
    )
    to_pasture = (Conversion.FOREST_TO_PASTURE, Conversion.RANGELAND_TO_PASTURE)

    def mask(members: tuple[Conversion, ...]) -> xarray.DataArray:
        return get_darray_for_data(data=[[conversion in members] * width])

    return ConversionRecord(
        conversion=get_darray_for_data(data=[[conversion] * width]),
        destination_dataset=get_darray_for_data(
            data=[
                [
                    DestinationDataset.LIAO_GACED30
                    if conversion in to_cropland
                    else DestinationDataset.GPW_GRASSLAND
                    if conversion in to_pasture
                    else 0
                ]
                * width
            ]
        ),
        from_forest=mask(FROM_FOREST),
        from_grassland=mask(FROM_GRASSLAND),
        to_cropland=mask(to_cropland),
        to_pasture=mask(to_pasture),
        year=get_darray_for_data(data=[[ASSESSMENT_YEAR] * width]),
    )


def get_pools_for(
    conversion: Conversion, is_peatland: bool = False
) -> ConversionEmissions:
    return get_conversion_emissions(
        climate_zones=get_darray_for_data(data=[[Zone.TROPICAL_DRY.value]]),
        conversion_record=get_record_for(conversion=conversion),
        forest_carbon=get_darray_for_data(data=[[FOREST_TCARBON_PER_HA]]),
        grassland_carbon=get_darray_for_data(data=[[GRASSLAND_TCARBON_PER_HA]]),
        is_peatland=get_darray_for_data(data=[[is_peatland]]),
        soil_organic_carbon=get_darray_for_data(data=[[SOIL_ORGANIC_CARBON]]),
    )


def test_a_half_resolved_pixel_is_charged_nothing() -> None:
    """A source with no destination, or a destination with no source, releases no pool.

    `get_record_for` cannot build this state, because it derives both halves from the conversion.
    The carbon is real, and `get_dropped_emissions` reports it, but no conversion fired, so
    nothing may charge it to a commodity.
    """

    def get_half_resolved_record(
        from_forest: bool, to_cropland: bool
    ) -> ConversionRecord:
        def mask(value: bool) -> xarray.DataArray:
            return get_darray_for_data(data=[[value]])

        return ConversionRecord(
            conversion=get_darray_for_data(data=[[Conversion.NONE]]),
            destination_dataset=get_darray_for_data(data=[[0]]),
            from_forest=mask(from_forest),
            from_grassland=mask(False),
            to_cropland=mask(to_cropland),
            to_pasture=mask(False),
            year=get_darray_for_data(data=[[ASSESSMENT_YEAR]]),
        )

    for from_forest, to_cropland in ((True, False), (False, True)):
        result = get_conversion_emissions(
            climate_zones=get_darray_for_data(data=[[Zone.TROPICAL_DRY.value]]),
            conversion_record=get_half_resolved_record(
                from_forest=from_forest, to_cropland=to_cropland
            ),
            forest_carbon=get_darray_for_data(data=[[FOREST_TCARBON_PER_HA]]),
            grassland_carbon=get_darray_for_data(data=[[GRASSLAND_TCARBON_PER_HA]]),
            is_peatland=get_darray_for_data(data=[[False]]),
            soil_organic_carbon=get_darray_for_data(data=[[SOIL_ORGANIC_CARBON]]),
        )
        assert float(result.vegetation.sum()) == 0.0
        assert float(result.soil.sum()) == 0.0


@pytest.mark.parametrize(
    ("conversion", "source_tcarbon_per_ha"),
    (
        pytest.param(Conversion.NONE, 0.0, id="outside the matrix nothing fires"),
        pytest.param(
            Conversion.FOREST_TO_CROPLAND,
            FOREST_TCARBON_PER_HA,
            id="forest to cropland releases the measured stand",
        ),
        pytest.param(
            Conversion.FOREST_TO_PASTURE,
            FOREST_TCARBON_PER_HA,
            id="and so does forest to pasture, the destination holding no stock to subtract",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_CROPLAND,
            GRASSLAND_TCARBON_PER_HA,
            id="rangeland to cropland charges the climate-zone table, never the forest stand",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_PASTURE,
            GRASSLAND_TCARBON_PER_HA,
            id="rangeland to pasture: one grassland stock covers both sides for now, and the "
            "source is still released whole rather than differenced against it",
        ),
        pytest.param(
            Conversion.PASTURE_TO_CROPLAND,
            GRASSLAND_TCARBON_PER_HA,
            id="pasture to cropland, which is 36-38% of the Cerrado's crop numerator",
        ),
    ),
)
def test_the_biomass_each_conversion_releases(
    conversion: Conversion, source_tcarbon_per_ha: float
) -> None:
    result = get_pools_for(conversion=conversion).vegetation
    assert result.name == "tco2e-per-ha"
    numpy.testing.assert_allclose(
        result.data, [[source_tcarbon_per_ha * CO2E_PER_CARBON]]
    )


@pytest.mark.parametrize(
    ("conversion", "is_peatland", "expected"),
    (
        pytest.param(
            Conversion.NONE, False, 0.0, id="outside the matrix nothing fires"
        ),
        pytest.param(
            Conversion.NONE,
            True,
            0.0,
            id="and peat with no conversion over it is peat nobody drained",
        ),
        pytest.param(
            Conversion.FOREST_TO_CROPLAND,
            False,
            MINERAL_SOIL_TCO2E_PER_HA,
            id="a cropland destination keeps only the fraction Table 5.5 gives it",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_CROPLAND,
            False,
            MINERAL_SOIL_TCO2E_PER_HA,
            id="whatever it was converted from",
        ),
        pytest.param(
            Conversion.PASTURE_TO_CROPLAND,
            False,
            MINERAL_SOIL_TCO2E_PER_HA,
            id="likewise",
        ),
        pytest.param(
            Conversion.FOREST_TO_PASTURE,
            False,
            0.0,
            id="a pasture destination leaves the soil at the grassland reference state, so the "
            "land-use and management factors are both 1.0 and the term is accounted and zero",
        ),
        pytest.param(Conversion.RANGELAND_TO_PASTURE, False, 0.0, id="likewise"),
        pytest.param(
            Conversion.FOREST_TO_PASTURE,
            True,
            PEATLAND_EMISSIONS_PULSE_TCO2E_PER_HA,
            id="peat releases its pulse even where the mineral term is zero",
        ),
        pytest.param(
            Conversion.PASTURE_TO_CROPLAND,
            True,
            PEATLAND_EMISSIONS_PULSE_TCO2E_PER_HA,
            id="and replaces the mineral term rather than adding to it",
        ),
    ),
)
def test_the_soil_each_conversion_disturbs(
    conversion: Conversion, is_peatland: bool, expected: float
) -> None:
    result = get_pools_for(conversion=conversion, is_peatland=is_peatland).soil
    assert result.name == "tco2e-per-ha"
    numpy.testing.assert_allclose(result.data, [[expected]])


def test_get_conversion_emissions_tolerates_missing_soil_carbon() -> None:
    # SoilGrids has genuine gaps over water, rock and ice; a missing stock must not poison the
    # pixel, which would silently discard its vegetation emissions too
    result = get_conversion_emissions(
        climate_zones=get_darray_for_data(data=[[Zone.TROPICAL_DRY.value] * 2]),
        conversion_record=get_record_for(
            conversion=Conversion.FOREST_TO_CROPLAND, width=2
        ),
        forest_carbon=get_darray_for_data(data=[[FOREST_TCARBON_PER_HA] * 2]),
        grassland_carbon=get_darray_for_data(data=[[GRASSLAND_TCARBON_PER_HA] * 2]),
        is_peatland=get_darray_for_data(data=[[False, True]]),
        soil_organic_carbon=get_darray_for_data(data=[[numpy.nan, numpy.nan]]),
    )
    assert not numpy.isnan(result.soil.data).any()
    # the mineral pixel falls back to a zero stock; the peat pulse never asked SoilGrids
    numpy.testing.assert_allclose(
        result.soil.data, [[0, PEATLAND_EMISSIONS_PULSE_TCO2E_PER_HA]]
    )


@pytest.mark.parametrize(
    ("destination", "is_peatland", "expected"),
    (
        pytest.param(
            True, True, PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA, id="drained peat"
        ),
        pytest.param(
            True,
            False,
            0.0,
            id="a cropland or pasture destination off peat drains nothing",
        ),
        pytest.param(
            False,
            True,
            0.0,
            id="peat whose destination no layer resolves is charged no occupation",
        ),
    ),
)
def test_get_peatland_occupation_emissions(
    destination: bool, is_peatland: bool, expected: float
) -> None:
    # Occupation prices the drained end state, so it asks the destination and nothing else: a pixel
    # whose source never resolved still occupies drained peat, and a pixel that lost forest into
    # nothing does not.  The workflow calls it once per destination.
    result = get_peatland_occupation_emissions(
        destination=get_darray_for_data(data=[[destination]]),
        is_peatland=get_darray_for_data(data=[[is_peatland]]),
    )
    assert result.name == "tco2e-per-ha"
    numpy.testing.assert_allclose(result.data, [[expected]])


def test_get_dropped_emissions() -> None:
    # A pixel that lost a class no destination claimed is charged to nobody, and the size of what
    # it drops is the reported number -- 69-96% of tropical forest-loss carbon.
    result = get_dropped_emissions(
        forest_carbon=get_darray_for_data(data=[[FOREST_TCARBON_PER_HA] * 5]),
        from_forest=get_darray_for_data(data=[[True, True, False, False, False]]),
        from_grassland=get_darray_for_data(data=[[False, False, True, True, False]]),
        grassland_carbon=get_darray_for_data(data=[[GRASSLAND_TCARBON_PER_HA] * 5]),
        has_destination=get_darray_for_data(data=[[False, True, False, True, False]]),
    )
    assert result.name == "tco2e-per-ha"
    numpy.testing.assert_allclose(
        result.data,
        [
            [
                FOREST_TCARBON_PER_HA * CO2E_PER_CARBON,
                0,
                GRASSLAND_TCARBON_PER_HA * CO2E_PER_CARBON,
                0,
                # nothing left a class here, so there is nothing to drop
                0,
            ]
        ],
    )


@pytest.mark.parametrize(
    ("conversion_year", "spans"),
    (
        pytest.param(2001, {(2000, 2005)}, id="the first year a conversion can carry"),
        pytest.param(
            2005, {(2000, 2005)}, id="a shared boundary belongs to the span it closes"
        ),
        pytest.param(2006, {(2005, 2010)}, id="and not to the one it opens"),
        pytest.param(
            ASSESSMENT_YEAR, {(2015, 2020)}, id="the last year a conversion can carry"
        ),
        pytest.param(0, set(), id="no conversion is charged to no span"),
    ),
)
def test_get_span_to_charge(conversion_year: int, spans: set[SpanType]) -> None:
    # Spans share their boundary years, so a charge landing in two of them would be counted twice
    # and one landing in none would vanish without a trace.
    result = get_span_to_charge(
        conversion_year=get_darray_for_data(data=[[conversion_year]]),
        darray=get_darray_for_data(data=[[1]]),
    )
    assert set(result) == set(SPAN_TO_LINEAR_DISCOUNT_WEIGHT)
    assert {span for span, darray in result.items() if float(darray.sum())} == spans


def test_get_dset_for_output_appends_each_band_its_units() -> None:
    # Every emitted band is keyed on this name, so a change here strands the zarr caches keyed on
    # the old one and KeyErrors the two legs that read it. The units come off the DataArray and
    # slot in after the first colon, which is what puts the span last.
    result = get_dset_for_output(
        name_to_darray={
            "conversion": get_darray_for_data(data=[[1]]).rename(None),
            "emissions:2015-2020": get_darray_for_data(data=[[1]]).rename(
                "tco2e-per-ha"
            ),
            "hectares-per-pixel": get_darray_for_data(data=[[1]]).rename("ha"),
        }
    )
    assert set(result) == {
        "conversion",
        "emissions:tco2e-per-ha:2015-2020",
        "hectares-per-pixel:ha",
    }


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


VEGETATION = 10.0
SOIL = 100.0


def get_dset_for_pixel(conversion: Conversion, is_peat: float) -> xarray.Dataset:
    """One pixel, charged the same emissions in every span."""
    return xarray.Dataset(
        {
            "conversion": get_darray_for_data(data=[[conversion]]),
            PEATLANDS.fully_qualified_band_name: get_darray_for_data(data=[[is_peat]]),
        }
        | {
            name: get_darray_for_data(data=[[value]])
            for before, after in SPAN_TO_LINEAR_DISCOUNT_WEIGHT
            for name, value in (
                (f"vegetation-emissions:tco2e-per-ha:{before:d}-{after:d}", VEGETATION),
                (f"soil-emissions:tco2e-per-ha:{before:d}-{after:d}", SOIL),
            )
        }
    )


@pytest.mark.parametrize(
    ("conversion", "is_peat", "source_to_expected"),
    (
        pytest.param(
            Conversion.FOREST_TO_CROPLAND,
            0,
            {"forest": VEGETATION + SOIL},
            id="a forest source claims both pools",
        ),
        pytest.param(
            Conversion.FOREST_TO_CROPLAND,
            numpy.nan,
            {"forest": VEGETATION + SOIL},
            id="and still does where the peat mask has no data, which is not peat",
        ),
        pytest.param(
            Conversion.PASTURE_TO_CROPLAND,
            0,
            {"grassland": VEGETATION + SOIL},
            id="and so does a grassland one",
        ),
        # Peat claims all the soil it sits under whatever the source class, so the class keeps its
        # vegetation and loses its soil
        pytest.param(
            Conversion.FOREST_TO_PASTURE,
            1,
            {"forest": VEGETATION, "peatland_conversion": SOIL},
            id="forest on peat",
        ),
        pytest.param(
            Conversion.RANGELAND_TO_PASTURE,
            1,
            {"grassland": VEGETATION, "peatland_conversion": SOIL},
            id="grassland on peat",
        ),
    ),
)
def test_get_span_to_source_to_emissions(
    conversion: Conversion, is_peat: float, source_to_expected: dict[str, float]
) -> None:
    # `emissions_mt` is computed from the span totals rather than by summing the component columns,
    # so the columns decompose it only if the sources claim every unit exactly once. Each case
    # names the sources that fire and must account for the whole pixel, which makes a pool
    # dropped, double-counted, or gated on the wrong conversion fail here.
    assert sum(source_to_expected.values()) == VEGETATION + SOIL
    span_to_component_to_emissions = get_span_to_component_to_emissions(
        dset=get_dset_for_pixel(conversion=conversion, is_peat=is_peat)
    )
    assert set(span_to_component_to_emissions) == set(SPAN_TO_LINEAR_DISCOUNT_WEIGHT)
    for component_to_emissions in span_to_component_to_emissions.values():
        assert set(component_to_emissions) == set(EmissionComponent)
        assert {
            component: float(darray.sum())
            for component, darray in component_to_emissions.items()
            if float(darray.sum())
        } == source_to_expected


@pytest.mark.parametrize(
    ("is_rangeland_by_year", "year"),
    (
        pytest.param([True] * len(LOOKBACK_YEARS_RANGE), 0, id="never departs"),
        pytest.param([False] * len(LOOKBACK_YEARS_RANGE), 0, id="never is sourced"),
        pytest.param(
            [year <= 2000 for year in LOOKBACK_YEARS_RANGE],
            2001,
            id="departs in the first year",
        ),
        pytest.param(
            [year <= 2011 for year in LOOKBACK_YEARS_RANGE],
            2012,
            id="departs in the middle",
        ),
        pytest.param(
            [year <= 2005 or 2010 <= year <= 2013 for year in LOOKBACK_YEARS_RANGE],
            2014,
            id="takes the last of several",
        ),
        pytest.param(
            [year >= 2010 for year in LOOKBACK_YEARS_RANGE],
            0,
            id="an arrival is not a departure",
        ),
    ),
)
def test_get_last_departure_year(is_rangeland_by_year: list[bool], year: int) -> None:
    result = get_last_departure_year(
        is_source=xarray.DataArray(
            is_rangeland_by_year,
            coords={"year": list(LOOKBACK_YEARS_RANGE)},
            dims="year",
        )
    )
    assert int(result) == year


NO_GRASSLAND = [float(Grassland.OTHER)] * len(LOOKBACK_YEARS_RANGE)


def get_dset_for_one_pixel(
    grassland_by_year: list[float],
    is_cropland: bool,
    loss_year: float | None,
    planting_year: float,
) -> xarray.Dataset:
    return xarray.Dataset(
        {
            GRASSLAND.fully_qualified_band_names[
                GRASSLAND_YEARS.index(grassland_year)
            ]: get_darray_for_data(data=[[grassland_by_year[offset]]])
            for offset, grassland_year in enumerate(LOOKBACK_YEARS_RANGE)
        }
        | {
            CROPLAND.fully_qualified_band_names[
                CROPLAND_YEARS.index(ASSESSMENT_YEAR)
            ]: get_darray_for_data(
                data=[[Cropland.CROPLAND if is_cropland else Cropland.NOT_CROPLAND]]
            ),
            OIL_PALM.fully_qualified_band_name: get_darray_for_data(
                data=[[planting_year]]
            ),
            TREE_COVER_LOSS.fully_qualified_band_name: get_darray_for_data(
                data=[[0 if loss_year is None else loss_year - LOSS_YEAR_OFFSET]]
            ),
        }
    )


@pytest.mark.parametrize(
    ("grassland_by_year", "is_cropland", "loss_year", "conversion", "year"),
    (
        pytest.param(
            NO_GRASSLAND,
            True,
            2012,
            Conversion.FOREST_TO_CROPLAND,
            2012,
            id="forest to cropland",
        ),
        pytest.param(
            [*NO_GRASSLAND[:-1], Grassland.CULTIVATED],
            False,
            2012,
            Conversion.FOREST_TO_PASTURE,
            2012,
            id="forest to pasture",
        ),
        pytest.param(
            NO_GRASSLAND,
            False,
            2012,
            Conversion.NONE,
            2012,
            id="forest whose destination no layer resolves is dropped",
        ),
        pytest.param(
            [
                Grassland.NATURAL if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            True,
            None,
            Conversion.RANGELAND_TO_CROPLAND,
            2012,
            id="rangeland to cropland",
        ),
        pytest.param(
            [
                Grassland.OPEN_SHRUBLAND if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            True,
            None,
            Conversion.RANGELAND_TO_CROPLAND,
            2012,
            id="open shrubland counts as rangeland",
        ),
        pytest.param(
            [
                Grassland.NATURAL if year <= 2011 else Grassland.CULTIVATED
                for year in LOOKBACK_YEARS_RANGE
            ],
            False,
            None,
            Conversion.RANGELAND_TO_PASTURE,
            2012,
            id="rangeland to pasture",
        ),
        pytest.param(
            [
                Grassland.CULTIVATED if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            True,
            None,
            Conversion.PASTURE_TO_CROPLAND,
            2012,
            id="pasture to cropland",
        ),
        pytest.param(
            [
                Grassland.NATURAL if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            False,
            None,
            Conversion.NONE,
            2012,
            id="rangeland that departs with no destination is dropped",
        ),
        pytest.param(
            [
                Grassland.CULTIVATED if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            False,
            None,
            Conversion.NONE,
            2012,
            id="pasture that departs with no destination is dropped",
        ),
        pytest.param(
            [Grassland.NATURAL] * len(LOOKBACK_YEARS_RANGE),
            False,
            None,
            Conversion.NONE,
            0,
            id="rangeland that never departs is no conversion",
        ),
        pytest.param(
            [Grassland.CULTIVATED] * len(LOOKBACK_YEARS_RANGE),
            False,
            None,
            Conversion.NONE,
            0,
            id="pasture that never departs is no conversion",
        ),
        pytest.param(
            [
                Grassland.CULTIVATED
                if year <= 2011 or year >= 2018
                else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            False,
            None,
            Conversion.NONE,
            2012,
            id="pasture that departs and returns is no conversion",
        ),
        pytest.param(
            [
                Grassland.NATURAL if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            True,
            2003,
            Conversion.FOREST_TO_CROPLAND,
            2003,
            id="tree-cover loss outranks a rangeland departure",
        ),
        pytest.param(
            [
                Grassland.CULTIVATED if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            True,
            2004,
            Conversion.FOREST_TO_CROPLAND,
            2004,
            id="tree-cover loss outranks a pasture departure",
        ),
        pytest.param(
            [*NO_GRASSLAND[:-1], Grassland.CULTIVATED],
            True,
            2012,
            Conversion.FOREST_TO_CROPLAND,
            2012,
            id="cropland outranks pasture at the assessment year",
        ),
        pytest.param(
            [
                Grassland.NATURAL
                if year <= 2005
                else Grassland.CULTIVATED
                if year <= 2014
                else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            True,
            None,
            Conversion.PASTURE_TO_CROPLAND,
            2015,
            id="the later of two departures names the source",
        ),
        pytest.param(
            NO_GRASSLAND,
            True,
            ASSESSMENT_YEAR,
            Conversion.FOREST_TO_CROPLAND,
            ASSESSMENT_YEAR,
            id="loss in the assessment year is inside the lookback",
        ),
        pytest.param(
            NO_GRASSLAND,
            True,
            LOOKBACK_YEARS_RANGE[1],
            Conversion.FOREST_TO_CROPLAND,
            LOOKBACK_YEARS_RANGE[1],
            id="the earliest loss the layer can carry is inside the lookback",
        ),
        pytest.param(
            NO_GRASSLAND,
            True,
            2024,
            Conversion.NONE,
            0,
            id="loss after the assessment year is outside the lookback",
        ),
        pytest.param(
            NO_GRASSLAND,
            True,
            LOOKBACK_YEARS_RANGE[0],
            Conversion.NONE,
            0,
            id="no loss is not a loss in the opening year",
        ),
        pytest.param(
            [float("nan")] * len(LOOKBACK_YEARS_RANGE),
            False,
            None,
            Conversion.NONE,
            0,
            id="no observation resolves nothing",
        ),
        pytest.param(
            NO_GRASSLAND,
            False,
            float("nan"),
            Conversion.NONE,
            0,
            id="an unobserved loss resolves nothing",
        ),
    ),
)
def test_get_conversion_record(
    grassland_by_year: list[float],
    is_cropland: bool,
    loss_year: float | None,
    conversion: Conversion,
    year: int,
) -> None:
    result = get_conversion_record(
        dset=get_dset_for_one_pixel(
            grassland_by_year=grassland_by_year,
            is_cropland=is_cropland,
            loss_year=loss_year,
            planting_year=0,
        )
    )
    assert result.conversion.data == [[conversion]]
    assert result.year.data == [[year]]


@pytest.mark.parametrize(
    (
        "is_cropland",
        "is_cultivated",
        "planting_year",
        "destination_dataset",
        "to_cropland",
        "to_pasture",
    ),
    (
        pytest.param(
            True,
            False,
            0,
            DestinationDataset.LIAO_GACED30,
            True,
            False,
            id="GACED30 alone claims it",
        ),
        pytest.param(
            False,
            True,
            0,
            DestinationDataset.GPW_GRASSLAND,
            False,
            True,
            id="GPW alone claims it",
        ),
        pytest.param(
            True,
            True,
            0,
            DestinationDataset.LIAO_GACED30 | DestinationDataset.GPW_GRASSLAND,
            True,
            False,
            id="both claim it, and the earlier member resolves it",
        ),
        pytest.param(False, False, 0, 0, False, False, id="neither claims it"),
        pytest.param(
            False,
            False,
            2012,
            DestinationDataset.DESCALS_OIL_PALM,
            True,
            False,
            id="oil palm claims what no other layer sees",
        ),
        pytest.param(
            False,
            True,
            2012,
            DestinationDataset.DESCALS_OIL_PALM | DestinationDataset.GPW_GRASSLAND,
            True,
            False,
            id="oil palm outranks pasture on a pixel both claim",
        ),
        pytest.param(
            False,
            False,
            ASSESSMENT_YEAR,
            DestinationDataset.DESCALS_OIL_PALM,
            True,
            False,
            id="a plantation established in the assessment year still counts",
        ),
        pytest.param(
            False,
            False,
            ASSESSMENT_YEAR + 1,
            0,
            False,
            False,
            id="a plantation established after it does not",
        ),
    ),
)
def test_get_conversion_record_destination(
    is_cropland: bool,
    is_cultivated: bool,
    planting_year: float,
    destination_dataset: DestinationDataset | int,
    to_cropland: bool,
    to_pasture: bool,
) -> None:
    """A destination keeps every bit that fired, but resolves to the earliest one."""
    result = get_conversion_record(
        dset=get_dset_for_one_pixel(
            grassland_by_year=[
                *NO_GRASSLAND[:-1],
                float(Grassland.CULTIVATED if is_cultivated else Grassland.OTHER),
            ],
            is_cropland=is_cropland,
            loss_year=2012,
            planting_year=planting_year,
        )
    )
    assert result.destination_dataset.data == [[destination_dataset]]
    assert result.to_cropland.data == [[to_cropland]]
    assert result.to_pasture.data == [[to_pasture]]


DEPARTS_RANGELAND_IN_2012 = [
    Grassland.NATURAL if year <= 2011 else Grassland.OTHER
    for year in LOOKBACK_YEARS_RANGE
]


@pytest.mark.parametrize(
    (
        "grassland_by_year",
        "is_cropland",
        "loss_year",
        "from_forest",
        "from_grassland",
        "has_destination",
    ),
    (
        pytest.param(
            NO_GRASSLAND, True, 2012, True, False, True, id="both halves resolve"
        ),
        pytest.param(
            NO_GRASSLAND,
            False,
            2012,
            True,
            False,
            False,
            id="a forest loss that reached no destination is still a forest source -- it is the "
            "pixel the dropped total is charged on",
        ),
        pytest.param(
            NO_GRASSLAND,
            True,
            None,
            False,
            False,
            True,
            id="a destination with no source is still drained, and still takes peat occupation",
        ),
        pytest.param(
            NO_GRASSLAND,
            False,
            2024,
            False,
            False,
            False,
            id="a loss past the lookback is no source",
        ),
        pytest.param(
            DEPARTS_RANGELAND_IN_2012,
            True,
            None,
            False,
            True,
            True,
            id="a grassland departure is the other source",
        ),
        pytest.param(
            DEPARTS_RANGELAND_IN_2012,
            True,
            2012,
            True,
            False,
            True,
            id="and yields to a forest loss on the same pixel, per the source priority",
        ),
        pytest.param(
            [*NO_GRASSLAND[:-1], Grassland.CULTIVATED],
            False,
            None,
            False,
            False,
            True,
            id="cultivated grassland at the assessment year is a destination too, and arriving in "
            "it is not a departure from it",
        ),
        pytest.param(
            [
                Grassland.CULTIVATED if year <= 2011 else Grassland.OTHER
                for year in LOOKBACK_YEARS_RANGE
            ],
            True,
            None,
            False,
            True,
            True,
            id="a pasture departure is a grassland source as much as a rangeland one",
        ),
        pytest.param(
            NO_GRASSLAND, False, None, False, False, False, id="nothing resolves at all"
        ),
    ),
)
def test_the_halves_a_conversion_record_resolves(
    grassland_by_year: list[float],
    is_cropland: bool,
    loss_year: float | None,
    from_forest: bool,
    from_grassland: bool,
    has_destination: bool,
) -> None:
    # The halves outlive the conversion they resolve into: one without the other names no
    # conversion, but each still drives something -- the dropped total and peat occupation.
    result = get_conversion_record(
        dset=get_dset_for_one_pixel(
            grassland_by_year=grassland_by_year,
            is_cropland=is_cropland,
            loss_year=loss_year,
            planting_year=0,
        )
    )
    assert bool(result.from_forest) == from_forest
    assert bool(result.from_grassland) == from_grassland
    assert bool(result.to_cropland | result.to_pasture) == has_destination
