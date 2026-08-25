import numpy
import pandas

from jdluc.attribute import Methodology
from jdluc.datasets.worldbank_jurisdictions import AdminLevel
from jdluc.trace import (
    attach_ratios,
    derive_jurisdictional_production_kg,
    derive_statistical_production_kg,
    iter_national_from_provincials,
)

JURISDICTIONAL_EMISSIONS = pandas.DataFrame.from_records(
    [
        # normal row
        (
            "PROVINCIAL",
            "MAIZE",
            "Delaware",
            "JURISDICTIONAL_DIRECT",
            "USA008",
            100.0,
            10.0,
            50.0,
            200.0,
            120.0,
            30.0,
        ),
        # zero production (crop_hectares == 0)
        (
            "PROVINCIAL",
            "SOYBEAN",
            "Delaware",
            "JURISDICTIONAL_DIRECT",
            "USA008",
            0.0,
            0.0,
            0.0,
            80.0,
            50.0,
            30.0,
        ),
        # unmatched yield + zero emissions
        (
            "PROVINCIAL",
            "WHEAT",
            "Iowa",
            "JURISDICTIONAL_DIRECT",
            "USA016",
            50.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ],
    columns=[
        "admin_level",
        "crop_name",
        "jurisdiction_name",
        "methodology",
        "admin_id",
        "crop_hectares",
        "peatland_crop_hectares",
        "peatland_occupation_emissions_mt",
        "emissions_mt",
        "forest_emissions_mt",
        "peatland_conversion_emissions_mt",
    ],
).set_index(["admin_level", "admin_id", "crop_name", "methodology"])
RAW_YIELDS = pandas.DataFrame.from_records(
    [
        # MAIZE: 2016 must be excluded; mean(8, 10, 12, 10) over 2017-2020 == 10
        *(
            ("PROVINCIAL", "USA008", "Delaware", "MAIZE", year, val)
            for year, val in (
                (2016, 999.0),
                (2017, 8.0),
                (2018, 10.0),
                (2019, 12.0),
                (2020, 10.0),
            )
        ),
        # SOYBEAN: mean == 30
        *(
            ("PROVINCIAL", "USA008", "Delaware", "SOYBEAN", year, 30.0)
            for year in (2017, 2018, 2019, 2020)
        ),
        # (USA016, WHEAT): deliberately absent -> unmatched
    ],
    columns=[
        "admin_level",
        "admin_id",
        "jurisdiction_name",
        "crop_name",
        "year",
        "yield_kg_per_ha",
    ],
).set_index(["admin_level", "admin_id", "jurisdiction_name", "crop_name", "year"])


def test_merge_jurisdictional_emissions_and_yields() -> None:
    result = attach_ratios(
        df=derive_jurisdictional_production_kg(
            emissions=JURISDICTIONAL_EMISSIONS, raw_yields=RAW_YIELDS
        )
    )
    assert isinstance(result, pandas.DataFrame)

    iter_result = result.iterrows()
    key, corn = next(iter_result)
    assert key == ("PROVINCIAL", "USA008", "MAIZE", "JURISDICTIONAL_DIRECT")
    assert corn["yield_kg_per_ha"] == 10.0  # 4-year mean, 2016 excluded
    assert corn["production_kg"] == 1000.0  # 100 ha x 10
    assert corn["emissions_factor_kgco2e_per_kg"] == 200.0  # 200 t x 1000 / 1000 kg
    assert corn["peatland_occupation_fraction"] == 0.25  # 50 / 200

    # zero production -> EF guarded to NaN (not inf); fraction still defined (0/80)
    key, soy = next(iter_result)
    assert key == ("PROVINCIAL", "USA008", "SOYBEAN", "JURISDICTIONAL_DIRECT")
    assert soy["production_kg"] == 0.0
    assert numpy.isnan(soy["emissions_factor_kgco2e_per_kg"])
    assert soy["peatland_occupation_fraction"] == 0.0

    # unmatched yield -> NaN yield/production/EF; zero emissions_mt -> NaN fraction
    key, wheat = next(iter_result)
    assert key == ("PROVINCIAL", "USA016", "WHEAT", "JURISDICTIONAL_DIRECT")
    assert numpy.isnan(wheat["yield_kg_per_ha"])
    assert numpy.isnan(wheat["production_kg"])
    assert numpy.isnan(wheat["emissions_factor_kgco2e_per_kg"])
    assert numpy.isnan(wheat["peatland_occupation_fraction"])


def test_iter_national_from_provincials_single_province() -> None:
    provincials = pandas.DataFrame.from_records(
        [
            {
                "crop_hectares": 1,
                "crop_name": "CROP_1",
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_crop_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
        ]
    )
    (result,) = iter_national_from_provincials(
        iso_3166="ISO_A3",
        methodology=Methodology.STATISTICAL,
        national_name="NAME",
        provincials=provincials,
    )
    assert result == {
        "admin_id": "ISO_A3",
        "admin_level": AdminLevel.NATIONAL.name,
        "crop_hectares": 1,
        "crop_name": "CROP_1",
        "emissions_factor_kgco2e_per_kg": 1000,
        "emissions_mt": 1,
        "forest_emissions_mt": 1,
        "jurisdiction_name": "NAME",
        "methodology": "STATISTICAL",
        "peatland_conversion_emissions_mt": 1,
        "peatland_crop_hectares": 1,
        "peatland_occupation_emissions_mt": 1,
        "peatland_occupation_fraction": 1,
        "production_kg": 1,
        "yield_kg_per_ha": 1,
    }


def test_iter_national_from_provincials_multi_province_crop() -> None:
    provincials = pandas.DataFrame.from_records(
        [
            {
                "admin_id": "ADM_0",
                "crop_hectares": 1,
                "crop_name": "CROP_0",
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_crop_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
            {
                "admin_id": "ADM_0",
                "crop_hectares": 1,
                "crop_name": "CROP_1",
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_crop_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
            {
                "admin_id": "ADM_1",
                "crop_hectares": 1,
                "crop_name": "CROP_0",
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_crop_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
        ]
    )
    (crop_0, crop_1) = iter_national_from_provincials(
        iso_3166="ISO_A3",
        methodology=Methodology.STATISTICAL,
        national_name="NAME",
        provincials=provincials,
    )
    assert crop_0 == {
        "admin_id": "ISO_A3",
        "admin_level": AdminLevel.NATIONAL.name,
        "crop_hectares": 2,
        "crop_name": "CROP_0",
        "emissions_factor_kgco2e_per_kg": 1000,
        "emissions_mt": 2,
        "forest_emissions_mt": 2,
        "jurisdiction_name": "NAME",
        "methodology": "STATISTICAL",
        "peatland_conversion_emissions_mt": 2,
        "peatland_crop_hectares": 2,
        "peatland_occupation_emissions_mt": 2,
        "peatland_occupation_fraction": 1,
        "production_kg": 2,
        "yield_kg_per_ha": 1,
    }
    assert crop_1 == {
        "admin_id": "ISO_A3",
        "admin_level": AdminLevel.NATIONAL.name,
        "crop_hectares": 1,
        "crop_name": "CROP_1",
        "emissions_factor_kgco2e_per_kg": 1000,
        "emissions_mt": 1,
        "forest_emissions_mt": 1,
        "methodology": "STATISTICAL",
        "jurisdiction_name": "NAME",
        "peatland_conversion_emissions_mt": 1,
        "peatland_crop_hectares": 1,
        "peatland_occupation_emissions_mt": 1,
        "peatland_occupation_fraction": 1,
        "production_kg": 1,
        "yield_kg_per_ha": 1,
    }


STATISTICAL_EMISSIONS = pandas.DataFrame.from_records(
    [
        (
            "PROVINCIAL",
            "CORN",
            "Delaware",
            "STATISTICAL",
            "USA008",
            100.0,
            10.0,
            50.0,
            200.0,
            5.0,
            120.0,
            30.0,
        ),
    ],
    columns=[
        "admin_level",
        "crop_name",
        "jurisdiction_name",
        "methodology",
        "admin_id",
        "crop_hectares",
        "peatland_crop_hectares",
        "peatland_occupation_emissions_mt",
        "emissions_mt",
        "production_mt",
        "forest_emissions_mt",
        "peatland_conversion_emissions_mt",
    ],
).set_index(["admin_level", "admin_id", "crop_name", "methodology"])


def test_derive_statistical_production_kg_stays_indexed() -> None:
    result = derive_statistical_production_kg(emissions=STATISTICAL_EMISSIONS)
    # regression guard: must return an indexed frame like its jurisdictional twin
    assert result.index.names == [
        "admin_level",
        "admin_id",
        "crop_name",
        "methodology",
    ]
    assert "production_mt" not in result.columns
    assert result.iloc[0]["production_kg"] == 5000.0  # 5 mt x 1000
    # .copy() guard: attach_ratios mutates in place, so the source frame must be untouched
    assert "production_kg" not in STATISTICAL_EMISSIONS.columns


def test_methodologies_share_a_schema() -> None:
    jurisdictional = attach_ratios(
        df=derive_jurisdictional_production_kg(
            emissions=JURISDICTIONAL_EMISSIONS, raw_yields=RAW_YIELDS
        )
    )
    statistical = attach_ratios(
        df=derive_statistical_production_kg(emissions=STATISTICAL_EMISSIONS)
    )
    assert set(jurisdictional.reset_index().columns) == set(
        statistical.reset_index().columns
    )
