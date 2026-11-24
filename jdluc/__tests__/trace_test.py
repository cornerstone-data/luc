import numpy
import pandas
import pytest

from jdluc import statistical
from jdluc.attribute import Methodology
from jdluc.datasets.worldbank_jurisdictions import AdminLevel
from jdluc.jurisdictional_direct_proxy import HERD_HEADS_COLUMN
from jdluc.taxonomy import InventedCommodityId
from jdluc.trace import (
    ADDITIVE_COLUMNS,
    attach_ratios,
    derive_jurisdictional_production_kg,
    derive_production_kg,
    iter_national_from_provincials,
    with_commodity_ids,
    with_withheld_factors,
)

JURISDICTIONAL_EMISSIONS = pandas.DataFrame.from_records(
    [
        # normal row
        (
            "USA008",
            "PROVINCIAL",
            2020,
            100.0,
            "MAIZE",
            200.0,
            120.0,
            "Delaware",
            "JURISDICTIONAL_DIRECT",
            10.0,
            30.0,
            50.0,
        ),
        # zero production (commodity_hectares == 0)
        (
            "USA008",
            "PROVINCIAL",
            2020,
            0.0,
            "SOYBEAN",
            80.0,
            50.0,
            "Delaware",
            "JURISDICTIONAL_DIRECT",
            0.0,
            30.0,
            0.0,
        ),
        # unmatched yield + zero emissions
        (
            "USA016",
            "PROVINCIAL",
            2020,
            50.0,
            "WHEAT",
            0.0,
            0.0,
            "Iowa",
            "JURISDICTIONAL_DIRECT",
            0.0,
            0.0,
            0.0,
        ),
    ],
    columns=[
        "admin_id",
        "admin_level",
        "assessment_year",
        "commodity_hectares",
        "commodity_name",
        "emissions_mt",
        "forest_emissions_mt",
        "jurisdiction_name",
        "methodology",
        "peatland_commodity_hectares",
        "peatland_conversion_emissions_mt",
        "peatland_occupation_emissions_mt",
    ],
).set_index(
    ["admin_id", "admin_level", "assessment_year", "commodity_name", "methodology"]
)
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
        "commodity_name",
        "year",
        "yield_kg_per_ha",
    ],
).set_index(["admin_level", "admin_id", "jurisdiction_name", "commodity_name", "year"])


def test_derive_jurisdictional_production_kg_needs_every_year_of_its_window() -> None:
    # The yields stop at 2020, so a window reaching 2021 has a year to average that NASS never gave
    with pytest.raises(AssertionError, match="NASS yields are missing a year"):
        derive_jurisdictional_production_kg(
            emissions=JURISDICTIONAL_EMISSIONS,
            raw_yields=RAW_YIELDS,
            years=(2018, 2019, 2020, 2021),
        )


def test_derive_jurisdictional_production_kg_then_attach_ratios() -> None:
    result = attach_ratios(
        df=derive_jurisdictional_production_kg(
            emissions=JURISDICTIONAL_EMISSIONS,
            raw_yields=RAW_YIELDS,
            years=(2017, 2018, 2019, 2020),
        )
    )
    assert isinstance(result, pandas.DataFrame)

    iter_result = result.iterrows()
    key, corn = next(iter_result)
    assert key == ("USA008", "PROVINCIAL", 2020, "MAIZE", "JURISDICTIONAL_DIRECT")
    assert corn["yield_kg_per_ha"] == 10.0  # 4-year mean, 2016 excluded
    assert corn["production_kg"] == 1000.0  # 100 ha x 10
    assert corn["emissions_factor_kgco2e_per_kg"] == 200.0  # 200 t x 1000 / 1000 kg
    assert corn["land_occupation_m2_year_per_kg"] == 1000.0  # 10,000 m² / 10 kg per ha

    # zero production -> EF guarded to NaN rather than inf
    key, soy = next(iter_result)
    assert key == ("USA008", "PROVINCIAL", 2020, "SOYBEAN", "JURISDICTIONAL_DIRECT")
    assert soy["production_kg"] == 0.0
    assert numpy.isnan(soy["emissions_factor_kgco2e_per_kg"])
    assert numpy.isnan(soy["land_occupation_m2_year_per_kg"])

    # unmatched yield -> NaN yield, production and EF
    key, wheat = next(iter_result)
    assert key == ("USA016", "PROVINCIAL", 2020, "WHEAT", "JURISDICTIONAL_DIRECT")
    assert numpy.isnan(wheat["yield_kg_per_ha"])
    assert numpy.isnan(wheat["production_kg"])
    assert numpy.isnan(wheat["emissions_factor_kgco2e_per_kg"])
    assert numpy.isnan(wheat["land_occupation_m2_year_per_kg"])


def test_iter_national_from_provincials_single_province() -> None:
    provincials = pandas.DataFrame.from_records(
        [
            {
                "commodity_hectares": 1,
                "commodity_id": 236,
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "grassland_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_commodity_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
        ]
    )
    (result,) = iter_national_from_provincials(
        assessment_year=2020,
        iso_3166="ISO_A3",
        methodology=Methodology.STATISTICAL,
        national_name="NAME",
        provincials=provincials,
    )
    assert result == {
        "admin_id": "ISO_A3",
        "admin_level": AdminLevel.NATIONAL.name,
        "assessment_year": 2020,
        "commodity_hectares": 1,
        "commodity_id": 236,
        "emissions_factor_kgco2e_per_kg": 1000,
        "emissions_mt": 1,
        "forest_emissions_mt": 1,
        "grassland_emissions_mt": 1,
        "jurisdiction_name": "NAME",
        "land_occupation_m2_year_per_kg": 10_000,
        "methodology": "STATISTICAL",
        "peatland_conversion_emissions_mt": 1,
        "peatland_commodity_hectares": 1,
        "peatland_occupation_emissions_mt": 1,
        "production_kg": 1,
        "yield_kg_per_ha": 1,
    }


def test_iter_national_from_provincials_multi_province_crop() -> None:
    provincials = pandas.DataFrame.from_records(
        [
            {
                "admin_id": "ADM_0",
                "commodity_hectares": 1,
                "commodity_id": 56,
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "grassland_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_commodity_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
            {
                "admin_id": "ADM_0",
                "commodity_hectares": 1,
                "commodity_id": 236,
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "grassland_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_commodity_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
            {
                "admin_id": "ADM_1",
                "commodity_hectares": 1,
                "commodity_id": 56,
                "emissions_mt": 1,
                "forest_emissions_mt": 1,
                "grassland_emissions_mt": 1,
                "peatland_conversion_emissions_mt": 1,
                "peatland_commodity_hectares": 1,
                "peatland_occupation_emissions_mt": 1,
                "production_kg": 1,
            },
        ]
    )
    (maize, soya_beans) = iter_national_from_provincials(
        assessment_year=2020,
        iso_3166="ISO_A3",
        methodology=Methodology.STATISTICAL,
        national_name="NAME",
        provincials=provincials,
    )
    assert maize == {
        "admin_id": "ISO_A3",
        "admin_level": AdminLevel.NATIONAL.name,
        "assessment_year": 2020,
        "commodity_hectares": 2,
        "commodity_id": 56,
        "emissions_factor_kgco2e_per_kg": 1000,
        "emissions_mt": 2,
        "forest_emissions_mt": 2,
        "grassland_emissions_mt": 2,
        "jurisdiction_name": "NAME",
        "land_occupation_m2_year_per_kg": 10_000,
        "methodology": "STATISTICAL",
        "peatland_conversion_emissions_mt": 2,
        "peatland_commodity_hectares": 2,
        "peatland_occupation_emissions_mt": 2,
        "production_kg": 2,
        "yield_kg_per_ha": 1,
    }
    assert soya_beans == {
        "admin_id": "ISO_A3",
        "admin_level": AdminLevel.NATIONAL.name,
        "assessment_year": 2020,
        "commodity_hectares": 1,
        "commodity_id": 236,
        "emissions_factor_kgco2e_per_kg": 1000,
        "emissions_mt": 1,
        "forest_emissions_mt": 1,
        "grassland_emissions_mt": 1,
        "methodology": "STATISTICAL",
        "jurisdiction_name": "NAME",
        "land_occupation_m2_year_per_kg": 10_000,
        "peatland_conversion_emissions_mt": 1,
        "peatland_commodity_hectares": 1,
        "peatland_occupation_emissions_mt": 1,
        "production_kg": 1,
        "yield_kg_per_ha": 1,
    }


def test_iter_national_from_provincials_land_occupation_comes_from_the_summed_totals() -> (
    None
):
    # Provinces occupying 10 and 2.5 m² a year per kg average to 6.25, but the country's
    # 2 ha over 5,000 kg occupy 4
    provincials = pandas.DataFrame.from_records(
        [
            dict.fromkeys(ADDITIVE_COLUMNS, 0.0)
            | {"commodity_hectares": 1.0, "commodity_id": 56, "production_kg": 1000.0},
            dict.fromkeys(ADDITIVE_COLUMNS, 0.0)
            | {"commodity_hectares": 1.0, "commodity_id": 56, "production_kg": 4000.0},
        ]
    )
    (result,) = iter_national_from_provincials(
        assessment_year=2020,
        iso_3166="ISO_A3",
        methodology=Methodology.STATISTICAL,
        national_name="NAME",
        provincials=provincials,
    )
    assert result["land_occupation_m2_year_per_kg"] == 4.0


PRODUCTION_YEARS = (2017, 2018, 2019, 2020)
STATISTICAL_EMISSIONS = (
    pandas.DataFrame.from_records(
        [
            ("BRA001", "RAPESEED", 10.0, numpy.nan),
            ("BRA001", "OTHER_OILCROPS", 3.0, numpy.nan),
            ("BRA002", "OTHER_OILCROPS", 1.0, numpy.nan),
            ("BRA001", "ARABICA_COFFEE", 1.0, numpy.nan),
            ("BRA001", "ROBUSTA_COFFEE", 2.0, numpy.nan),
            ("BRA001", "BEEF_CATTLE", 1.0, 100.0),
            ("BRA001", "PASTURE", 1.0, numpy.nan),
            ("BRA001", "DROPPED", 0.0, numpy.nan),
        ],
        columns=[
            "admin_id",
            "commodity_name",
            "commodity_hectares",
            statistical.HERD_HEADS_COLUMN,
        ],
    )
    .assign(
        admin_level="PROVINCIAL",
        assessment_year=2020,
        emissions_mt=1.0,
        forest_emissions_mt=1.0,
        jurisdiction_name="Acre",
        methodology="STATISTICAL",
        peatland_commodity_hectares=0.0,
        peatland_conversion_emissions_mt=0.0,
        peatland_occupation_emissions_mt=0.0,
    )
    .set_index(
        ["admin_id", "admin_level", "assessment_year", "commodity_name", "methodology"]
    )
)
STATISTICAL_CROPS = pandas.DataFrame.from_records(
    [
        *(("BRA", 270, year, 100.0) for year in PRODUCTION_YEARS),
        ("BRA", 292, 2020, 20.0),
        *(("BRA", 260, year, 50.0) for year in PRODUCTION_YEARS),
        ("BRA", 333, 2019, 10.0),
        *(("BRA", 656, year, 300.0) for year in PRODUCTION_YEARS),
    ],
    columns=["admin_id", "commodity_id", "year", "production_kg"],
).set_index(["admin_id", "commodity_id", "year"])
STATISTICAL_LIVESTOCK = pandas.DataFrame.from_records(
    [
        *(("BRA", "CATTLE", 866, year, 100.0, numpy.nan) for year in PRODUCTION_YEARS),
        *(("BRA", "CATTLE", 867, year, numpy.nan, 1000.0) for year in PRODUCTION_YEARS),
    ],
    columns=["admin_id", "species", "commodity_id", "year", "heads", "production_kg"],
).set_index(["admin_id", "commodity_id", "year"])


def test_derive_production_kg_for_the_statistical_leg() -> None:
    """Brazil's rapeseed takes mustard's 20 kg beside its own 100, and its other oil crops olives'
    50 and linseed's 10, split 3:1 between two provinces. Arabica and robusta divide coffee's 300 by
    their area, and its cattle weigh 10 kg of meat a head over 100 head.
    """
    result = derive_production_kg(
        crops=STATISTICAL_CROPS,
        emissions=STATISTICAL_EMISSIONS,
        heads_column=statistical.HERD_HEADS_COLUMN,
        livestock=STATISTICAL_LIVESTOCK,
        methodology=Methodology.STATISTICAL,
        years=PRODUCTION_YEARS,
    )
    production_kg = result["production_kg"].droplevel(
        ["admin_level", "assessment_year", "methodology"]
    )
    assert production_kg.to_dict() == pytest.approx(
        {
            ("BRA001", "RAPESEED"): 120.0,
            ("BRA001", "OTHER_OILCROPS"): 45.0,
            ("BRA002", "OTHER_OILCROPS"): 15.0,
            ("BRA001", "ARABICA_COFFEE"): 100.0,
            ("BRA001", "ROBUSTA_COFFEE"): 200.0,
            ("BRA001", "BEEF_CATTLE"): 1000.0,
            ("BRA001", "PASTURE"): numpy.nan,
            ("BRA001", "DROPPED"): numpy.nan,
        },
        nan_ok=True,
    )


def test_methodologies_share_a_schema() -> None:
    jurisdictional = attach_ratios(
        df=derive_jurisdictional_production_kg(
            emissions=JURISDICTIONAL_EMISSIONS,
            raw_yields=RAW_YIELDS,
            years=PRODUCTION_YEARS,
        )
    )
    statistical_production = attach_ratios(
        df=derive_production_kg(
            crops=STATISTICAL_CROPS,
            emissions=STATISTICAL_EMISSIONS,
            heads_column=statistical.HERD_HEADS_COLUMN,
            livestock=STATISTICAL_LIVESTOCK,
            methodology=Methodology.STATISTICAL,
            years=PRODUCTION_YEARS,
        )
    )
    assert set(jurisdictional.reset_index().columns) == set(
        statistical_production.reset_index().columns
    )


def test_with_commodity_ids_adds_together_the_names_sharing_an_item() -> None:
    emissions = pandas.DataFrame.from_records(
        [
            (
                "PROVINCIAL",
                "BRA001",
                "ARABICA_COFFEE",
                "STATISTICAL",
                2020,
                "Acre",
                1.0,
                10.0,
            ),
            (
                "PROVINCIAL",
                "BRA001",
                "ROBUSTA_COFFEE",
                "STATISTICAL",
                2020,
                "Acre",
                2.0,
                20.0,
            ),
            ("PROVINCIAL", "BRA001", "MAIZE", "STATISTICAL", 2020, "Acre", 4.0, 40.0),
        ],
        columns=[
            "admin_level",
            "admin_id",
            "commodity_name",
            "methodology",
            "assessment_year",
            "jurisdiction_name",
            "commodity_hectares",
            "production_kg",
        ],
    ).set_index(
        ["admin_id", "admin_level", "assessment_year", "commodity_name", "methodology"]
    )
    result = with_commodity_ids(df=emissions, methodology=Methodology.STATISTICAL)
    assert result.to_dict(orient="index") == {
        ("BRA001", "PROVINCIAL", 2020, 56, "STATISTICAL"): {
            "commodity_hectares": 4.0,
            "jurisdiction_name": "Acre",
            "production_kg": 40.0,
        },
        ("BRA001", "PROVINCIAL", 2020, 656, "STATISTICAL"): {
            "commodity_hectares": 3.0,
            "jurisdiction_name": "Acre",
            "production_kg": 30.0,
        },
    }


def test_with_commodity_ids_keeps_a_missing_yield_missing() -> None:
    result = with_commodity_ids(
        df=derive_jurisdictional_production_kg(
            emissions=JURISDICTIONAL_EMISSIONS,
            raw_yields=RAW_YIELDS,
            years=(2017, 2018, 2019, 2020),
        ),
        methodology=Methodology.JURISDICTIONAL_DIRECT,
    )
    assert pandas.isna(
        result.loc[
            ("USA016", "PROVINCIAL", 2020, 15, "JURISDICTIONAL_DIRECT"),
            "production_kg",
        ]
    )


def test_derive_production_kg_for_the_proxy_leg() -> None:
    """Brazil's maize is FAOSTAT's 2019-2021 mean of 200 kg, split over 50 ha of maize and popcorn,
    which share its item. Its cattle weigh FAOSTAT's 1050 kg of meat a year over 100 head, with
    2021's meat unreported and 2018 outside the years. Its soya beans have no CROPGRIDS area, so no
    row takes them.
    """
    crops = pandas.DataFrame.from_records(
        [
            {
                "admin_id": "BRA",
                "commodity_id": 56,
                "production_kg": 100.0,
                "year": 2019,
            },
            {
                "admin_id": "BRA",
                "commodity_id": 56,
                "production_kg": 200.0,
                "year": 2020,
            },
            {
                "admin_id": "BRA",
                "commodity_id": 56,
                "production_kg": 300.0,
                "year": 2021,
            },
            {
                "admin_id": "BRA",
                "commodity_id": 236,
                "production_kg": 7.0,
                "year": 2020,
            },
        ]
    ).set_index(["admin_id", "commodity_id", "year"])
    livestock = pandas.DataFrame.from_records(
        [
            *(
                ("BRA", "CATTLE", 866, year, 100.0, numpy.nan)
                for year in (2018, 2019, 2020, 2021)
            ),
            ("BRA", "CATTLE", 867, 2018, numpy.nan, 99999.0),
            ("BRA", "CATTLE", 867, 2019, numpy.nan, 900.0),
            ("BRA", "CATTLE", 867, 2020, numpy.nan, 1200.0),
            ("BRA", "CATTLE", 882, 2019, 20.0, 3000.0),
            ("BRA", "CATTLE", 882, 2020, 20.0, 3000.0),
            ("BRA", "CATTLE", 882, 2021, 20.0, 6000.0),
        ],
        columns=[
            "admin_id",
            "species",
            "commodity_id",
            "year",
            "heads",
            "production_kg",
        ],
    ).set_index(["admin_id", "commodity_id", "year"])
    emissions = pandas.DataFrame.from_records(
        [
            ("BRA001", "MAIZE", 30.0, numpy.nan),
            ("BRA002", "MAIZE", 10.0, numpy.nan),
            ("BRA001", "POPCORN", 10.0, numpy.nan),
            ("BRA001", "ALFALFA", 5.0, numpy.nan),
            ("BRA001", "BEEF_CATTLE", 1.0, 100.0),
            ("BRA001", "MILK_CATTLE", 1.0, 100.0),
            ("BRA001", "PASTURE", 1.0, numpy.nan),
            ("BRA001", "DROPPED", 0.0, numpy.nan),
        ],
        columns=["admin_id", "commodity_name", "commodity_hectares", HERD_HEADS_COLUMN],
    ).assign(
        admin_level="PROVINCIAL",
        assessment_year=2020,
        methodology="JURISDICTIONAL_DIRECT_PROXY",
    )
    emissions = emissions.set_index(
        ["admin_id", "admin_level", "assessment_year", "commodity_name", "methodology"]
    )
    result = derive_production_kg(
        crops=crops,
        emissions=emissions,
        heads_column=HERD_HEADS_COLUMN,
        livestock=livestock,
        methodology=Methodology.JURISDICTIONAL_DIRECT_PROXY,
        years=(2019, 2020, 2021),
    )
    production_kg = result["production_kg"].droplevel(
        ["admin_level", "methodology", "assessment_year"]
    )
    assert production_kg[("BRA001", "MAIZE")] == pytest.approx(120.0)
    assert production_kg[("BRA002", "MAIZE")] == pytest.approx(40.0)
    assert production_kg[("BRA001", "POPCORN")] == pytest.approx(40.0)
    assert production_kg[("BRA001", "BEEF_CATTLE")] == pytest.approx(1050.0)
    # The same 100 heads at 40 kg of milk each: mean milk over mean stocks, 2019-2021
    assert production_kg[("BRA001", "MILK_CATTLE")] == pytest.approx(4000.0)
    assert (
        production_kg[[("BRA001", name) for name in ("ALFALFA", "PASTURE", "DROPPED")]]
        .isna()
        .all()
    )
    assert HERD_HEADS_COLUMN not in result.columns


def test_with_withheld_factors_blanks_only_the_factors_of_withheld_pairs() -> None:
    traced = pandas.DataFrame(
        data={
            "emissions_factor_kgco2e_per_kg": [1.0, 2.0, 3.0, 4.0],
            "emissions_mt": [10.0, 20.0, 30.0, 40.0],
            "yield_kg_per_ha": [5.0, 6.0, 7.0, 8.0],
        },
        index=pandas.MultiIndex.from_tuples(
            [
                ("CHN", InventedCommodityId.OTHER_OILCROPS),
                ("TWN", InventedCommodityId.OTHER_OILCROPS),
                ("TWN", 27),
                ("TWN001", InventedCommodityId.OTHER_OILCROPS),
            ],
            names=["admin_id", "commodity_id"],
        ),
    )
    result = with_withheld_factors(df=traced)
    assert result["emissions_factor_kgco2e_per_kg"].isna().tolist() == [
        False,
        True,
        False,
        True,
    ]
    pandas.testing.assert_frame_equal(
        result.drop(columns="emissions_factor_kgco2e_per_kg"),
        traced.drop(columns="emissions_factor_kgco2e_per_kg"),
    )
