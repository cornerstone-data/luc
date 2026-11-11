import numpy
import pandas
import pytest

from jdluc.attribute import Methodology
from jdluc.datasets import tang_cropgrids
from jdluc.datasets.worldbank_jurisdictions import AdminLevel
from jdluc.jurisdictional_direct_proxy import HERD_HEADS_COLUMN
from jdluc.statistical import HERD_HEADS_COLUMNS
from jdluc.taxonomy import (
    CO_PRODUCT_ITEM_CODE_TO_ITEM_CODE,
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID,
    UNMAPPED_CROP_ITEM_CODES,
)
from jdluc.trace import (
    attach_ratios,
    derive_jurisdictional_production_kg,
    derive_proxy_production_kg,
    derive_statistical_production_kg,
    iter_national_from_provincials,
    with_commodity_ids,
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

    # zero production -> EF guarded to NaN rather than inf
    key, soy = next(iter_result)
    assert key == ("USA008", "PROVINCIAL", 2020, "SOYBEAN", "JURISDICTIONAL_DIRECT")
    assert soy["production_kg"] == 0.0
    assert numpy.isnan(soy["emissions_factor_kgco2e_per_kg"])

    # unmatched yield -> NaN yield, production and EF
    key, wheat = next(iter_result)
    assert key == ("USA016", "PROVINCIAL", 2020, "WHEAT", "JURISDICTIONAL_DIRECT")
    assert numpy.isnan(wheat["yield_kg_per_ha"])
    assert numpy.isnan(wheat["production_kg"])
    assert numpy.isnan(wheat["emissions_factor_kgco2e_per_kg"])


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
        "peatland_conversion_emissions_mt": 1,
        "peatland_commodity_hectares": 1,
        "peatland_occupation_emissions_mt": 1,
        "production_kg": 1,
        "yield_kg_per_ha": 1,
    }


STATISTICAL_EMISSIONS = pandas.DataFrame.from_records(
    [
        (
            "USA008",
            "PROVINCIAL",
            2020,
            100.0,
            "CORN",
            200.0,
            120.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "Delaware",
            "STATISTICAL",
            10.0,
            30.0,
            50.0,
            5.0,
        ),
        (
            "BRA001",
            "PROVINCIAL",
            2020,
            100.0,
            "BEEF_CATTLE",
            200.0,
            200.0,
            100.0,
            100.0,
            100.0,
            100.0,
            "Acre",
            "STATISTICAL",
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        # A country FAOSTAT reports no herd for
        (
            "NIU001",
            "PROVINCIAL",
            2020,
            100.0,
            "BEEF_CATTLE",
            200.0,
            200.0,
            100.0,
            100.0,
            100.0,
            100.0,
            "Niue",
            "STATISTICAL",
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            "PRY001",
            "PROVINCIAL",
            2020,
            100.0,
            "BEEF_CATTLE",
            200.0,
            200.0,
            100.0,
            100.0,
            100.0,
            100.0,
            "Concepción",
            "STATISTICAL",
            0.0,
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
        "herd_heads_2000",
        "herd_heads_2005",
        "herd_heads_2010",
        "herd_heads_2020",
        "jurisdiction_name",
        "methodology",
        "peatland_commodity_hectares",
        "peatland_conversion_emissions_mt",
        "peatland_occupation_emissions_mt",
        "production_mt",
    ],
).set_index(
    ["admin_id", "admin_level", "assessment_year", "commodity_name", "methodology"]
)


# The 2020 assessment's snapshot weights
SNAPSHOT_YEAR_TO_WEIGHT = {2000: 0.03125, 2005: 0.125, 2010: 0.46875, 2020: 0.375}


# Brazil's cattle rate doubles each snapshot, and its sheep rate catches a herd priced as another
# species. Paraguay reports meat for its herd only in 2020.
STOCKS = pandas.DataFrame.from_records(
    [
        *(
            ("NATIONAL", "BRA", "Brazil", "CATTLE", year, 1000.0, kg)
            for year, kg in (
                (2000, 10_000.0),
                (2005, 20_000.0),
                (2010, 40_000.0),
                (2020, 80_000.0),
            )
        ),
        ("NATIONAL", "BRA", "Brazil", "SHEEP", 2020, 1.0, 1_000_000.0),
        *(
            ("NATIONAL", "PRY", "Paraguay", "CATTLE", year, 1000.0, kg)
            for year, kg in (
                (2000, numpy.nan),
                (2005, numpy.nan),
                (2010, numpy.nan),
                (2020, 80_000.0),
            )
        ),
    ],
    columns=[
        "admin_level",
        "admin_id",
        "jurisdiction_name",
        "commodity_name",
        "year",
        "stocks_head",
        "production_kg",
    ],
).set_index(["admin_level", "admin_id", "jurisdiction_name", "commodity_name", "year"])


def test_derive_statistical_production_kg_stays_indexed() -> None:
    result = derive_statistical_production_kg(
        emissions=STATISTICAL_EMISSIONS,
        snapshot_year_to_weight=SNAPSHOT_YEAR_TO_WEIGHT,
        stocks=STOCKS,
    )
    # regression guard: must return an indexed frame like its jurisdictional twin
    assert result.index.names == [
        "admin_id",
        "admin_level",
        "assessment_year",
        "commodity_name",
        "methodology",
    ]
    assert "production_mt" not in result.columns
    assert not set(HERD_HEADS_COLUMNS) & set(result.columns)
    assert (
        result.loc[
            ("USA008", "PROVINCIAL", 2020, "CORN", "STATISTICAL"), "production_kg"
        ]
        == 5000.0  # 5 mt x 1000
    )
    # .copy() guard: attach_ratios mutates in place, so the source frame must be untouched
    assert "production_kg" not in STATISTICAL_EMISSIONS.columns


def test_derive_statistical_production_kg_prices_a_herd_at_each_snapshots_rate() -> (
    None
):
    result = derive_statistical_production_kg(
        emissions=STATISTICAL_EMISSIONS,
        snapshot_year_to_weight=SNAPSHOT_YEAR_TO_WEIGHT,
        stocks=STOCKS,
    )
    # 100 heads at 10, 20, 40 and 80 kg, weighted 3.125%, 12.5%, 46.875% and 37.5%
    assert result.loc[
        ("BRA001", "PROVINCIAL", 2020, "BEEF_CATTLE", "STATISTICAL"), "production_kg"
    ] == pytest.approx(5156.25)
    assert (
        result.loc[
            ("NIU001", "PROVINCIAL", 2020, "BEEF_CATTLE", "STATISTICAL"),
            "production_kg",
        ]
        == 0.0
    )
    # 100 heads at 80 kg, weighted 37.5%, in 2020 alone
    assert result.loc[
        ("PRY001", "PROVINCIAL", 2020, "BEEF_CATTLE", "STATISTICAL"), "production_kg"
    ] == pytest.approx(3000.0)


def test_methodologies_share_a_schema() -> None:
    jurisdictional = attach_ratios(
        df=derive_jurisdictional_production_kg(
            emissions=JURISDICTIONAL_EMISSIONS,
            raw_yields=RAW_YIELDS,
            years=(2017, 2018, 2019, 2020),
        )
    )
    statistical = attach_ratios(
        df=derive_statistical_production_kg(
            emissions=STATISTICAL_EMISSIONS,
            snapshot_year_to_weight=SNAPSHOT_YEAR_TO_WEIGHT,
            stocks=STOCKS,
        )
    )
    assert set(jurisdictional.reset_index().columns) == set(
        statistical.reset_index().columns
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


def test_derive_proxy_production_kg() -> None:
    """Brazil's maize is FAOSTAT's 2019-2021 mean of 200 kg, split over 50 ha of maize and popcorn,
    which share its item. Its cattle weigh FAOSTAT's 1050 kg of meat a year over 100 head, with
    2021's meat unreported and 2018 outside the years. Its soya beans have no CROPGRIDS area, so no
    row takes them.
    """
    proxy_name_to_commodity_id = METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
        Methodology.JURISDICTIONAL_DIRECT_PROXY
    ]
    # Every crop item, so the coverage assertion holds
    every_item_code = (
        {
            proxy_name_to_commodity_id[crop.name]
            for crop in tang_cropgrids.Crop
            if proxy_name_to_commodity_id[crop.name] > 0
        }
        | set(CO_PRODUCT_ITEM_CODE_TO_ITEM_CODE)
        | UNMAPPED_CROP_ITEM_CODES
    )
    crops = pandas.DataFrame.from_records(
        [
            {
                "admin_id": "ZZZ",
                "commodity_id": code,
                "production_kg": 0.0,
                "year": 2020,
            }
            for code in sorted(every_item_code)
        ]
        + [
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
    stocks = pandas.DataFrame.from_records(
        [
            ("BRA", "CATTLE", 2018, 99999.0, 100.0),
            ("BRA", "CATTLE", 2019, 900.0, 100.0),
            ("BRA", "CATTLE", 2020, 1200.0, 100.0),
            ("BRA", "CATTLE", 2021, numpy.nan, 100.0),
        ],
        columns=["admin_id", "commodity_name", "year", "production_kg", "stocks_head"],
    ).set_index(["admin_id", "commodity_name", "year"])
    emissions = pandas.DataFrame.from_records(
        [
            ("BRA001", "MAIZE", 30.0, numpy.nan),
            ("BRA002", "MAIZE", 10.0, numpy.nan),
            ("BRA001", "POPCORN", 10.0, numpy.nan),
            ("BRA001", "ALFALFA", 5.0, numpy.nan),
            ("BRA001", "BEEF_CATTLE", 1.0, 100.0),
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
    result = derive_proxy_production_kg(
        crops=crops, emissions=emissions, stocks=stocks, years=(2019, 2020, 2021)
    )
    production_kg = result["production_kg"].droplevel(
        ["admin_level", "methodology", "assessment_year"]
    )
    assert production_kg[("BRA001", "MAIZE")] == pytest.approx(120.0)
    assert production_kg[("BRA002", "MAIZE")] == pytest.approx(40.0)
    assert production_kg[("BRA001", "POPCORN")] == pytest.approx(40.0)
    assert production_kg[("BRA001", "BEEF_CATTLE")] == pytest.approx(1050.0)
    assert (
        production_kg[[("BRA001", name) for name in ("ALFALFA", "PASTURE", "DROPPED")]]
        .isna()
        .all()
    )
    assert HERD_HEADS_COLUMN not in result.columns
