import math

import pandas
import pytest

from jdluc.datasets import faostat
from jdluc.datasets.worldbank_jurisdictions import AdminLevel


def get_faostat_frame(
    column_name: str, key_to_value: dict[tuple[str, str, int], float]
) -> pandas.DataFrame:
    """A table shaped like `faostat.load`'s livestock tables, one column over each species' live
    animals by (country, species, year).
    """
    return pandas.DataFrame.from_records(
        data=[
            {
                "admin_level": AdminLevel.NATIONAL.name,
                "admin_id": iso_3166,
                "jurisdiction_name": iso_3166,
                "species": species_name,
                "commodity_id": faostat.SPECIES_TO_LIVE_ANIMALS_ITEM_CODE[
                    faostat.Species[species_name]
                ],
                "year": year,
                column_name: value,
            }
            for (iso_3166, species_name, year), value in key_to_value.items()
        ]
    ).set_index(faostat.LIVESTOCK_IDX_COLUMN_NAMES)


NO_LIVESTOCK_UNITS = dict.fromkeys(faostat.Species, 0.0)
CATTLE = faostat.Species.CATTLE


@pytest.mark.parametrize(
    ("livestock_units", "stocks", "expected"),
    (
        pytest.param(
            {
                ("BRA", "CATTLE", 2000): 70.0,
                ("BRA", "CATTLE", 2005): 70.0,
                ("BRA", "CATTLE", 2010): 95.0,
            },
            {
                ("BRA", "CATTLE", 2000): 100.0,
                ("BRA", "CATTLE", 2005): 100.0,
                ("BRA", "CATTLE", 2010): 100.0,
            },
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="a-year-whose-heads-were-since-revised-does-not-move-the-median",
        ),
        pytest.param(
            # Buffalo as FAOSTAT gives South America's, at zero, and sheep with no figure at all:
            # neither falls back to another coefficient
            {("BRA", "BUFFALO", 2020): 0.0, ("BRA", "CATTLE", 2020): 70.0},
            {
                ("BRA", "BUFFALO", 2020): 100.0,
                ("BRA", "CATTLE", 2020): 100.0,
                ("BRA", "SHEEP", 2020): 100.0,
            },
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="a-grazer-given-zero-or-no-livestock-units-counts-none",
        ),
        pytest.param(
            {("BRA", "CATTLE", 2000): 10.0, ("BRA", "CATTLE", 2020): 70.0},
            {("BRA", "CATTLE", 2000): 0.0, ("BRA", "CATTLE", 2020): 100.0},
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="a-year-without-heads-is-left-out-rather-than-divided-by",
        ),
        pytest.param(
            {("ARG", "CATTLE", 2020): 50.0, ("BRA", "CATTLE", 2020): 70.0},
            {("ARG", "CATTLE", 2020): 100.0, ("BRA", "CATTLE", 2020): 100.0},
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="only-the-countrys-own-rows-count",
        ),
    ),
)
def test_get_species_to_livestock_units_per_head(
    livestock_units: dict[tuple[str, str, int], float],
    stocks: dict[tuple[str, str, int], float],
    expected: dict[faostat.Species, float],
) -> None:
    assert faostat.get_species_to_livestock_units_per_head(
        iso_3166="BRA",
        livestock=get_faostat_frame(column_name="heads", key_to_value=stocks),
        livestock_units=get_faostat_frame(
            column_name="livestock_units", key_to_value=livestock_units
        ),
    ) == pytest.approx(expected)


AREA = faostat.AREA_HARVESTED_ELEMENT_CODE
PRODUCTION = faostat.PRODUCTION_ELEMENT_CODE


def get_bulk_row(
    cpc_code: str, element_code: int, flag: str, item_code: int
) -> dict[str, str]:
    """A row of the Production archive, for Brazil in 2020: 10 ha harvested or 3 t produced."""
    return {
        "Area Code (M49)": "'076",
        "Area": "Brazil",
        "Item Code": str(item_code),
        "Item Code (CPC)": cpc_code,
        "Item": f"item {item_code:d}",
        "Element Code": str(element_code),
        "Year": "2020",
        "Value": {AREA: "10", PRODUCTION: "3"}[element_code],
        "Flag": flag,
    }


def test_get_crop_records() -> None:
    rows = [
        # Maize carries area and production, in kg
        get_bulk_row(cpc_code="'0112", element_code=AREA, flag="A", item_code=56),
        get_bulk_row(cpc_code="'0112", element_code=PRODUCTION, flag="A", item_code=56),
        # Brazil nuts report no area, which reads NaN
        get_bulk_row(
            cpc_code="'01377", element_code=PRODUCTION, flag="A", item_code=216
        ),
        # Barley reports no production, and chick peas' is missing: both are dropped
        get_bulk_row(cpc_code="'0115", element_code=AREA, flag="A", item_code=44),
        get_bulk_row(cpc_code="'01703", element_code=AREA, flag="A", item_code=191),
        get_bulk_row(
            cpc_code="'01703",
            element_code=PRODUCTION,
            flag=faostat.MISSING_FLAG,
            item_code=191,
        ),
        # Livestock and aggregates are not crops
        get_bulk_row(
            cpc_code="'21111.01", element_code=PRODUCTION, flag="A", item_code=867
        ),
        get_bulk_row(
            cpc_code="F1717", element_code=PRODUCTION, flag="A", item_code=1717
        ),
    ]
    column_names = ["commodity_id", "area_hectares", "production_kg"]
    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_records(
            columns=column_names, data=faostat.get_crop_records(rows=rows)
        ),
        pandas.DataFrame.from_records(
            columns=column_names, data=[(56, 10.0, 3000.0), (216, math.nan, 3000.0)]
        ),
        check_dtype=False,
    )


def test_get_livestock_records() -> None:
    # Each item is its own record, named as FAOSTAT names it; items of no grazer are dropped
    rows = [
        {
            "Area Code (M49)": m49_code,
            "Area": area,
            "Item Code": str(item_code),
            "Item": item_name,
            "Element Code": str(element_code),
            "Year": "2020",
            "Value": value,
            "Flag": "A",
        }
        for m49_code, area, item_code, item_name, element_code, value in (
            ("'076", "Brazil", 866, "Cattle", faostat.STOCKS_ELEMENT_CODE, "1000"),
            ("'076", "Brazil", 867, "Meat of cattle", PRODUCTION, "50"),
            ("'076", "Brazil", 882, "Raw milk of cattle", PRODUCTION, "200"),
            (
                "'076",
                "Brazil",
                882,
                "Raw milk of cattle",
                faostat.MILK_ANIMALS_ELEMENT_CODE,
                "100",
            ),
            ("'076", "Brazil", 1057, "Chickens", faostat.STOCKS_ELEMENT_CODE, "9"),
        )
    ]
    column_names = [
        "commodity_id",
        "commodity_name",
        "heads",
        "production_kg",
        "species",
    ]
    records = faostat.get_livestock_records(
        element_columns=faostat.LIVESTOCK_ELEMENT_COLUMNS, rows=rows
    )
    assert all(list(record) == sorted(record) for record in records)
    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_records(data=records)[column_names],
        pandas.DataFrame.from_records(
            columns=column_names,
            data=[
                (866, "Cattle", 1000.0, math.nan, "CATTLE"),
                (867, "Meat of cattle", math.nan, 50_000.0, "CATTLE"),
                (882, "Raw milk of cattle", 100.0, 200_000.0, "CATTLE"),
            ],
        ),
    )


# Cattle (866) and the cows milked (882), in heads
CATTLE_HEADS = (
    pandas.DataFrame.from_records(
        columns=["admin_id", "commodity_id", "year", "heads"],
        data=[
            ("BRA", 866, 2018, 100.0),
            ("BRA", 882, 2018, 100.0),
            ("BRA", 866, 2019, 100.0),
            ("BRA", 882, 2019, 10.0),
            ("BRA", 866, 2020, 300.0),
            ("BRA", 882, 2020, 50.0),
            ("PRY", 866, 2020, 100.0),
        ],
    )
    .assign(
        admin_level=AdminLevel.NATIONAL.name,
        jurisdiction_name="",
        species=CATTLE.name,
    )
    .set_index(faostat.LIVESTOCK_IDX_COLUMN_NAMES)
)


@pytest.mark.parametrize(
    ("iso_3166", "expected"),
    (
        pytest.param(
            "BRA", 30.0 / 200.0, id="a-ratio-of-means-over-the-years-asked-for"
        ),
        pytest.param("PRY", 0.0, id="no-milking-cows-reported-is-no-dairy-herd"),
    ),
)
def test_get_dairy_share(iso_3166: str, expected: float) -> None:
    assert faostat.get_dairy_share(
        iso_3166=iso_3166, livestock=CATTLE_HEADS, years=(2019, 2020)
    ) == pytest.approx(expected)
