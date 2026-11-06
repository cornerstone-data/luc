import math

import pandas
import pytest

from jdluc.datasets import faostat
from jdluc.datasets.worldbank_jurisdictions import AdminLevel


def get_faostat_frame(
    column_name: str, key_to_value: dict[tuple[str, str, int], float]
) -> pandas.DataFrame:
    """A table shaped like `faostat.load`'s, one column over (country, species, year)."""
    return pandas.DataFrame.from_records(
        data=[
            {
                "admin_level": AdminLevel.NATIONAL.name,
                "admin_id": iso_3166,
                "jurisdiction_name": iso_3166,
                "commodity_name": commodity_name,
                "year": year,
                column_name: value,
            }
            for (iso_3166, commodity_name, year), value in key_to_value.items()
        ]
    ).set_index(faostat.IDX_COLUMN_NAMES)


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
        livestock_units=get_faostat_frame(
            column_name="stocks_livestock_units", key_to_value=livestock_units
        ),
        stocks=get_faostat_frame(column_name="stocks_head", key_to_value=stocks),
    ) == pytest.approx(expected)


def get_bulk_row(
    cpc_code: str, element_code: int, flag: str, item_code: str, value: str
) -> dict[str, str]:
    """A row of the Production archive, for Brazil in 2020."""
    return {
        "Area Code (M49)": "'076",
        "Area": "Brazil",
        "Item Code": item_code,
        "Item Code (CPC)": cpc_code,
        "Item": f"item {item_code:s}",
        "Element Code": str(element_code),
        "Year": "2020",
        "Value": value,
        "Flag": flag,
    }


AREA = faostat.AREA_HARVESTED_ELEMENT_CODE
PRODUCTION = faostat.PRODUCTION_ELEMENT_CODE


@pytest.mark.parametrize(
    ("rows", "expected"),
    (
        pytest.param(
            [
                get_bulk_row(
                    cpc_code="'0112",
                    element_code=AREA,
                    flag="A",
                    item_code="56",
                    value="10",
                ),
                get_bulk_row(
                    cpc_code="'0112",
                    element_code=PRODUCTION,
                    flag="A",
                    item_code="56",
                    value="3",
                ),
            ],
            [(56, 10.0, 3000.0)],
            id="a-crop-carries-area-and-production-in-kg",
        ),
        pytest.param(
            [
                get_bulk_row(
                    cpc_code="'01377",
                    element_code=PRODUCTION,
                    flag="A",
                    item_code="216",
                    value="3",
                )
            ],
            [(216, math.nan, 3000.0)],
            id="a-crop-with-no-area-reads-nan",
        ),
        pytest.param(
            [
                get_bulk_row(
                    cpc_code="'0112",
                    element_code=AREA,
                    flag="A",
                    item_code="56",
                    value="10",
                )
            ],
            [],
            id="a-crop-with-no-production-is-dropped",
        ),
        pytest.param(
            [
                get_bulk_row(
                    cpc_code="'0112",
                    element_code=AREA,
                    flag="A",
                    item_code="56",
                    value="10",
                ),
                get_bulk_row(
                    cpc_code="'0112",
                    element_code=PRODUCTION,
                    flag=faostat.MISSING_FLAG,
                    item_code="56",
                    value="3",
                ),
            ],
            [],
            id="a-missing-value-counts-as-none",
        ),
        pytest.param(
            [
                get_bulk_row(
                    cpc_code="'21111.01",
                    element_code=PRODUCTION,
                    flag="A",
                    item_code="867",
                    value="3",
                ),
                get_bulk_row(
                    cpc_code="F1717",
                    element_code=PRODUCTION,
                    flag="A",
                    item_code="1717",
                    value="3",
                ),
            ],
            [],
            id="livestock-and-aggregates-are-not-crops",
        ),
    ),
)
def test_get_crop_records(
    rows: list[dict[str, str]], expected: list[tuple[int, float, float]]
) -> None:
    records = faostat.get_crop_records(rows=rows)
    column_names = ["commodity_id", "area_hectares", "production_kg"]
    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_records(columns=column_names, data=records),
        pandas.DataFrame.from_records(columns=column_names, data=expected),
        check_dtype=False,
    )
    assert all(record["admin_id"] == "BRA" for record in records)
