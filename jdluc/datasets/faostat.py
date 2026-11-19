"""Food and Agriculture Organization of the United Nations | FAOSTAT Production, Livestock Patterns

license: CC BY 4.0

year: 1961, ..., 2024 (Production); 1961, ..., 2023 (Livestock Patterns)

FAO. 2025. Production: Crops and livestock products. FAOSTAT. Rome.
FAO. 2025. Livestock Patterns. FAOSTAT. Rome.

https://www.fao.org/faostat/en/#data/QCL
https://www.fao.org/faostat/en/#data/EK
https://files-faostat.fao.org/production/EK/EK_e.pdf
https://bulks-faostat.fao.org/production/

# Methodology

- National statistics from annual member-country questionnaires, with FAO estimates and imputations
  where a country does not report
- Area harvested counts a field once per harvest, so a double-cropped field counts twice
- Livestock Patterns' stocks in livestock units are Production's heads times FAO's livestock-unit
  coefficient for the country's region (FAO 2011, after Chilonda and Otte 2006), fixed over time

Two bulk archives, read into three tables:

- `CROP_DATASET`: area harvested and production of every crop item, keyed by item code in
  `commodity_id`. Yield is not carried: area and production add across items, and yield does not
- `LIVESTOCK_DATASET`: the items of five grazers, keyed by item code in `commodity_id` with its
  `species`: stocks of live animals, production of meat (carcass weight, with the bone), and
  cattle's milking cows and production of raw milk, as `heads` and `production_kg`
- `LIVESTOCK_PATTERNS_DATASET`: stocks in livestock units of the same five grazers, from Livestock
  Patterns; `get_species_to_livestock_units_per_head` recovers a country's coefficients from it
"""

import collections.abc
import csv
import dataclasses
import enum
import functools
import io
import logging
import math
import os
import tempfile
import zipfile

import iso3166
import pandas

from jdluc import config, storage, utils
from jdluc.datasets import base, worldbank_jurisdictions

logger = logging.getLogger(__name__)

BULK_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
)
BULK_MEMBER_NAME = "Production_Crops_Livestock_E_All_Data_(Normalized).csv"
LIVESTOCK_PATTERNS_BULK_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Environment_LivestockPatterns_E_All_Data_(Normalized).zip"
)
LIVESTOCK_PATTERNS_BULK_MEMBER_NAME = (
    "Environment_LivestockPatterns_E_All_Data_(Normalized).csv"
)
# Copied rather than taken from `trace`, which the datasets layer may not import.
KG_PER_TONNE = 1000

AREA_HARVESTED_ELEMENT_CODE = 5312
PRODUCTION_ELEMENT_CODE = 5510
STOCKS_ELEMENT_CODE = 5111
# Milking animals, in heads
MILK_ANIMALS_ELEMENT_CODE = 5318
# Livestock Patterns' stocks, in livestock units rather than heads
LIVESTOCK_UNITS_ELEMENT_CODE = 5118
# "Missing value; data cannot exist". Estimated and imputed figures are kept.
MISSING_FLAG = "M"
# FAOSTAT's crop items are exactly CPC division 01, quoted in the bulk (`'0112` is maize). It
# includes co-products such as cotton seed (329).
CROP_CPC_PREFIX = "'01"


class Species(enum.StrEnum):
    BUFFALO = enum.auto()
    CATTLE = enum.auto()
    GOAT = enum.auto()
    HORSE = enum.auto()
    SHEEP = enum.auto()


# Live animals in heads (e.g., "Cattle"): the standing herd at a reference date in the year, every
# age and purpose -- calves and dairy cows included. Not the animals slaughtered in the year, which
# FAOSTAT counts under the meat items. Meat over stocks is therefore carcass weight per standing
# head a year -- the share of the herd slaughtered times the carcass weight -- which assumes every
# head in a country yields alike in a given year, though breeding regions slaughter a smaller share
# than finishing ones.
SPECIES_TO_STOCKS_ITEM_CODE = {
    Species.BUFFALO: 946,
    Species.CATTLE: 866,
    Species.GOAT: 1016,
    Species.HORSE: 1096,
    Species.SHEEP: 976,
}
# Carcass weight in kg, bone in (e.g., "Meat of cattle with the bone, fresh or chilled"): the
# dressed carcass, without the offal, fat and hides FAOSTAT lists as items of their own. Not live
# weight or boneless meat, either of which would scale a comparison by a dressing or deboning yield.
SPECIES_TO_MEAT_ITEM_CODE = {
    Species.BUFFALO: 947,
    Species.CATTLE: 867,
    Species.GOAT: 1017,
    Species.HORSE: 1097,
    Species.SHEEP: 977,
}
assert (
    set(SPECIES_TO_STOCKS_ITEM_CODE) == set(SPECIES_TO_MEAT_ITEM_CODE) == set(Species)
)
# Raw milk, read for cattle only
SPECIES_TO_MILK_ITEM_CODE = {
    Species.CATTLE: 882,
}
LIVESTOCK_ITEM_CODE_TO_SPECIES = {
    item_code: species
    for species_to_item_code in (
        SPECIES_TO_STOCKS_ITEM_CODE,
        SPECIES_TO_MEAT_ITEM_CODE,
        SPECIES_TO_MILK_ITEM_CODE,
    )
    for species, item_code in species_to_item_code.items()
}


@dataclasses.dataclass(frozen=True)
class ElementColumn:
    """A FAOSTAT element carried as one column of a table."""

    column_name: str
    element_code: int
    # Multiplies a reported value into the column's unit
    scale: float


# An item's heads are its standing herd for live animals and the animals milked for milk; FAOSTAT
# reports one or the other, never both
LIVESTOCK_ELEMENT_COLUMNS = (
    ElementColumn(column_name="heads", element_code=MILK_ANIMALS_ELEMENT_CODE, scale=1),
    ElementColumn(column_name="heads", element_code=STOCKS_ELEMENT_CODE, scale=1),
    ElementColumn(
        column_name="production_kg",
        element_code=PRODUCTION_ELEMENT_CODE,
        scale=KG_PER_TONNE,
    ),
)
LIVESTOCK_UNITS_ELEMENT_COLUMNS = (
    ElementColumn(
        column_name="livestock_units",
        element_code=LIVESTOCK_UNITS_ELEMENT_CODE,
        scale=1,
    ),
)


def get_iso_3166(m49_code: str) -> str | None:
    """The ISO 3166-1 alpha-3 for a FAOSTAT M49 code, or None where it names no country.

    None is exactly FAOSTAT's aggregates ("World", "European Union (27)") and dissolved states
    ("USSR"). The code arrives quoted and zero-padded (`'004`); mainland China is 156, Taiwan 158.
    """
    numeric = m49_code.strip("'").zfill(3)
    if numeric in iso3166.countries_by_numeric:
        return iso3166.countries_by_numeric[numeric].alpha3
    else:
        return None


def iter_rows(
    member_name: str, path_to_zip: str
) -> collections.abc.Iterator[dict[str, str]]:
    """Every row of an archive's data member, streamed: Production's is 520 MiB unzipped."""
    with (
        zipfile.ZipFile(file=path_to_zip) as zf,
        zf.open(member_name) as member,
    ):
        yield from csv.DictReader(
            io.TextIOWrapper(member, encoding="utf8", errors="replace")
        )


def get_livestock_records(
    element_columns: tuple[ElementColumn, ...],
    rows: collections.abc.Iterable[dict[str, str]],
) -> list[dict[str, str | float]]:
    """One record per (country, livestock item, year) carrying every element, from an archive's
    rows. An element the item does not report reads NaN.
    """
    element_code_to_column = {
        element_column.element_code: element_column
        for element_column in element_columns
    }
    accumulated: dict[tuple[str, int, int], dict[str, float]] = {}
    iso_3166_to_jurisdiction_name: dict[str, str] = {}
    commodity_id_to_commodity_name: dict[int, str] = {}
    seen = 0

    for row in rows:
        seen += 1
        element_code = int(row["Element Code"])
        commodity_id = int(row["Item Code"])
        if (
            row["Flag"] != MISSING_FLAG
            and row["Value"]
            and element_code in element_code_to_column
            and commodity_id in LIVESTOCK_ITEM_CODE_TO_SPECIES
            and (iso_3166 := get_iso_3166(m49_code=row["Area Code (M49)"])) is not None
        ):
            element_column = element_code_to_column[element_code]
            key = (iso_3166, commodity_id, int(row["Year"]))
            if key not in accumulated:
                accumulated[key] = {}
            assert element_column.column_name not in accumulated[key], key
            accumulated[key][element_column.column_name] = (
                float(row["Value"]) * element_column.scale
            )
            iso_3166_to_jurisdiction_name[iso_3166] = row["Area"]
            commodity_id_to_commodity_name[commodity_id] = row["Item"]
    logger.info(f"Read {seen:d} rows; kept {len(accumulated):d} keys")

    records: list[dict[str, str | float]] = [
        {
            "admin_id": worldbank_jurisdictions.iso_3166_str(s=iso_3166),
            "admin_level": worldbank_jurisdictions.AdminLevel.NATIONAL.name,
            "commodity_id": commodity_id,
            "commodity_name": commodity_id_to_commodity_name[commodity_id],
            "jurisdiction_name": iso_3166_to_jurisdiction_name[iso_3166],
            "species": LIVESTOCK_ITEM_CODE_TO_SPECIES[commodity_id].name,
            "year": year,
            **dict.fromkeys(
                [element_column.column_name for element_column in element_columns],
                math.nan,
            ),
            **values,
        }
        for (iso_3166, commodity_id, year), values in sorted(accumulated.items())
    ]
    # NB: columns sorted by name, so the element columns sit among the others
    return [dict(sorted(record.items())) for record in records]


def get_crop_records(
    rows: collections.abc.Iterable[dict[str, str]],
) -> list[dict[str, str | float]]:
    """One record per (country, crop item, year), from the Production archive's rows.

    Production is required. Area harvested reads NaN where FAOSTAT reports none (Brazil nuts).
    """
    accumulated: dict[tuple[str, int, int], dict[int, float]] = {}
    iso_3166_to_jurisdiction_name: dict[str, str] = {}
    commodity_id_to_commodity_name: dict[int, str] = {}
    seen = 0

    for row in rows:
        seen += 1
        element_code = int(row["Element Code"])
        if (
            row["Item Code (CPC)"].startswith(CROP_CPC_PREFIX)
            and row["Flag"] != MISSING_FLAG
            and row["Value"]
            and element_code in (AREA_HARVESTED_ELEMENT_CODE, PRODUCTION_ELEMENT_CODE)
            and (iso_3166 := get_iso_3166(m49_code=row["Area Code (M49)"])) is not None
        ):
            commodity_id = int(row["Item Code"])
            key = (iso_3166, commodity_id, int(row["Year"]))
            if key not in accumulated:
                accumulated[key] = {AREA_HARVESTED_ELEMENT_CODE: math.nan}
            accumulated[key][element_code] = float(row["Value"])
            iso_3166_to_jurisdiction_name[iso_3166] = row["Area"]
            commodity_id_to_commodity_name[commodity_id] = row["Item"]
    logger.info(f"Read {seen:d} rows; kept {len(accumulated):d} keys")

    records: list[dict[str, str | float]] = [
        {
            "admin_id": worldbank_jurisdictions.iso_3166_str(s=iso_3166),
            "admin_level": worldbank_jurisdictions.AdminLevel.NATIONAL.name,
            "area_hectares": element_code_to_value[AREA_HARVESTED_ELEMENT_CODE],
            "commodity_id": commodity_id,
            "commodity_name": commodity_id_to_commodity_name[commodity_id],
            "jurisdiction_name": iso_3166_to_jurisdiction_name[iso_3166],
            "production_kg": element_code_to_value[PRODUCTION_ELEMENT_CODE]
            * KG_PER_TONNE,
            "year": year,
        }
        for (iso_3166, commodity_id, year), element_code_to_value in sorted(
            accumulated.items()
        )
        if PRODUCTION_ELEMENT_CODE in element_code_to_value
    ]
    logger.info(f"After requiring production: {len(records):d} records")
    return records


def get_crop_records_for_tile(tile_id: str) -> list[dict[str, str | float]]:
    with tempfile.TemporaryDirectory() as local_dir:
        path_to_zip = os.path.join(local_dir, "data.zip")
        utils.save_remote_url_to_local_path(
            local_path=path_to_zip, params={}, remote_url=BULK_URL
        )
        return get_crop_records(
            rows=iter_rows(member_name=BULK_MEMBER_NAME, path_to_zip=path_to_zip)
        )


def _get_records_for_tile(
    tile_id: str,
    bulk_url: str,
    element_columns: tuple[ElementColumn, ...],
    member_name: str,
) -> list[dict[str, str | float]]:
    with tempfile.TemporaryDirectory() as local_dir:
        path_to_zip = os.path.join(local_dir, "data.zip")
        utils.save_remote_url_to_local_path(
            local_path=path_to_zip, params={}, remote_url=bulk_url
        )
        return get_livestock_records(
            element_columns=element_columns,
            rows=iter_rows(member_name=member_name, path_to_zip=path_to_zip),
        )


LIVESTOCK_IDX_COLUMN_NAMES = [
    "admin_id",
    "admin_level",
    "commodity_id",
    "year",
]
CROP_IDX_COLUMN_NAMES = [
    "admin_level",
    "admin_id",
    "jurisdiction_name",
    "commodity_id",
    "year",
]

CROP_DATASET = base.TabularDataset(
    get_records_for_tile_id=get_crop_records_for_tile,
    idx_column_names=CROP_IDX_COLUMN_NAMES,
    product_name="production-crops",
    source_name="faostat",
    # v1: crop_name -> commodity_name
    # v2: coffee as COFF
    # v3: every crop item, keyed by commodity_id
    version="v3",
)
LIVESTOCK_DATASET = base.TabularDataset(
    get_records_for_tile_id=functools.partial(
        _get_records_for_tile,
        bulk_url=BULK_URL,
        element_columns=LIVESTOCK_ELEMENT_COLUMNS,
        member_name=BULK_MEMBER_NAME,
    ),
    idx_column_names=LIVESTOCK_IDX_COLUMN_NAMES,
    product_name="production-livestock",
    source_name="faostat",
    # v1: every item, keyed by commodity_id, with cattle's milk
    version="v1",
)
LIVESTOCK_PATTERNS_DATASET = base.TabularDataset(
    get_records_for_tile_id=functools.partial(
        _get_records_for_tile,
        bulk_url=LIVESTOCK_PATTERNS_BULK_URL,
        element_columns=LIVESTOCK_UNITS_ELEMENT_COLUMNS,
        member_name=LIVESTOCK_PATTERNS_BULK_MEMBER_NAME,
    ),
    idx_column_names=LIVESTOCK_IDX_COLUMN_NAMES,
    product_name="livestock-patterns",
    source_name="faostat",
    # v1: keyed by commodity_id, as livestock_units
    version="v1",
)


@utils.threadsafe_cache
def load(dataset: base.TabularDataset) -> pandas.DataFrame:
    uri = storage.join_uri(
        prefix=dataset.get_prefix(tile_id="world"),
        root=config.Config.from_dot_env().ingest_root,
    )
    logger.info(f"Loading {dataset.product_name:s} from {uri=:s}")
    return pandas.read_parquet(path=uri)


def get_species_to_livestock_units_per_head(
    iso_3166: str, livestock_units: pandas.DataFrame, stocks: pandas.DataFrame
) -> dict[Species, float]:
    """A country's livestock units per head of each grazer, as FAOSTAT counts them.

    Each is the median over the years of `LIVESTOCK_PATTERNS_DATASET`'s stocks in livestock units
    over `LIVESTOCK_DATASET`'s stocks in heads. FAO holds a region's coefficients fixed over time,
    but Livestock Patterns was computed from an older vintage of the heads, so single years drift
    off the coefficient wherever Production has since revised a herd; the median lands on it
    wherever it holds most of the years. A grazer FAOSTAT gives no livestock units, such as South
    America's buffalo, counts none.
    """
    key = ["admin_id", "commodity_id", "year"]
    merged = livestock_units.reset_index()[[*key, "livestock_units"]].merge(
        stocks.reset_index()[[*key, "heads", "species"]], how="inner", on=key
    )
    merged = merged[(merged["admin_id"] == iso_3166) & (merged["heads"] > 0)]
    species_name_to_units_per_head = (
        (merged["livestock_units"] / merged["heads"])
        .groupby(merged["species"])
        .median()
    )
    return {
        species: (
            float(species_name_to_units_per_head[species.name])
            if species.name in species_name_to_units_per_head.index
            else 0.0
        )
        for species in Species
    }
