"""Food and Agriculture Organization of the United Nations | FAOSTAT Production

license: CC-BY-4.0

year: 1961, ..., 2024

FAO. 2025. Production: Crops and livestock products. FAOSTAT. Rome.

https://www.fao.org/faostat/en/#data/QCL
https://bulks-faostat.fao.org/production/

# Methodology

- National statistics from annual member-country questionnaires, with FAO estimates and imputations
  where a country does not report
- Area harvested counts a field once per harvest, so a double-cropped field counts twice

One bulk archive, read into two tables:

- `CROP_DATASET`: area harvested and production of each MapSPAM crop FAOSTAT names one-for-one.
  Yield is not carried, since a MapSPAM group's yield is not its members'; area and production are
  additive, so a consumer sums the items it wants and divides
- `LIVESTOCK_DATASET`: stocks (live animals) and meat production (carcass weight, with the bone) of
  five grazers
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
from jdluc.datasets import base, ifpri_mapspam, worldbank_jurisdictions

logger = logging.getLogger(__name__)

BULK_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
)
BULK_MEMBER_NAME = "Production_Crops_Livestock_E_All_Data_(Normalized).csv"
# Copied rather than taken from `trace`, which the datasets layer may not import.
KG_PER_TONNE = 1000

AREA_HARVESTED_ELEMENT_CODE = 5312
PRODUCTION_ELEMENT_CODE = 5510
STOCKS_ELEMENT_CODE = 5111
# "Missing value; data cannot exist". Estimated and imputed figures are kept.
MISSING_FLAG = "M"


class Crop(enum.IntEnum):
    """The MapSPAM crops FAOSTAT reports one-for-one."""

    BANA = enum.auto()
    BARL = enum.auto()
    BEAN = enum.auto()
    CASS = enum.auto()
    CHIC = enum.auto()
    CNUT = enum.auto()
    COCO = enum.auto()
    COTT = enum.auto()
    COWP = enum.auto()
    GROU = enum.auto()
    LENT = enum.auto()
    MAIZ = enum.auto()
    OILP = enum.auto()
    PIGE = enum.auto()
    PLNT = enum.auto()
    POTA = enum.auto()
    RAPE = enum.auto()
    RICE = enum.auto()
    SESA = enum.auto()
    SORG = enum.auto()
    SOYB = enum.auto()
    SUGB = enum.auto()
    SUGC = enum.auto()
    SUNF = enum.auto()
    SWPO = enum.auto()
    TEAS = enum.auto()
    TOBA = enum.auto()
    WHEA = enum.auto()
    YAMS = enum.auto()


# Keyed on code, since FAOSTAT spells an item's name differently across its own files. COTT is
# seed cotton (328), not ginned lint (767), and OILP is fruit bunches (254), not palm oil (257),
# both as MapSPAM: either mistake would scale a comparison by the milling yield.
CROP_TO_ITEM_CODE = {
    Crop.BANA: 486,
    Crop.BARL: 44,
    Crop.BEAN: 176,
    Crop.CASS: 125,
    Crop.CHIC: 191,
    Crop.CNUT: 249,
    Crop.COCO: 661,
    Crop.COTT: 328,
    Crop.COWP: 195,
    Crop.GROU: 242,
    Crop.LENT: 201,
    Crop.MAIZ: 56,
    Crop.OILP: 254,
    Crop.PIGE: 197,
    Crop.PLNT: 489,
    Crop.POTA: 116,
    Crop.RAPE: 270,
    Crop.RICE: 27,
    Crop.SESA: 289,
    Crop.SORG: 83,
    Crop.SOYB: 236,
    Crop.SUGB: 157,
    Crop.SUGC: 156,
    Crop.SUNF: 267,
    Crop.SWPO: 122,
    Crop.TEAS: 667,
    Crop.TOBA: 826,
    Crop.WHEA: 15,
    Crop.YAMS: 137,
}


# One FAOSTAT item that MapSPAM splits in two, so neither half can take the figure.
SPLIT_CROP_NAMES = {
    ifpri_mapspam.CANONICAL_CROP_CLS.ACOF.name,  # both are 656, Coffee, green
    ifpri_mapspam.CANONICAL_CROP_CLS.RCOF.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.PMIL.name,  # both are 79, Millet
    ifpri_mapspam.CANONICAL_CROP_CLS.SMIL.name,
}
# MapSPAM groups that aggregate many FAOSTAT items, by a member list that is MapSPAM's.
SPAM_GROUP_CROP_NAMES = {
    ifpri_mapspam.CANONICAL_CROP_CLS.OCER.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.OFIB.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.OOIL.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.OPUL.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.ORTS.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.REST.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.TEMF.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.TROF.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.VEGE.name,
}
# Every canonical crop is mapped, split or grouped, so a MapSPAM rename cannot drop one silently.
assert {crop.name for crop in Crop} | SPLIT_CROP_NAMES | SPAM_GROUP_CROP_NAMES == {
    crop.name for crop in ifpri_mapspam.CANONICAL_CROP_CLS
}
assert not {crop.name for crop in Crop} & (SPLIT_CROP_NAMES | SPAM_GROUP_CROP_NAMES)
assert set(CROP_TO_ITEM_CODE) == set(Crop)
assert len(set(CROP_TO_ITEM_CODE.values())) == len(CROP_TO_ITEM_CODE)


class Species(enum.StrEnum):
    BUFFALO = enum.auto()
    CATTLE = enum.auto()
    GOAT = enum.auto()
    HORSE = enum.auto()
    SHEEP = enum.auto()


SPECIES_TO_STOCKS_ITEM_CODE = {
    Species.BUFFALO: 946,
    Species.CATTLE: 866,
    Species.GOAT: 1016,
    Species.HORSE: 1096,
    Species.SHEEP: 976,
}
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


@dataclasses.dataclass(frozen=True)
class ElementColumn:
    """A FAOSTAT element carried as one column of a table, for the items that table names."""

    column_name: str
    commodity_name_to_item_code: dict[str, int]
    required: bool
    # Multiplies a reported value into the column's unit
    scale: float


CROP_ELEMENT_COLUMNS = {
    AREA_HARVESTED_ELEMENT_CODE: ElementColumn(
        column_name="area_hectares",
        commodity_name_to_item_code={
            crop.name: item_code for crop, item_code in CROP_TO_ITEM_CODE.items()
        },
        required=True,
        scale=1,
    ),
    PRODUCTION_ELEMENT_CODE: ElementColumn(
        column_name="production_kg",
        commodity_name_to_item_code={
            crop.name: item_code for crop, item_code in CROP_TO_ITEM_CODE.items()
        },
        required=True,
        scale=KG_PER_TONNE,
    ),
}
LIVESTOCK_ELEMENT_COLUMNS = {
    STOCKS_ELEMENT_CODE: ElementColumn(
        column_name="stocks_head",
        commodity_name_to_item_code={
            species.name: item_code
            for species, item_code in SPECIES_TO_STOCKS_ITEM_CODE.items()
        },
        required=True,
        scale=1,
    ),
    PRODUCTION_ELEMENT_CODE: ElementColumn(
        column_name="production_kg",
        commodity_name_to_item_code={
            species.name: item_code
            for species, item_code in SPECIES_TO_MEAT_ITEM_CODE.items()
        },
        # NB: FAOSTAT reports stocks without meat for many country-species-years -- horses above
        # all, and India's cattle -- and those stocks are kept
        required=False,
        scale=KG_PER_TONNE,
    ),
}


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


def iter_rows(path_to_zip: str) -> collections.abc.Iterator[dict[str, str]]:
    """Every row of the archive's data member, streamed: it is 520 MiB unzipped."""
    with (
        zipfile.ZipFile(file=path_to_zip) as zf,
        zf.open(BULK_MEMBER_NAME) as member,
    ):
        yield from csv.DictReader(
            io.TextIOWrapper(member, encoding="utf8", errors="replace")
        )


def get_records_for_path(
    element_columns: dict[int, ElementColumn], path_to_zip: str
) -> list[dict[str, str | float]]:
    """One record per (country, commodity, year) carrying every element, from a local archive.

    Split from the retrieval so a caller already holding the archive parses it without fetching
    32 MiB again. The file is long format, so a key's elements are rows arbitrarily far apart and
    accumulate across the pass. A key missing a required element is dropped, and an optional one it
    lacks reads NaN.
    """
    element_code_to_item_code_to_commodity_name = {
        element_code: {
            item_code: commodity_name
            for commodity_name, item_code in element_column.commodity_name_to_item_code.items()
        }
        for element_code, element_column in element_columns.items()
    }
    accumulated: dict[tuple[str, str, int], dict[str, float]] = {}
    names: dict[str, str] = {}
    seen = 0

    for row in iter_rows(path_to_zip=path_to_zip):
        seen += 1
        element_code = int(row["Element Code"])
        item_code = int(row["Item Code"])
        if (
            row["Flag"] != MISSING_FLAG
            and row["Value"]
            and element_code in element_columns
            and item_code in element_code_to_item_code_to_commodity_name[element_code]
            and (iso_3166 := get_iso_3166(m49_code=row["Area Code (M49)"])) is not None
        ):
            element_column = element_columns[element_code]
            key = (
                iso_3166,
                element_code_to_item_code_to_commodity_name[element_code][item_code],
                int(row["Year"]),
            )
            if key not in accumulated:
                accumulated[key] = {}
            accumulated[key][element_column.column_name] = (
                float(row["Value"]) * element_column.scale
            )
            names[iso_3166] = row["Area"]
    logger.info(f"Read {seen:d} rows; kept {len(accumulated):d} keys")

    column_names = [
        element_column.column_name for element_column in element_columns.values()
    ]
    required_column_names = [
        element_column.column_name
        for element_column in element_columns.values()
        if element_column.required
    ]
    records: list[dict[str, str | float]] = [
        {
            "admin_id": worldbank_jurisdictions.iso_3166_str(s=iso_3166),
            "admin_level": worldbank_jurisdictions.AdminLevel.NATIONAL.name,
            "commodity_name": commodity_name,
            "jurisdiction_name": names[iso_3166],
            "year": year,
            # NB: keys follow column_names, since every key in values is one of them
            **dict.fromkeys(column_names, math.nan),
            **values,
        }
        for (iso_3166, commodity_name, year), values in sorted(accumulated.items())
        if all(column_name in values for column_name in required_column_names)
    ]
    logger.info(f"After requiring the required elements: {len(records):d} records")
    return records


def _get_records_for_tile(
    tile_id: str, element_columns: dict[int, ElementColumn]
) -> list[dict[str, str | float]]:
    with tempfile.TemporaryDirectory() as local_dir:
        path_to_zip = os.path.join(local_dir, "data.zip")
        utils.save_remote_url_to_local_path(
            local_path=path_to_zip, params={}, remote_url=BULK_URL
        )
        return get_records_for_path(
            element_columns=element_columns, path_to_zip=path_to_zip
        )


IDX_COLUMN_NAMES = [
    "admin_level",
    "admin_id",
    "jurisdiction_name",
    "commodity_name",
    "year",
]

CROP_DATASET = base.TabularDataset(
    get_records_for_tile_id=functools.partial(
        _get_records_for_tile, element_columns=CROP_ELEMENT_COLUMNS
    ),
    idx_column_names=IDX_COLUMN_NAMES,
    product_name="production-crops",
    source_name="faostat",
    # v1: crop_name -> commodity_name
    version="v1",
)
LIVESTOCK_DATASET = base.TabularDataset(
    get_records_for_tile_id=functools.partial(
        _get_records_for_tile, element_columns=LIVESTOCK_ELEMENT_COLUMNS
    ),
    idx_column_names=IDX_COLUMN_NAMES,
    product_name="production-livestock",
    source_name="faostat",
    version="v0",
)


def load(dataset: base.TabularDataset) -> pandas.DataFrame:
    uri = storage.join_uri(
        prefix=dataset.get_prefix(tile_id="world"),
        root=config.Config.from_dot_env().ingest_root,
    )
    logger.info(f"Loading {dataset.product_name:s} from {uri=:s}")
    return pandas.read_parquet(path=uri)
