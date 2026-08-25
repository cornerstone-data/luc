"""Food and Agriculture Organization of the United Nations | FAOSTAT Production

license: CC-BY-4.0

year: 1961, ..., 2024

FAO. 2025. Production: Crops and livestock products. FAOSTAT. Rome.

https://www.fao.org/faostat/en/#data/QCL
https://bulks-faostat.fao.org/production/

# Methodology

- Tabular national statistics (NOT remote sensing) from annual member-country questionnaires,
  with FAO estimating or imputing where a country does not report
- National grain only; provincial production comes from `usda_nass_quickstats` for the US and
  from MapSPAM's surfaces elsewhere
- Area harvested counts a field once per harvest, so a doubly-cropped field is counted twice and
  the total exceeds physical cropland extent

Area harvested and production are carried; FAOSTAT's yield column is not. MapSPAM's group crops
each aggregate several FAOSTAT items, and a group's yield is not the sum of its constituents', so a
stored yield would be right for the one-to-one crops and quietly wrong for the rest. Area and
production are additive, so a consumer sums the items it wants and divides.
"""

import collections.abc
import csv
import dataclasses
import enum
import io
import logging
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

# The two additive elements out of the file's twenty; the rest are livestock, a fifth of its 4.2
# million rows.
AREA_HARVESTED_ELEMENT_CODE = 5312
PRODUCTION_ELEMENT_CODE = 5510
# "Missing value; data cannot exist". Official, estimated, imputed and external figures are all
# kept: dropping the estimated ones would thin the countries that report least.
MISSING_FLAG = "M"


class ItemCode(enum.IntEnum):
    """The FAOSTAT item each MapSPAM crop corresponds to one-for-one.

    Keyed on item code because the name is not a stable join key: FAOSTAT spells the same item
    "Cassava, fresh" in its data file and "Cassava; fresh" in its own code table.

    COTT is seed cotton (328) not ginned lint (767), and OILP is fruit bunches (254) not palm oil
    (257), both matching MapSPAM. Either mistake would scale a comparison by the milling yield.
    """

    BANA = 486  # Bananas
    BARL = 44  # Barley
    BEAN = 176  # Beans, dry
    CASS = 125  # Cassava, fresh
    CHIC = 191  # Chick peas, dry
    CNUT = 249  # Coconuts, in shell
    COCO = 661  # Cocoa beans
    COTT = 328  # Seed cotton, unginned
    COWP = 195  # Cow peas, dry
    GROU = 242  # Groundnuts, excluding shelled
    LENT = 201  # Lentils, dry
    MAIZ = 56  # Maize (corn)
    OILP = 254  # Oil palm fruit
    PIGE = 197  # Pigeon peas, dry
    PLNT = 489  # Plantains and cooking bananas
    POTA = 116  # Potatoes
    RAPE = 270  # Rape or colza seed
    RICE = 27  # Rice
    SESA = 289  # Sesame seed
    SORG = 83  # Sorghum
    SOYB = 236  # Soya beans
    SUGB = 157  # Sugar beet
    SUGC = 156  # Sugar cane
    SUNF = 267  # Sunflower seed
    SWPO = 122  # Sweet potatoes
    TEAS = 667  # Tea leaves
    TOBA = 826  # Unmanufactured tobacco
    WHEA = 15  # Wheat
    YAMS = 137  # Yams


# MapSPAM splits one FAOSTAT item in two, so giving either name the figure invents the split and
# giving both it doubles the total.
SPLIT_CROP_NAMES = {
    ifpri_mapspam.CANONICAL_CROP_CLS.ACOF.name,  # both are 656, Coffee, green
    ifpri_mapspam.CANONICAL_CROP_CLS.RCOF.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.PMIL.name,  # both are 79, Millet
    ifpri_mapspam.CANONICAL_CROP_CLS.SMIL.name,
}
# MapSPAM aggregates many FAOSTAT items under one name, and the member list is MapSPAM's to define.
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
assert {item.name for item in ItemCode} | SPLIT_CROP_NAMES | SPAM_GROUP_CROP_NAMES == {
    crop.name for crop in ifpri_mapspam.CANONICAL_CROP_CLS
}
assert not {item.name for item in ItemCode} & (SPLIT_CROP_NAMES | SPAM_GROUP_CROP_NAMES)
assert len({item.value for item in ItemCode}) == len(ItemCode)

ITEM_CODE_TO_CROP_NAME = {int(item.value): item.name for item in ItemCode}


def get_iso_3166(m49_code: str) -> str | None:
    """The ISO 3166-1 alpha-3 for a FAOSTAT M49 code, or None where it names no country.

    M49 is ISO 3166-1 numeric for countries, so this doubles as the aggregate filter: 202 of
    FAOSTAT's 244 areas resolve, and the 42 that do not are exactly its aggregates ("World",
    "European Union (27)") and dissolved states ("USSR", "Czechoslovakia").

    FAOSTAT quotes and zero-pads the code (`'004`). Its own "China" aggregate (159) is among the
    codes outside the standard, so mainland arrives as 156 and Taiwan separately as 158.
    """
    country = iso3166.countries_by_numeric.get(m49_code.strip("'").zfill(3))
    return None if country is None else country.alpha3


@dataclasses.dataclass
class Production:
    admin_id: str
    admin_level: str
    area_hectares: float
    crop_name: str
    jurisdiction_name: str
    production_kg: float
    year: int


def iter_rows(path_to_zip: str) -> collections.abc.Iterator[dict[str, str]]:
    """Every row of the archive's data member, streamed: it is 520 MiB unzipped."""
    with (
        zipfile.ZipFile(file=path_to_zip) as zf,
        zf.open(BULK_MEMBER_NAME) as member,
    ):
        yield from csv.DictReader(
            io.TextIOWrapper(member, encoding="utf8", errors="replace")
        )


def get_records_for_path(path_to_zip: str) -> list[dict[str, str | float]]:
    """Area harvested and production per (country, MapSPAM crop, year), from a local archive.

    Split from the retrieval so a caller already holding the archive parses it instead of fetching
    32 MiB again -- a test with a fixture, or a validation run reading its own digest-pinned copy.

    Accumulated across one pass because the file is long format: a country-crop-year's area and
    production are two rows, arbitrarily far apart. A key missing either is dropped, since a
    yield taken from one of them alone would be wrong rather than partial.
    """
    element_codes = {AREA_HARVESTED_ELEMENT_CODE, PRODUCTION_ELEMENT_CODE}
    accumulated: dict[tuple[str, str, int], dict[str, float]] = {}
    names: dict[str, str] = {}
    seen = 0

    for row in iter_rows(path_to_zip=path_to_zip):
        seen += 1
        if row["Flag"] == MISSING_FLAG or not row["Value"]:
            continue
        element_code = int(row["Element Code"])
        if element_code not in element_codes:
            continue
        crop_name = ITEM_CODE_TO_CROP_NAME.get(int(row["Item Code"]))
        if crop_name is None:
            continue
        iso_3166 = get_iso_3166(m49_code=row["Area Code (M49)"])
        if iso_3166 is None:
            continue
        names[iso_3166] = row["Area"]
        values = accumulated.setdefault((iso_3166, crop_name, int(row["Year"])), {})
        if element_code == AREA_HARVESTED_ELEMENT_CODE:
            values["area_hectares"] = float(row["Value"])
        else:
            values["production_kg"] = float(row["Value"]) * KG_PER_TONNE
    logger.info(f"Read {seen:d} rows; kept {len(accumulated):d} country-crop-years")

    productions = [
        Production(
            admin_id=worldbank_jurisdictions.iso_3166_str(iso_3166),
            admin_level=worldbank_jurisdictions.AdminLevel.NATIONAL.name,
            area_hectares=float(values["area_hectares"]),
            crop_name=crop_name,
            jurisdiction_name=names[iso_3166],
            production_kg=float(values["production_kg"]),
            year=year,
        )
        for (iso_3166, crop_name, year), values in sorted(accumulated.items())
        if "area_hectares" in values
        if "production_kg" in values
    ]
    logger.info(f"After requiring both elements: {len(productions):d} records")

    return list(map(dataclasses.asdict, productions))


def get_records_for_tile(tile_id: str) -> list[dict[str, str | float]]:
    """Retrieve the bulk archive and read it."""
    del tile_id  # whole-world partitioning, so there is only ever one
    with tempfile.TemporaryDirectory() as local_dir:
        path_to_zip = os.path.join(local_dir, "data.zip")
        utils.save_remote_url_to_local_path(
            local_path=path_to_zip, params={}, remote_url=BULK_URL
        )
        return get_records_for_path(path_to_zip=path_to_zip)


DATASET = base.TabularDataset(
    get_records_for_tile_id=get_records_for_tile,
    idx_column_names=[
        "admin_level",
        "admin_id",
        "jurisdiction_name",
        "crop_name",
        "year",
    ],
    product_name="production-crops",
    source_name="faostat",
    version="v0",
)


def load() -> pandas.DataFrame:
    uri = storage.join_uri(
        root=config.Config.from_dot_env().ingest_root,
        prefix=DATASET.get_prefix(tile_id="world"),
    )
    logger.info(f"Loading production from {uri=:s}")
    return pandas.read_parquet(path=uri)
