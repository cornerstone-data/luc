"""USDA National Agricultural Statistics Service | Quick Stats (survey)

license: public domain

year: 1866, ..., 2025

United States Department of Agriculture (USDA) National Agricultural Statistics Service (NASS), Quick Stats: USDA NASS, Washington, D.C.

https://quickstats.nass.usda.gov/
https://www.nass.usda.gov/Surveys/

# Methodology

- Tabular statistics from farmer/rancher questionnaires (NOT remote sensing)
- Sample-based: hundreds of surveys per year (e.g. Crop Production,
  Agricultural Prices, Grain Stocks, Cattle/Hog inventory, ARMS)
- Samples drawn from a list frame (known operations) plus an area frame
  (land segments), combined to cover operations missing from the list
- Collected by mail, phone (CATI), online (agcounts.usda.gov), and field
  enumeration
- Responses expanded to population estimates; final published values set by
  the Agricultural Statistics Board
- Only aggregated values published; small/identifying cells suppressed for
  confidentiality (Title 7 U.S.C., CIPSEA)
- API access: max 50,000 records/request (use bulk file downloads for more)
"""

import csv
import dataclasses
import enum
import io
import logging
import typing
import urllib.parse

import pandas

from jdluc import config, storage, utils
from jdluc.datasets import base, worldbank_jurisdictions

logger = logging.getLogger(__name__)

HA_PER_ACRE = 0.40468564
KG_PER_LB = 0.45359237


@dataclasses.dataclass(frozen=True, eq=False)
class Series:
    """The one NASS series that measures a crop's yield, and what a unit of it weighs.

    NASS publishes yield per acre in four units, and which one applies is a property of the
    commodity, so the request cannot pin a single one. `lb_per_unit` is the marketing bushel for
    the grains, from table 6 of USDA Agricultural Handbook 697 (ERS, June 1992): barley 48, shelled
    corn 56, sorghum grain 56, soybeans 60, wheat 60. Not the grading test weight of 7 CFR 810,
    which is a different quantity (barley 47.0, sorghum 57.0, and soybeans carries none at all).

    `commodity_desc` is carried only where NASS spells the commodity differently from the crop;
    `CropSeries.commodity_desc` falls back to the member's own name. `reported_fraction` is the
    share of the harvested crop the published yield weighs, 1 for everything but cotton, which NASS
    reports ginned.

    Compared by identity rather than by field, because two crops can share a unit and a weight --
    soybean and wheat do, as do maize and sorghum -- and under structural equality the second of
    each pair would become an alias of the first inside `CropSeries`, silently dropping it from
    iteration.
    """

    unit_desc: str
    lb_per_unit: float
    commodity_desc: str | None = None
    class_descs: tuple[str, ...] = ("ALL CLASSES",)
    util_practice_desc: str = "ALL UTILIZATION PRACTICES"
    reported_fraction: float = 1.0


@enum.unique
class CropSeries(enum.Enum):
    """Which NASS series measures each `jurisdictional_direct.Crop`.

    Members are named for the jdLUC crop rather than the NASS commodity, so the ingested table
    carries crop names the rest of the pipeline already speaks and needs no crosswalk: NASS
    `BEANS` is `BEAN`, `CORN` is `MAIZE`, `POTATOES` is `POTATO`.
    """

    BARLEY = Series(unit_desc="BU / ACRE", lb_per_unit=48)
    # NASS renamed the rolled-up dry-bean class in 2019 and publishes no ALL CLASSES to fall back
    # on, so spanning a multi-year window takes both names. They are not the same quantity: the
    # earlier one includes chickpeas, which the CDL scores separately as CHICK_PEAS.
    BEAN = Series(
        commodity_desc="BEANS",
        unit_desc="LB / ACRE",
        lb_per_unit=1,
        class_descs=("DRY EDIBLE, INCL CHICKPEAS", "DRY EDIBLE, (EXCL CHICKPEAS)"),
    )
    # NASS reports ginned lint, where MapSPAM, FAOSTAT and WRI all carry seed cotton -- see the
    # ItemCode docstring in `faostat_production`. 0.36 is what FAOSTAT's US seed cotton implies
    # against this series over 2011-2020 (mean 0.361), and agrees with the ~35% gin turnout.
    # FAOSTAT implies ~0.40 before 2011, so revisit this if the trace window moves earlier.
    COTTON = Series(unit_desc="LB / ACRE", lb_per_unit=1, reported_fraction=0.36)
    MAIZE = Series(
        commodity_desc="CORN",
        unit_desc="BU / ACRE",
        lb_per_unit=56,
        util_practice_desc="GRAIN",
    )
    POTATO = Series(commodity_desc="POTATOES", unit_desc="CWT / ACRE", lb_per_unit=100)
    RICE = Series(unit_desc="LB / ACRE", lb_per_unit=1)
    SORGHUM = Series(unit_desc="BU / ACRE", lb_per_unit=56, util_practice_desc="GRAIN")
    SOYBEAN = Series(commodity_desc="SOYBEANS", unit_desc="BU / ACRE", lb_per_unit=60)
    SUGARBEET = Series(
        commodity_desc="SUGARBEETS", unit_desc="TONS / ACRE", lb_per_unit=2000
    )
    # Sugarcane publishes no ALL UTILIZATION PRACTICES row at all; SUGAR & SEED is its total.
    SUGARCANE = Series(
        unit_desc="TONS / ACRE", lb_per_unit=2000, util_practice_desc="SUGAR & SEED"
    )
    WHEAT = Series(unit_desc="BU / ACRE", lb_per_unit=60)

    @property
    def commodity_desc(self) -> str:
        """What NASS calls this crop: its own name, unless its series overrides it."""
        return self.value.commodity_desc or self.name


STATE_FIPS_TO_ADMIN_ID = {
    1: "USA001",  # Alabama
    2: "USA002",  # Alaska
    4: "USA003",  # Arizona
    5: "USA004",  # Arkansas
    6: "USA005",  # California
    8: "USA006",  # Colorado
    9: "USA007",  # Connecticut
    10: "USA008",  # Delaware
    11: "USA009",  # District of Columbia
    12: "USA010",  # Florida
    13: "USA011",  # Georgia
    15: "USA012",  # Hawaii
    16: "USA013",  # Idaho
    17: "USA014",  # Illinois
    18: "USA015",  # Indiana
    19: "USA016",  # Iowa
    20: "USA017",  # Kansas
    21: "USA018",  # Kentucky
    22: "USA019",  # Louisiana
    23: "USA020",  # Maine
    24: "USA021",  # Maryland
    25: "USA022",  # Massachusetts
    26: "USA023",  # Michigan
    27: "USA024",  # Minnesota
    28: "USA025",  # Mississippi
    29: "USA026",  # Missouri
    30: "USA027",  # Montana
    31: "USA028",  # Nebraska
    32: "USA029",  # Nevada
    33: "USA030",  # New Hampshire
    34: "USA031",  # New Jersey
    35: "USA032",  # New Mexico
    36: "USA033",  # New York
    37: "USA034",  # North Carolina
    38: "USA035",  # North Dakota
    39: "USA036",  # Ohio
    40: "USA037",  # Oklahoma
    41: "USA038",  # Oregon
    42: "USA039",  # Pennsylvania
    44: "USA040",  # Rhode Island
    45: "USA041",  # South Carolina
    46: "USA042",  # South Dakota
    47: "USA043",  # Tennessee
    48: "USA044",  # Texas
    49: "USA045",  # Utah
    50: "USA046",  # Vermont
    51: "USA047",  # Virginia
    53: "USA048",  # Washington
    54: "USA049",  # West Virginia
    55: "USA050",  # Wisconsin
    56: "USA051",  # Wyoming
}


@dataclasses.dataclass
class Yield:
    admin_id: str
    admin_level: str
    crop_name: str
    jurisdiction_name: str
    year: int
    yield_kg_per_ha: float

    @classmethod
    def from_dict(
        cls, crop_series: CropSeries, d: dict[str, float | str]
    ) -> typing.Self:
        series = crop_series.value
        per_acre = float(str(d["Value"]).replace(",", ""))
        return cls(
            admin_id=STATE_FIPS_TO_ADMIN_ID[int(d["state_fips_code"])],
            admin_level=worldbank_jurisdictions.AdminLevel.PROVINCIAL.name,
            crop_name=crop_series.name,
            jurisdiction_name=str(d["state_name"]),
            year=int(d["year"]),
            yield_kg_per_ha=(
                per_acre
                * series.lb_per_unit
                * KG_PER_LB
                / HA_PER_ACRE
                / series.reported_fraction
            ),
        )


def get_yield_dicts_from_api(
    api_key: str, crop_series: CropSeries
) -> list[dict[str, float | str]]:
    series = crop_series.value
    param_tuples = (
        ("key", api_key),
        ("source_desc", "SURVEY"),
        ("sector_desc", "CROPS"),
        ("statisticcat_desc", "YIELD"),
        ("agg_level_desc", "STATE"),
        ("unit_desc", series.unit_desc),
        ("freq_desc", "ANNUAL"),
        ("reference_period_desc", "YEAR"),
        *(("class_desc", c) for c in series.class_descs),
        ("prodn_practice_desc", "ALL PRODUCTION PRACTICES"),
        ("commodity_desc", crop_series.commodity_desc),
        ("util_practice_desc", series.util_practice_desc),
        ("format", "CSV"),
    )
    with utils.get_requests_session().request(
        method="GET",
        params=urllib.parse.urlencode(param_tuples),
        url="https://quickstats.nass.usda.gov/api/api_GET/",
    ) as response:
        response.raise_for_status()
        body = response.text

    return list(csv.DictReader(io.StringIO(body)))


def _get_records_for_tile(tile_id: str) -> list[dict[str, str | float]]:
    api_key = config.Config.from_dot_env().usda_nass_api_key
    yields = [
        Yield.from_dict(crop_series=crop_series, d=yield_dict)
        for crop_series in CropSeries
        for yield_dict in get_yield_dicts_from_api(
            api_key=api_key, crop_series=crop_series
        )
        if int(yield_dict["state_fips_code"]) in STATE_FIPS_TO_ADMIN_ID
        if str(yield_dict["Value"]).strip() not in {"(D)", "(Z)", "(S)", "(NA)"}
    ]
    logger.info(
        f"After filtering: {len(yields):d} yields over {len(CropSeries):d} crops"
    )

    return list(map(dataclasses.asdict, yields))


DATASET = base.TabularDataset(
    get_records_for_tile_id=_get_records_for_tile,
    idx_column_names=[
        "admin_level",
        "admin_id",
        "jurisdiction_name",
        "crop_name",
        "year",
    ],
    product_name="quickstats",
    source_name="usda-nass",
    # 2025b: expand from three to eleven crops
    version="2025b",
)


def load() -> pandas.DataFrame:
    uri = storage.join_uri(
        root=config.Config.from_dot_env().ingest_root,
        prefix=DATASET.get_prefix(tile_id="world"),
    )
    logger.info(f"Loading yields from {uri=:s}")
    return pandas.read_parquet(path=uri)
