"""Build the per-(jurisdiction, commodity) emissions-factor table.

Starting from the per-(admin, commodity) attribution rollup (`attribute.workflow`), determines
each commodity's total production — the only step that depends on methodology:
  - JURISDICTIONAL_DIRECT: production = crop area x NASS QuickStats yield (4-year mean).
  - STATISTICAL: production = MAPSPAM production for crops, carried straight through the
    attribute, and the attribute's GPW heads x FAOSTAT carcass weight per head for livestock.

It then derives, identically for both, each commodity's yield (kg per hectare) and emissions
factor (kgCO2e per kg), and rolls the provincials up to national totals. Returns a cached
pandas.DataFrame indexed by (admin_level, admin_id, commodity_id, methodology), where
`commodity_id` is a FAOSTAT item code or a `taxonomy.InventedCommodityId`, and the `commodity_name`
column holds its name: FAOSTAT's, from its crop table, or `taxonomy`'s.

The countries come from the positional ISO 3166 alpha-3 codes, or -- with `--backfill` --
from every country in the World Bank admin-0 layer.

Example invocations:
  uv run python jdluc/trace.py --methodology-name STATISTICAL USA
  uv run python jdluc/trace.py --methodology-name STATISTICAL --backfill
"""

import argparse
import collections.abc
import logging
import typing

import pandas

from jdluc import attribute, emit, statistical, storage, taxonomy
from jdluc.datasets import faostat, usda_nass_quickstats, worldbank_jurisdictions

logger = logging.getLogger(__name__)


NASS_YIELD_YEARS = (2017, 2018, 2019, 2020)
KG_PER_TONNE = 1000
CANONICAL_KEY = ("admin_level", "admin_id", "commodity_id", "methodology")
ADDITIVE_COLUMNS = (
    "commodity_hectares",
    "emissions_mt",
    "peatland_commodity_hectares",
    "peatland_occupation_emissions_mt",
    "production_kg",
    *(component.column for component in emit.EmissionComponent),
)


def derive_jurisdictional_production_kg(
    emissions: pandas.DataFrame, raw_yields: pandas.DataFrame
) -> pandas.DataFrame:
    reduced_yields = (
        raw_yields[raw_yields.index.get_level_values("year").isin(NASS_YIELD_YEARS)]
        .groupby(level=["admin_id", "commodity_name"])["yield_kg_per_ha"]
        .mean()
        .reset_index()
    )
    merged = emissions.reset_index().merge(
        reduced_yields, how="left", on=["admin_id", "commodity_name"]
    )
    unmatched = int(merged["yield_kg_per_ha"].isna().sum())
    if unmatched:
        logger.warning(f"{unmatched:d} (admin, crop) row(s) had no matching NASS yield")
    merged["production_kg"] = merged["commodity_hectares"] * merged["yield_kg_per_ha"]
    return (
        merged.drop(columns="yield_kg_per_ha")
        .set_index(emissions.index.names)
        .sort_index()
    )


def derive_statistical_production_kg(
    emissions: pandas.DataFrame, stocks: pandas.DataFrame
) -> pandas.DataFrame:
    """A crop's production is MapSPAM's, carried through `attribute`. A herd's is its heads at
    each snapshot times FAOSTAT's national carcass weight per standing head that year, reduced
    over the window with the weights `statistical.get_discounted_snapshot_mean` gives a crop's.
    """
    livestock = stocks.reset_index()
    # No meat in a year, or no herd in FAOSTAT, prices at 0
    kg_per_head = (
        (livestock["production_kg"] / livestock["stocks_head"])
        .fillna(0)
        .set_axis(
            pandas.MultiIndex.from_frame(
                livestock[["admin_id", "commodity_name", "year"]]
            )
        )
    )
    ret = emissions.copy()
    ret["production_kg"] = ret["production_mt"] * KG_PER_TONNE
    for livestock_commodity, (species,) in statistical.LIVESTOCK_TO_SPECIES.items():
        is_herd = (
            ret.index.get_level_values("commodity_name") == livestock_commodity.name
        )
        iso_3166s = ret.index.get_level_values("admin_id")[is_herd].str[:3]
        ret.loc[is_herd, "production_kg"] = sum(
            weight
            * ret.loc[is_herd, statistical.get_herd_heads_column(year=year)]
            * kg_per_head.reindex(
                fill_value=0,
                index=pandas.MultiIndex.from_arrays(
                    [
                        iso_3166s,
                        [species.name] * len(iso_3166s),
                        [year] * len(iso_3166s),
                    ]
                ),
            ).to_numpy()
            for year, weight in statistical.SNAPSHOT_YEAR_TO_WEIGHT.items()
        )
    return ret.drop(columns=["production_mt", *statistical.HERD_HEADS_COLUMNS])


def with_commodity_ids(
    df: pandas.DataFrame, methodology: attribute.Methodology
) -> pandas.DataFrame:
    """Re-key rows named by the leg's own map onto `taxonomy`'s codes, adding together the rows that
    share a code, such as arabica and robusta coffee.
    """
    commodity_name_to_commodity_id = (
        taxonomy.METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[methodology]
    )
    flat = df.reset_index()
    flat["commodity_id"] = [
        int(commodity_name_to_commodity_id[commodity_name])
        for commodity_name in flat["commodity_name"]
    ]
    grouped = flat.drop(columns="commodity_name").groupby(by=list(CANONICAL_KEY))
    # min_count keeps a row with no production at all, such as one NASS has no yield for, at NaN
    return grouped.sum(min_count=1, numeric_only=True).assign(
        jurisdiction_name=grouped["jurisdiction_name"].first()
    )


def get_commodity_id_to_name(crops: pandas.DataFrame) -> dict[int, str]:
    """Every code's name: FAOSTAT's for its crop items, from its crop table, and `taxonomy`'s for
    the rest.
    """
    pairs = crops.reset_index()[["commodity_id", "commodity_name"]].drop_duplicates()
    assert pairs["commodity_id"].is_unique
    crop_id_to_name = dict(
        zip(
            map(int, pairs["commodity_id"]),
            map(str, pairs["commodity_name"]),
            strict=True,
        )
    )
    assert not set(crop_id_to_name) & set(taxonomy.COMMODITY_ID_TO_NAME)
    return crop_id_to_name | taxonomy.COMMODITY_ID_TO_NAME


def attach_ratios(df: pandas.DataFrame) -> pandas.DataFrame:
    df["yield_kg_per_ha"] = (
        df["production_kg"]
        .div(df["commodity_hectares"])
        .where(df["commodity_hectares"] > 0)
    )
    df["emissions_factor_kgco2e_per_kg"] = (
        (df["emissions_mt"] * KG_PER_TONNE)
        .div(df["production_kg"])
        .where(df["production_kg"] > 0)
    )
    return df


def iter_national_from_provincials(
    iso_3166: str,
    methodology: attribute.Methodology,
    national_name: str,
    provincials: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, str | float]]:
    for commodity_id, group in sorted(provincials.groupby(by="commodity_id")):

        def safe_div(numer: float, denom: float) -> float:
            return float("nan") if denom == 0 else numer / denom

        ret = {
            column_name: float(group[column_name].sum())
            for column_name in ADDITIVE_COLUMNS
        }
        yield ret | {
            "admin_level": worldbank_jurisdictions.AdminLevel.NATIONAL.name,
            "commodity_id": int(typing.cast(int, commodity_id)),
            "jurisdiction_name": national_name,
            "admin_id": iso_3166,
            "yield_kg_per_ha": safe_div(
                numer=ret["production_kg"], denom=ret["commodity_hectares"]
            ),
            "emissions_factor_kgco2e_per_kg": safe_div(
                numer=ret["emissions_mt"] * KG_PER_TONNE, denom=ret["production_kg"]
            ),
            "methodology": methodology.name,
        }  # type: ignore


@storage.cache_to_parquet(ignored_args=["concurrency"], version=2)
def workflow(
    concurrency: int,
    commodity_names: tuple[str, ...],
    iso_3166s: tuple[str, ...],
    methodology: attribute.Methodology,
) -> pandas.DataFrame:
    emissions = attribute.workflow(
        concurrency=concurrency,
        commodity_names=commodity_names,
        iso_3166s=iso_3166s,
        methodology=methodology,
    )
    assert emissions.index.names == [
        "admin_level",
        "admin_id",
        "commodity_name",
        "methodology",
    ]
    production = (
        derive_jurisdictional_production_kg(
            emissions=emissions, raw_yields=usda_nass_quickstats.load()
        )
        if methodology == attribute.Methodology.JURISDICTIONAL_DIRECT
        else derive_statistical_production_kg(
            emissions=emissions,
            stocks=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
        )
    )
    provincials = attach_ratios(
        df=with_commodity_ids(df=production, methodology=methodology)
    )
    assert all(
        admin_level_name == worldbank_jurisdictions.AdminLevel.PROVINCIAL.name
        for admin_level_name in map(
            str, provincials.index.get_level_values(level="admin_level")
        )
    )
    iter_national = (
        national
        for iso_3166 in iso_3166s
        for national in iter_national_from_provincials(
            iso_3166=iso_3166,
            methodology=methodology,
            national_name=str(
                worldbank_jurisdictions.get_jurisdiction_for_admin_level(
                    admin_level=worldbank_jurisdictions.AdminLevel.NATIONAL
                ).loc[iso_3166]["name"]
            ),
            provincials=provincials[
                provincials.index.get_level_values("admin_id").str.startswith(iso_3166)
            ],
        )
    )
    ret = pandas.concat(
        [
            provincials,
            pandas.DataFrame.from_records(iter_national).set_index(
                provincials.index.names
            ),
        ]
    )
    commodity_id_to_name = get_commodity_id_to_name(
        crops=faostat.load(dataset=faostat.CROP_DATASET)
    )
    ret["commodity_name"] = [
        commodity_id_to_name[int(commodity_id)]
        for commodity_id in ret.index.get_level_values("commodity_id")
    ]
    return ret


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "iso_3166s",
        nargs=argparse.ZERO_OR_MORE,
        type=worldbank_jurisdictions.iso_3166_str,
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="trace every country in the World Bank admin-0 layer",
    )
    parser.add_argument("--concurrency", default=8, type=int)
    parser.add_argument("--display-results", action="store_true")
    parser.add_argument(
        "--methodology-name",
        choices=sorted(e.name for e in attribute.Methodology),
        default=attribute.Methodology.STATISTICAL.name,
    )
    args = parser.parse_args()
    assert bool(args.iso_3166s) ^ bool(args.backfill), (
        "pass either one-or-more iso_3166s or --backfill"
    )

    methodology = attribute.Methodology[str(args.methodology_name)]
    df = workflow(
        concurrency=int(args.concurrency),
        commodity_names=attribute.get_commodity_names(methodology=methodology),
        iso_3166s=tuple(
            sorted(
                worldbank_jurisdictions.get_all_iso_3166s()
                if args.backfill
                else args.iso_3166s
            )
        ),
        methodology=methodology,
    )
    if args.display_results:
        print(df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
