"""Build the per-(jurisdiction, commodity) emissions-factor table.

Starting from the per-(admin, commodity) attribution rollup (`attribute.workflow`), determines
each commodity's total production — the only step that depends on methodology:
  - JURISDICTIONAL_DIRECT: production = crop area x NASS QuickStats yield, averaged over the 3
    years centered on the assessment year.
  - STATISTICAL: production = MAPSPAM production for crops, carried straight through the
    attribute, and the attribute's GPW heads x FAOSTAT carcass weight per head for livestock.
  - JURISDICTIONAL_DIRECT_PROXY: production = FAOSTAT national production over the 3 years
    centered on the assessment year, split among provinces by CROPGRIDS crop area for crops, and
    the attribute's 2020 GPW heads x FAOSTAT carcass weight per head over the same years for
    livestock.

It then derives, identically for all three, each commodity's yield (kg per hectare) and emissions
factor (kgCO2e per kg), and rolls the provincials up to national totals. Returns a cached
pandas.DataFrame indexed by (admin_level, admin_id, commodity_id, methodology), where
`commodity_id` is a FAOSTAT item code or a `taxonomy.InventedCommodityId`, and the `commodity_name`
column holds its name: FAOSTAT's, from its crop table, or `taxonomy`'s.

The countries come from the positional ISO 3166 alpha-3 codes, or -- with `--backfill` --
from every country in the World Bank admin-0 layer.

Example invocations:
  uv run python jdluc/trace.py --methodology-name STATISTICAL USA
  uv run python jdluc/trace.py --assessment-year 2024 --methodology-name STATISTICAL --backfill
"""

import argparse
import collections.abc
import logging
import typing

import numpy
import pandas

from jdluc import (
    attribute,
    emit,
    jurisdictional_direct_proxy,
    statistical,
    storage,
    taxonomy,
)
from jdluc.datasets import (
    faostat,
    gpw_livestock,
    tang_cropgrids,
    usda_nass_quickstats,
    worldbank_jurisdictions,
)

logger = logging.getLogger(__name__)


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
    emissions: pandas.DataFrame, raw_yields: pandas.DataFrame, years: tuple[int, ...]
) -> pandas.DataFrame:
    """A crop's production is its area times NASS's state yield, averaged over `years`."""
    yield_years = raw_yields.index.get_level_values("year")
    assert set(years) <= set(yield_years), f"NASS yields are missing a year of {years}"
    reduced_yields = (
        raw_yields[yield_years.isin(years)]
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


def get_herd_production_kg(
    heads: pandas.Series,
    species: gpw_livestock.Species,
    stocks: pandas.DataFrame,
    years: tuple[int, ...],
) -> numpy.ndarray:
    """Heads times FAOSTAT's national carcass weight per head: mean meat over mean stocks across
    `years`. No meat, or no herd in FAOSTAT, prices at 0.
    """
    livestock = stocks.reset_index()
    means = (
        livestock[livestock["year"].isin(years)]
        .groupby(["admin_id", "commodity_name"])[["production_kg", "stocks_head"]]
        .mean()
    )
    kg_per_head = (means["production_kg"] / means["stocks_head"]).fillna(0)
    iso_3166s = heads.index.get_level_values("admin_id").str[:3]
    return (
        heads.to_numpy()
        * kg_per_head.reindex(
            fill_value=0,
            index=pandas.MultiIndex.from_arrays(
                [iso_3166s, [species.name] * len(iso_3166s)]
            ),
        ).to_numpy()
    )


def derive_statistical_production_kg(
    emissions: pandas.DataFrame,
    snapshot_year_to_weight: dict[int, float],
    stocks: pandas.DataFrame,
) -> pandas.DataFrame:
    """A crop's production is MapSPAM's, carried through `attribute`. A herd's is its heads at
    each snapshot times FAOSTAT's national carcass weight per standing head that year, reduced
    over the window with the weights `statistical.get_discounted_snapshot_mean` gives a crop's.
    """
    ret = emissions.copy()
    ret["production_kg"] = ret["production_mt"] * KG_PER_TONNE
    for livestock, (species,) in statistical.LIVESTOCK_TO_SPECIES.items():
        is_herd = ret.index.get_level_values("commodity_name") == livestock.name
        ret.loc[is_herd, "production_kg"] = sum(
            weight
            * get_herd_production_kg(
                heads=ret.loc[is_herd, statistical.get_herd_heads_column(year=year)],
                species=species,
                stocks=stocks,
                years=(year,),
            )
            for year, weight in snapshot_year_to_weight.items()
        )
    return ret.drop(columns=["production_mt", *statistical.HERD_HEADS_COLUMNS])


def derive_proxy_production_kg(
    crops: pandas.DataFrame,
    emissions: pandas.DataFrame,
    stocks: pandas.DataFrame,
    years: tuple[int, ...],
) -> pandas.DataFrame:
    """A crop's production is FAOSTAT's national production of its item over `years`, split
    among provinces by CROPGRIDS crop area. A herd's is its heads times FAOSTAT's carcass weight per
    head. Rows with no FAOSTAT item have none.
    """
    commodity_name_to_commodity_id = (
        taxonomy.METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
            attribute.Methodology.JURISDICTIONAL_DIRECT_PROXY
        ]
    )
    crop_item_codes = {
        commodity_name_to_commodity_id[crop.name]
        for crop in tang_cropgrids.Crop
        if commodity_name_to_commodity_id[crop.name] > 0
    }
    table = crops.reset_index()
    # Every crop item is mapped or left out on purpose
    assert (
        set(table["commodity_id"])
        == crop_item_codes
        | set(taxonomy.CO_PRODUCT_ITEM_CODE_TO_ITEM_CODE)
        | taxonomy.UNMAPPED_CROP_ITEM_CODES
    )
    assert set(years) <= set(table["year"]), (
        f"FAOSTAT production is missing a year of {years}"
    )
    assert set(years) <= set(stocks.index.get_level_values("year")), (
        f"FAOSTAT stocks are missing a year of {years}"
    )
    national_production = (
        table[table["year"].isin(years)]
        .groupby(["admin_id", "commodity_id"])["production_kg"]
        .mean()
    )

    flat = emissions.reset_index()
    flat["commodity_id"] = [
        commodity_name_to_commodity_id[commodity_name]
        for commodity_name in flat["commodity_name"]
    ]
    flat["iso_3166"] = flat["admin_id"].str[:3]
    national_hectares = (
        flat.groupby(["iso_3166", "commodity_id"])["commodity_hectares"]
        .sum()
        .rename_axis(index=["admin_id", "commodity_id"])
    )
    row_keys = pandas.MultiIndex.from_frame(
        flat[["iso_3166", "commodity_id"]], names=["admin_id", "commodity_id"]
    )
    item_hectares = national_hectares.reindex(row_keys)
    flat["production_kg"] = (
        national_production.reindex(row_keys).to_numpy()
        * flat["commodity_hectares"].to_numpy()
        / item_hectares.where(item_hectares > 0).to_numpy()
    )

    # FAOSTAT production with no CROPGRIDS area to split it by
    traced = national_production[
        national_production.index.get_level_values("admin_id").isin(
            set(flat["iso_3166"])
        )
        & ~national_production.index.get_level_values("commodity_id").isin(
            list(taxonomy.CO_PRODUCT_ITEM_CODE_TO_ITEM_CODE)
        )
    ]
    unplaced = traced[
        national_hectares.reindex(traced.index, fill_value=0).to_numpy() == 0
    ]
    if unplaced.sum() > 0:
        logger.warning(
            f"{len(unplaced):d} FAOSTAT (country, item) production(s), "
            f"{unplaced.sum() / KG_PER_TONNE**2:.3f} kt in all, have no CROPGRIDS area to "
            f"split them by: {sorted(unplaced.index)}"
        )

    ret = flat.drop(columns=["commodity_id", "iso_3166"]).set_index(
        emissions.index.names
    )
    for livestock, (species,) in statistical.LIVESTOCK_TO_SPECIES.items():
        is_herd = ret.index.get_level_values("commodity_name") == livestock.name
        ret.loc[is_herd, "production_kg"] = get_herd_production_kg(
            heads=ret.loc[is_herd, jurisdictional_direct_proxy.HERD_HEADS_COLUMN],
            species=species,
            stocks=stocks,
            years=years,
        )
    return ret.drop(columns=jurisdictional_direct_proxy.HERD_HEADS_COLUMN)


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


@storage.cache_to_parquet(ignored_args=["concurrency"], version=0)
def workflow(
    assessment_year: int,
    commodity_names: tuple[str, ...],
    concurrency: int,
    iso_3166s: tuple[str, ...],
    methodology: attribute.Methodology,
) -> pandas.DataFrame:
    emissions = attribute.workflow(
        assessment_year=assessment_year,
        commodity_names=commodity_names,
        concurrency=concurrency,
        iso_3166s=iso_3166s,
        methodology=methodology,
    )
    assert emissions.index.names == [
        "admin_level",
        "admin_id",
        "commodity_name",
        "methodology",
    ]
    match methodology:
        case attribute.Methodology.JURISDICTIONAL_DIRECT:
            production = derive_jurisdictional_production_kg(
                emissions=emissions,
                raw_yields=usda_nass_quickstats.load(),
                years=tuple(range(assessment_year - 1, assessment_year + 2)),
            )
        case attribute.Methodology.JURISDICTIONAL_DIRECT_PROXY:
            production = derive_proxy_production_kg(
                crops=faostat.load(dataset=faostat.CROP_DATASET),
                emissions=emissions,
                stocks=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
                years=tuple(range(assessment_year - 1, assessment_year + 2)),
            )
        case attribute.Methodology.STATISTICAL:
            production = derive_statistical_production_kg(
                emissions=emissions,
                snapshot_year_to_weight=statistical.get_snapshot_year_to_weight(
                    assessment_year=assessment_year
                ),
                stocks=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
            )
        case _:
            raise ValueError(methodology)
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
        "--assessment-year",
        choices=emit.ASSESSMENT_YEARS,
        default=emit.DEFAULT_ASSESSMENT_YEAR,
        type=int,
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
        assessment_year=int(args.assessment_year),
        commodity_names=attribute.get_commodity_names(methodology=methodology),
        concurrency=int(args.concurrency),
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
