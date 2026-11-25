"""Build the per-(jurisdiction, commodity) emissions-factor table.

Starting from the per-(admin, commodity) attribution rollup (`attribute.workflow`), determines
each commodity's total production — the only step that depends on methodology:
  - JURISDICTIONAL_DIRECT: production = crop area x NASS QuickStats yield, averaged over the 4
    years ending at the assessment year.
  - STATISTICAL and JURISDICTIONAL_DIRECT_PROXY: production = FAOSTAT national production over
    the same 4 years, split among provinces by the leg's crop area (MapSPAM 2020's or CROPGRIDS')
    for crops, and the attribute's 2020 GPW heads x FAOSTAT meat or milk per head over the same
    years for livestock.

It then derives, identically for all three, each commodity's yield (kg per hectare), land occupation
(m² for one year per kg, the reciprocal of yield) and emissions factor (kgCO2e per kg), and rolls
the provincials up to national totals. Returns a cached
pandas.DataFrame indexed by `CANONICAL_KEY`, (admin_id, admin_level, assessment_year, commodity_id,
methodology), where `commodity_id` is a FAOSTAT item code or a `taxonomy.InventedCommodityId`,
and the `commodity_name` column holds its name: FAOSTAT's, from its crop table, or `taxonomy`'s.

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
    usda_nass_quickstats,
    worldbank_jurisdictions,
)

logger = logging.getLogger(__name__)


KG_PER_TONNE = 1000
CANONICAL_KEY = (
    "admin_id",
    "admin_level",
    "assessment_year",
    "commodity_id",
    "methodology",
)
ADDITIVE_COLUMNS = (
    "commodity_hectares",
    "emissions_mt",
    "peatland_commodity_hectares",
    "peatland_occupation_emissions_mt",
    "production_kg",
    *(component.column for component in emit.EmissionComponent),
)
# Emissions factors left blank; the rows keep their other columns
WITHHELD_ISO_3166_COMMODITY_IDS = {
    # MapSPAM 2010 puts 29 times FAOSTAT's oil crop area on Taiwan
    ("TWN", taxonomy.InventedCommodityId.OTHER_OILCROPS),
}


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


def derive_production_kg(
    crops: pandas.DataFrame,
    emissions: pandas.DataFrame,
    heads_column: str,
    livestock: pandas.DataFrame,
    methodology: attribute.Methodology,
    years: tuple[int, ...],
) -> pandas.DataFrame:
    """A crop's production is FAOSTAT's national production of its items over `years`, split
    among provinces by the leg's crop area. A herd's is its heads times FAOSTAT's production of its
    item per head. Rows with no FAOSTAT item have none.
    """
    commodity_name_to_commodity_id = (
        taxonomy.METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[methodology]
    )
    item_code_to_commodity_id = {
        item_code: commodity_id
        for commodity_id, item_codes in taxonomy.METHODOLOGY_TO_CROP_ID_TO_ITEM_CODES[
            methodology
        ].items()
        for item_code in item_codes
    }
    table = crops.reset_index()
    assert set(years) <= set(table["year"]), (
        f"FAOSTAT production is missing a year of {years}"
    )
    assert set(years) <= set(livestock.index.get_level_values("year")), (
        f"FAOSTAT livestock is missing a year of {years}"
    )
    # Each item's mean over the years it reports, added up over the items a crop takes
    national_production = (
        table[
            table["year"].isin(years)
            & table["commodity_id"].isin(list(item_code_to_commodity_id))
        ]
        .groupby(["admin_id", "commodity_id"])["production_kg"]
        .mean()
        .rename(index=item_code_to_commodity_id, level="commodity_id")
        .groupby(level=["admin_id", "commodity_id"])
        .sum(min_count=1)
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

    # FAOSTAT production with no crop area to split it by
    traced = national_production[
        national_production.index.get_level_values("admin_id").isin(
            set(flat["iso_3166"])
        )
    ]
    unplaced = traced[
        national_hectares.reindex(traced.index, fill_value=0).to_numpy() == 0
    ]
    if unplaced.sum() > 0:
        logger.warning(
            f"{len(unplaced):d} FAOSTAT (country, crop) production(s), "
            f"{unplaced.sum() / KG_PER_TONNE**2:.3f} kt in all, have no crop area to "
            f"split them by: {sorted(unplaced.index)}"
        )
    # Crop area with no FAOSTAT production
    crop_hectares = national_hectares[
        national_hectares.index.get_level_values("commodity_id").isin(
            list(taxonomy.METHODOLOGY_TO_CROP_ID_TO_ITEM_CODES[methodology])
        )
        & (national_hectares > 0)
    ]
    unproduced = crop_hectares[
        national_production.reindex(crop_hectares.index).fillna(0).to_numpy() == 0
    ]
    if len(unproduced) > 0:
        logger.warning(
            f"{len(unproduced):d} (country, crop) area(s), {unproduced.sum():.0f} ha in all, "
            f"have no FAOSTAT production: {sorted(unproduced.index)}"
        )

    ret = flat.drop(columns=["commodity_id", "iso_3166"]).set_index(
        emissions.index.names
    )
    # A herd's output per head is its item's mean production over the herd's mean head count; it is
    # 0 where FAOSTAT has no production or no herd
    herds = livestock.reset_index()
    herds = herds[herds["year"].isin(years)]
    for commodity, species in gpw_livestock.LIVESTOCK_TO_SPECIES.items():
        heads = herds[
            herds["commodity_id"]
            == faostat.SPECIES_TO_LIVE_ANIMALS_ITEM_CODE[faostat.Species[species.name]]
        ]
        item = herds[
            herds["commodity_id"] == commodity_name_to_commodity_id[commodity.name]
        ]
        kg_per_head = (
            item.groupby("admin_id")["production_kg"].mean()
            / heads.groupby("admin_id")["heads"].mean()
        ).fillna(0)
        is_herd = ret.index.get_level_values("commodity_name") == commodity.name
        ret.loc[is_herd, "production_kg"] = (
            ret.loc[is_herd, heads_column].to_numpy()
            * kg_per_head.reindex(
                fill_value=0,
                index=ret.index[is_herd].get_level_values("admin_id").str[:3],
            ).to_numpy()
        )
    return ret.drop(columns=heads_column)


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


def get_commodity_id_to_name(
    crops: pandas.DataFrame, livestock: pandas.DataFrame
) -> dict[int, str]:
    """Every code's name: FAOSTAT's for its crop and livestock items, from its tables, and
    `taxonomy`'s for the rest.
    """
    pairs = pandas.concat(
        [
            table.reset_index()[["commodity_id", "commodity_name"]]
            for table in (crops, livestock)
        ]
    ).drop_duplicates()
    assert pairs["commodity_id"].is_unique
    item_id_to_name = dict(
        zip(
            map(int, pairs["commodity_id"]),
            map(str, pairs["commodity_name"]),
            strict=True,
        )
    )
    assert not set(item_id_to_name) & set(taxonomy.COMMODITY_ID_TO_NAME)
    return item_id_to_name | taxonomy.COMMODITY_ID_TO_NAME


def attach_ratios(df: pandas.DataFrame) -> pandas.DataFrame:
    df["yield_kg_per_ha"] = (
        df["production_kg"]
        .div(df["commodity_hectares"])
        .where(df["commodity_hectares"] > 0)
    )
    df["land_occupation_m2_year_per_kg"] = (
        emit.M2_PER_HECTARE / df["yield_kg_per_ha"]
    ).where(df["yield_kg_per_ha"] > 0)
    df["emissions_factor_kgco2e_per_kg"] = (
        (df["emissions_mt"] * KG_PER_TONNE)
        .div(df["production_kg"])
        .where(df["production_kg"] > 0)
    )
    return df


def with_withheld_factors(df: pandas.DataFrame) -> pandas.DataFrame:
    is_withheld = pandas.MultiIndex.from_arrays(
        [
            df.index.get_level_values("admin_id").str[:3],
            df.index.get_level_values("commodity_id"),
        ]
    ).isin(list(WITHHELD_ISO_3166_COMMODITY_IDS))
    return df.assign(
        emissions_factor_kgco2e_per_kg=df["emissions_factor_kgco2e_per_kg"].mask(
            cond=is_withheld
        )
    )


def iter_national_from_provincials(
    assessment_year: int,
    iso_3166: str,
    methodology: attribute.Methodology,
    national_name: str,
    provincials: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, str | float]]:
    for commodity_id, group in sorted(provincials.groupby(by="commodity_id")):

        def safe_div(numer: float, denom: float) -> float:
            return float("nan") if denom == 0 else numer / denom

        # NB: min_count keeps a column NaN, rather than 0, where every province lacks it
        ret = {
            column_name: float(group[column_name].sum(min_count=1))
            for column_name in ADDITIVE_COLUMNS
        }
        yield_kg_per_ha = safe_div(
            numer=ret["production_kg"], denom=ret["commodity_hectares"]
        )
        yield ret | {
            "admin_level": worldbank_jurisdictions.AdminLevel.NATIONAL.name,
            "commodity_id": int(typing.cast(int, commodity_id)),
            "jurisdiction_name": national_name,
            "admin_id": iso_3166,
            "yield_kg_per_ha": yield_kg_per_ha,
            "land_occupation_m2_year_per_kg": safe_div(
                numer=emit.M2_PER_HECTARE, denom=yield_kg_per_ha
            ),
            "emissions_factor_kgco2e_per_kg": safe_div(
                numer=ret["emissions_mt"] * KG_PER_TONNE, denom=ret["production_kg"]
            ),
            "methodology": methodology.name,
            "assessment_year": assessment_year,
        }  # type: ignore


@storage.cache_to_parquet(ignored_args=["concurrency"], version=2)
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
    assert emissions.index.names == list(attribute.INDEX_NAMES)
    production_years = emit.get_production_years_range(assessment_year=assessment_year)
    match methodology:
        case attribute.Methodology.JURISDICTIONAL_DIRECT:
            production = derive_jurisdictional_production_kg(
                emissions=emissions,
                raw_yields=usda_nass_quickstats.load(),
                years=production_years,
            )
        case attribute.Methodology.JURISDICTIONAL_DIRECT_PROXY:
            production = derive_production_kg(
                crops=faostat.load(dataset=faostat.CROP_DATASET),
                emissions=emissions,
                heads_column=jurisdictional_direct_proxy.HERD_HEADS_COLUMN,
                livestock=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
                methodology=methodology,
                years=production_years,
            )
        case attribute.Methodology.STATISTICAL:
            production = derive_production_kg(
                crops=faostat.load(dataset=faostat.CROP_DATASET),
                emissions=emissions,
                heads_column=statistical.HERD_HEADS_COLUMN,
                livestock=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
                methodology=methodology,
                years=production_years,
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
            assessment_year=assessment_year,
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
    ret = with_withheld_factors(
        df=pandas.concat(
            [
                provincials,
                pandas.DataFrame.from_records(iter_national).set_index(
                    provincials.index.names
                ),
            ]
        )
    )
    commodity_id_to_name = get_commodity_id_to_name(
        crops=faostat.load(dataset=faostat.CROP_DATASET),
        livestock=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
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
