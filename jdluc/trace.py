"""Build the per-(jurisdiction, crop) emissions-factor table.

Starting from the per-(admin, crop) attribution rollup (`attribute.workflow`), determines
each crop's total production — the only step that depends on methodology:
  - JURISDICTIONAL_DIRECT: production = crop area x NASS QuickStats yield (4-year mean).
  - STATISTICAL: production = MAPSPAM production, carried straight through the attribute.

It then derives, identically for both, each crop's emissions factor (kgCO2e per kg) and
peatland-occupation fraction, and rolls the provincials up to national totals. Returns a
pandas.DataFrame indexed by (admin level, crop, jurisdiction) which is cached.

The countries come from the positional ISO 3166 alpha-3 codes, or -- with `--backfill` --
from every country in the World Bank admin-0 layer.

Example invocations:
  uv run python jdluc/trace.py --methodology-name STATISTICAL USA
  uv run python jdluc/trace.py --methodology-name STATISTICAL --backfill
"""

import argparse
import collections.abc
import logging

import pandas

from jdluc import attribute, storage
from jdluc.datasets import usda_nass_quickstats, worldbank_jurisdictions

logger = logging.getLogger(__name__)


NASS_YIELD_YEARS = (2017, 2018, 2019, 2020)
KG_PER_TONNE = 1000
CANONICAL_KEY = ("admin_level", "admin_id", "crop_name", "methodology")
ADDITIVE_COLUMNS = (
    "crop_hectares",
    "forest_emissions_mt",
    "peatland_crop_hectares",
    "peatland_conversion_emissions_mt",
    "peatland_occupation_emissions_mt",
    "emissions_mt",
    "production_kg",
)


def derive_jurisdictional_production_kg(
    emissions: pandas.DataFrame, raw_yields: pandas.DataFrame
) -> pandas.DataFrame:
    reduced_yields = (
        raw_yields[raw_yields.index.get_level_values("year").isin(NASS_YIELD_YEARS)]
        .groupby(level=["admin_id", "crop_name"])["yield_kg_per_ha"]
        .mean()
        .reset_index()
    )
    merged = emissions.reset_index().merge(
        reduced_yields, how="left", on=["admin_id", "crop_name"]
    )
    unmatched = int(merged["yield_kg_per_ha"].isna().sum())
    if unmatched:
        logger.warning(f"{unmatched:d} (admin, crop) row(s) had no matching NASS yield")
    merged["production_kg"] = merged["crop_hectares"] * merged["yield_kg_per_ha"]
    return (
        merged.drop(columns="yield_kg_per_ha")
        .set_index(list(CANONICAL_KEY))
        .sort_index()
    )


def derive_statistical_production_kg(emissions: pandas.DataFrame) -> pandas.DataFrame:
    ret = emissions.copy()
    ret["production_kg"] = ret["production_mt"] * KG_PER_TONNE
    return ret.drop(columns="production_mt")


def attach_ratios(df: pandas.DataFrame) -> pandas.DataFrame:
    df["yield_kg_per_ha"] = (
        df["production_kg"].div(df["crop_hectares"]).where(df["crop_hectares"] > 0)
    )
    df["emissions_factor_kgco2e_per_kg"] = (
        (df["emissions_mt"] * KG_PER_TONNE)
        .div(df["production_kg"])
        .where(df["production_kg"] > 0)
    )
    df["peatland_occupation_fraction"] = (
        df["peatland_occupation_emissions_mt"]
        .div(df["emissions_mt"])
        .where(df["emissions_mt"] > 0)
    )
    return df


def iter_national_from_provincials(
    iso_3166: str,
    methodology: attribute.Methodology,
    national_name: str,
    provincials: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, str | float]]:
    for crop_name, group in sorted(provincials.groupby(by="crop_name")):

        def safe_div(numer: float, denom: float) -> float:
            return float("nan") if denom == 0 else numer / denom

        ret = {
            column_name: float(group[column_name].sum())
            for column_name in ADDITIVE_COLUMNS
        }
        yield ret | {
            "admin_level": worldbank_jurisdictions.AdminLevel.NATIONAL.name,
            "crop_name": str(crop_name),
            "jurisdiction_name": national_name,
            "admin_id": iso_3166,
            "yield_kg_per_ha": safe_div(
                numer=ret["production_kg"], denom=ret["crop_hectares"]
            ),
            "emissions_factor_kgco2e_per_kg": safe_div(
                numer=ret["emissions_mt"] * KG_PER_TONNE, denom=ret["production_kg"]
            ),
            "peatland_occupation_fraction": safe_div(
                numer=ret["peatland_occupation_emissions_mt"], denom=ret["emissions_mt"]
            ),
            "methodology": methodology.name,
        }  # type: ignore


@storage.cache_to_parquet(version=0)
def workflow(
    crop_names: tuple[str, ...],
    iso_3166s: tuple[str, ...],
    methodology: attribute.Methodology,
    skip_glad_crop_filter: bool,
) -> pandas.DataFrame:
    emissions = attribute.workflow(
        concurrency=attribute.DEFAULT_CONCURRENCY,
        crop_names=crop_names,
        iso_3166s=iso_3166s,
        methodology=methodology,
        skip_glad_crop_filter=skip_glad_crop_filter,
    )
    assert tuple(emissions.index.names) == CANONICAL_KEY
    emissions_and_yields = (
        derive_jurisdictional_production_kg(
            emissions=emissions, raw_yields=usda_nass_quickstats.load()
        )
        if methodology == attribute.Methodology.JURISDICTIONAL_DIRECT
        else derive_statistical_production_kg(emissions=emissions)
    )
    provincials = attach_ratios(df=emissions_and_yields)
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
    return pandas.concat(
        [
            provincials,
            pandas.DataFrame.from_records(iter_national).set_index(
                provincials.index.names
            ),
        ]
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--methodology-name",
        choices=sorted(e.name for e in attribute.Methodology),
        default=attribute.Methodology.STATISTICAL.name,
    )
    parser.add_argument("--skip-display", action="store_true")
    parser.add_argument("--skip-glad-crop-filter", action="store_true")
    args = parser.parse_args()
    assert bool(args.iso_3166s) ^ bool(args.backfill), (
        "pass either one-or-more iso_3166s or --backfill"
    )

    methodology = attribute.Methodology[str(args.methodology_name)]
    df = workflow(
        crop_names=attribute.get_crop_names(methodology=methodology),
        iso_3166s=tuple(
            sorted(
                worldbank_jurisdictions.get_all_iso_3166s()
                if args.backfill
                else args.iso_3166s
            )
        ),
        methodology=methodology,
        skip_glad_crop_filter=args.skip_glad_crop_filter,
    )
    if not args.skip_display:
        print(df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
