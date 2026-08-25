"""Roll up per-pixel emissions to per-(jurisdiction, crop) totals.

Dispatches each ten-degree tile to a per-methodology workflow once per overlapping ISO
3166 country and sums the partials, since a jurisdiction may straddle a tile boundary:
  - STATISTICAL (default): downscales the per-pixel emissions to the MAPSPAM grid and
    attributes them across crops by MAPSPAM crop-expansion shares.
  - JURISDICTIONAL_DIRECT: masks the per-pixel emissions to each crop's CDL codes.
Both clip to provincial (World Bank admin-1) polygons, restrict to GLAD 2020 cropland
unless `--skip-glad-crop-filter` is set, and sum crop area, peatland crop area,
peatland-occupation emissions, and total emissions. Returns a pandas.DataFrame indexed by
(admin_level, admin_id, crop_name, methodology); the per-(country, tile) sub-workflows are
cached and `--concurrency` bounds how many tiles are in flight at once.

The countries come from the positional ISO 3166 alpha-3 codes, or -- with `--backfill` --
from every country in the World Bank admin-0 layer.

Example invocations:
  uv run python jdluc/attribute.py --methodology-name STATISTICAL USA
  uv run python jdluc/attribute.py --methodology-name STATISTICAL --backfill
"""

import argparse
import collections.abc
import concurrent.futures
import enum
import logging

import pandas

from jdluc import (
    jurisdictional_direct,
    statistical,
)
from jdluc.datasets import worldbank_jurisdictions

logger = logging.getLogger(__name__)


class Methodology(enum.IntEnum):
    JURISDICTIONAL_DIRECT = enum.auto()
    STATISTICAL = enum.auto()


def merge_dfs(*dfs: pandas.DataFrame) -> pandas.DataFrame:
    combined = pandas.concat(list(dfs))
    jurisdiction_name = (
        combined["jurisdiction_name"].groupby(level=combined.index.names).first()
    )
    return (
        combined.select_dtypes("number")
        .groupby(level=combined.index.names)
        .sum()
        .assign(jurisdiction_name=jurisdiction_name)
    )


def workflow(
    concurrency: int,
    crop_names: tuple[str, ...],
    iso_3166s: collections.abc.Iterable[str],
    methodology: Methodology,
    skip_glad_crop_filter: bool,
) -> pandas.DataFrame:
    workflow_for_tile = (
        jurisdictional_direct.workflow
        if methodology == Methodology.JURISDICTIONAL_DIRECT
        else statistical.workflow
    )

    # NB: have each thread handle a tile, to avoid races in harmonize, emit, or downscale
    tile_id_to_iso_3166s: dict[str, list[str]] = collections.defaultdict(list)
    for iso_3166 in sorted(iso_3166s):
        for tile_id in sorted(
            worldbank_jurisdictions.get_ten_degree_tile_ids_for_admin_id(
                admin_id=iso_3166,
                admin_level=worldbank_jurisdictions.AdminLevel.NATIONAL,
            )
        ):
            tile_id_to_iso_3166s[tile_id].append(iso_3166)

    def workflow_for_tile_id(tile_id: str) -> list[pandas.DataFrame]:
        return [
            workflow_for_tile(
                crop_names=crop_names,
                iso_3166=iso_3166,
                skip_glad_crop_filter=skip_glad_crop_filter,
                tile_id=tile_id,
            )
            for iso_3166 in tile_id_to_iso_3166s[tile_id]
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(workflow_for_tile_id, tile_id=tile_id)
            for tile_id in sorted(tile_id_to_iso_3166s)
        ]
        dfs = [df for future in futures for df in future.result()]

    # NB: ensure empty jurisdictions aren't dropped
    all_jurisdiction_crops = pandas.DataFrame.from_records(
        data=[
            {
                "admin_level": jurisdiction.level,
                "admin_id": jurisdiction.id,
                "crop_name": crop_name,
                "jurisdiction_name": jurisdiction.name,
            }
            for tile_id, iso_3166s_for_tile in sorted(tile_id_to_iso_3166s.items())
            for iso_3166 in iso_3166s_for_tile
            for jurisdiction in worldbank_jurisdictions.iter_jurisdiction_for_iso_3166_tile_id(
                admin_level=worldbank_jurisdictions.AdminLevel.PROVINCIAL,
                iso_3166=iso_3166,
                tile_id=tile_id,
            )
            for crop_name in crop_names
        ]
    ).set_index(keys=["admin_level", "admin_id", "crop_name"])

    ret = (
        merge_dfs(all_jurisdiction_crops, *dfs)
        .assign(methodology=methodology.name)
        .set_index("methodology", append=True)
        .sort_index()
    )
    assert not ret.index.duplicated().any()
    return ret


def get_crop_names(methodology: Methodology) -> tuple[str, ...]:
    if methodology == Methodology.STATISTICAL:
        return tuple(sorted(c.name for c in statistical.Crop))
    else:
        assert methodology == Methodology.JURISDICTIONAL_DIRECT
        return tuple(sorted(c.name for c in jurisdictional_direct.Crop))


DEFAULT_CONCURRENCY = 6


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
        help="attribute every country in the World Bank admin-0 layer",
    )
    parser.add_argument(
        "--concurrency",
        default=DEFAULT_CONCURRENCY,
        type=int,
    )
    parser.add_argument(
        "--methodology-name",
        choices=sorted(e.name for e in Methodology),
        default=Methodology.STATISTICAL.name,
    )
    parser.add_argument("--skip-display", action="store_true")
    parser.add_argument("--skip-glad-crop-filter", action="store_true")
    args = parser.parse_args()
    assert bool(args.iso_3166s) ^ bool(args.backfill), (
        "pass either one-or-more iso_3166s or --backfill"
    )

    methodology = Methodology[str(args.methodology_name)]
    df = workflow(
        concurrency=int(args.concurrency),
        crop_names=get_crop_names(methodology=methodology),
        iso_3166s=(
            worldbank_jurisdictions.get_all_iso_3166s()
            if args.backfill
            else args.iso_3166s
        ),
        methodology=methodology,
        skip_glad_crop_filter=args.skip_glad_crop_filter,
    )
    if not args.skip_display:
        print(df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
