"""Ingest a named dataset over a set of ten-degree tiles.

The positionals are `dataset_name`, from `datasets.DatasetName`, then the ISO 3166 alpha-3
codes to cover. The codes resolve to tiles via
`worldbank_jurisdictions.get_ten_degree_tile_ids_for_iso_3166s`, the same derivation
`attribute.workflow` uses, so ingesting ahead of attribution covers precisely what attribution
will ask for. Whole-world datasets are ingested over the "world" tile based on their
partitioning, so the tile set only bounds tiled datasets.

The tile set comes from the positional ISO 3166 alpha-3 codes -- or, with `--backfill`,
from `tiling.GLOBAL_FOREST_WATCH_TILE_IDS`.

Example invocations:
  uv run python -m jdluc.ingest IPCC_CLIMATE_ZONES USA
  uv run python -m jdluc.ingest USDA_NASS_CDL USA MEX --concurrency=8
  uv run python -m jdluc.ingest GLAD_GLCLUC BRA --overwrite
  uv run python -m jdluc.ingest GLAD_GLCLUC --backfill
"""

import argparse
import collections.abc
import concurrent.futures
import logging

from jdluc import config, datasets, tiling
from jdluc.datasets import base, worldbank_jurisdictions

logger = logging.getLogger(__name__)


def workflow(
    concurrency: int,
    dataset: base.RasterDataset | base.TabularDataset | base.VectorDataset,
    overwrite: bool,
    root: str,
    tile_ids: collections.abc.Iterable[str],
) -> dict[str, str | Exception]:
    if dataset.partitioning == tiling.Partitioning.WHOLE_WORLD:
        tile_ids = (tiling.WHOLE_WORLD_TILE_ID,)
        logger.warning(f"Ingesting {dataset=} for {tile_ids=}")
    tile_id_is_valid_func = tiling.PARTITIONING_TO_IS_VALID_TILE_ID[
        dataset.partitioning
    ]
    if not all(map(tile_id_is_valid_func, tile_ids)):
        raise ValueError(f"{tile_ids=:} are not valid for {dataset=:}")
    results: dict[str, str | Exception] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[concurrent.futures.Future[str], str] = {}
        for tile_id in sorted(tile_ids):
            futures[
                executor.submit(
                    dataset.ingest_a_tile,
                    overwrite=overwrite,
                    root=root,
                    tile_id=tile_id,
                )
            ] = tile_id
        for future in concurrent.futures.as_completed(futures):
            tile_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                results[tile_id] = exc
            else:
                results[tile_id] = result
    return results


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_name", choices=sorted(e.name for e in datasets.DatasetName)
    )
    parser.add_argument(
        "iso_3166s",
        help="cover exactly the tiles these countries' boundaries touch",
        nargs=argparse.ZERO_OR_MORE,
        type=worldbank_jurisdictions.iso_3166_str,
    )
    parser.add_argument("--backfill", action="store_true", help="cover all GFW tiles")
    parser.add_argument("--concurrency", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    assert bool(args.iso_3166s) ^ bool(args.backfill), (
        "pass either one-or-more iso_3166s or --backfill"
    )

    dataset = datasets.NAME_TO_CLS[datasets.DatasetName[str(args.dataset_name)]]
    tile_id_to_path_or_exc: dict[str, str | Exception] = workflow(
        concurrency=int(args.concurrency),
        dataset=dataset,
        overwrite=args.overwrite,
        root=config.Config.from_dot_env().ingest_root,
        tile_ids=(
            tiling.GLOBAL_FOREST_WATCH_TILE_IDS
            if args.backfill
            else worldbank_jurisdictions.get_ten_degree_tile_ids_for_iso_3166s(
                iso_3166s=args.iso_3166s
            )
        ),
    )
    for tile_id, path_or_exc in tile_id_to_path_or_exc.items():
        print(tile_id, repr(path_or_exc))
    return sum(
        1 for result in tile_id_to_path_or_exc.values() if isinstance(result, Exception)
    )


if __name__ == "__main__":
    raise SystemExit(main())
