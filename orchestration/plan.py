"""What a run consists of, as the JSON lists the workflow loops over.

The dataset lists are a hand-kept claim about what jdluc reads: no jdluc constant declares it
all. A laptop run ingests lazily, from inside harmonize, so it never needs them. A workflow
ingests up front instead, so that compute runs with ``skip_ingest`` and a missing dataset fails
at ingest rather than surfacing as an all-nodata band (``ignore_missing_tiles``). In particular
``faostat.load`` and ``usda_nass_quickstats.load`` read the ingest root directly and never
ingest, so those tables appear here by name. When jdluc starts reading a new dataset, add it
here.
"""

import collections.abc
import json
import logging
import os

from jdluc import (
    attribute,
    config,
    harmonize,
    jurisdictional_direct,
    statistical,
    tiling,
)
from jdluc.datasets import NAME_TO_CLS, DatasetName, worldbank_jurisdictions

logger = logging.getLogger(__name__)

# Read by every tile lookup and every per-jurisdiction rollup, but named by no workflow
BOUNDARY_DATASET_NAMES = (
    DatasetName.WORLD_BANK_ADMIN_0,
    DatasetName.WORLD_BANK_ADMIN_1,
)

# Read with `load()` straight from the ingest root, never ingested on demand
METHODOLOGY_TO_LOADED_DATASET_NAMES = {
    attribute.Methodology.JURISDICTIONAL_DIRECT: (DatasetName.USDA_NASS_QUICKSTATS,),
    attribute.Methodology.STATISTICAL: (
        DatasetName.FAOSTAT_LIVESTOCK_PATTERNS,
        DatasetName.FAOSTAT_PRODUCTION_LIVESTOCK,
    ),
}

METHODOLOGY_TO_HARMONIZED_DATASET_NAMES = {
    attribute.Methodology.JURISDICTIONAL_DIRECT: jurisdictional_direct.DATASET_NAMES,
    attribute.Methodology.STATISTICAL: statistical.DATASET_NAMES,
}


# The scope of a run over every country and GNW tile, as jdluc's `--backfill`; also what its
# run-wide outputs are named, in place of every country's code joined
WORLD = "world"


def get_dataset_names(
    methodology: attribute.Methodology, partitioning: tiling.Partitioning
) -> list[DatasetName]:
    """Every dataset of one partitioning that a run of ``methodology`` reads."""
    dataset_names = {
        *BOUNDARY_DATASET_NAMES,
        *(name for stack in harmonize.Stack for name in stack.value),
        *METHODOLOGY_TO_HARMONIZED_DATASET_NAMES[methodology],
        *METHODOLOGY_TO_LOADED_DATASET_NAMES[methodology],
    }
    return sorted(
        (
            name
            for name in dataset_names
            if NAME_TO_CLS[name].partitioning == partitioning
        ),
        key=lambda name: name.name,
    )


def scope_str(s: str) -> str:
    """One element of a run's scope: an ISO 3166 alpha-3 code, or ``world``."""
    return s if s == WORLD else worldbank_jurisdictions.iso_3166_str(s=s)


def is_world(scope: tuple[str, ...]) -> bool:
    assert scope, "the scope names no country"
    assert scope == (WORLD,) or WORLD not in scope, (
        f"{scope=} mixes {WORLD} with countries"
    )
    return scope == (WORLD,)


def get_iso_3166s(scope: tuple[str, ...]) -> tuple[str, ...]:
    """The scope's countries: those named, or for ``world`` every country, as jdluc's
    ``--backfill`` reads them."""
    return tuple(
        sorted(
            worldbank_jurisdictions.get_all_iso_3166s() if is_world(scope) else scope
        )
    )


def get_name(scope: tuple[str, ...]) -> str:
    """What the run's run-wide outputs, such as its emissions-factor table, are named."""
    return WORLD if is_world(scope) else "-".join(sorted(scope))


def get_tile_ids(scope: tuple[str, ...]) -> list[str]:
    """The tiles the scope's countries touch, or for ``world`` every GNW tile, as jdluc's
    ``--backfill`` reads them: tiles no country touches included."""
    return sorted(
        tiling.GLOBAL_NATURE_WATCH_TILE_IDS
        if is_world(scope)
        else worldbank_jurisdictions.get_ten_degree_tile_ids_for_iso_3166s(
            iso_3166s=get_iso_3166s(scope=scope)
        )
    )


def ingest_boundaries() -> None:
    """The tile list is read off the boundaries, so the plan ingests them first.

    A no-op when they are already in the ingest root, as after any earlier run.
    """
    root = config.Config.from_dot_env().ingest_root
    for name in BOUNDARY_DATASET_NAMES:
        logger.info(f"Ingesting {name.name:s} if absent")
        NAME_TO_CLS[name].ingest_a_tile(
            overwrite=False, root=root, tile_id=tiling.WHOLE_WORLD_TILE_ID
        )


def check_allow_failures(
    allow_failures: tuple[str, ...], methodology: attribute.Methodology
) -> None:
    tiled = get_dataset_names(
        methodology=methodology, partitioning=tiling.Partitioning.TEN_DEGREE_TILE
    )
    unknown = sorted(set(allow_failures) - {name.name for name in tiled})
    assert not unknown, f"{unknown} are not ten-degree datasets this run ingests"


def get_task_to_items(
    allow_failures: tuple[str, ...],
    methodology: attribute.Methodology,
    scope: tuple[str, ...],
) -> dict[str, list]:
    """Each list the workflow loops over, by task name. Writes nothing, so that a wrapper can
    add lists of its own, or keys to each tile's unit, before `write_task_to_items`.

    Reads the boundaries for the tile list: `ingest_boundaries` first, on a fresh ingest root.

    ``tile-units`` groups the units by tile, for the example workflow, which runs each tile's
    ingest and compute as a DAG of its own. The other lists hold the same units flat, one per
    stage across every tile, for a workflow that fans out stage by stage instead.
    """
    check_allow_failures(allow_failures=allow_failures, methodology=methodology)
    tiled = get_dataset_names(
        methodology=methodology, partitioning=tiling.Partitioning.TEN_DEGREE_TILE
    )
    tile_ids = get_tile_ids(scope=scope)
    # One flat list of (dataset, tile), so that the workflow needs no nested loop
    ingest_units = [
        {"dataset": name.name, "tile_id": tile_id}
        for name in tiled
        for tile_id in tile_ids
    ]
    return {
        "ingest-world": [
            name.name
            for name in get_dataset_names(
                methodology=methodology, partitioning=tiling.Partitioning.WHOLE_WORLD
            )
            if name not in BOUNDARY_DATASET_NAMES
        ],
        "ingest-tiles": [u for u in ingest_units if u["dataset"] not in allow_failures],
        "ingest-tiles-allowed-failures": [
            u for u in ingest_units if u["dataset"] in allow_failures
        ],
        "tiles": [{"tile_id": tile_id} for tile_id in tile_ids],
        "tile-units": [
            {
                "tile_id": tile_id,
                "ingests": [
                    u["dataset"]
                    for u in ingest_units
                    if u["tile_id"] == tile_id and u["dataset"] not in allow_failures
                ],
                "ingests_allowed_failures": [
                    u["dataset"]
                    for u in ingest_units
                    if u["tile_id"] == tile_id and u["dataset"] in allow_failures
                ],
            }
            for tile_id in tile_ids
        ],
    }


def write_task_to_items(
    output_dir: str, task_to_items: collections.abc.Mapping[str, list]
) -> None:
    """Write each list to ``output_dir``, as ``<task>.json``."""
    os.makedirs(output_dir, exist_ok=True)
    for task, items in task_to_items.items():
        path = os.path.join(output_dir, f"{task:s}.json")
        logger.info(f"Writing {len(items):d} items to {path=:s}")
        with open(path, "w") as fp:
            json.dump(items, fp)


def write_plan(
    allow_failures: tuple[str, ...],
    methodology: attribute.Methodology,
    output_dir: str,
    scope: tuple[str, ...],
) -> None:
    """Write each list the workflow loops over to ``output_dir``, as ``<task>.json``.

    The run covers ``scope``: one or more countries, or ``world`` for every country and GNW
    tile.

    ``allow_failures`` names ten-degree datasets whose failed tiles must not stop the run.
    Their units go to their own list, so that the workflow can run them as a task that
    compute does not wait to succeed.
    """
    # Before the boundaries' ingest, so that a typo fails fast
    check_allow_failures(allow_failures=allow_failures, methodology=methodology)
    ingest_boundaries()
    write_task_to_items(
        output_dir=output_dir,
        task_to_items=get_task_to_items(
            allow_failures=allow_failures, methodology=methodology, scope=scope
        ),
    )
