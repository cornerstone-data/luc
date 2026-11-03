import argparse
import collections.abc
import logging

import orchestration
from jdluc import (
    attribute,
    config,
    emit,
    harmonize,
    ingest,
    jurisdictional_direct,
    statistical,
    storage,
    tiling,
    trace,
)
from jdluc.datasets import NAME_TO_CLS, DatasetName, worldbank_jurisdictions
from orchestration import plan

logger = logging.getLogger(__name__)


def ingest_tile(dataset: DatasetName, tile_id: str) -> None:
    """One ten-degree dataset on one tile, into the ingest root. Raises if the tile fails."""
    (path_or_exc,) = ingest.workflow(
        concurrency=1,
        dataset=NAME_TO_CLS[dataset],
        overwrite=False,
        root=config.Config.from_dot_env().ingest_root,
        tile_ids=(tile_id,),
    ).values()
    if isinstance(path_or_exc, Exception):
        raise path_or_exc
    logger.info(f"ingested {dataset.name:s} {tile_id=:s} to {path_or_exc:s}")


def get_missing_ingests(
    allow_failures: tuple[str, ...], methodology: attribute.Methodology, tile_id: str
) -> list[str]:
    """The URIs of the tile's inputs absent from the ingest root: its ten-degree datasets
    other than ``allow_failures``, and every whole-world dataset."""
    root = config.Config.from_dot_env().ingest_root
    name_and_tile_ids = [
        *(
            (name, tile_id)
            for name in plan.get_dataset_names(
                methodology=methodology,
                partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
            )
            if name.name not in allow_failures
        ),
        *(
            (name, tiling.WHOLE_WORLD_TILE_ID)
            for name in plan.get_dataset_names(
                methodology=methodology, partitioning=tiling.Partitioning.WHOLE_WORLD
            )
        ),
    ]
    uris = [
        storage.join_uri(
            prefix=NAME_TO_CLS[name].get_prefix(tile_id=tile_id), root=root
        )
        for name, tile_id in name_and_tile_ids
    ]
    return [uri for uri in uris if not storage.path_exists(uri=uri)]


def compute(
    allow_failures: tuple[str, ...],
    check_ingests: bool,
    methodology: attribute.Methodology,
    scope: tuple[str, ...],
    tile_id: str,
) -> None:
    """harmonize -> emit -> attribute for one tile, in one process.

    harmonize goes first with ``skip_ingest=True``: the later stages call it again with
    ``skip_ingest=False``, and the cache key ignores the flag, so warming it here is what
    keeps source credentials off compute pods.

    With ``check_ingests``, raises first if any input other than ``allow_failures`` is
    missing. harmonize would otherwise read it as an all-nodata band, and cache that.
    """
    if check_ingests:
        missing = get_missing_ingests(
            allow_failures=allow_failures, methodology=methodology, tile_id=tile_id
        )
        if missing:
            raise FileNotFoundError(f"{tile_id=:s} is missing ingests: {missing}")
    for stack in harmonize.Stack:
        logger.info(f"harmonize {stack.name:s} {tile_id=:s}")
        harmonize.workflow(
            dataset_names=stack.value,
            ignore_missing_tiles=True,
            skip_ingest=True,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.GLAD,
        )
    logger.info(f"emit {tile_id=:s}")
    emit.workflow(tile_id=tile_id)

    workflow_for_tile = {
        attribute.Methodology.JURISDICTIONAL_DIRECT: jurisdictional_direct.workflow,
        attribute.Methodology.STATISTICAL: statistical.workflow,
    }[methodology]
    # A tile can straddle several of the run's countries; trace sums exactly these
    # per-(tile, country) partials, so compute has to produce all of them
    for iso_3166 in get_iso_3166s_touching_tile(
        iso_3166s=plan.get_iso_3166s(scope=scope), tile_id=tile_id
    ):
        logger.info(f"attribute {methodology.name:s} {tile_id=:s} {iso_3166=:s}")
        workflow_for_tile(
            commodity_names=attribute.get_commodity_names(methodology=methodology),
            iso_3166=iso_3166,
            tile_id=tile_id,
        )


def get_iso_3166s_touching_tile(
    iso_3166s: collections.abc.Iterable[str], tile_id: str
) -> list[str]:
    return sorted(
        iso_3166
        for iso_3166 in iso_3166s
        if tile_id
        in worldbank_jurisdictions.get_ten_degree_tile_ids_for_iso_3166s(
            iso_3166s=(iso_3166,)
        )
    )


def trace_countries(methodology: attribute.Methodology, scope: tuple[str, ...]) -> None:
    """The cross-tile merge, written as the emissions-factor table.

    Cheap only if compute ran with the same methodology."""
    uri = trace.export_workflow(
        commodity_names=attribute.get_commodity_names(methodology=methodology),
        iso_3166s=plan.get_iso_3166s(scope=scope),
        methodology=methodology,
        name=plan.get_name(scope=scope),
    )
    logger.info(f"exported emissions factors to {uri:s}")


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    func: collections.abc.Callable[..., None],
) -> argparse.ArgumentParser:
    """A subcommand that calls ``func`` with its flags, which must be exactly ``func``'s
    parameters."""
    subparser = subparsers.add_parser(name, help=(func.__doc__ or "").split("\n")[0])
    subparser.set_defaults(func=func)
    return subparser


def add_scope(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "scope",
        help=f"ISO 3166 alpha-3 codes, or {plan.WORLD:s} for every country and GNW tile",
        nargs=argparse.ONE_OR_MORE,
        type=plan.scope_str,
    )


def add_methodology(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--methodology",
        choices=list(attribute.Methodology),
        metavar="{" + ",".join(m.name for m in attribute.Methodology) + "}",
        required=True,
        type=attribute.Methodology.__getitem__,
    )


def add_tile_id(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--tile-id", required=True)


def add_allow_failures(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--allow-failures",
        default=(),
        help="comma-separated ten-degree datasets whose failed tiles must not stop the run",
        type=lambda s: tuple(name for name in s.split(",") if name),
    )


def build_parser() -> tuple[
    argparse.ArgumentParser, argparse._SubParsersAction[argparse.ArgumentParser]
]:
    """The CLI, and its subparsers for a wrapper to add subcommands to before `run`."""
    parser = argparse.ArgumentParser(
        description=orchestration.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(required=True)

    subparser = add_subparser(subparsers, "plan", plan.write_plan)
    add_scope(subparser)
    add_methodology(subparser)
    add_allow_failures(subparser)
    subparser.add_argument("--output-dir", required=True)

    subparser = add_subparser(subparsers, "ingest-tile", ingest_tile)
    subparser.add_argument(
        "--dataset",
        choices=list(DatasetName),
        metavar="DATASET",
        required=True,
        # NB: not `DatasetName.__getitem__`, which a StrEnum resolves to `str.__getitem__`
        type=lambda s: DatasetName[s],
    )
    add_tile_id(subparser)

    subparser = add_subparser(subparsers, "compute", compute)
    add_scope(subparser)
    add_methodology(subparser)
    add_tile_id(subparser)
    add_allow_failures(subparser)
    subparser.add_argument(
        "--check-ingests",
        action="store_true",
        help="fail if an input other than --allow-failures is missing from the ingest root",
    )

    subparser = add_subparser(subparsers, "trace", trace_countries)
    add_scope(subparser)
    add_methodology(subparser)

    return parser, subparsers


def run(
    parser: argparse.ArgumentParser, args: collections.abc.Sequence[str] | None = None
) -> int:
    """Parse ``args`` (by default the command line) and call the chosen subcommand."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    # Each subcommand's flags are exactly its function's parameters
    kwargs = vars(parser.parse_args(args))
    func = kwargs.pop("func")
    func(**{k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()})
    return 0


def main() -> int:
    parser, _ = build_parser()
    return run(parser)


if __name__ == "__main__":
    raise SystemExit(main())
