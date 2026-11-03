import json
import pathlib
import re
import sys
import types

import pytest

from jdluc import attribute, datasets, tiling, trace
from jdluc.datasets import worldbank_jurisdictions
from orchestration import __main__ as cli
from orchestration import plan

TILE_IDS = {"10N_010W", "20N_010W", "30N_010W"}
# A stand-in for every GNW tile: more than the countries' tiles, as the real list is
WORLD_TILE_IDS = ("00N_010W", *sorted(TILE_IDS), "40N_010W")


@pytest.fixture
def ingested(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Stub the boundaries and every dataset; return the (dataset, tile) pairs ingested."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        worldbank_jurisdictions,
        "get_ten_degree_tile_ids_for_iso_3166s",
        lambda iso_3166s: TILE_IDS,
    )
    monkeypatch.setattr(
        worldbank_jurisdictions, "get_all_iso_3166s", lambda: {"AAA", "BBB"}
    )
    monkeypatch.setattr(tiling, "GLOBAL_NATURE_WATCH_TILE_IDS", WORLD_TILE_IDS)
    for name, cls in list(datasets.NAME_TO_CLS.items()):

        def ingest_a_tile(
            overwrite: bool, root: str, tile_id: str, name: str = name.name
        ) -> str:
            calls.append((name, tile_id))
            return f"{tile_id:s}.tif"

        monkeypatch.setitem(
            datasets.NAME_TO_CLS,
            name,
            types.SimpleNamespace(
                get_prefix=lambda tile_id, name=name.name: f"{name:s}/{tile_id:s}",
                ingest_a_tile=ingest_a_tile,
                partitioning=cls.partitioning,
            ),
        )
    return calls


def write_plan(
    tmp_path: pathlib.Path,
    allow_failures: tuple[str, ...],
    scope: tuple[str, ...] = ("AAA",),
) -> dict:
    plan.write_plan(
        allow_failures=allow_failures,
        methodology=attribute.Methodology.STATISTICAL,
        output_dir=str(tmp_path),
        scope=scope,
    )
    return {p.stem: json.loads(p.read_text()) for p in tmp_path.glob("*.json")}


def test_plan_ingests_the_boundaries_it_reads(
    ingested: list[tuple[str, str]], tmp_path: pathlib.Path
) -> None:
    task_to_items = write_plan(tmp_path=tmp_path, allow_failures=())
    assert sorted(ingested) == [
        ("WORLD_BANK_ADMIN_0", tiling.WHOLE_WORLD_TILE_ID),
        ("WORLD_BANK_ADMIN_1", tiling.WHOLE_WORLD_TILE_ID),
    ]
    assert "WORLD_BANK_ADMIN_0" not in task_to_items["ingest-world"]


def test_allowed_failures_get_their_own_list(
    ingested: list[tuple[str, str]], tmp_path: pathlib.Path
) -> None:
    task_to_items = write_plan(tmp_path=tmp_path, allow_failures=("DESCALS_OIL_PALM",))
    allowed = task_to_items["ingest-tiles-allowed-failures"]
    assert {u["dataset"] for u in allowed} == {"DESCALS_OIL_PALM"}
    assert {u["tile_id"] for u in allowed} == TILE_IDS
    assert "DESCALS_OIL_PALM" not in {
        u["dataset"] for u in task_to_items["ingest-tiles"]
    }


def test_an_unknown_allowed_failure_fails(
    ingested: list[tuple[str, str]], tmp_path: pathlib.Path
) -> None:
    with pytest.raises(AssertionError, match="DESCALS_OIL_PLAM"):
        write_plan(tmp_path=tmp_path, allow_failures=("DESCALS_OIL_PLAM",))


@pytest.mark.parametrize("scope", [("AAA",), (plan.WORLD,)])
def test_each_planned_unit_ingests_exactly_its_tile(
    ingested: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    scope: tuple[str, ...],
    tmp_path: pathlib.Path,
) -> None:
    task_to_items = write_plan(tmp_path=tmp_path, allow_failures=(), scope=scope)
    for unit in task_to_items["ingest-tiles"]:
        ingested.clear()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "orchestration",
                "ingest-tile",
                f"--dataset={unit['dataset']:s}",
                f"--tile-id={unit['tile_id']:s}",
            ],
        )
        assert cli.main() == 0
        assert ingested == [(unit["dataset"], unit["tile_id"])]


def test_a_failed_tile_fails_its_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    def ingest_a_tile(overwrite: bool, root: str, tile_id: str) -> str:
        raise RuntimeError(f"no {tile_id:s}")

    monkeypatch.setitem(
        datasets.NAME_TO_CLS,
        datasets.DatasetName.DESCALS_OIL_PALM,
        types.SimpleNamespace(
            ingest_a_tile=ingest_a_tile,
            partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
        ),
    )
    with pytest.raises(RuntimeError, match="no 10N_010W"):
        cli.ingest_tile(
            dataset=datasets.DatasetName.DESCALS_OIL_PALM, tile_id="10N_010W"
        )


def test_world_plans_every_gnw_tile(
    ingested: list[tuple[str, str]], tmp_path: pathlib.Path
) -> None:
    task_to_items = write_plan(
        tmp_path=tmp_path, allow_failures=(), scope=(plan.WORLD,)
    )
    assert [u["tile_id"] for u in task_to_items["tiles"]] == list(WORLD_TILE_IDS)
    assert {u["tile_id"] for u in task_to_items["ingest-tiles"]} == set(WORLD_TILE_IDS)


@pytest.mark.parametrize("scope", [(), (plan.WORLD, "AAA")])
def test_a_scope_is_countries_or_world_alone(scope: tuple[str, ...]) -> None:
    with pytest.raises(AssertionError):
        plan.get_tile_ids(scope=scope)


def test_world_names_its_run_wide_outputs_world(
    ingested: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def record(**kwargs: object) -> str:
        calls.append(kwargs)
        return "uri"

    monkeypatch.setattr(trace, "export_workflow", record)
    monkeypatch.setattr(
        sys, "argv", ["orchestration", "trace", "world", "--methodology=STATISTICAL"]
    )
    assert cli.main() == 0

    (trace_kwargs,) = calls
    assert trace_kwargs["name"] == "world"
    assert trace_kwargs["iso_3166s"] == ("AAA", "BBB")


def test_a_country_run_names_its_outputs_by_its_countries() -> None:
    assert plan.get_name(scope=("HND", "GTM")) == "GTM-HND"


def test_get_task_to_items_ingests_nothing_and_matches_the_plan(
    ingested: list[tuple[str, str]], tmp_path: pathlib.Path
) -> None:
    task_to_items = plan.get_task_to_items(
        allow_failures=(),
        methodology=attribute.Methodology.STATISTICAL,
        scope=("AAA",),
    )
    assert ingested == []
    assert task_to_items == write_plan(tmp_path=tmp_path, allow_failures=())


def test_a_wrapper_adds_a_list_and_a_subcommand(
    ingested: list[tuple[str, str]], tmp_path: pathlib.Path
) -> None:
    """The extension points a wrapper builds on: a list added to the plan, and a subcommand
    added to the CLI, with no edit to this package."""
    task_to_items = plan.get_task_to_items(
        allow_failures=(),
        methodology=attribute.Methodology.STATISTICAL,
        scope=("AAA",),
    )
    task_to_items["extra-tiles"] = [u["tile_id"] for u in task_to_items["tiles"]]
    plan.write_task_to_items(output_dir=str(tmp_path), task_to_items=task_to_items)
    assert json.loads((tmp_path / "extra-tiles.json").read_text()) == sorted(TILE_IDS)

    calls: list[dict] = []

    def extra(scope: tuple[str, ...], tile_id: str) -> None:
        calls.append({"scope": scope, "tile_id": tile_id})

    parser, subparsers = cli.build_parser()
    subparser = cli.add_subparser(subparsers, "extra", extra)
    cli.add_scope(subparser)
    cli.add_tile_id(subparser)
    assert cli.run(parser, ["extra", "AAA", "--tile-id=10N_010W"]) == 0
    assert calls == [{"scope": ("AAA",), "tile_id": "10N_010W"}]


@pytest.mark.parametrize("scope", [("AAA",), (plan.WORLD,)])
def test_tile_units_hold_the_flat_lists_by_tile(
    ingested: list[tuple[str, str]], scope: tuple[str, ...], tmp_path: pathlib.Path
) -> None:
    task_to_items = write_plan(
        tmp_path=tmp_path,
        allow_failures=("DESCALS_OIL_PALM",),
        scope=scope,
    )
    tile_units = task_to_items["tile-units"]
    assert [u["tile_id"] for u in tile_units] == [
        u["tile_id"] for u in task_to_items["tiles"]
    ]
    for key, task in (
        ("ingests", "ingest-tiles"),
        ("ingests_allowed_failures", "ingest-tiles-allowed-failures"),
    ):
        assert sorted(
            (dataset, u["tile_id"]) for u in tile_units for dataset in u[key]
        ) == sorted((u["dataset"], u["tile_id"]) for u in task_to_items[task])


def test_each_tile_lists_each_tiled_dataset_once(
    ingested: list[tuple[str, str]], tmp_path: pathlib.Path
) -> None:
    task_to_items = write_plan(tmp_path=tmp_path, allow_failures=("DESCALS_OIL_PALM",))
    tiled = {
        name.name
        for name in plan.get_dataset_names(
            methodology=attribute.Methodology.STATISTICAL,
            partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
        )
    }
    for u in task_to_items["tile-units"]:
        assert u["ingests_allowed_failures"] == ["DESCALS_OIL_PALM"]
        assert sorted([*u["ingests"], *u["ingests_allowed_failures"]]) == sorted(tiled)


@pytest.fixture
def present_uris(
    ingested: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> set[str]:
    """Every input of tile 10N_010W, as the ingest root would hold them; remove some to
    make them missing."""
    root = cli.config.Config.from_dot_env().ingest_root
    uris = {
        cli.storage.join_uri(prefix=f"{name.name:s}/{tile_id:s}", root=root)
        for partitioning, tile_id in (
            (tiling.Partitioning.TEN_DEGREE_TILE, "10N_010W"),
            (tiling.Partitioning.WHOLE_WORLD, tiling.WHOLE_WORLD_TILE_ID),
        )
        for name in plan.get_dataset_names(
            methodology=attribute.Methodology.STATISTICAL, partitioning=partitioning
        )
    }
    monkeypatch.setattr(cli.storage, "path_exists", lambda uri: uri in uris)
    return uris


def test_compute_checks_ingests_before_harmonize(
    monkeypatch: pytest.MonkeyPatch, present_uris: set[str]
) -> None:
    (missing,) = [uri for uri in present_uris if "GPW_GRASSLAND/" in uri]
    present_uris.remove(missing)

    def harmonize(**kwargs: object) -> None:
        raise AssertionError("harmonize ran on a tile missing an ingest")

    monkeypatch.setattr(cli.harmonize, "workflow", harmonize)
    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        cli.compute(
            allow_failures=("DESCALS_OIL_PALM",),
            check_ingests=True,
            methodology=attribute.Methodology.STATISTICAL,
            scope=("AAA",),
            tile_id="10N_010W",
        )


def test_an_allowed_failure_is_not_missing(present_uris: set[str]) -> None:
    present_uris -= {uri for uri in present_uris if "DESCALS_OIL_PALM/" in uri}
    assert (
        cli.get_missing_ingests(
            allow_failures=("DESCALS_OIL_PALM",),
            methodology=attribute.Methodology.STATISTICAL,
            tile_id="10N_010W",
        )
        == []
    )
    assert (
        cli.get_missing_ingests(
            allow_failures=(),
            methodology=attribute.Methodology.STATISTICAL,
            tile_id="10N_010W",
        )
        != []
    )
