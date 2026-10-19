import sys
import types

import pytest

from jdluc import config, datasets, ingest, tiling
from jdluc.datasets import worldbank_jurisdictions

TILE_IDS = {"10N_010W", "20N_010W", "30N_010W"}


@pytest.fixture
def ingested_tile_ids(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the config, the boundaries and a tiled dataset; return the tiles it was asked to ingest."""
    calls: list[str] = []

    def ingest_a_tile(overwrite: bool, root: str, tile_id: str) -> str:
        calls.append(tile_id)
        return f"{tile_id:s}.tif"

    monkeypatch.setattr(
        config.Config,
        "from_dot_env",
        lambda: types.SimpleNamespace(ingest_root="unused"),
    )
    monkeypatch.setattr(
        worldbank_jurisdictions,
        "get_ten_degree_tile_ids_for_iso_3166s",
        lambda iso_3166s: TILE_IDS,
    )
    monkeypatch.setitem(
        datasets.NAME_TO_CLS,
        datasets.DatasetName.GNW_HARRIS_AGB,
        types.SimpleNamespace(
            ingest_a_tile=ingest_a_tile,
            partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
        ),
    )
    return calls


@pytest.mark.parametrize(
    ("shard_args", "expected_tile_ids"),
    [
        pytest.param([], ["10N_010W", "20N_010W", "30N_010W"], id="every-tile"),
        pytest.param(
            ["--modulus=2", "--residues=0"], ["10N_010W", "30N_010W"], id="first-shard"
        ),
        pytest.param(["--modulus=3", "--residues=1"], ["20N_010W"], id="one-tile"),
        pytest.param(
            ["--modulus=3", "--residues=0", "--residues=2"],
            ["10N_010W", "30N_010W"],
            id="two-shards",
        ),
    ],
)
def test_shards_select_tiles_by_their_sorted_index(
    ingested_tile_ids: list[str],
    monkeypatch: pytest.MonkeyPatch,
    expected_tile_ids: list[str],
    shard_args: list[str],
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["ingest.py", "GNW_HARRIS_AGB", "AAA", *shard_args]
    )
    assert ingest.main() == 0
    assert sorted(ingested_tile_ids) == expected_tile_ids
