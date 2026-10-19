import types

import pandas
import pytest

from jdluc import attribute, statistical
from jdluc.datasets import worldbank_jurisdictions

ISO_3166_TO_TILE_IDS = {"AAA": {"10N_010W", "20N_010W"}, "BBB": {"20N_010W"}}


@pytest.fixture
def attributed_tile_ids(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the boundaries and the per-tile leg; return the tiles the leg was called for."""
    calls: list[str] = []

    def workflow_for_tile(
        commodity_names: tuple[str, ...], iso_3166: str, tile_id: str
    ) -> pandas.DataFrame:
        calls.append(tile_id)
        return pandas.DataFrame.from_records(
            [("PROVINCIAL", f"{iso_3166:s}001", "MAIZE", iso_3166, 1.0)],
            columns=[
                "admin_level",
                "admin_id",
                "commodity_name",
                "jurisdiction_name",
                "emissions_mt",
            ],
        ).set_index(["admin_level", "admin_id", "commodity_name"])

    monkeypatch.setattr(
        worldbank_jurisdictions,
        "get_ten_degree_tile_ids_for_admin_id",
        lambda admin_id, admin_level: ISO_3166_TO_TILE_IDS[admin_id],
    )
    monkeypatch.setattr(
        worldbank_jurisdictions,
        "iter_jurisdiction_for_iso_3166_tile_id",
        lambda admin_level, iso_3166, tile_id: iter(
            [
                types.SimpleNamespace(
                    id=f"{iso_3166:s}001", level="PROVINCIAL", name=iso_3166
                )
            ]
        ),
    )
    monkeypatch.setattr(statistical, "workflow", workflow_for_tile)
    return calls


@pytest.mark.parametrize(
    ("residues", "expected_tile_ids"),
    [
        pytest.param(None, ["10N_010W", "20N_010W", "20N_010W"], id="every-tile"),
        pytest.param([0], ["10N_010W"], id="first-shard"),
        # A tile shared by two countries is one shard, so both its partials land together
        pytest.param([1], ["20N_010W", "20N_010W"], id="second-shard"),
    ],
)
def test_shards_select_tiles_by_their_sorted_index(
    attributed_tile_ids: list[str],
    expected_tile_ids: list[str],
    residues: list[int] | None,
) -> None:
    attribute.workflow(
        commodity_names=("MAIZE",),
        concurrency=1,
        iso_3166s=("AAA", "BBB"),
        methodology=attribute.Methodology.STATISTICAL,
        modulus=2,
        residues=residues,
    )
    assert sorted(attributed_tile_ids) == expected_tile_ids
