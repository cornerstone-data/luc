"""Build the committed set of countries the pipeline can produce a factor for -- E1's second half.

Eligibility cuts some thirty thousand possible (country, crop) pairs to a few hundred, and all but
one of its filters read anchors `sources.lock.json` already pins by sha256. `prepare.get_eligible`
derives those on every report run, in about a tenth of a second, rather than any of them being
committed a second time.

This is the exception. Deciding which countries intersect at least one of the 280 ten-degree GFW
tiles needs a spatial join against a 93 MiB GeoPackage and geopandas, neither of which belongs on the
reporting path. So the join happens here and its result is committed -- a few kilobytes, and it moves
only when the tile set or the World Bank layer does, not when the WRI pin does.

The committed set is that join **minus** `worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S`, and the
subtraction is the part worth reading the source for: the join alone answers a question about
geometry, where E1 needs one about production. The constant is imported rather than restated so that
this set and `get_all_iso_3166s` -- what `--backfill` runs -- cannot drift apart; see it for why the
difference is a list rather than a predicate.

**The output is the artifact, and a broken one exits nonzero.** A bad committed set is not a finding,
it is a defect: a spatial join that silently returned a handful of countries would fail every pair at
E1, which the reporting path would report as a mis-specified filter rather than as a bad input.

To see the shortlist this makes possible, and choose from it, use the report rather than this tool:

  uv run --with geopandas python tools/build-tiled-countries.py
  uv run python -m validation --section eligible --show 40
"""

import argparse
import collections.abc
import dataclasses
import json
import logging
import pathlib

import geopandas

from jdluc import tiling
from jdluc.datasets import worldbank_jurisdictions
from validation import prepare, pull, targets

logger = logging.getLogger(__name__)

# Below this the join did not merely lose a country, it broke. 220 of the layer's 244 countries
# survive the join and the carve-out below, so the headroom is wide and only a structural failure
# trips it.
MINIMUM_TILED_SHARE = 0.75


@dataclasses.dataclass(frozen=True)
class Check:
    """One property the finished set must have, and what was found instead where it does not."""

    name: str
    passed: bool
    detail: str


def get_tiled_iso_3166s(path_to_geopackage: pathlib.Path) -> tuple[set[str], int]:
    """Countries intersecting at least one ten-degree GFW tile, and how many the layer holds.

    The GFW tile set covers 280 ten-degree cells rather than all of them, so this filter genuinely
    bites: a country entirely outside it has no tree-cover-loss data and cannot be computed. Read
    from the GeoPackage `build-national-mappings` downloads, so this needs no ingest -- run that
    tool first on a cold cache, or this one fails on a missing file.
    """
    world_bank = geopandas.read_file(path_to_geopackage)
    tiles = geopandas.GeoDataFrame(
        geometry=[
            tiling.get_box_for_tile_id(tile_id=tile_id)
            for tile_id in sorted(tiling.GLOBAL_FOREST_WATCH_TILE_IDS)
        ],
        crs=world_bank.crs,
    )
    joined = geopandas.sjoin(world_bank, tiles, how="inner", predicate="intersects")
    return set(joined["ISO_A3"].dropna()), int(world_bank["ISO_A3"].dropna().nunique())


def iter_checks(
    countries: int, joined: set[str], tiled: set[str]
) -> collections.abc.Iterator[Check]:
    """Verify the finished set rather than trusting the join that produced it.

    These read the artifact that gets committed, so a bad set fails whichever route produced it --
    including a hand-edited file. `joined` is the raw spatial join and `tiled` is what survives the
    carve-out, because two of the checks below can only tell a real change from a stale list by
    comparing them.
    """
    share = len(tiled) / countries if countries else 0.0
    yield Check(
        name="coverage-floor",
        passed=share >= MINIMUM_TILED_SHARE,
        detail=(
            f"{len(tiled):d} of {countries:d} countries ({share:.1%}) intersect a tile, against a "
            f"{MINIMUM_TILED_SHARE:.0%} floor"
            + (
                "; a collapse this size usually means the layer renamed a column or lost its CRS"
                if share < MINIMUM_TILED_SHARE
                else ""
            )
        ),
    )

    malformed = sorted(
        iso_3166
        for iso_3166 in tiled
        if len(iso_3166) != 3 or not iso_3166.isupper() or not iso_3166.isalpha()
    )
    yield Check(
        name="well-formed-codes",
        passed=not malformed,
        detail=(
            f"{len(malformed):d} entry(ies) are not ISO 3166 alpha-3: {', '.join(malformed[:6])}"
            if malformed
            else f"all {len(tiled):d} entries are ISO 3166 alpha-3"
        ),
    )

    # A carved-out country the join no longer returns means the list is claiming credit for a
    # removal the geometry already made -- harmless today, and a lie the next time someone reads it
    # to find out what the carve-out costs.
    stale = sorted(worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S - joined)
    yield Check(
        name="carve-out-current",
        passed=not stale,
        detail=(
            f"{len(stale):d} carved-out country(ies) no longer intersect a tile, so "
            f"UNPRODUCTIVE_ISO_3166S is stale: {', '.join(stale)}"
            if stale
            else (
                f"all {len(worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S):d} carved-out "
                f"countries still intersect a "
                f"tile, so each is removed by the carve-out rather than by the join"
            )
        ),
    )

    # A chosen target that fails E1 cannot be computed at all, so the set silently dropping one
    # would empty that pair from the report rather than raising anywhere.
    missing = sorted(
        {
            target.iso_3166
            for target in targets.iter_targets()
            if target.iso_3166 not in tiled
        }
    )
    yield Check(
        name="targets-tiled",
        passed=not missing,
        detail=(
            f"{len(missing):d} country(ies) named by {targets.TARGETS.name:s} intersect no tile: "
            f"{', '.join(missing)}"
            if missing
            else f"every country in {targets.TARGETS.name:s} intersects a tile"
        ),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geopackage",
        type=pathlib.Path,
        default=pull.CACHE / "world_bank_admin_1.gpkg",
        help="the World Bank admin-1 GeoPackage build-national-mappings downloads",
    )
    args = parser.parse_args()
    assert args.geopackage.exists(), (
        f"{args.geopackage} is absent; run `uv run --with geopandas python "
        "tools/build-national-mappings.py` first, which is what downloads it"
    )

    joined, countries = get_tiled_iso_3166s(path_to_geopackage=args.geopackage)
    tiled = joined - worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S
    pull.DATA.mkdir(parents=True, exist_ok=True)
    # No timestamp: the inputs' versions are the identity, and a clock would make two branches
    # disagree about an identical set.
    prepare.TILED_ISO_3166S.write_text(
        json.dumps(
            {
                "note": [
                    "DERIVED -- do not hand-edit. Regenerate with",
                    "`uv run --with geopandas python tools/build-tiled-countries.py`.",
                    "E1's second half: the countries intersecting at least one GFW tile,",
                    "less `carved_out_iso_3166s` -- countries that do intersect a tile but",
                    "carry no crop production to attribute anything to. See",
                    "worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S for why that is a list.",
                    "Committed because deriving it needs a spatial join against a 93 MiB",
                    "GeoPackage, where the rest of eligibility is pandas over anchors the lock",
                    "already pins -- so the rest is derived on demand rather than committed twice.",
                ],
                "world_bank_source": {
                    "dataset": worldbank_jurisdictions.ADMIN_1_DATASET.product_name,
                    "version": worldbank_jurisdictions.ADMIN_1_DATASET.version,
                },
                "tile_ids": len(tiling.GLOBAL_FOREST_WATCH_TILE_IDS),
                "carved_out_iso_3166s": sorted(
                    worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S
                ),
                "tiled_iso_3166s": sorted(tiled),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"\n  {len(tiled):d} tiled country(ies) "
        f"({len(joined):d} joined, less "
        f"{len(worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S):d} carved out)  ->  "
        f"{prepare.TILED_ISO_3166S}\n"
    )

    checks = tuple(iter_checks(countries=countries, joined=joined, tiled=tiled))
    for check in checks:
        print(
            f"  {'ok  ' if check.passed else 'FAIL'} {check.name:20s} {check.detail:s}"
        )
    failed = [check.name for check in checks if not check.passed]
    if failed:
        # The set is still on disk, deliberately: a failed check is easier to diagnose against the
        # written file than against a run that refused to produce one.
        print(
            f"\n{len(failed):d} check(s) failed: {', '.join(failed)}. "
            f"{prepare.TILED_ISO_3166S} is not safe to commit."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
