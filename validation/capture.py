"""Run the pipeline over the target set and write the artifact every comparison reads.

The one stage that runs the pipeline itself, so it is run by hand rather than by
`python -m validation`, and its output is cached rather than recomputed per report.

**One capture, both legs.** `efs.parquet` holds sLUC and jdLUC together, indexed on
`trace.CANONICAL_KEY` -- `(admin_level, admin_id, crop_name, methodology)` -- so the US
sLUC-versus-jdLUC head-to-head is a filter on `methodology` within one table rather than a join
between two files. A join there would invite exactly the alignment failure the comparison exists to
rule out.

**Every crop, not just the targets.** The conservation bound compares a jurisdiction's forest pool
against the sum of `forest_emissions_mt` over *all* crops, so restricting the run to target crops
would shrink the numerator and let the bound pass by construction. It is also cheaper than it looks:
`statistical.get_downscaled_luc_emissions` is keyed on `(skip_glad_crop_filter, tile_id)` alone, so
the expensive per-tile layer is shared and a crop adds only its share arithmetic.

**Merging is per (ISO, methodology), and rows keep their own `code_version`.** A capture scoped with
`--isos` leaves other countries in place, so the artifact can span versions and a reader has to be
able to see that -- `report.get_provenance_findings` raises a DEFECT when it does. That is why
provenance travels in the row rather than in a sidecar.

Run it from the repo root, so `storage`'s cache keying resolves the module path:

  uv run python -m validation.capture --dry-run        resolve the plan, touch nothing
  uv run python -m validation.capture                  every target, both legs
  uv run python -m validation.capture --isos BRA PRY   those countries only, merged in

`--dry-run` does not run the pipeline: it resolves the target set, the crop set and the
tile count and prints what a real run would compute. Use it to confirm the plan before spending a
capture, since the run itself is hours of compute against a warm cache and rather more against a
cold one.
"""

import argparse
import collections.abc
import logging
import pathlib

import pandas

from jdluc import attribute, emit, geo, statistical, trace
from jdluc.datasets import worldbank_jurisdictions
from validation import pull, schema, targets

logger = logging.getLogger(__name__)

# jdLUC asserts `iso_3166 == "USA"` in its own workflow, since it needs the USDA CDL raster and NASS
# yields, both US-only. Naming it here keeps the reason with the filter.
JURISDICTIONAL_DIRECT_ISO_3166 = "USA"


def get_iso_3166s(isos: tuple[str, ...]) -> tuple[str, ...]:
    """The countries to compute, defaulting to every one the target set names.

    An explicit `--isos` is checked against the target set rather than passed through, because a
    typo would otherwise capture a country nothing compares and silently leave a target missing.
    """
    wanted = tuple(sorted({target.iso_3166 for target in targets.iter_targets()}))
    if not isos:
        return wanted
    unknown = sorted(set(isos) - set(wanted))
    assert not unknown, (
        f"{', '.join(unknown)} name no target in {targets.TARGETS}; capture computes the target "
        f"set, and a country outside it has nothing to compare against"
    )
    return tuple(sorted(set(isos)))


def iter_methodologies(
    iso_3166s: tuple[str, ...],
) -> collections.abc.Iterator[tuple[attribute.Methodology, tuple[str, ...]]]:
    """Each leg, with the countries it can run over.

    sLUC is global. jdLUC is US-only and is skipped rather than asserted when the USA is out of
    scope, so `--isos BRA` is a valid capture rather than an error.
    """
    yield attribute.Methodology.STATISTICAL, iso_3166s
    if JURISDICTIONAL_DIRECT_ISO_3166 in iso_3166s:
        yield (
            attribute.Methodology.JURISDICTIONAL_DIRECT,
            (JURISDICTIONAL_DIRECT_ISO_3166,),
        )


def get_tile_ids(iso_3166s: tuple[str, ...]) -> tuple[str, ...] | None:
    """Every ten-degree tile the capture touches, deduplicated, or None without the ingested layer.

    Worth reporting because tiles are the unit of cost: two countries sharing a tile pay for it
    once, which is why tile overlap is a criterion when the target set is chosen.

    None rather than raising, because this reads the ingested admin-0 layer from `ingest_root` and
    `--dry-run` has to work without credentials -- checking the plan before spending a capture is
    most useful from a machine that cannot spend one. The rest of the plan is `targets.json` and
    resolves offline.
    """
    try:
        return tuple(
            sorted(
                {
                    tile_id
                    for iso_3166 in iso_3166s
                    for tile_id in worldbank_jurisdictions.get_ten_degree_tile_ids_for_admin_id(
                        admin_id=iso_3166,
                        admin_level=worldbank_jurisdictions.AdminLevel.NATIONAL,
                    )
                }
            )
        )
    except Exception as error:
        # Broad on purpose: the read goes through GDAL and pyogrio, which surface a missing
        # credential as any of several driver errors, and none of them is worth a dependency here.
        logger.warning(f"Cannot resolve tiles without the ingested layer: {error!r}")
        return None


def read_efs() -> pandas.DataFrame | None:
    """The existing artifact, or None on the first capture."""
    if not pull.EFS.exists():
        return None
    return pandas.read_parquet(pull.EFS)


def merge_efs(
    captured: pandas.DataFrame, existing: pandas.DataFrame | None
) -> pandas.DataFrame:
    """Replace what was recomputed, keep everything else exactly as it was.

    Keyed on (ISO, methodology) rather than on the whole index: a country recomputed for one leg
    must lose all of that leg's rows, including any crop that no longer produces one, or a stale row
    would survive beside its replacements and be summed with them.
    """
    if existing is None:
        return captured
    recomputed = {
        (str(admin_id)[:3], str(methodology))
        for admin_id, methodology in zip(
            captured.index.get_level_values("admin_id"),
            captured.index.get_level_values("methodology"),
            strict=True,
        )
    }
    keep = [
        (str(admin_id)[:3], str(methodology)) not in recomputed
        for admin_id, methodology in zip(
            existing.index.get_level_values("admin_id"),
            existing.index.get_level_values("methodology"),
            strict=True,
        )
    ]
    logger.info(
        f"Keeping {sum(keep):,d} row(s) from the previous capture and replacing "
        f"{len(keep) - sum(keep):,d}"
    )
    return pandas.concat([existing[keep], captured]).sort_index()


def workflow(iso_3166s: tuple[str, ...], repo_root: pathlib.Path) -> pandas.DataFrame:
    """Both legs over the given countries, stamped with the code that produced them."""
    frames = []
    for methodology, wanted in iter_methodologies(iso_3166s=iso_3166s):
        crop_names = attribute.get_crop_names(methodology=methodology)
        logger.info(
            f"Capturing {methodology.name:s} over {len(wanted):d} country(ies) and "
            f"{len(crop_names):d} crops"
        )
        frames.append(
            trace.workflow(
                crop_names=crop_names,
                iso_3166s=wanted,
                methodology=methodology,
                skip_glad_crop_filter=False,
            )
        )
    captured = pandas.concat(frames)
    captured["code_version"] = schema.get_code_version(repo_root=repo_root)
    return captured


def get_forest_pool_tonnes(iso_3166: str) -> float:
    """The forest conversion a country's 2020 cropland contains, with no crop share applied.

    The same GHGP-discounted sum each crop's `forest_emissions_mt` is drawn from, minus the
    expansion share -- so it bounds what any allocation can hand out, and a country attributing more
    than it has an attribution bug rather than a disagreement.

    Computed here rather than emitted by the pipeline, which carries no such column. A second pass
    but not a recomputation: `get_downscaled_luc_emissions` is cached on
    `(skip_glad_crop_filter, tile_id)` alone, so this reads back the bands the capture attributed
    from, and the weights and pixel areas come from `emit` rather than being restated.

    Running in the same process at the same commit as the capture beside it is what makes the bound
    freezable rather than indicative.
    """
    from rioxarray.exceptions import NoDataInBounds

    tonnes = 0.0
    for tile_id in sorted(
        worldbank_jurisdictions.get_ten_degree_tile_ids_for_admin_id(
            admin_id=iso_3166, admin_level=worldbank_jurisdictions.AdminLevel.NATIONAL
        )
    ):
        # Iterated rather than looked up, mirroring `statistical.workflow`: at the national grain
        # this yields the one jurisdiction, and yields nothing where the tile meets the country's
        # bounding box but not its geometry.
        for (
            jurisdiction
        ) in worldbank_jurisdictions.iter_jurisdiction_for_iso_3166_tile_id(
            admin_level=worldbank_jurisdictions.AdminLevel.NATIONAL,
            iso_3166=iso_3166,
            tile_id=tile_id,
        ):
            logger.info(f"Summing the forest pool for {iso_3166=:s} {tile_id=:s}")
            try:
                clipped = geo.clip_dset(
                    dset=statistical.get_downscaled_luc_emissions(
                        skip_glad_crop_filter=False, tile_id=tile_id
                    ),
                    geometry=jurisdiction.geometry,
                )
            except NoDataInBounds as error:
                logger.warning(repr(error))
                continue
            bands = {
                (before, after): clipped[f"forest:tco2e-per-ha:{before:d}-{after:d}"]
                for (before, after) in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT
            }
            hectares = emit.get_hectares_per_pixel(darray=next(iter(bands.values())))
            tonnes += float(
                emit.get_linear_discounted_total(
                    span_to_value={
                        span: band * hectares for span, band in bands.items()
                    }
                )
                .sum()
                .compute()
            )
    return tonnes


def get_forest_pools(
    captured: pandas.DataFrame, iso_3166s: tuple[str, ...]
) -> pandas.DataFrame:
    """Per country, the pool against the sum attributed to crops out of it.

    The shape `report.get_conservation_findings` reads. National rows only, and the statistical leg
    only: jdLUC derives no unallocated pool, and summing the two legs would double-count the USA.

    The attributed figure is summed across every crop, which is the quantity the bound constrains --
    restricting it to target crops would shrink the numerator and let the bound pass by
    construction.
    """
    national = captured[
        (captured.index.get_level_values("admin_level") == schema.NATIONAL)
        & (captured.index.get_level_values("methodology") == schema.STATISTICAL)
    ]
    assert len(national), (
        "the capture holds no national statistical rows, so the conservation bound has nothing to "
        "read; a provincial-only artifact means the rollup did not run"
    )
    attributed = national.groupby(level="admin_id")["forest_emissions_mt"].sum()
    return pandas.DataFrame.from_records(
        [
            {
                "iso_3166": iso_3166,
                "attributed_tonnes": float(attributed.loc[iso_3166]),
                "pool_tonnes": get_forest_pool_tonnes(iso_3166=iso_3166),
            }
            for iso_3166 in iso_3166s
            if iso_3166 in attributed.index
        ]
    )


def render_plan(iso_3166s: tuple[str, ...]) -> str:
    """What a real run would compute, resolved without running the pipeline."""
    armed = {target.slug for target in targets.iter_control_targets()}
    tile_ids = get_tile_ids(iso_3166s=iso_3166s)
    lines = [
        f"{len(iso_3166s):d} country(ies): {', '.join(iso_3166s)}",
        f"{len(tile_ids):d} ten-degree tile(s), the unit of cost"
        if tile_ids is not None
        else "tile count unavailable: the ingested admin-0 layer has not been read",
    ]
    for methodology, wanted in iter_methodologies(iso_3166s=iso_3166s):
        crop_names = attribute.get_crop_names(methodology=methodology)
        lines.append(
            f"{methodology.name:s}: {len(wanted):d} country(ies) x {len(crop_names):d} crops"
        )
    covered = set(iso_3166s)
    missing = [
        target.slug
        for target in targets.iter_control_targets()
        if target.iso_3166 not in covered
    ]
    lines.append(f"{len(armed):d} target(s) carry a control")
    if missing:
        # A control whose target was not computed removes its own guard, and does so silently.
        lines.append(
            f"**{len(missing):d} of them are out of scope and will keep whatever the previous "
            f"capture left**: {', '.join(missing)}"
        )
    return "\n".join(f"  {line:s}" for line in lines)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isos",
        nargs=argparse.ONE_OR_MORE,
        default=(),
        type=worldbank_jurisdictions.iso_3166_str,
        help="capture only these countries and merge them into the existing artifact; "
        "defaults to every country the target set names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the plan without running the pipeline",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="the jdluc checkout, read for code_version",
    )
    args = parser.parse_args()

    iso_3166s = get_iso_3166s(isos=tuple(args.isos))
    print(render_plan(iso_3166s=iso_3166s))
    if args.dry_run:
        return 0

    captured = workflow(iso_3166s=iso_3166s, repo_root=args.repo_root)
    merged = merge_efs(captured=captured, existing=read_efs())
    pull.CAPTURE.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(pull.EFS)
    print(f"\nWrote {len(merged):,d} row(s) to {pull.EFS}")

    # Only for the countries this run recomputed: a pool beside emissions from a different run
    # compares a numerator and a denominator built by different code.
    pools = get_forest_pools(captured=merged, iso_3166s=iso_3166s)
    pools["code_version"] = schema.get_code_version(repo_root=args.repo_root)
    pools.to_parquet(pull.FOREST_POOLS)
    over = pools[pools["attributed_tonnes"] > pools["pool_tonnes"]]
    print(
        f"Wrote {len(pools):d} forest pool(s) to {pull.FOREST_POOLS}; {len(over):d} jurisdiction(s) "
        f"attribute more than the pool holds{':' if len(over) else '.'}"
    )
    for row in over.to_dict("records"):
        print(
            f"  {row['iso_3166']!s} {row['attributed_tonnes'] / row['pool_tonnes']:.1%}"
        )
    # The exit code does not depend on the findings; see `__main__`. A conservation overrun is a
    # result to be reported, and a capture that ran is a capture that succeeded.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
