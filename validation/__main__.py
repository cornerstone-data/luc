"""One entrypoint. Retrieve the anchors, compare against them, print the result.

  uv run python -m validation                       everything the anchors support
  uv run python -m validation --stage pull          retrieve and lock, nothing else
  uv run python -m validation --section comparisons only that section
  uv run python -m validation --year 2015           a different FAOSTAT vintage

**Sections are individually selectable because the whole document will not always fit.** GitHub caps a
pull-request body at 65,536 characters and the comparison table alone will exceed that once provincial
rows land, so a slice has to be pastable on its own. Every section stands alone for that reason.

**The exit code does not depend on the findings.** It is 0 whenever the run completed, however bad the
news, because a report that fails on a bad result cannot be used to characterize bad results. It fails
only on unreadable input or a missing join key. `tools/` is the deliberate exception: a generator that
emits a broken committed artifact exits nonzero.

`capture` is not a stage here. It runs our own pipeline, so it is `validation.capture`, run
by hand; until it has been run, the conservation bound and every measure with a term of ours is
reported as not-yet-run rather than omitted -- a missing section reads as a passing one.
"""

import argparse
import enum
import logging
import pathlib
import typing

from validation import prepare, pull, report, targets

logger = logging.getLogger(__name__)

# The eligible shortlist runs to hundreds of pairs, and the whole document has to fit a GitHub
# pull-request body. The findings are computed over every pair regardless of what is tabulated.
DEFAULT_ELIGIBLE_ROWS = 20


class Stage(enum.StrEnum):
    """What to run. `capture` is not here; see the module docstring."""

    PULL = enum.auto()
    REPORT = enum.auto()


class Section(enum.StrEnum):
    """The report's sections, in the order they are printed.

    Ordering is deliberate. `ELIGIBLE` comes first because it establishes the universe the rest is
    drawn from, and because it carries the one BLOCKING finding measurable without a capture: the
    pairs the pipeline cannot produce a comparable number for at all outrank any disagreement about
    a number it can. Denominators come next, because a product-form mismatch makes every factor
    built on that denominator incomparable. `COMPARISONS` carries the conservation bound and the
    findings, which outrank the anchor tables inside it.
    """

    ELIGIBLE = enum.auto()
    YIELDS = enum.auto()
    ANCHORS = enum.auto()
    COMPARISONS = enum.auto()


def get_document(
    repo_root: pathlib.Path, show: int, wanted: tuple[Section, ...], year: int
) -> str:
    """Render the requested sections, in `Section` order, joined into one document.

    Everything is computed once whether or not the section that uses it was asked for. None of these
    reads is expensive -- the largest is a 1.4 MiB parquet -- so a compute-only-what-you-need branch per
    section bought a little speed at the cost of four conditionals around the code that does the work.
    """
    wri_yields = prepare.read_wri_yields(grain_name="national")
    comparison = prepare.get_yield_comparison(
        faostat_yields=prepare.get_faostat_yields(), wri_yields=wri_yields, year=year
    )
    agreements = tuple(prepare.iter_yield_agreements(comparison=comparison))
    emissions = prepare.read_wri_national_emissions()
    stability = prepare.get_anchor_stability(emissions=emissions)
    deforestation = prepare.get_deforestation_share(
        emissions=emissions,
        faostat_areas=prepare.read_faostat_areas(year=year),
        year=year,
    )
    comparisons = prepare.get_comparisons(
        repo_root=repo_root, deforestation=deforestation
    )

    eligible = prepare.get_eligible()
    rendered = {
        Section.ELIGIBLE: report.render_eligible(
            eligible=eligible,
            findings=prepare.get_perennial_findings(eligible=eligible),
            show=show,
        ),
        Section.YIELDS: report.render_yield_agreement(
            agreements=agreements,
            findings=prepare.get_yield_findings(agreements=agreements, year=year),
            unpaired=prepare.get_unpaired_crop_names(
                comparison=comparison, wri_yields=wri_yields
            ),
            year=year,
        ),
        Section.ANCHORS: report.render_anchor_emissions(
            deforestation=deforestation,
            findings=(
                prepare.get_stability_findings(
                    stability=stability, tolerance=targets.DEFAULT_TOLERANCE
                )
                + prepare.get_deforestation_share_findings(deforestation=deforestation)
                + prepare.get_anchor_disagreement_findings(
                    agreement=prepare.get_anchor_shape_agreement()
                )
                + prepare.get_orbae_findings(frame=prepare.read_orbae())
            ),
            scope_ratios=prepare.get_scope_difference()["scope_ratio"],
            stability=stability,
        ),
        Section.COMPARISONS: report.render(
            comparisons=comparisons,
            # None until a capture has run: the pool needs a raster pass over cached layers, so
            # `validation.capture` derives it and writes it beside the emissions it bounds.
            forest_pools=prepare.read_forest_pools(),
            unanchored=prepare.get_unanchored_targets(comparisons=comparisons),
            # Target-keyed rather than comparison-keyed, so a pair no anchor reaches still says its
            # anchor contradicts itself; see `get_target_anchor_consistency_findings`.
            extra_findings=prepare.get_target_anchor_consistency_findings(
                deforestation=deforestation
            ),
        ),
    }
    assert set(rendered) == set(Section), "a Section has no rendered text"
    document = "\n\n".join(
        rendered[section] for section in Section if section in wanted
    )
    # Appended here rather than inside a section: only this function sees the assembled length, and
    # a section measuring itself would stay silent exactly when the whole set went over.
    return document + report.get_size_note(document=document)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=Stage,
        choices=tuple(Stage),
        default=Stage.REPORT,
        help="which stage to run",
    )
    parser.add_argument(
        "--section",
        action="append",
        type=Section,
        choices=tuple(Section),
        default=[],
        help="render only this section; repeatable, defaults to all of them",
    )
    parser.add_argument(
        "--overwrite",
        action="append",
        default=[],
        choices=("wri",),
        help="pull stage only: re-retrieve this source even where the digest matches",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        # `validation/` sits at the repo root once landed, so its parent is the checkout. While it
        # lives in a scratch tree outside the repo, pass --repo-root explicitly.
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="the jdluc checkout, read for code_version",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=DEFAULT_ELIGIBLE_ROWS,
        help="how many of the eligible pairs to tabulate; the findings cover all of them",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=prepare.REFERENCE_YEAR,
        help="the FAOSTAT year to compare WRI's undated yield against",
    )
    args = parser.parse_args()

    # Bound to a typed local so `assert_never` can see the enum: argparse hands back Any, against
    # which exhaustiveness cannot be checked. With the annotation, adding a Stage without a case is a
    # mypy error naming the member, and at runtime it raises rather than falling through to return 0.
    stage: Stage = args.stage
    match stage:
        case Stage.PULL:
            remotes = tuple(pull.iter_remotes())
            counts = pull.workflow(overwrite=tuple(args.overwrite), remotes=remotes)
            # Orbae is supplied rather than retrieved, so `pull` cannot fetch it -- but the lock
            # should still record every anchor's bytes, and five baselines rest on these.
            if prepare.ORBAE_EXPORT.exists():
                prepare.pin_orbae_export()
            print(
                f"\n{len(remotes):d} file(s): {counts['downloaded']:d} retrieved, "
                f"{counts['reused']:d} already current, {counts['changed']:d} changed upstream"
            )
        case Stage.REPORT:
            document = get_document(
                repo_root=args.repo_root,
                show=args.show,
                wanted=tuple(args.section) or tuple(Section),
                year=args.year,
            )
            print(document)
            if len(document) > report.PULL_REQUEST_BODY_LIMIT:
                # To stderr, so piping the document somewhere does not carry the warning into it.
                logger.warning(
                    f"{len(document):,d} characters exceeds GitHub's "
                    f"{report.PULL_REQUEST_BODY_LIMIT:,d}-character pull-request body limit; "
                    "render fewer sections with --section"
                )
        case _:
            typing.assert_never(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
