"""One entrypoint. Retrieve the anchors, compare a `trace` table against them, print the result.

  uv run python -m validation --table v202611.parquet                       every section
  uv run python -m validation --table v202611.parquet --section comparisons only that section
  uv run python -m validation --table v202611.parquet --year 2015           another FAOSTAT year
  uv run python -m validation --stage pull                                  retrieve and lock

`--table` is a parquet `trace` wrote: a release's table, or a branch's `trace.py <countries>
--methodology-name ...` runs merged into one, every leg in it compared at once. A check before a
release therefore needs a `trace` run of the target countries.

**Sections are individually selectable because the whole document will not always fit.** GitHub caps a
pull-request body at 65,536 characters and the comparison table alone will exceed that once provincial
rows land, so a slice has to be pastable on its own. Every section stands alone for that reason.

**The exit code does not depend on the findings.** It is 0 whenever the run completed, however bad the
news, because a report that fails on a bad result cannot be used to characterize bad results. It fails
only on unreadable input or a missing join key. `tools/` is the deliberate exception: a generator that
emits a broken committed artifact exits nonzero.
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
    """What to run."""

    PULL = enum.auto()
    REPORT = enum.auto()


class Section(enum.StrEnum):
    """The report's sections, in the order they are printed.

    Ordering is deliberate. `ELIGIBLE` comes first because it establishes the universe the rest is
    drawn from, and because it carries the one BLOCKING finding measurable from the anchors alone: the
    pairs the pipeline cannot produce a comparable number for at all outrank any disagreement about
    a number it can. `COMPARISONS` carries the findings, which outrank the anchor tables inside it.
    `TOTALS` comes last: it carries no findings, only our totals beside published ones.
    """

    ELIGIBLE = enum.auto()
    ANCHORS = enum.auto()
    COMPARISONS = enum.auto()
    TOTALS = enum.auto()


def get_document(
    show: int, table_path: pathlib.Path, wanted: tuple[Section, ...], year: int
) -> str:
    """Render the requested sections, in `Section` order, joined into one document.

    The document opens by naming the `trace` table it compares, by file name and digest: every
    figure of ours comes from that one table, so the line is what identifies them.

    Everything is computed once whether or not the section that uses it was asked for. None of these
    reads is expensive -- the largest is the `trace` table -- so a compute-only-what-you-need branch per
    section bought a little speed at the cost of four conditionals around the code that does the work.
    """
    table = prepare.read_trace_table(path=table_path)
    commodity_id_to_name = prepare.get_commodity_id_to_name()
    emissions = prepare.read_wri_national_emissions()
    stability = prepare.get_anchor_stability(emissions=emissions)
    deforestation = prepare.get_deforestation_share(
        emissions=emissions,
        faostat_areas=prepare.read_faostat_areas(year=year),
        year=year,
    )
    comparisons = prepare.get_comparisons(
        commodity_id_to_name=commodity_id_to_name,
        deforestation=deforestation,
        table=table,
    )

    uncompared = prepare.get_uncompared_targets(comparisons=comparisons, table=table)
    eligible = prepare.get_eligible(commodity_id_to_name=commodity_id_to_name)
    rendered = {
        Section.ELIGIBLE: report.render_eligible(
            eligible=eligible,
            findings=prepare.get_perennial_findings(eligible=eligible),
            show=show,
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
            extra_findings=prepare.get_target_anchor_consistency_findings(
                commodity_id_to_name=commodity_id_to_name, deforestation=deforestation
            ),
            uncompared=uncompared,
        ),
        Section.TOTALS: report.render_totals(
            iso_3166s=tuple(
                sorted({target.iso_3166 for target in targets.iter_targets()})
            ),
            leg_totals=prepare.get_leg_totals(table=table),
            outside=prepare.get_outside_totals(year=year),
        ),
    }
    assert set(rendered) == set(Section), "a Section has no rendered text"
    document = "\n\n".join(
        [
            f"_Compared against `{table_path.name:s}`, "
            f"sha256 `{pull.get_sha256(path=table_path)[:12]:s}`._",
            *[rendered[section] for section in Section if section in wanted],
        ]
    )
    # Appended here rather than inside a section: only this function sees the assembled length, and
    # a section measuring itself would stay silent exactly when the whole set went over.
    return document + report.get_size_note(document=document)


def main() -> int:
    logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=tuple(Stage),
        default=Stage.REPORT,
        help="which stage to run",
        type=Stage,
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=tuple(Section),
        default=[],
        help="render only this section; repeatable, defaults to all of them",
        type=Section,
    )
    parser.add_argument(
        "--overwrite",
        action="append",
        choices=("faostat", "wri"),
        default=[],
        help="pull stage only: re-retrieve this source even where the digest matches",
    )
    parser.add_argument(
        "--table",
        help="report stage only: the `trace` parquet to compare against the anchors",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--show",
        default=DEFAULT_ELIGIBLE_ROWS,
        help="how many of the eligible pairs to tabulate; the findings cover all of them",
        type=int,
    )
    parser.add_argument(
        "--year",
        default=prepare.REFERENCE_YEAR,
        help="the FAOSTAT year of harvested area the deforestation share divides by",
        type=int,
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
            assert args.table, "the report stage compares a `trace` table; pass --table"
            document = get_document(
                show=args.show,
                table_path=args.table,
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
