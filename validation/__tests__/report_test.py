"""Rendering and provenance checks that a real run cannot make for us.

What these have in common is that the failure is invisible in the output. The report is Markdown, so
a malformed cell is not an error -- it renders as a broken table nobody notices -- and a body that
overruns GitHub's limit is not an error either, it is silently truncated. So: a pipe that ends a cell
early, an empty table reading as agreement, a shape-only ratio indistinguishable from a level, a
control checked with the wrong arithmetic, and a table drawn from two runs of the pipeline.
"""

import pandas
import pytest

from validation import report, schema
from validation.__tests__ import fixture


def test_a_pipe_in_a_cell_is_escaped() -> None:
    """Every provincial jurisdiction_name looks like "Angola | Bengo", which would end the cell."""
    table = report.format_markdown_table(
        headers=("Jurisdiction", "ratio"), rows=[["Angola | Bengo", "0.400"]]
    )
    row = table.splitlines()[-1]
    assert row == r"| Angola \| Bengo | 0.400 |"
    # Three cells, not four: the escape is what keeps the row the width of its header.
    assert row.count(" | ") == len(("Jurisdiction", "ratio")) - 1


def test_an_empty_table_says_so() -> None:
    """A header with no body reads as "nothing wrong" rather than "nothing measured"."""
    assert report.format_markdown_table(headers=("a", "b"), rows=[]) == "_(no rows)_"


def test_a_missing_value_renders_as_a_dash() -> None:
    assert report.format_number(None) == "—"
    assert report.format_number(float("nan")) == "—"
    assert report.format_number(0.4) == "0.400"


def test_conservation_fires_only_above_the_pool() -> None:
    """The bound is the one check that needs no anchor, so it has to be right on its own."""
    pools = pandas.DataFrame.from_records(
        data=[
            {"iso_3166": "XAA", "attributed_tonnes": 9.0e6, "pool_tonnes": 1.0e7},
            {"iso_3166": "XAB", "attributed_tonnes": 2.5e7, "pool_tonnes": 1.0e7},
        ]
    )
    (finding,) = report.get_conservation_findings(forest_pools=pools)
    assert finding.severity == schema.Severity.BLOCKING
    assert finding.affected_iso_3166s == ("XAB",)
    assert finding.magnitude_tonnes == 1.5e7


@pytest.mark.parametrize(
    ("slug", "fires"),
    (
        pytest.param(
            "control-xaf-soybean-orbae_over_wri",
            False,
            id="0.057-in-correlation-units-is-inside-the-0.10-band",
        ),
        pytest.param(
            "control-xag-soybean-orbae_over_wri",
            True,
            id="0.243-is-outside-it-under-either-arithmetic",
        ),
        pytest.param(
            "control-xaa-maize-sluc_over_wri",
            False,
            id="a-ratio-within-its-relative-band",
        ),
        pytest.param(
            "control-xab-soybean-sluc_over_wri",
            True,
            id="and-one-well-outside-it",
        ),
    ),
)
def test_a_control_fires_by_the_arithmetic_its_statistic_implies(
    slug: str, fires: bool
) -> None:
    """A relative band makes the weakest-agreeing control the twitchiest, which is backwards.

    XAF is the row where the two arithmetics disagree: +0.443 to +0.500 is 0.057 in correlation
    units, inside the 0.10 band, but 12.9% of the baseline, so a relative test fires on it and an
    absolute one must not. `schema.Statistic` promises a correlation is never "tolerance-checked as
    though it were a ratio"; this is the row that holds that promise.
    """
    fired = {
        finding.slug
        for finding in report.get_control_findings(
            comparisons=fixture.get_comparisons()
        )
    }
    assert (slug in fired) is fires


def test_a_control_whose_anchors_moved_is_advisory_not_a_defect() -> None:
    """A moved yardstick is not a moved pipeline, and calling it a DEFECT books one as the other.

    XAH is 0.900 against a baseline of 0.443 -- far outside any tolerance -- so if the anchor
    version were ignored it would be the loudest DEFECT in the fixture.
    """
    findings = {
        finding.slug: finding
        for finding in report.get_control_findings(
            comparisons=fixture.get_comparisons()
        )
    }
    assert "control-xah-soybean-orbae_over_wri" not in findings
    stale = findings["stale-baseline-xah-soybean-orbae_over_wri"]
    assert stale.severity == schema.Severity.ADVISORY
    assert fixture.SUPERSEDED_SOURCE_VERSION in stale.message


def test_borrowed_evidence_is_reported_at_all() -> None:
    """BORROWED means no recorded provenance, which the figure agreeing does not repair.

    `schema.SourceTier` calls it sufficient for ranking a magnitude and not for a claim, so the row
    has to be named even when nothing about its number looks wrong.
    """
    findings = {
        finding.slug: finding
        for finding in report.get_provenance_findings(
            comparisons=fixture.get_comparisons()
        )
    }
    borrowed = findings["borrowed-evidence"]
    assert borrowed.severity == schema.Severity.ADVISORY
    assert borrowed.confidence == schema.Confidence.LOW
    assert borrowed.affected_rows == 1
    assert borrowed.affected_iso_3166s == ("XAB",)


def test_a_table_spanning_two_code_versions_is_a_defect() -> None:
    """A per-ISO merge keeps rows it did not recompute, so one artifact can hold two versions.

    This is a disagreement between two runs of ours rather than between two sources, so no anchor
    can catch it and nothing about the numbers looks wrong.
    """
    comparisons = fixture.get_comparisons()
    assert "mixed-code-version" not in {
        finding.slug
        for finding in report.get_provenance_findings(comparisons=comparisons)
    }
    comparisons.loc[comparisons.index[0], "code_version"] = (
        fixture.SUPERSEDED_CODE_VERSION
    )
    findings = {
        finding.slug: finding
        for finding in report.get_provenance_findings(comparisons=comparisons)
    }
    mixed = findings["mixed-code-version"]
    assert mixed.severity == schema.Severity.DEFECT
    # Both versions, since the fix is to recapture whichever of the two is the stale one.
    assert fixture.CODE_VERSION in mixed.message
    assert fixture.SUPERSEDED_CODE_VERSION in mixed.message
    assert mixed.affected_rows == len(comparisons)


def test_every_pattern_only_row_is_labelled_as_shape_and_not_level() -> None:
    """A shape-only ratio and a level one are the same number; only the column separates them.

    Asserted over the whole table rather than one row, because labelling most of them is the same
    failure as labelling none: a reader who finds the column trustworthy reads every unlabelled row
    as a level.
    """
    comparisons = fixture.get_comparisons()
    # Every invented country is XA-something, which is what distinguishes a body row from the
    # header and the alignment row without parsing the table.
    rows = [
        line
        for line in report.render_comparisons(comparisons=comparisons).splitlines()
        if line.startswith("| XA")
    ]
    assert len(rows) == len(comparisons)
    assert len([line for line in rows if "| pattern_only |" in line]) == int(
        (comparisons["comparability"] == schema.Comparability.PATTERN_ONLY).sum()
    )
    # One pair, two pools: WRI publishes no production at the provincial grain, so the per-kg
    # factor carries shape while the rollup carries level. The label is the only difference.
    (shape,) = [line for line in rows if line.startswith("| XAC OILPALM | forest |")]
    (level,) = [line for line in rows if line.startswith("| XAC OILPALM | total |")]
    assert "| pattern_only |" in shape
    assert "| level |" in level
    # And the rollup carries its coverage, because 62% of a country on our own provincial weights
    # is not the anchor's published national figure.
    assert "| rolled_up (62%) |" in level


def test_the_document_leads_with_findings_then_conservation() -> None:
    """A table invites a conclusion the findings may already have disqualified, so order matters."""
    document = report.render(
        comparisons=fixture.get_comparisons(),
        forest_pools=fixture.get_forest_pools(),
        unanchored=fixture.get_unanchored_targets(),
    )
    for earlier, later in (
        ("### Findings", "### Forest-pool conservation"),
        ("### Forest-pool conservation", "### Comparisons"),
        ("### Comparisons", "### Coverage"),
    ):
        assert document.index(earlier) < document.index(later), (
            f"{earlier} after {later}"
        )


def test_coverage_names_the_targets_no_anchor_covered() -> None:
    """Silence is not agreement; an omitted target reads as a checked one."""
    document = report.render(
        comparisons=fixture.get_comparisons(),
        forest_pools=fixture.get_forest_pools(),
        unanchored=fixture.get_unanchored_targets(),
    )
    assert "XAE-WHEAT" in document


def test_the_size_note_appears_only_past_the_limit() -> None:
    """GitHub truncates an over-long body silently, so the boundary is the whole behaviour."""
    assert report.get_size_note(document="x" * report.PULL_REQUEST_BODY_LIMIT) == ""
    over = "x" * (report.PULL_REQUEST_BODY_LIMIT + 1)
    note = report.get_size_note(document=over)
    # Both numbers: the note is only actionable if it says how far over the document is.
    assert f"{len(over):,d}" in note
    assert f"{report.PULL_REQUEST_BODY_LIMIT:,d}" in note


def test_conservation_says_it_has_not_run_rather_than_going_missing() -> None:
    """`forest_pools` is None until a capture exists, so this is the path every run takes today.

    An omitted section reads as a passing one, and this is the check that outranks every anchor: a
    country attributing more forest emissions than its pool holds makes its comparisons moot.
    """
    document = report.render(
        comparisons=fixture.get_comparisons(),
        forest_pools=None,
        unanchored=fixture.get_unanchored_targets(),
    )
    assert "### Forest-pool conservation" in document
    assert "Not run" in document
    # Still in its place ahead of the tables, so nothing below reads as qualified by it.
    assert document.index("### Forest-pool conservation") < document.index(
        "### Comparisons"
    )


def test_a_blocking_finding_cannot_be_buried_under_a_larger_advisory() -> None:
    """Severity first, then magnitude: an advisory worth 90 Mt still sorts below a blocking one."""

    def get_finding(
        slug: str, severity: schema.Severity, magnitude: float | None
    ) -> schema.Finding:
        return schema.Finding(
            slug=slug,
            severity=severity,
            message="Invented",
            confidence=schema.Confidence.HIGH,
            magnitude_tonnes=magnitude,
        )

    rendered = report.render_findings(
        findings=[
            get_finding("big-advisory", schema.Severity.ADVISORY, 9.0e7),
            get_finding("small-blocking", schema.Severity.BLOCKING, 1.0e6),
            get_finding("big-blocking", schema.Severity.BLOCKING, 5.0e7),
            get_finding("unsized-defect", schema.Severity.DEFECT, None),
        ]
    )
    assert [
        line.split("**")[1] for line in rendered.splitlines() if line.startswith("- `")
    ] == ["big-blocking", "small-blocking", "unsized-defect", "big-advisory"]
