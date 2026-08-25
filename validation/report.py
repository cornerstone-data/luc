"""Render the comparisons into Markdown: what broke, what disagrees, and what went unchecked.

Every filter, rescale, factorization and rollup has already happened in `prepare`, so a section here
is a pivot plus a caption. Nothing in this module reads a file or computes a quantity; if a renderer
needs a number that is not in its arguments, the number belongs in `prepare`.

Three ordering choices matter more than they look.

The conservation bound goes first, ahead of every anchor. A country whose per-crop forest emissions
exceed the pool available to allocate has an attribution bug, and every anchor comparison for it is
moot until that is fixed -- so it is not a caveat on the tables below, it is the reason not to read
them yet.

Findings come before tables, sorted by severity then magnitude, because a table invites the reader
to draw a conclusion the findings may have already disqualified.

Coverage comes last but is not optional. A report that silently omits the targets no anchor covered
reads as though it checked them, and an empty cell is indistinguishable from agreement unless
something says otherwise.

Sections are individually renderable so a slice can be pasted into a PR description: GitHub caps a
body at 65,536 characters, and the full set over provincial rows will exceed it. No HTML and no
footnote syntax, both unreliable there.
"""

import typing

import pandas

from validation import schema, targets

# GitHub's pull-request body limit. Exceeding it truncates silently, which is worse than refusing.
PULL_REQUEST_BODY_LIMIT = 65_536


def format_number(value: float | None, precision: int = 3) -> str:
    if value is None or pandas.isna(value):
        return "—"
    return f"{value:.{precision}f}"


def escape_cell(value: str) -> str:
    """Escape a pipe, which would otherwise end the cell and break the row.

    Not hypothetical: every provincial `jurisdiction_name` is of the form "Angola | Bengo", so any
    table naming a province emits broken markdown without this. `tabulate` does not escape them
    either, so reaching for a library would not have covered it.
    """
    return value.replace("|", "\\|")


def format_markdown_table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    """A table, or an honest note that there was nothing to put in one.

    An empty table renders as a header with no body, which reads as "nothing wrong" rather than
    "nothing measured", so the two are distinguished here.
    """
    if not rows:
        return "_(no rows)_"
    alignments = ["---" if index == 0 else "--:" for index in range(len(headers))]
    return "\n".join(
        [
            "| " + " | ".join(escape_cell(value=header) for header in headers) + " |",
            "| " + " | ".join(alignments) + " |",
            *[
                "| " + " | ".join(escape_cell(value=value) for value in row) + " |"
                for row in rows
            ],
        ]
    )


def get_conservation_findings(forest_pools: pandas.DataFrame) -> list[schema.Finding]:
    """A finding per jurisdiction attributing more forest emissions than its pool holds.

    The pool is forest conversion over all 2020 cropland with no crop share applied, so it bounds
    what
    any allocation can distribute. Exceeding it is an attribution error rather than a disagreement,
    which is why these are BLOCKING and are reported before any comparison.

    The pool and the attributed total must come from the same run: mixing a pool from one code
    version
    with a capture from another compares a numerator and denominator computed by different code.
    """
    findings = []
    for row in forest_pools.to_dict("records"):
        attributed, pool = float(row["attributed_tonnes"]), float(row["pool_tonnes"])
        share = attributed / pool
        if share <= 1.0:
            continue
        findings.append(
            schema.Finding(
                slug="forest-pool-conservation",
                severity=schema.Severity.BLOCKING,
                message=(
                    f"{row['iso_3166']!s} attributes {share:.1%} of its forest pool "
                    f"({attributed:,.0f} t against {pool:,.0f} t). No "
                    "allocation can distribute more than the pool holds, so this is an error and "
                    "every comparison for this jurisdiction is moot until it is fixed"
                ),
                confidence=schema.Confidence.HIGH,
                magnitude_tonnes=attributed - pool,
                affected_rows=1,
                affected_iso_3166s=(str(row["iso_3166"]),),
            )
        )
    return findings


def get_control_findings(comparisons: pandas.DataFrame) -> list[schema.Finding]:
    """A finding per control that has moved beyond tolerance from its frozen baseline.

    Controls fire against `baseline` only. An inherited figure is a sanity check applied when the
    baseline is set, not a threshold; see `targets.Control`.

    Three outcomes, not two. A control with no baseline is carried rather than armed and is filtered
    out here. One whose anchors have moved since it was frozen is ADVISORY: its baseline describes a
    different yardstick, so the movement cannot be attributed, and calling that a DEFECT would book
    an anchor revision as a change in our own pipeline. Only a control measured against the anchors
    it was frozen against can produce a DEFECT -- and a measure with no external term skips that
    check entirely, having no yardstick that could have moved.

    How far a control has moved depends on what its baseline is. A RATIO is compared relatively,
    having no natural scale. A RANK_CORRELATION is compared absolutely, because rho lives on
    [-1, 1]: dividing by the baseline would make the weakest-agreeing control the twitchiest -- at
    +0.443 a relative 10% fires on a move of 0.044 where +0.886 tolerates 0.089 -- and a baseline
    near zero would fire on noise. `schema.Statistic` exists to keep the two apart, and it is read
    strictly so that a row which cannot say which it is fails rather than defaulting to the ratio
    arithmetic.
    """
    controls = comparisons[comparisons["is_control"] & comparisons["baseline"].notna()]
    findings = []
    for row in controls.to_dict("records"):
        value, baseline = float(row["ratio"]), float(row["baseline"])
        tolerance = float(row["tolerance"])
        frozen_against, ran_against = (
            str(row["baseline_source_version"]),
            str(row["source_version"]),
        )
        if row["measure"].is_anchored and frozen_against != ran_against:
            # Not a DEFECT: nothing is wrong, the yardstick is simply a different one, and reading
            # the movement would book an anchor revision as a change in our pipeline.
            findings.append(
                schema.Finding(
                    slug=(
                        f"stale-baseline-{str(row['iso_3166']).lower():s}"
                        f"-{str(row['crop_name']).lower():s}"
                        f"-{row['measure'].name.lower():s}"
                    ),
                    severity=schema.Severity.ADVISORY,
                    message=(
                        f"{row['iso_3166']!s} {row['crop_name']!s} {row['measure'].name:s} is "
                        f"frozen against {frozen_against:s} but this run read {ran_against:s}, so "
                        f"its baseline of {baseline:.3f} describes a different yardstick and the "
                        f"movement to {value:.3f} cannot be attributed. Re-freeze it against the "
                        "current anchors rather than reading it"
                    ),
                    confidence=schema.Confidence(row["confidence"]),
                    affected_rows=1,
                    affected_iso_3166s=(str(row["iso_3166"]),),
                )
            )
            continue
        # pandas does not preserve enum identity through a frame, so restore it from the value.
        statistic = schema.Statistic(row["statistic"])
        match statistic:
            case schema.Statistic.RATIO:
                movement = abs(value - baseline) / abs(baseline)
                moved = f"{movement:.1%} away with a tolerance of {tolerance:.0%}"
                addendum = ""
            case schema.Statistic.RANK_CORRELATION:
                movement = abs(value - baseline)
                moved = (
                    f"{movement:.3f} away in correlation units with a tolerance of "
                    f"{tolerance:.3f}"
                )
                addendum = (
                    ". This one holds a rank correlation rather than a ratio, so the movement is "
                    "in the ordering across jurisdictions, not in a magnitude"
                )
            case _:
                typing.assert_never(statistic)
        if movement <= tolerance:
            continue
        findings.append(
            schema.Finding(
                slug=(
                    f"control-{str(row['iso_3166']).lower():s}-{str(row['crop_name']).lower():s}"
                    f"-{row['measure'].name.lower():s}"
                ),
                severity=schema.Severity.DEFECT,
                message=(
                    f"{row['iso_3166']!s} {row['crop_name']!s} {row['measure'].name:s} is "
                    f"{value:.3f} against a baseline of {baseline:.3f}, {moved:s}. A control "
                    "moved, so something changed that was not supposed to" + addendum
                ),
                # pandas stores an IntEnum column as int64, so the enum identity does not survive
                # a round trip through the frame and has to be restored here
                confidence=schema.Confidence(row["confidence"]),
                affected_rows=1,
                affected_iso_3166s=(str(row["iso_3166"]),),
            )
        )
    return findings


def get_provenance_findings(comparisons: pandas.DataFrame) -> list[schema.Finding]:
    """Borrowed evidence, and tables built from more than one `code_version`.

    A mixed-version table is a defect: a cross-country comparison drawn from two pipeline versions
    is not a comparison. Checked at read time, so it holds regardless of how the artifact was
    assembled.
    """
    findings = []
    versions = sorted(set(comparisons["code_version"]))
    if len(versions) > 1:
        findings.append(
            schema.Finding(
                slug="mixed-code-version",
                severity=schema.Severity.DEFECT,
                message=(
                    f"These rows span {len(versions):d} code versions "
                    f"({', '.join(versions)}). Rows carried forward by a per-ISO merge keep "
                    "their own version, so a cross-country comparison here would mix "
                    "pipeline versions"
                ),
                confidence=schema.Confidence.HIGH,
                affected_rows=len(comparisons),
            )
        )
    borrowed = comparisons[comparisons["worst_tier"] == schema.SourceTier.BORROWED]
    if len(borrowed):
        findings.append(
            schema.Finding(
                slug="borrowed-evidence",
                severity=schema.Severity.ADVISORY,
                message=(
                    f"{len(borrowed):d} comparison(s) rest on borrowed inputs with no "
                    "provenance. Usable for ranking a magnitude, never for a claim"
                ),
                confidence=schema.Confidence.LOW,
                affected_rows=len(borrowed),
                affected_iso_3166s=tuple(sorted(set(borrowed["iso_3166"]))),
            )
        )
    return findings


def render_findings(findings: list[schema.Finding]) -> str:
    """Severity first, then magnitude, so a blocking result cannot be buried under advisories."""
    if not findings:
        return "### Findings\n\n_None._"
    ordered = sorted(
        findings,
        key=lambda finding: (
            list(schema.Severity).index(finding.severity),
            -(finding.magnitude_tonnes or 0.0),
        ),
    )
    lines = []
    for finding in ordered:
        magnitude = (
            f" [{finding.magnitude_tonnes / schema.TONNES_PER_MEGATONNE:,.1f} Mt]"
            if finding.magnitude_tonnes
            else ""
        )
        lines.append(
            f"- `{finding.severity.marker:s}` **{finding.slug:s}**{magnitude:s} "
            f"(confidence {finding.confidence.name.lower():s}) — {finding.message:s}"
        )
    return "### Findings\n\n" + "\n".join(lines)


def render_conservation(forest_pools: pandas.DataFrame) -> str:
    rows = [
        [
            str(row["iso_3166"]),
            format_number(
                float(row["attributed_tonnes"]) / schema.TONNES_PER_MEGATONNE, 2
            ),
            format_number(float(row["pool_tonnes"]) / schema.TONNES_PER_MEGATONNE, 2),
            f"{float(row['attributed_tonnes']) / float(row['pool_tonnes']):.1%}",
            "over"
            if float(row["attributed_tonnes"]) > float(row["pool_tonnes"])
            else "ok",
        ]
        for row in forest_pools.sort_values(
            "attributed_tonnes", ascending=False
        ).to_dict("records")
    ]
    return (
        "### Forest-pool conservation\n\n"
        + format_markdown_table(
            headers=("Country", "attributed (Mt)", "pool (Mt)", "share", ""), rows=rows
        )
        + "\n\n_The pool is forest conversion over all 2020 cropland with no crop share "
        "applied, so it bounds what any allocation can hand out. Needs no anchor and no "
        "baseline, which is why it is reported first: a country over 100% has an attribution "
        "bug, and its anchor comparisons below are not yet worth reading._"
    )


def render_comparisons(comparisons: pandas.DataFrame) -> str:
    """The target rows in full, and everything else summarised by measure.

    `comparisons` reaches well beyond the chosen set -- every country where an anchor and our capture
    happen to overlap -- and tabulating all of it costs more characters than a pull-request body
    holds. The chosen pairs are what the report is about, so those are listed; the rest is a
    distribution per measure rather than nothing, because a row that is silently dropped reads as a
    row that was never computed.

    `comparability` and `aggregation` are columns rather than footnotes because a `PATTERN_ONLY`
    ratio and a `LEVEL` one look identical otherwise, and a rollup on our own weights is not the
    anchor's published figure.
    """
    on_target = comparisons[comparisons["is_target"]]
    rows = [
        [
            f"{row['iso_3166']!s} {row['crop_name']!s}",
            str(row["emission_pool"]),
            str(row["measure"].name),
            str(row["statistic"]),
            format_number(float(row["ratio"]), 3),
            format_number(
                None if pandas.isna(row["baseline"]) else float(row["baseline"]), 3
            ),
            str(row["comparability"]),
            str(row["aggregation"])
            + (
                f" ({float(row['coverage_fraction']):.0%})"
                if row["aggregation"] == schema.Aggregation.ROLLED_UP
                else ""
            ),
            format_number(row["anchor_deforestation_share"], 3),
            schema.Confidence(row["confidence"]).name.lower(),
        ]
        for row in on_target.sort_values(
            ["iso_3166", "crop_name", "emission_pool"]
        ).to_dict("records")
    ]
    rest = comparisons[~comparisons["is_target"]]
    # Iterated over the distinct measures rather than grouped, because a groupby key types as
    # Hashable and a measure carries the name this table is sorted and labeled by.
    summary = []
    for measure in sorted(set(rest["measure"]), key=lambda value: value.name):
        group = rest[rest["measure"] == measure]
        summary.append(
            [
                f"{measure.name:s} ({group['statistic'].iloc[0]!s})",
                f"{len(group):d}",
                format_number(float(group["ratio"].median()), 3),
                format_number(float(group["ratio"].quantile(0.1)), 3),
                format_number(float(group["ratio"].quantile(0.9)), 3),
            ]
        )
    return "\n\n".join(
        [
            f"### Comparisons — {len(on_target):d} rows over "
            f"{len(set(zip(on_target['iso_3166'], on_target['crop_name'], strict=True))):d} "
            "chosen pairs",
            format_markdown_table(
                headers=(
                    "Target",
                    "Pool",
                    "Measure",
                    "statistic",
                    "value",
                    "baseline",
                    "comparability",
                    "aggregation",
                    "deforestation share",
                    "confidence",
                ),
                rows=rows,
            ),
            "_`pattern_only` rows have no production at the provincial grain, so they carry shape "
            "and not level. A `rank_correlation` row's value is a correlation rather than a ratio, "
            "and its baseline moves in correlation units. `rolled_up` rows use our provincial "
            "weights over the coverage shown, so they are not the anchor's own national figure — "
            "the difference between the two is itself a measurement. `deforestation share` is how much of "
            "the crop's harvested area WRI treats as deforestation-linked; above 1.0 the anchor "
            "contradicts itself, since a crop cannot be grown on more land than it is harvested "
            "from._",
            f"#### The other {len(rest):d} comparisons, which reach beyond the chosen set",
            format_markdown_table(
                headers=("Measure", "rows", "median", "p10", "p90"), rows=summary
            ),
            "_These are every pair where an anchor and the capture happen to overlap. They are not "
            "targets and carry no expectation, but they are what a chosen pair's figure should be "
            "read against — a control at the median of its own measure is a different claim from "
            "one in the tail._",
        ]
    )


def render_coverage(
    comparisons: pandas.DataFrame, unanchored: tuple[targets.Target, ...]
) -> str:
    """What was not checked, which an empty cell cannot say for itself."""
    lines = [
        f"- {len(comparisons):d} comparison(s) over "
        f"{comparisons['iso_3166'].nunique():d} countries.",
        f"- {int(comparisons['is_target'].sum()):d} of them are targets.",
    ]
    if unanchored:
        lines.append(
            f"- **{len(unanchored):d} target(s) had no anchor at all**: "
            + ", ".join(target.slug for target in unanchored)
            + ". Their silence is not agreement."
        )
    pattern_only = int(
        (comparisons["comparability"] == schema.Comparability.PATTERN_ONLY).sum()
    )
    if pattern_only:
        lines.append(
            f"- {pattern_only:d} comparison(s) are shape-only and must not be read as levels."
        )
    lines.append(
        "- Grassland and peat have no external anchor in any run: WRI is forest-only, Orbae "
        "is defective on both, and our carbon densities are the datasets we would check "
        "against. Those pools can be sized, never confirmed."
    )
    return "### Coverage\n\n" + "\n".join(lines)


def render_anchor_emissions(
    deforestation: pandas.DataFrame,
    findings: list[schema.Finding],
    scope_ratios: pandas.Series,
    stability: pandas.DataFrame,
) -> str:
    """WRI measured against itself, which is all an emissions section can do before a capture.

    Three facts, none needing a number of ours. The reporting-year spread bounds what a comparison
    against WRI can resolve while the year is unpinned; the scope ratio removes any guessing about
    which gas basis an outside figure used; and the deforestation share is a share against a share,
    which is what our expansion share should be compared against once it exists.
    """
    quantiles = [0.5, 0.9, 0.99]
    rows = [
        [
            "WRI reporting-year spread, max/min over 2020-2024",
            *[
                format_number(float(stability["spread"].quantile(q)), 2)
                for q in quantiles
            ],
        ],
        [
            "CO2e / CO2, same country, crop and year",
            *[format_number(float(scope_ratios.quantile(q)), 4) for q in quantiles],
        ],
        [
            "deforestation-linked share of harvested area",
            *[
                format_number(
                    float(deforestation["deforestation_share"].quantile(q)), 3
                )
                for q in quantiles
            ],
        ],
    ]
    return "\n\n".join(
        [
            "### WRI against itself",
            format_markdown_table(headers=("Measure", "p50", "p90", "p99"), rows=rows),
            render_findings(findings=findings),
            "_The spread is WRI's own five reporting years for one country and crop. Those are a "
            "series rather than five estimates of one quantity -- each column slides the LSRS "
            "20-year window forward over both the loss series and the production denominator -- so "
            "the spread is the price of leaving the reporting year unpinned, and the ceiling on "
            "what any sLUC/WRI comparison can resolve without pinning it._",
        ]
    )


def render_eligible(
    eligible: pandas.DataFrame, findings: list[schema.Finding], show: int
) -> str:
    """The eligible shortlist and the figures to choose from -- not a chosen set.

    Ordered by the anchor's own deforestation figure, so the table reads top-down. The choosing
    happens in `data/targets.json`, where each pair carries a written reason.
    """
    rows = [
        [
            f"{row['iso_3166']!s} {row['crop_name']!s}",
            f"{float(row['deforestation_tonnes']) / schema.TONNES_PER_MEGATONNE:,.2f}",
            f"{float(row['production_kg']) / schema.KG_PER_TONNE / 1e6:,.2f}",
            "".join(
                marker
                for marker, present in (
                    ("P", row["is_perennial"]),
                    ("D", row["is_decomposed_group_crop"]),
                )
                if present
            )
            or "\u2014",
            f"{int(row['provincial_units']):d} / {float(row['area_coverage']):.0%}",
        ]
        for row in eligible.head(show).to_dict("records")
    ]
    return "\n\n".join(
        [
            f"### Eligible pairs \u2014 {len(eligible):d} passed E1\u2013E4, "
            f"largest {min(show, len(eligible)):d} by WRI deforestation",
            format_markdown_table(
                headers=(
                    "Pair",
                    "WRI deforestation (Mt)",
                    "production (Mt)",
                    "flags",
                    "provinces / area",
                ),
                rows=rows,
            ),
            render_findings(findings=findings),
            "_`P` a woody perennial, which GLAD\u2019s cropland class excludes by construction; "
            "`D` a crop "
            "MapSPAM decomposes rather than observes. This is a shortlist to choose from, not a "
            "selected set \u2014 see `data/targets.json` for what was chosen and why._",
        ]
    )


def render_yield_agreement(
    agreements: tuple[schema.YieldAgreement, ...],
    findings: list[schema.Finding],
    unpaired: dict[schema.UnpairedReason, tuple[str, ...]],
    year: int,
) -> str:
    """WRI's yields against FAOSTAT's, per crop.

    A denominator section rather than an emissions one, and it belongs ahead of the comparisons for
    the same reason conservation does: a crop whose two yields are on different product forms has no
    comparable emissions factor, so its rows below are not worth reading.
    """
    rows = [
        [
            agreement.crop_name,
            str(agreement.countries),
            format_number(agreement.median_ratio, 3),
            format_number(agreement.lowest_ratio, 2),
            format_number(agreement.highest_ratio, 2),
            (
                "product form"
                if agreement.is_product_form_mismatch
                else "off"
                if agreement.is_beyond_tolerance
                else "ok"
            ),
        ]
        for agreement in sorted(agreements, key=lambda a: a.median_ratio)
    ]
    sections = [
        f"### Yield agreement, WRI against FAOSTAT {year:d}",
        format_markdown_table(
            headers=("Crop", "countries", "median", "min", "max", ""), rows=rows
        ),
        render_findings(findings=findings),
    ]
    if unpaired:
        total = sum(len(names) for names in unpaired.values())
        sections.append(
            "\n".join(
                [
                    f"**{total:d} of WRI's crops are not compared.** Their absence is not "
                    "agreement.",
                    "",
                    *[
                        f"- `{', '.join(names)}` — {reason.value:s}"
                        for reason, names in unpaired.items()
                    ],
                ]
            )
        )
    return "\n\n".join(sections)


def render(
    comparisons: pandas.DataFrame,
    forest_pools: pandas.DataFrame | None,
    unanchored: tuple[targets.Target, ...],
    extra_findings: list[schema.Finding] | None = None,
) -> str:
    """The whole document, or as much of it as the available data supports.

    `forest_pools` is None until a capture exists, because the pool needs a raster pass over cached
    layers. The conservation section then says it has not run rather than being omitted: a missing
    section reads as a passing one, and this is the check that outranks every anchor.
    """
    findings = (
        (
            get_conservation_findings(forest_pools=forest_pools)
            if forest_pools is not None
            else []
        )
        + get_control_findings(comparisons=comparisons)
        + get_provenance_findings(comparisons=comparisons)
        + (extra_findings or [])
    )
    document = "\n\n".join(
        [
            "## LUC validation",
            render_findings(findings=findings),
            render_conservation(forest_pools=forest_pools)
            if forest_pools is not None
            else (
                "### Forest-pool conservation\n\n_Not run: the pool needs a raster pass over "
                "cached layers, so it arrives with the first capture. **This is the check that "
                "outranks every anchor**, so nothing below has been qualified by it — a country "
                "attributing more forest emissions than its pool holds would make its comparisons "
                "moot, and that has not yet been tested._"
            ),
            render_comparisons(comparisons=comparisons),
            render_coverage(comparisons=comparisons, unanchored=unanchored),
        ]
    )
    return document


def get_size_note(document: str) -> str:
    """A note appended where the *whole* document will not fit a pull-request body.

    It belongs to whatever assembles the sections, not to one of them: a section that measured only
    itself and then described "this document" would stay silent exactly when the assembled set went
    over, which is the case that matters.
    """
    if len(document) <= PULL_REQUEST_BODY_LIMIT:
        return ""
    return (
        f"\n\n_This document is {len(document):,d} characters and will not fit in a GitHub "
        f"pull-request body ({PULL_REQUEST_BODY_LIMIT:,d}). Render a subset of sections._"
    )
