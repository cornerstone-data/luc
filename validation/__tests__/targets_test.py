"""Checks on the target set's reader, which a well-formed file cannot make for us.

`data/targets.json` satisfies every check in `check_document`, which is the difficulty: those
refusals are what stop a control from being dropped silently, taking its guard with it, and a passing
file exercises none of them. They are asserted here against invented documents.

The invariants over the real file are a different kind of check and are here too: not that the reader
refuses bad input, but that what it hands back preserves what the file said -- order, uniqueness, a
frozen baseline travelling with the yardstick it was frozen against, and a pair naming a crop and a
country that something downstream can resolve.
"""

import contextlib
import typing

import iso3166
import pytest

from jdluc import statistical
from validation import pull, schema, targets


def get_document(**overrides: typing.Any) -> dict[str, typing.Any]:
    """One target carrying one control, well-formed, with `overrides` applied to the control.

    Keyed on the control, where all but four of the checks look. A test names only the field it is
    breaking, so the diff from a document that passes is what the test is about.
    """
    control: dict[str, typing.Any] = {
        "emission_pool": schema.EmissionPool.FOREST.name,
        "measure": targets.Measure.SLUC_OVER_WRI.name,
        "inherited": 0.40,
        "baseline": 0.40,
        "baseline_source_versions": {"WRI": "fixture00000"},
        "note": "Invented",
    }
    control.update(overrides)
    return {
        "provenance": {"wri_revision": pull.WRI_REVISION},
        "targets": [
            {
                "iso_3166": "XAA",
                "crop_name": "MAIZE",
                "basis": "control",
                "reason": "Invented, to exercise the reader",
                "controls": [control],
            }
        ],
    }


def get_context(refusal: str | None) -> typing.Any:
    """Refuse with a message matching `refusal`, or accept when there is none to match.

    The checks below are truth tables rather than lists of failures: what a reader needs is the line
    between accepted and refused, and a table of refusals alone does not draw it.
    """
    if refusal is None:
        return contextlib.nullcontext()
    return pytest.raises(AssertionError, match=refusal)


def test_a_well_formed_document_is_accepted() -> None:
    """The anchor for every refusal below, each of which changes one field of this document."""
    targets.check_document(document=get_document())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param("measure", "SLUC_OVER_ATLANTIS", id="an-unknown-measure"),
        pytest.param("emission_pool", "MANTLE", id="an-unknown-emission-pool"),
        pytest.param("statistic", "VIBES", id="an-unknown-statistic"),
    ),
)
def test_a_control_naming_something_that_does_not_exist_is_refused(
    field: str, value: str
) -> None:
    """An unresolvable name would drop the control, and a dropped control takes its guard with it."""
    with pytest.raises(AssertionError, match=value):
        targets.check_document(document=get_document(**{field: value}))


@pytest.mark.parametrize(
    ("tolerance", "refusal"),
    (
        pytest.param(0.10, None, id="the-default-band"),
        pytest.param(0.01, None, id="a-tight-band-is-a-choice-not-a-mistake"),
        pytest.param(
            1.00, None, id="one-is-the-inclusive-end-useless-but-not-malformed"
        ),
        pytest.param(0.00, "outside", id="zero-fires-on-everything"),
        pytest.param(1.50, "outside", id="above-one-fires-on-nothing"),
        pytest.param(-0.10, "outside", id="negative-is-not-a-band-at-all"),
    ),
)
def test_a_tolerance_is_read_against_the_unit_interval(
    tolerance: float, refusal: str | None
) -> None:
    """The bound holds under both readings of the number, which is why only the message differs."""
    with get_context(refusal=refusal):
        targets.check_document(document=get_document(tolerance=tolerance))


@pytest.mark.parametrize(
    ("measure", "baseline", "versions", "refusal"),
    (
        pytest.param(
            targets.Measure.SLUC_OVER_WRI,
            0.40,
            {"WRI": "fixture00000"},
            None,
            id="frozen-against-the-anchors-it-was-measured-with",
        ),
        pytest.param(
            targets.Measure.SLUC_OVER_WRI,
            None,
            None,
            None,
            id="not-yet-frozen-so-carried-rather-than-armed",
        ),
        pytest.param(
            targets.Measure.SLUC_OVER_WRI,
            0.40,
            None,
            "baseline_source_versions",
            id="frozen-with-no-yardstick-cannot-attribute-its-own-movement",
        ),
        pytest.param(
            targets.Measure.SLUC_OVER_WRI,
            None,
            {"WRI": "fixture00000"},
            "baseline_source_versions",
            id="a-yardstick-with-nothing-frozen-against-it-describes-nothing",
        ),
        pytest.param(
            targets.Measure.SLUC_OVER_JDLUC,
            0.40,
            None,
            None,
            id="both-terms-ours-so-code-version-is-the-whole-provenance",
        ),
        pytest.param(
            targets.Measure.SLUC_OVER_JDLUC,
            0.40,
            {"WRI": "fixture00000"},
            "divides no external anchor",
            id="an-anchor-version-on-a-measure-with-no-anchor-is-a-fiction",
        ),
    ),
)
def test_a_baseline_travels_with_the_anchors_it_was_frozen_against(
    measure: targets.Measure,
    baseline: float | None,
    versions: dict[str, str] | None,
    refusal: str | None,
) -> None:
    """Either half alone cannot tell an anchor revision from a change in our own pipeline."""
    with get_context(refusal=refusal):
        targets.check_document(
            document=get_document(
                measure=measure.name,
                baseline=baseline,
                baseline_source_versions=versions,
            )
        )


@pytest.mark.parametrize(
    ("statistic", "field", "value", "refusal"),
    (
        pytest.param(
            schema.Statistic.RATIO,
            "baseline",
            2.50,
            None,
            id="a-ratio-above-one-is-ordinary",
        ),
        pytest.param(
            schema.Statistic.RATIO,
            "baseline",
            0.008,
            None,
            id="and-so-is-a-very-small-one",
        ),
        pytest.param(
            schema.Statistic.RATIO,
            "baseline",
            0.00,
            "baseline",
            id="but-zero-is-not-a-ratio",
        ),
        pytest.param(
            schema.Statistic.RATIO,
            "baseline",
            -0.40,
            "baseline",
            id="nor-is-a-negative-one",
        ),
        pytest.param(
            schema.Statistic.RATIO,
            "inherited",
            -0.40,
            "inherited",
            id="checked-on-both-numbers",
        ),
        pytest.param(
            schema.Statistic.RANK_CORRELATION,
            "baseline",
            -0.40,
            None,
            id="an-ordering-may-disagree-so-a-negative-rho-is-fine",
        ),
        pytest.param(
            schema.Statistic.RANK_CORRELATION,
            "baseline",
            1.00,
            None,
            id="perfect-agreement-is-the-inclusive-bound",
        ),
        pytest.param(
            schema.Statistic.RANK_CORRELATION,
            "baseline",
            1.40,
            "baseline",
            id="past-it-is-not-a-correlation",
        ),
        pytest.param(
            schema.Statistic.RANK_CORRELATION,
            "inherited",
            -1.40,
            "inherited",
            id="in-either-direction-and-on-both-numbers",
        ),
    ),
)
def test_a_number_is_read_in_the_units_its_statistic_implies(
    statistic: schema.Statistic, field: str, value: float, refusal: str | None
) -> None:
    """The same -0.40 is malformed as a ratio and ordinary as a correlation, which is the point."""
    with get_context(refusal=refusal):
        targets.check_document(
            document=get_document(statistic=statistic.name, **{field: value})
        )


@pytest.mark.parametrize(
    ("basis", "has_controls", "refusal"),
    (
        pytest.param("control", True, None, id="a-control-pair-carries-one"),
        pytest.param("reserved", True, None, id="a-reserved-pair-carries-one-too"),
        pytest.param("ranked", False, None, id="a-ranked-pair-carries-none"),
        pytest.param(
            "gap", False, None, id="a-gap-pair-is-here-to-fail-so-it-arms-nothing"
        ),
        pytest.param(
            "control",
            False,
            "expected at least one",
            id="a-control-pair-with-none-is-silently-disarmed",
        ),
        pytest.param(
            "reserved", False, "expected at least one", id="and-so-is-a-reserved-one"
        ),
        pytest.param(
            "ranked",
            True,
            "expected none",
            id="a-ranked-pair-with-one-arms-a-pair-nobody-chose-to-arm",
        ),
        pytest.param("gap", True, "expected none", id="as-does-a-gap-pair"),
        pytest.param(
            "elective", True, "unknown basis", id="a-basis-outside-the-four-is-refused"
        ),
    ),
)
def test_a_basis_and_its_controls_must_agree(
    basis: str, has_controls: bool, refusal: str | None
) -> None:
    """Every basis, both ways: the contradiction is silent in both directions, so both are checked."""
    document = get_document()
    document["targets"][0]["basis"] = basis
    if not has_controls:
        document["targets"][0].pop("controls")
    with get_context(refusal=refusal):
        targets.check_document(document=document)


def test_the_document_must_name_a_distinct_and_reasoned_set() -> None:
    """A repeated pair makes it ambiguous which row armed it, and a blank reason justifies nothing.

    Not parametrized: the three malformations are three different shapes of document rather than
    three values of one field, and a table of mutating callables reads worse than the three cases do.
    """
    with pytest.raises(AssertionError, match="names no targets"):
        targets.check_document(
            document={"provenance": {"wri_revision": pull.WRI_REVISION}, "targets": []}
        )
    document = get_document()
    document["targets"].append(dict(document["targets"][0]))
    with pytest.raises(AssertionError, match="repeats XAA-MAIZE"):
        targets.check_document(document=document)
    document = get_document()
    document["targets"][0]["reason"] = "   "
    with pytest.raises(AssertionError, match="no reason given"):
        targets.check_document(document=document)


def test_a_set_chosen_against_another_wri_release_is_refused() -> None:
    """Every written reason quotes that release's deforestation figures, so a moved pin leaves them
    describing numbers the tool does not read."""
    document = get_document()
    document["provenance"]["wri_revision"] = "0" * len(pull.WRI_REVISION)
    with pytest.raises(AssertionError, match="re-derive the shortlist"):
        targets.check_document(document=document)


@pytest.mark.parametrize(
    "measure",
    tuple(pytest.param(measure, id=measure.name) for measure in targets.Measure),
)
def test_a_measure_names_its_two_sources_in_order(measure: targets.Measure) -> None:
    """The name is what the report prints and the value is what the arithmetic reads, so a measure
    whose two disagree would label a ratio with its own inverse."""
    assert (
        measure.name == f"{measure.numerator.name:s}_OVER_{measure.denominator.name:s}"
    )


def test_only_a_measure_with_both_terms_ours_is_unanchored() -> None:
    """`report.get_control_findings` skips the stale-anchor check on exactly these.

    Asserted as a whole set rather than per measure: what matters is that adding one forces a
    decision about which side of that check it falls on, which a per-member test cannot ask.
    """
    assert {measure for measure in targets.Measure if not measure.is_anchored} == {
        targets.Measure.SLUC_OVER_JDLUC
    }


@pytest.mark.parametrize(
    "statistic",
    tuple(pytest.param(statistic, id=statistic.name) for statistic in schema.Statistic),
)
def test_every_statistic_has_a_default_band(statistic: schema.Statistic) -> None:
    """A new `Statistic` member has to choose a band rather than inheriting the ratio one.

    Only that a band exists and is usable: DEFAULT_TOLERANCE and DEFAULT_RANK_TOLERANCE are both
    0.10 today, so nothing asserted here can tell the two branches apart. What separates them is the
    units, and `report_test`'s rank-versus-ratio pair is where that is pinned.
    """
    assert 0.0 < targets.get_default_tolerance(statistic=statistic) <= 1.0


def test_a_control_with_no_baseline_is_carried_rather_than_armed() -> None:
    """Every control in the file is frozen, so this state exists only here -- and it is the state a
    newly added control arrives in, where reading it as armed would fire on the first capture."""
    carried = targets.Control(
        target=targets.Target(
            iso_3166="XAA", crop_name="MAIZE", basis="control", reason="Invented"
        ),
        emission_pool=schema.EmissionPool.FOREST,
        measure=targets.Measure.SLUC_OVER_WRI,
        inherited=0.40,
        baseline=None,
        note="Invented",
    )
    assert not carried.is_frozen
    assert carried.tolerance == targets.DEFAULT_TOLERANCE
    assert carried.statistic == schema.Statistic.RATIO


def test_a_target_slug_is_the_pair_a_report_prints() -> None:
    """`render_coverage` prints these and `iter_control_targets` dedups on them, so the separator is
    load-bearing in two places that never see each other."""
    target = targets.Target(
        iso_3166="XAA", crop_name="MAIZE", basis="ranked", reason="Invented"
    )
    assert target.slug == "XAA-MAIZE"


def test_the_control_targets_are_the_pairs_carrying_controls_in_file_order() -> None:
    """A capture is driven off this list, so a pair missing from it silently loses its guard.

    Deduplicated, because a pair carrying three controls is one target to compute and not three, and
    in file order, which is materiality order by construction.
    """
    slugs = [target.slug for target in targets.iter_control_targets()]
    assert len(slugs) == len(set(slugs))
    assert set(slugs) == {control.target.slug for control in targets.iter_controls()}
    assert slugs == [
        target.slug for target in targets.iter_targets() if target.slug in set(slugs)
    ]


def test_a_frozen_control_carries_the_anchors_it_was_frozen_against() -> None:
    """`check_document` asserts this of the file; this asserts `iter_controls` carries it through,
    which it builds conditionally and so could drop.
    """
    frozen = [control for control in targets.iter_controls() if control.is_frozen]
    assert frozen, "no control in the set is frozen, so this proves nothing"
    for control in frozen:
        assert (
            control.baseline_source_version is not None
        ) == control.measure.is_anchored
        if control.baseline_source_version is not None:
            # Sorted by `schema.get_source_version_key`, which is what lets `report` compare it
            # against a row's own `source_version` as one string.
            anchors = control.baseline_source_version.split(",")
            assert anchors == sorted(anchors)


def test_every_target_names_a_crop_and_a_country_that_resolve() -> None:
    """A name nothing downstream resolves is malformed input, and this is where it surfaces.

    `prepare.get_target_anchor_consistency_findings` indexes `statistical.Crop` directly, so an
    unknown crop raises there rather than dropping the pair's guard and reporting agreement.  The
    country half has no reader that would raise, and `check_document` catches neither: both fields
    are well-formed strings.

    Every offender at once rather than a parametrized case per pair, because parametrizing over the
    file's own contents reads it while the module is being collected: a malformed set then fails
    collection for the whole file instead of failing the one test that describes it.
    """
    unresolvable = sorted(
        target.slug
        for target in targets.iter_targets()
        if target.crop_name not in statistical.Crop.__members__
        or target.iso_3166 not in iso3166.countries_by_alpha3
    )
    assert not unresolvable, (
        f"{', '.join(unresolvable)} name a crop or country nothing resolves"
    )


def test_the_target_set_on_disk_is_well_formed() -> None:
    """One named test for the real file, so a malformed set says so as a failure rather than as an
    error raised out of whichever test happened to read it first."""
    targets.check_document(document=targets.read_document())
