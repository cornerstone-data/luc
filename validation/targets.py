"""Read the target set, and the relationships within it expected to hold still.

`data/targets.json` is the single source of truth: which pairs are validated, why each is there, and
which carry an expectation. This module reads it and refuses malformed input -- it does not hold the
set. Selection is a person's judgement over the shortlist `prepare.get_eligible` derives; see
`docs/validation.md` section 3.

Each control holds two numbers. `inherited` is a figure from earlier work, used once as a sanity check
when the first capture sets `baseline`; a large difference between the two is itself reported.
Controls fire against `baseline` alone, and a null baseline means not yet frozen. Tolerance is a
flagging threshold, not a noise band: the pipeline is deterministic, so movement comes only from a
code or data change.

What these controls do not cover: WRI shares this leg's MapSPAM expansion-share allocation, so an
sLUC/WRI agreement does not independently confirm allocation. WRI remains independent on the forest
pool and on carbon density, which is what these controls guard. Independent evidence about allocation
comes from Orbae (per-H3, jdLUC-family), the US sLUC-versus-jdLUC comparison, and the forest
conservation bound.
"""

import collections.abc
import dataclasses
import enum
import functools
import json
import typing

from validation import pull, schema

TARGETS = pull.DATA / "targets.json"
# A fraction of the baseline, for RATIO controls.
DEFAULT_TOLERANCE = 0.10
# Absolute correlation units, for RANK_CORRELATION controls. Rho lives on [-1, 1], so a fraction of
# the baseline makes the weakest-agreeing control the twitchiest: at +0.443 a relative 10% fires on a
# move of 0.044 where +0.886 tolerates 0.089, and a baseline near zero would fire on noise.
DEFAULT_RANK_TOLERANCE = 0.10
# `gap` marks a pair included *because* it is expected
# to fail -- IDN oil palm is in the set to fail at 0.008 -- which `ranked` would misrepresent.
BASES = frozenset({"ranked", "control", "reserved", "gap"})
# The two bases that carry an expectation. A pair outside them with a `controls` list, or inside them
# without one, is a contradiction rather than an omission, so reading asserts both directions.
BASES_WITH_CONTROLS = frozenset({"control", "reserved"})


class Measure(enum.Enum):
    """A ratio between two sources. The value holds them, so a measure carries its own arithmetic.

    ORBAE_OVER_WRI contains no term of ours. A change in our pipeline moves SLUC_OVER_WRI and
    SLUC_OVER_ORBAE together while leaving ORBAE_OVER_WRI unchanged, so an anchor revision is
    distinguishable from a change in our own figures.
    """

    SLUC_OVER_WRI = (schema.Source.SLUC, schema.Source.WRI)
    SLUC_OVER_ORBAE = (schema.Source.SLUC, schema.Source.ORBAE)
    ORBAE_OVER_WRI = (schema.Source.ORBAE, schema.Source.WRI)
    SLUC_OVER_JDLUC = (schema.Source.SLUC, schema.Source.JDLUC)
    # The only method-family-matched comparison in the design: Orbae's every row is Method=jdLUC, and
    # the USA is the one jurisdiction where we run a jdLUC leg too. Our leg is the numerator, matching
    # every measure but ORBAE_OVER_WRI, which is anchor-against-anchor and has no term of ours.
    JDLUC_OVER_ORBAE = (schema.Source.JDLUC, schema.Source.ORBAE)

    @property
    def numerator(self) -> schema.Source:
        numerator, _ = self.value
        return numerator

    @property
    def denominator(self) -> schema.Source:
        _, denominator = self.value
        return denominator

    @property
    def is_anchored(self) -> bool:
        """Whether either side of the ratio is an external anchor.

        SLUC_OVER_JDLUC is the one measure where neither is: both terms are ours, so nothing outside
        this repository can move it. It therefore has no anchor identity to freeze a baseline
        against, and needs none -- `code_version` already identifies everything that can move it,
        which is exactly why a disagreement here implicates one of our own legs.
        """
        ours = {schema.Source.SLUC, schema.Source.JDLUC}
        return not {self.numerator, self.denominator} <= ours


@dataclasses.dataclass(frozen=True)
class Target:
    iso_3166: str
    crop_name: str
    basis: str
    reason: str

    @property
    def slug(self) -> str:
        return f"{self.iso_3166:s}-{self.crop_name:s}"


@dataclasses.dataclass(frozen=True)
class Control:
    """A relationship expected to hold still, with the source of the expectation.

    `tolerance` is in the units its `statistic` implies: a fraction of the baseline for a RATIO,
    absolute correlation units for a RANK_CORRELATION. `iter_controls` supplies the appropriate
    default when the file names none; the field default here is the RATIO one.
    """

    target: Target
    emission_pool: schema.EmissionPool
    measure: Measure
    inherited: float | None
    baseline: float | None
    note: str
    statistic: schema.Statistic = schema.Statistic.RATIO
    tolerance: float = DEFAULT_TOLERANCE
    # The anchors the baseline was frozen against, as `schema.get_source_version_key` renders them.
    # Null exactly when `baseline` is: a frozen number without the yardstick it was measured with
    # cannot distinguish a pipeline change from an anchor revision, so the two travel together.
    baseline_source_version: str | None = None

    @property
    def is_frozen(self) -> bool:
        """Whether a baseline exists to fire against. Until then a control is carried, not armed."""
        return self.baseline is not None


def check_document(document: dict[str, typing.Any]) -> None:
    """Refuse malformed input.

    Every failure here is malformed input rather than a finding, so it raises: a control naming a
    measure that does not exist would otherwise be dropped silently, taking its guard with it.

    Takes a document rather than reading one, so a refusal can be exercised against input that was
    never on disk. `data/targets.json` satisfies every check below, which is exactly why: a check
    nothing can reach is one that stops holding without saying so.
    """
    # The set was chosen against a particular WRI release, and its deforestation figures and counts
    # are that release's. If the pin moved underneath it, every written reason would describe
    # numbers the tool does not read.
    pinned = document["provenance"]["wri_revision"]
    assert pinned == pull.WRI_REVISION, (
        f"{TARGETS} was chosen against WRI {pinned[:12]:s} but pull is pinned to "
        f"{pull.WRI_REVISION[:12]:s}; re-derive the shortlist and re-freeze the baselines rather "
        "than reading the set against a release it was not chosen from"
    )
    targets = document["targets"]
    assert targets, f"{TARGETS} names no targets"
    slugs = [f"{target['iso_3166']:s}-{target['crop_name']:s}" for target in targets]
    duplicated = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    assert not duplicated, f"{TARGETS} repeats {', '.join(duplicated)}"
    for target, slug in zip(targets, slugs, strict=True):
        assert target["basis"] in BASES, f"{slug}: unknown basis {target['basis']!r}"
        assert target["reason"].strip(), f"{slug}: no reason given"
        rows = target.get("controls", [])
        expects = target["basis"] in BASES_WITH_CONTROLS
        assert bool(rows) == expects, (
            f"{slug}: basis {target['basis']!r} with {len(rows):d} control(s); "
            f"{'expected at least one' if expects else 'expected none'}"
        )
        for row in rows:
            assert row["measure"] in Measure.__members__, (
                f"{slug}: unknown measure {row['measure']!r}"
            )
            assert row["emission_pool"] in schema.EmissionPool.__members__, (
                f"{slug}: unknown emission pool {row['emission_pool']!r}"
            )
            # (0, 1] holds under both readings; what differs is what the number means, so the
            # message says so rather than the bound changing.
            assert 0.0 < row.get("tolerance", DEFAULT_TOLERANCE) <= 1.0, (
                f"{slug}: tolerance {row.get('tolerance')!r} outside (0, 1] -- a fraction of the "
                f"baseline for a RATIO, correlation units for a RANK_CORRELATION"
            )
            statistic = row.get("statistic", schema.Statistic.RATIO.name)
            assert statistic in schema.Statistic.__members__, (
                f"{slug}: unknown statistic {statistic!r}"
            )
            # A baseline and the anchors it was frozen against travel together -- a frozen number
            # with no yardstick recorded would report an anchor revision as a pipeline change, and a
            # yardstick with nothing frozen against it describes nothing. A measure with no external
            # term is the exception: there is no yardstick, and recording one would be a fiction.
            frozen_against = row.get("baseline_source_versions")
            if Measure[row["measure"]].is_anchored:
                assert (row["baseline"] is None) == (not frozen_against), (
                    f"{slug}: baseline {row['baseline']!r} with "
                    f"{'no' if not frozen_against else 'a'} baseline_source_versions; a frozen "
                    "baseline needs the anchor versions it was measured against, and vice versa"
                )
            else:
                assert not frozen_against, (
                    f"{slug}: {row['measure']} divides no external anchor, so it cannot be frozen "
                    "against one; what can move it is our own code, which `code_version` records"
                )
            # A correlation outside [-1, 1] is not a correlation, and a ratio is never negative.
            for field in ("inherited", "baseline"):
                value = row[field]
                if value is None:
                    continue
                if statistic == schema.Statistic.RANK_CORRELATION.name:
                    assert -1.0 <= value <= 1.0, (
                        f"{slug}: {field} {value!r} outside [-1, 1]"
                    )
                else:
                    assert value > 0.0, (
                        f"{slug}: {field} {value!r} is not a positive ratio"
                    )


@functools.cache
def read_document() -> dict[str, typing.Any]:
    """The parsed file, checked on the way through."""
    document = json.loads(TARGETS.read_text())
    check_document(document=document)
    return document


def get_default_tolerance(statistic: schema.Statistic) -> float:
    """The band a control gets when `targets.json` names none.

    A `match` closed with `assert_never`, so a new `Statistic` member is a mypy error naming it
    rather than a rank control silently inheriting the ratio band.
    """
    match statistic:
        case schema.Statistic.RATIO:
            return DEFAULT_TOLERANCE
        case schema.Statistic.RANK_CORRELATION:
            return DEFAULT_RANK_TOLERANCE
        case _:
            typing.assert_never(statistic)


def iter_targets() -> collections.abc.Iterator[Target]:
    """Every validated target, in file order, which is materiality order by construction."""
    for target in read_document()["targets"]:
        yield Target(
            iso_3166=target["iso_3166"],
            crop_name=target["crop_name"],
            basis=target["basis"],
            reason=target["reason"],
        )


def iter_controls() -> collections.abc.Iterator[Control]:
    """Every control, flattened across targets, so a caller need not know which pair holds which."""
    for target in read_document()["targets"]:
        for row in target.get("controls", []):
            # Bound before the yield: the default tolerance depends on it, and the two are in
            # different units.
            statistic = schema.Statistic[
                row.get("statistic", schema.Statistic.RATIO.name)
            ]
            yield Control(
                target=Target(
                    iso_3166=target["iso_3166"],
                    crop_name=target["crop_name"],
                    basis=target["basis"],
                    reason=target["reason"],
                ),
                emission_pool=schema.EmissionPool[row["emission_pool"]],
                measure=Measure[row["measure"]],
                inherited=row["inherited"],
                baseline=row["baseline"],
                note=row["note"],
                statistic=statistic,
                tolerance=row.get(
                    "tolerance", get_default_tolerance(statistic=statistic)
                ),
                baseline_source_version=(
                    schema.get_source_version_key(
                        versions=row["baseline_source_versions"]
                    )
                    if row.get("baseline_source_versions")
                    else None
                ),
            )


def iter_control_targets() -> collections.abc.Iterator[Target]:
    """The distinct targets carrying a control, in file order.

    A capture must include all of these: a control whose target was not computed removes its own
    guard, and does so silently.
    """
    seen: set[str] = set()
    for control in iter_controls():
        if control.target.slug not in seen:
            seen.add(control.target.slug)
            yield control.target
