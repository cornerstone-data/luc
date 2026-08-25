"""Measure the change in emissions-factor results caused by a code change.

Capture the emissions-factor table at a baseline commit and again at the working tree, diff the two,
summarize the diff. Each side writes one parquet per (methodology, country) plus a manifest, so
`--compare-only` re-runs every check on artifacts already on disk, and so any question this tool
does not answer can be asked of those parquets directly.

**The cache is content-blind, so a clean result can be a lie.** `storage.get_cache_decorator` keys
on module path, qualname, a hand-written `version=` int and the call arguments, never on the code
itself, so two commits can resolve to the *same* parquet URI -- and a head capture will then read
the baseline's answer back and report that nothing moved. Diagnosing that is yours: the capture
summary flags byte-identical parquets, and jdluc.storage logs the URI each side loaded, so both
cache keys are in the run log. `--isolated` sidesteps it with a private SCRATCH_ROOT per side, at
the price of the full pipeline rather than a cache read.

Every crop in the capture gets checked -- Target.crop_names is the same
attribute.get_crop_names call that drives the capture, so the two cannot diverge. The cache key
includes crop_names, so asking for a subset would also cold-miss every layer an ordinary
`uv run python jdluc/trace.py USA` populated; captures are one country at a time over the full
per-methodology list. Scoring the same list means each sum drift is a true all-crop total, and
production_kg must hold to float32 noise under any within-group redistribution.

That full list carries its own controls. The 13 crops in ifpri_mapspam.SHARED_CROP_NAMES -- MAIZE,
SOYBEAN, WHEAT, RICE and the rest -- take the simple-lookup branch of get_canonical_quantity, and
get_crop_to_share builds its denominator from raw per-year snapshots rather than decomposed ones,
so nothing about the 2000 decomposition can reach them. Any change confined to that decomposition
must leave all 13 at exactly zero on every column; movement there is a wiring break, not drift.

Whether the drift it reports is welcome is a judgement for the reader, so there is no verdict and
no exit code to read: it prints and exits 0, or fails loudly on an assertion when it cannot compare
at all. stdout is deterministic and free of log noise, so redirecting two runs and diffing them
compares one branch against another.

Must run from the repo root -- the cache resolves module paths through git, and Config finds .env
by walking up from the working directory.

  uv run python tools/measure-drift.py --baseline 121d40c
  uv run python tools/measure-drift.py --baseline 121d40c --iso USA --isolated
  uv run python tools/measure-drift.py --baseline 121d40c --compare-only 2>/dev/null > now.txt
"""

import argparse
import collections.abc
import dataclasses
import datetime
import enum
import hashlib
import itertools
import json
import logging
import math
import os
import pathlib
import shutil
import subprocess
import textwrap
import typing

import pandas

from jdluc import attribute, config, storage, trace

logger = logging.getLogger(__name__)


TOOL_NAME = pathlib.Path(__file__).name
# Countries chosen for distinct emissions regimes rather than coverage: temperate row crops,
# a tropical forest frontier, and tropical peat. Every added country multiplies capture cost, so
# widen this deliberately -- and use --iso to work on a subset meanwhile. HND and NIC are the
# cheap exception: both fall entirely inside the one ten-degree tile 20N_090W, and the expensive
# layer -- statistical.get_downscaled_luc_emissions -- is keyed on (skip_glad_crop_filter,
# tile_id) alone, so whichever of the two runs second pays for little beyond its province clips.
#
# Central America earns five slots because it carries every 2000 crop group ifpri_mapspam still
# decomposes, and the weight sits in different groups either side of a border: 44% of Honduran
# OOIL production takes the fallback path against 17% of Nicaraguan, and 77% of Nicaraguan BANP
# against 1% of Honduran. All five also run forest conversion, peatland conversion and peatland
# occupation together over the same hectares.
METHODOLOGY_TO_ISO_3166S: dict[attribute.Methodology, tuple[str, ...]] = {
    attribute.Methodology.JURISDICTIONAL_DIRECT: ("USA",),
    attribute.Methodology.STATISTICAL: (
        "BLZ",
        "BRA",
        "GTM",
        "HND",
        "IDN",
        "MEX",
        "NIC",
        "SLV",
        "USA",
    ),
}
# Uncommitted changes anywhere else cannot move the numbers, so they do not belong in the
# manifest's dirtiness record.
RESULT_BEARING_PATHS = ("jdluc", "pyproject.toml", "uv.lock")
NATIONAL = "NATIONAL"
PROVINCIAL = "PROVINCIAL"
# The emissions columns are megatonnes while an emissions factor is kg CO2e per kg
KG_PER_TONNE = 1000
# Two sides can be written with different index levels, so both are re-keyed onto this. admin_id
# rather than jurisdiction_name: a machine identifier, where a display string could be renamed.
CANONICAL_KEY = ("admin_level", "admin_id", "crop_name", "methodology")
# float32 eps is 1.19e-07, so movement at that scale is summation order rather than a real change.
DEFAULT_RTOL = 1.19e-07
# Reported as counts, because one max|rel| cannot distinguish 3 rows moving 50% from 128 of them.
DRIFT_THRESHOLDS = (1e-07, 1e-03, 1e-01)
# Exactly the columns iter_national_from_provincials sums; everything else is a ratio derived from
# them, and summing a ratio is meaningless.
ADDITIVE_COLUMNS = (
    "crop_hectares",
    "forest_emissions_mt",
    "peatland_crop_hectares",
    "peatland_conversion_emissions_mt",
    "peatland_occupation_emissions_mt",
    "emissions_mt",
    "production_kg",
)
# Invariants are re-derivations within one side, so they hold to float64 epsilon. Anything above
# this is a wiring break rather than arithmetic noise.
INVARIANT_RTOL = 1e-12
# Summation-order noise is symmetric, so a lopsided up/down split is a systematic shift even where
# every individual magnitude is negligible.
SIGN_ALPHA = 0.05


class Side(enum.StrEnum):
    BASELINE = enum.auto()
    HEAD = enum.auto()


class Severity(enum.StrEnum):
    BLOCKING = enum.auto()
    DEFECT = enum.auto()
    ADVISORY = enum.auto()

    @property
    def marker(self) -> str:
        match self:
            case Severity.BLOCKING:
                return "!!"
            case Severity.DEFECT:
                return "XX"
            case _:
                return "--"


class Invariant(enum.Enum):
    """Relationships that must hold inside a single side, whatever the other side says."""

    ROLLUP = "national totals equal the sum of their provincials"
    EMISSIONS_FACTOR = "emissions factor equals emissions x 1000 / production"
    YIELD = "yield equals production / hectares"
    PEAT_FRACTION = "peatland occupation fraction lies in [0, 1]"
    NON_NEGATIVE = "additive columns are non-negative"


@dataclasses.dataclass(frozen=True)
class Finding:
    severity: Severity
    message: str


@dataclasses.dataclass(frozen=True)
class Drift:
    """How far one column moved between the two sides, over shared rows.

    Not a Finding: drift is the measurement, a Finding means it is not to be trusted.
    """

    slug: str
    column: str
    rows: int
    max_absolute: float
    max_relative: float
    # Row counts over each of DRIFT_THRESHOLDS, in order
    counts_over: tuple[int, ...]
    # A value going NaN <-> finite is a categorically different event from one that moved
    nan_flips: int
    # Direction, which magnitude cannot show: noise splits evenly, a systematic shift does not
    moved_down: int
    moved_up: int
    sign_p_value: float

    def exceeds(self, rtol: float) -> bool:
        return (
            self.max_relative > rtol
            or bool(self.nan_flips)
            or self.sign_p_value < SIGN_ALPHA
        )


@dataclasses.dataclass(frozen=True)
class SumDrift:
    """A whole-target total, where per-row movement can cancel out or compound."""

    slug: str
    column: str
    before: float
    after: float

    @property
    def relative(self) -> float:
        return (
            (self.after - self.before) / abs(self.before)
            if self.before
            else float("nan")
        )

    def exceeds(self, rtol: float) -> bool:
        return abs(self.relative) > rtol if self.before else self.after != 0


@dataclasses.dataclass(frozen=True)
class Target:
    methodology: attribute.Methodology
    iso_3166: str

    @property
    def crop_names(self) -> tuple[str, ...]:
        # The same list the capture hands the pipeline, so what gets checked can never be less
        # than what ran.  A curated subset scores a partial sum, which moves whenever a crop
        # trades with a sibling outside it -- indistinguishable from mass being created.
        return attribute.get_crop_names(methodology=self.methodology)

    @property
    def slug(self) -> str:
        return f"{self.methodology.name:s}-{self.iso_3166:s}"


def iter_targets(
    iso_3166s: tuple[str, ...] | None,
) -> collections.abc.Iterator[Target]:
    for methodology, methodology_iso_3166s in sorted(METHODOLOGY_TO_ISO_3166S.items()):
        for iso_3166 in sorted(methodology_iso_3166s):
            if iso_3166s is None or iso_3166 in iso_3166s:
                yield Target(methodology=methodology, iso_3166=iso_3166)


@dataclasses.dataclass
class Manifest:
    """How one side's parquets were produced, in enough detail to distrust them later."""

    sha: str
    # `git status --porcelain` lines, so "?? path" rather than "path"
    dirty_paths: list[str]
    captured_at: str
    scratch_root: str
    row_counts: dict[str, int]
    # Differing digests prove little: pyarrow embeds library metadata that moves with the
    # environment. Only identical ones are informative.
    parquet_sha256: dict[str, str]

    FILENAME: typing.ClassVar[str] = "manifest.json"

    def write(self, directory: pathlib.Path) -> None:
        path = directory / self.FILENAME
        path.write_text(
            json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        )
        logger.info(f"Wrote {path}")

    @classmethod
    def read(cls, directory: pathlib.Path) -> typing.Self:
        """Ignores keys it does not recognize, so --compare-only still reads a capture written by
        an older version of this tool. A *missing* field still raises, loudly."""
        payload = json.loads((directory / cls.FILENAME).read_text())
        names = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in names})


def git(*arguments: str, cwd: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def get_repo_root() -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    root = pathlib.Path(git("rev-parse", "--show-toplevel", cwd=cwd))
    assert root == cwd, (
        f"Run from the repo root ({root}), not {cwd}: the cache keys module paths relative to "
        "the repo root and Config finds .env by walking up from the working directory"
    )
    return root


def resolve_commit(sha: str, cwd: pathlib.Path) -> str:
    return git("rev-parse", f"{sha:s}^{{commit}}", cwd=cwd)


def get_dirty_paths(cwd: pathlib.Path) -> list[str]:
    porcelain = git("status", "--porcelain", "--", *RESULT_BEARING_PATHS, cwd=cwd)
    return porcelain.splitlines() if porcelain else []


def load_capture(
    directory: pathlib.Path, slug: str
) -> tuple[pandas.DataFrame, tuple[str, ...]]:
    """Read one side's parquet onto CANONICAL_KEY, whatever index it was written with.

    Whichever of admin_id and jurisdiction_name is not in the index is a column, so resetting and
    re-keying realigns two sides written differently. Also returns the index each was *written*
    with, since a schema difference and a value disagreement look identical once rows fail to line
    up.
    """
    raw = pandas.read_parquet(directory / f"{slug:s}.parquet")
    written = tuple(str(name) for name in raw.index.names if name is not None)
    df = raw.reset_index()
    missing = [name for name in CANONICAL_KEY if name not in df.columns]
    assert not missing, (
        f"{slug:s} in {directory} lacks {missing}; is it the schema you think it is?"
    )
    return df.set_index(list(CANONICAL_KEY)).sort_index(), written


def check_alignment(
    baseline: pandas.DataFrame,
    crop_names: tuple[str, ...],
    head: pandas.DataFrame,
    slug: str,
    written: tuple[tuple[str, ...], tuple[str, ...]],
) -> list[Finding]:
    """Do the two sides describe the same rows? Everything downstream assumes they do."""
    findings: list[Finding] = []
    shared = baseline.index.intersection(head.index)
    print(
        f"\n  {slug:s}: {len(baseline):d} baseline row(s), {len(head):d} head row(s), "
        f"{len(shared):d} shared"
    )
    if not len(shared):
        # A column table of zeros would read as agreement when not one value was compared. The
        # written index levels are the diagnostic here and nowhere else: a mismatch confined to one
        # level is a schema difference that load_capture failed to reconcile, not a numeric one.
        findings.append(
            Finding(
                severity=Severity.BLOCKING,
                message=(
                    f"{slug:s} has no shared rows, so NOTHING was compared -- the keys do not "
                    f"line up. Written as {' + '.join(written[0])} against "
                    f"{' + '.join(written[1])}, re-keyed onto {' + '.join(CANONICAL_KEY)}"
                ),
            )
        )
    for label, df in (("baseline", baseline), ("head", head)):
        # Duplicates would make .loc[shared] return more rows than shared, misaligning everything
        # downstream rather than failing
        if duplicated := int(df.index.duplicated().sum()):
            findings.append(
                Finding(
                    severity=Severity.BLOCKING,
                    message=(
                        f"{slug:s} has {duplicated:d} duplicate key(s) on the {label:s} side, so "
                        "no row-to-row comparison is well defined"
                    ),
                )
            )
        # A target that silently stopped existing would otherwise be reported as nothing at all
        if absent := sorted(
            set(crop_names) - set(df.index.get_level_values("crop_name"))
        ):
            findings.append(
                Finding(
                    severity=Severity.BLOCKING,
                    message=(
                        f"{slug:s} is missing {absent} from the {label:s} capture, so a "
                        "hardcoded target is going unmeasured"
                    ),
                )
            )
        if len(only := df.index.difference(shared)):
            findings.append(
                Finding(
                    severity=Severity.ADVISORY,
                    message=(
                        f"{slug:s} has {len(only):d} row(s) only on the {label:s} side, "
                        f"e.g. {list(only)[:3]}"
                    ),
                )
            )
    return findings


def print_worst_rows(
    baseline: pandas.DataFrame, columns: list[str], head: pandas.DataFrame, limit: int
) -> None:
    """Name the rows that moved most, so a large max|rel| points somewhere specific."""
    records: list[tuple[float, str, tuple[str, ...], float, float]] = []
    for column in columns:
        before, after = baseline[column], head[column]
        relative = (after - before).abs().div(before.abs().where(before.abs() > 0))
        for key, value in relative.nlargest(limit).items():
            row = typing.cast(tuple[str, ...], key)
            records.append(
                (
                    float(value),
                    column,
                    row,
                    float(before.loc[row]),
                    float(after.loc[row]),
                )
            )
    if not records:
        return
    print(f"\n    worst {limit:d} row(s) by relative movement")
    print(
        f"      {'crop':<16}{'admin':<9}{'column':<34}{'before':>13}{'after':>13}{'rel':>11}"
    )
    for value, column, row, before_value, after_value in sorted(records, reverse=True)[
        :limit
    ]:
        admin_id, crop_name = row[1], row[2]
        print(
            f"      {crop_name!s:<16}{admin_id!s:<9}{column:<34}"
            f"{before_value:>13.5g}{after_value:>13.5g}{value:>11.3e}"
        )


def check_excluded_crops(
    baseline: pandas.DataFrame, crops: list[str], head: pandas.DataFrame, rtol: float
) -> list[Finding]:
    """Say what the hardcoded crop filter left unchecked, and whether it moved.

    The filter is deliberate, but a tool that checks a tenth of the rows reads as though it checked
    the artifact. Across every numeric column rather than the emissions factor alone, because a
    factor is a ratio: it sits still whenever its numerator and denominator move together, so a
    column that has not moved says nothing about the emissions and hectares underneath it.
    """
    outside = ~baseline.index.get_level_values("crop_name").isin(crops)
    left, right = baseline[outside], head[outside]
    shared = left.index.intersection(right.index)
    excluded = sorted(set(left.index.get_level_values("crop_name")))
    if not len(shared) or not excluded:
        return []
    worst, where, flips, blind = 0.0, "", 0, 0
    for column in sorted(
        set(left.select_dtypes("number").columns)
        & set(right.select_dtypes("number").columns)
    ):
        before, after = left.loc[shared, column], right.loc[shared, column]
        relative = (after - before).abs().div(before.abs().where(before.abs() > 0))
        flips += int((before.isna() != after.isna()).sum())
        # NaN on both sides everywhere: unmeasurable, which is not the same as unmoved
        if not relative.notna().any():
            blind += 1
        elif float(relative.max()) > worst:
            row = typing.cast(tuple[str, ...], relative.idxmax())
            worst, where = float(relative.max()), f"{column:s} {row[2]:s}/{row[1]:s}"
    # Three distinct outcomes, and the first two used to print alike: something moved, nothing was
    # comparable, and everything was comparable and sat exactly still
    measured = (
        f"worst {worst:.3e} ({where:s})"
        if where
        else "nothing measurable"
        if blind
        else "no movement at all"
    )
    print(
        f"    not checked: {len(shared):d} row(s) over {len(excluded):d} crop(s) outside the "
        f"hardcoded subset; {measured:s}, {flips:d} NaN flip(s), "
        f"{blind:d} column(s) NaN throughout"
    )
    if worst <= rtol and not flips:
        return []
    return [
        Finding(
            severity=Severity.ADVISORY,
            message=(
                f"{len(excluded):d} crop(s) outside the checked set moved where nothing "
                f"checked them: {measured:s}, plus {flips:d} NaN flip(s). Target.crop_names "
                "should cover every crop the capture holds, so this means they diverged"
            ),
        )
    ]


def check_value_drift(
    baseline: pandas.DataFrame, head: pandas.DataFrame, rtol: float, target: Target
) -> tuple[list[Drift], list[Finding]]:
    """Per column, how far did the values move on rows both sides have?

    Filtered to the hardcoded crops, which is what that subset is for; alignment ran over the
    whole table.
    """
    crops = list(target.crop_names)
    left_all = baseline[baseline.index.get_level_values("crop_name").isin(crops)]
    right_all = head[head.index.get_level_values("crop_name").isin(crops)]
    shared = left_all.index.intersection(right_all.index)
    columns = sorted(
        set(left_all.select_dtypes("number").columns)
        & set(right_all.select_dtypes("number").columns)
    )
    print(
        f"\n  {target.slug:s}: {len(shared):d} shared row(s) over "
        f"{len(crops):d} hardcoded crop(s)"
    )
    if not len(shared) or not columns:
        return [], []

    left, right = left_all.loc[shared], right_all.loc[shared]
    thresholds = "".join(f"{f'>{threshold:g}':>9s}" for threshold in DRIFT_THRESHOLDS)
    print(
        f"    {'column':<34}{'max|abs|':>13}{'max|rel|':>12}{thresholds:s}{'nan':>5}"
        f"{'up/dn':>10}{'sign p':>8}"
    )
    drifts: list[Drift] = []
    findings: list[Finding] = []
    for column in columns:
        before, after = left[column], right[column]
        # NaN in the same place on both sides is agreement, not a difference
        both_nan = before.isna() & after.isna()
        absolute = (after - before).abs().where(~both_nan)
        relative = absolute.div(before.abs().where(before.abs() > 0))
        down, up = int((after < before).sum()), int((after > before).sum())
        drift = Drift(
            slug=target.slug,
            column=column,
            rows=len(shared),
            max_absolute=float(absolute.max()) if absolute.notna().any() else 0.0,
            max_relative=float(relative.max()) if relative.notna().any() else 0.0,
            counts_over=tuple(
                int((relative > threshold).sum()) for threshold in DRIFT_THRESHOLDS
            ),
            nan_flips=int((before.isna() != after.isna()).sum()),
            moved_down=down,
            moved_up=up,
            sign_p_value=two_sided_sign_p_value(down=down, up=up),
        )
        drifts.append(drift)
        counts = "".join(f"{count:>9d}" for count in drift.counts_over)
        print(
            f"    {column:<34}{drift.max_absolute:>13.5g}{drift.max_relative:>12.3e}"
            f"{counts:s}{drift.nan_flips:>5d}{f'{up:d}/{down:d}':>10}"
            f"{drift.sign_p_value:>8.3f}"
        )
        if drift.nan_flips:
            findings.append(
                Finding(
                    severity=Severity.ADVISORY,
                    message=(
                        f"{target.slug:s} {column:s} changed NaN-ness on {drift.nan_flips:d} "
                        "row(s) -- a value appearing or disappearing, not merely moving; usually "
                        "rows gaining or losing production entirely"
                    ),
                )
            )
        if drift.sign_p_value < SIGN_ALPHA:
            findings.append(
                Finding(
                    severity=Severity.ADVISORY,
                    message=(
                        f"{target.slug:s} {column:s} moved in one direction more than chance "
                        f"allows: {up:d} up against {down:d} down (p={drift.sign_p_value:.3f}). "
                        "Summation noise is symmetric, so this is a systematic shift even where "
                        "the magnitudes are small"
                    ),
                )
            )
    print_worst_rows(baseline=left, columns=columns, head=right, limit=5)
    findings += check_excluded_crops(
        baseline=baseline, crops=crops, head=head, rtol=rtol
    )
    return drifts, findings


def check_sum_drift(
    baseline: pandas.DataFrame, head: pandas.DataFrame, target: Target
) -> list[SumDrift]:
    """Totals for the whole target, which answer a different question from per-row drift.

    Signed, so opposing movements show as cancellation rather than as calm. Summed over NATIONAL
    rows only: each national row is already the sum of its provincials, so both levels would double
    count.
    """

    def nationals(df: pandas.DataFrame) -> pandas.DataFrame:
        return df[
            (df.index.get_level_values("admin_level") == NATIONAL)
            & df.index.get_level_values("crop_name").isin(list(target.crop_names))
        ]

    left, right = nationals(baseline), nationals(head)
    print(
        f"\n  {target.slug:s}: {len(left):d} vs {len(right):d} national row(s) over "
        f"{len(target.crop_names):d} hardcoded crop(s)"
    )
    print(f"    {'column':<34}{'baseline':>15}{'head':>15}{'delta':>14}{'rel':>11}")
    totals = [
        SumDrift(
            slug=target.slug,
            column=column,
            before=float(left[column].sum()),
            after=float(right[column].sum()),
        )
        for column in ADDITIVE_COLUMNS
        if column in left.columns and column in right.columns
    ]
    # The aggregate factor these crops carry, which no single row's factor gives
    by_column = {total.column: total for total in totals}
    if {"emissions_mt", "production_kg"} <= set(by_column):
        emissions, production = by_column["emissions_mt"], by_column["production_kg"]
        totals.append(
            SumDrift(
                slug=target.slug,
                column="implied_emissions_factor",
                before=(
                    emissions.before * KG_PER_TONNE / production.before
                    if production.before
                    else float("nan")
                ),
                after=(
                    emissions.after * KG_PER_TONNE / production.after
                    if production.after
                    else float("nan")
                ),
            )
        )
    for total in totals:
        # Scientific, not percent: a percentage rounded for readability prints +0.00% for a 1e-06
        # move, which reads as agreement while the verdict counts it as drift
        print(
            f"    {total.column:<34}{total.before:>15.6g}{total.after:>15.6g}"
            f"{total.after - total.before:>+14.4g}{total.relative:>+11.2e}"
        )
    return totals


def two_sided_sign_p_value(down: int, up: int) -> float:
    """Exact two-sided binomial p-value for an up/down split, no scipy and no approximation.

    n is at most the provinces in a country, so the exact sum is cheap. Ties are excluded rather
    than split.
    """
    total = up + down
    if not total:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(max(up, down), total + 1))
    return min(1.0, 2 * tail / 2**total)


def check_invariants(df: pandas.DataFrame) -> dict[Invariant, str]:
    """Relationships that must hold within one side, whatever the other side says.

    Returns what broke rather than findings, because severity depends on the other side: see
    compare_invariants.
    """

    def worst_relative(left: pandas.Series, right: pandas.Series) -> float:
        relative = (left - right).abs().div(left.abs().where(left.abs() > 0)).dropna()
        return float(relative.max()) if len(relative) else 0.0

    broken: dict[Invariant, str] = {}
    levels = df.index.get_level_values("admin_level")
    national = df[levels == NATIONAL].reset_index().set_index("crop_name")
    provincial = df[levels == PROVINCIAL].reset_index().groupby("crop_name")
    columns = [column for column in ADDITIVE_COLUMNS if column in df.columns]

    rollup = {
        column: worst_relative(national[column], provincial[column].sum())
        for column in columns
    }
    if rollup and max(rollup.values()) > INVARIANT_RTOL:
        column = max(rollup, key=lambda name: rollup[name])
        broken[Invariant.ROLLUP] = (
            f"worst {rollup[column]:.3e} on {column:s}, over {len(national):d} national row(s)"
        )

    # Both admin levels, because attach_ratios derives these for provincials while
    # iter_national_from_provincials derives them again for nationals -- two separate code paths
    factor = worst_relative(
        df["emissions_factor_kgco2e_per_kg"],
        (df["emissions_mt"] * KG_PER_TONNE)
        .div(df["production_kg"])
        .where(df["production_kg"] > 0),
    )
    if factor > INVARIANT_RTOL:
        broken[Invariant.EMISSIONS_FACTOR] = f"worst {factor:.3e}"

    per_hectare = worst_relative(
        df["yield_kg_per_ha"],
        df["production_kg"].div(df["crop_hectares"]).where(df["crop_hectares"] > 0),
    )
    if per_hectare > INVARIANT_RTOL:
        broken[Invariant.YIELD] = f"worst {per_hectare:.3e}"

    fraction = df["peatland_occupation_fraction"].dropna()
    outside = int(((fraction < 0) | (fraction > 1)).sum())
    if outside:
        broken[Invariant.PEAT_FRACTION] = (
            f"{outside:d} row(s) outside [0, 1], range "
            f"[{fraction.min():.4f}, {fraction.max():.4f}]"
        )

    negative = {column: int((df[column] < 0).sum()) for column in columns}
    if any(negative.values()):
        broken[Invariant.NON_NEGATIVE] = ", ".join(
            f"{column:s}={count:d}" for column, count in negative.items() if count
        )
    return broken


def compare_invariants(
    baseline: dict[Invariant, str], head: dict[Invariant, str], slug: str
) -> list[Finding]:
    """Report per-side breaks, as a defect only where the head broke what the baseline did not.

    A break the baseline shares predates this change: worth knowing, but not this change's doing.
    """
    findings: list[Finding] = []
    for invariant in Invariant:
        if invariant not in baseline and invariant not in head:
            continue
        broke_here = invariant in head and invariant not in baseline
        findings.append(
            Finding(
                severity=Severity.DEFECT if broke_here else Severity.ADVISORY,
                message=(
                    f"{slug:s} breaks '{invariant.value:s}' -- baseline "
                    f"{baseline.get(invariant, 'holds'):s}, head "
                    f"{head.get(invariant, 'holds'):s}."
                    + (
                        " The change broke it, and drift columns can look fine while a table is "
                        "self-inconsistent"
                        if broke_here
                        else " Not introduced by this change"
                    )
                ),
            )
        )
    return findings


@dataclasses.dataclass(frozen=True)
class LegAgreement:
    """How far apart the two attribution legs price one crop, at one commit."""

    crop_name: str
    statistical: float
    jurisdictional_direct: float

    @property
    def ratio(self) -> float:
        return self.statistical / self.jurisdictional_direct


def check_leg_agreement(
    statistical: pandas.DataFrame, jurisdictional_direct: pandas.DataFrame
) -> list[LegAgreement]:
    """Implied national emission factor per crop on each leg, for the crops both legs price.

    The legs divide the SAME per-pixel emissions layer and differ only in how, so a ratio far from
    one is a methodological disagreement rather than noise. The jurisdictional-direct leg has no
    yield for most crops, leaving their production at zero and their factor undefined -- not zero
    -- so those are dropped rather than counted as perfect agreement.
    """

    def factors(df: pandas.DataFrame) -> pandas.Series:
        national = df[df.index.get_level_values("admin_level") == NATIONAL]
        priced = national[national["production_kg"] > 0]
        factor = priced["emissions_mt"] * KG_PER_TONNE / priced["production_kg"]
        return factor.groupby(level="crop_name").first()

    left, right = factors(statistical), factors(jurisdictional_direct)
    return [
        LegAgreement(
            crop_name=str(crop_name),
            statistical=float(left[crop_name]),
            jurisdictional_direct=float(right[crop_name]),
        )
        for crop_name in sorted(set(left.index) & set(right.index))
        if right[crop_name]
    ]


def compare_leg_agreement(
    baseline: list[LegAgreement], head: list[LegAgreement], iso_3166: str
) -> list[Finding]:
    """Print both sides' leg ratios, and report a change that widens the disagreement.

    Only the statistical leg can move: the jurisdictional-direct one reads CDL per pixel and never
    touches MapSPAM. So a ratio walking away from parity is the statistical leg drifting from the
    one that corroborates it -- which per-column drift cannot show, because every column can move a
    defensible amount while the two methods agree less than they did.
    """
    print(f"\n=== {iso_3166:s}: agreement between the legs ===")
    if not head:
        print("  no crop is priced by both legs, so there is nothing to compare")
        return []
    before = {agreement.crop_name: agreement for agreement in baseline}
    print(
        f"  {'crop':<16}{'jurisdictional':>16}{'statistical':>14}{'ratio':>8}"
        f"{'was':>14}{'ratio':>8}"
    )
    for agreement in head:
        was = before.get(agreement.crop_name)
        print(
            f"  {agreement.crop_name:<16}{agreement.jurisdictional_direct:>16.4f}"
            f"{agreement.statistical:>14.4f}{agreement.ratio:>8.2f}"
            + (
                f"{was.statistical:>14.4f}{was.ratio:>8.2f}"
                if was is not None
                else f"{'--':>14}{'--':>8}"
            )
        )
    # Distance from parity in log space, so 0.5x and 2.0x read as equally far apart
    distance = lambda agreements: max(
        (abs(math.log(one.ratio)) for one in agreements if one.ratio > 0), default=0.0
    )
    worst_before, worst_after = distance(baseline), distance(head)
    print(
        f"  worst disagreement {math.exp(worst_before):.2f}x -> "
        f"{math.exp(worst_after):.2f}x (parity is 1.00)"
    )
    if worst_after <= worst_before:
        return []
    return [
        Finding(
            severity=Severity.ADVISORY,
            message=(
                f"{iso_3166:s} the legs agree less than they did: worst ratio "
                f"{math.exp(worst_before):.2f}x to {math.exp(worst_after):.2f}x. Only the "
                "statistical leg can move here, so this is that leg drifting from the one that "
                "corroborates it. Per-column drift cannot show this: every column can move a "
                "defensible amount while the two methods diverge"
            ),
        )
    ]


def print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("\n  no findings")
        return
    print()
    # Severity order, so advisories cannot bury a defect or a blocked comparison
    for finding in sorted(findings, key=lambda f: list(Severity).index(f.severity)):
        print(
            textwrap.fill(
                finding.message,
                initial_indent=f"  {finding.severity.marker:s} ",
                subsequent_indent="     ",
                width=96,
            )
        )


def capture(directory: pathlib.Path, label: Side, targets: tuple[Target, ...]) -> None:
    """Trace every target into `directory`, one parquet per (methodology, country).

    Writes a manifest rather than returning one, so the driver only ever trusts what is on disk --
    all it can trust when the capture ran in a clone, in another process.
    """
    directory.mkdir(parents=True, exist_ok=True)
    cwd = get_repo_root()
    row_counts: dict[str, int] = {}
    parquet_sha256: dict[str, str] = {}
    for index, target in enumerate(targets, start=1):
        # The full crop list, so this reuses the caches an ordinary trace.py run populates
        crop_names = attribute.get_crop_names(methodology=target.methodology)
        logger.info(
            f"[{label!s}] tracing {target.slug:s} with {len(crop_names):d} crops "
            f"({index:d}/{len(targets):d})"
        )
        df = trace.workflow(
            crop_names=crop_names,
            iso_3166s=(target.iso_3166,),
            methodology=target.methodology,
            skip_glad_crop_filter=False,
        )
        path = directory / f"{target.slug:s}.parquet"
        df.to_parquet(path)
        row_counts[target.slug] = len(df)
        parquet_sha256[target.slug] = hashlib.sha256(path.read_bytes()).hexdigest()
        logger.info(f"Wrote {path} ({len(df):d} rows)")

    Manifest(
        sha=resolve_commit("HEAD", cwd=cwd),
        dirty_paths=get_dirty_paths(cwd=cwd),
        captured_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        scratch_root=config.Config.from_dot_env().scratch_root,
        row_counts=row_counts,
        parquet_sha256=parquet_sha256,
    ).write(directory=directory)


def materialize_commit(
    repo_root: pathlib.Path,
    sha: str,
    directory: pathlib.Path,
    scratch_root: str | None,
    patch: str | None,
) -> pathlib.Path:
    """Lay down a checkout of `sha` that can run a capture, reusing it if already present.

    `git clone --shared` rather than `git worktree add`, which writes into the source repo's .git
    and so fails when that repo is read-only. Nothing is copied: the clone borrows the source
    object store through alternates.
    """
    if (directory / ".git").exists():
        logger.info(f"Reusing clone at {directory}")
    else:
        git(
            "clone",
            "--shared",
            "--no-checkout",
            str(repo_root),
            str(directory),
            cwd=repo_root,
        )
        logger.info(f"Cloned {repo_root} to {directory}")

    # Reset to a pristine `sha` on every run, so a reused clone neither carries last run's patch
    # nor double-applies this one. Cleaning is scoped to the pipeline paths to spare the clone's
    # .venv, which is untracked and expensive to rebuild.
    git("checkout", "--force", "--detach", sha, cwd=directory)
    git("clean", "--force", "-d", "--", *RESULT_BEARING_PATHS, cwd=directory)
    logger.info(f"Checked out {sha[:12]:s} in {directory}")

    if patch:
        logger.info(f"Applying {len(patch.splitlines()):d} lines of working-tree diff")
        subprocess.run(
            ["git", "apply", "-"], check=True, cwd=directory, input=patch, text=True
        )

    # .env is gitignored, so the clone has none and Config.from_dot_env would raise. Config reads
    # the file through dotenv_values and never consults the environment, so an isolated cache root
    # has to be written into the file rather than exported.
    lines = (repo_root / ".env").read_text().splitlines(keepends=True)
    if scratch_root is not None:
        replaced = [
            line for line in lines if line.split("=")[0].strip() == "SCRATCH_ROOT"
        ]
        assert len(replaced) == 1, f"Expected one SCRATCH_ROOT in .env, got {replaced=}"
        lines = [
            f"SCRATCH_ROOT = '{scratch_root:s}'\n" if line in replaced else line
            for line in lines
        ]
    (directory / ".env").write_text("".join(lines))
    logger.info(f"Wrote {directory / '.env'} ({scratch_root=})")

    # Capture with today's code, so both sides write the same artifacts against the same targets
    # even when the baseline predates this tool
    (directory / "tools").mkdir(exist_ok=True)
    shutil.copy(pathlib.Path(__file__).resolve(), directory / "tools" / TOOL_NAME)
    return directory


def capture_side(
    label: Side,
    sha: str,
    repo_root: pathlib.Path,
    out: pathlib.Path,
    targets: tuple[Target, ...],
    iso_3166s: tuple[str, ...] | None,
    isolated: bool,
) -> Manifest:
    directory = out / str(label)
    if label == Side.HEAD and not isolated:
        # Nothing to isolate and no .env to rewrite, so run the working tree as it stands --
        # which also carries untracked modules that a clone would silently drop
        capture(directory=directory, label=label, targets=targets)
        return Manifest.read(directory=directory)

    # The baseline runs its own code untouched; only the head side carries the working tree, and
    # then only because isolating the cache means it cannot run in place
    patch = None
    if label == Side.HEAD:
        patch = git("diff", "HEAD", "--", *RESULT_BEARING_PATHS, cwd=repo_root) or None
        untracked = [
            line for line in get_dirty_paths(cwd=repo_root) if line.startswith("??")
        ]
        assert not untracked, (
            "A clone cannot carry untracked result-bearing paths, and dropping them would "
            f"measure code you are not running; git add them or drop --isolated: {untracked}"
        )

    scratch_root = None
    if isolated:
        # A cache root private to this exact code, so an isolated run cannot read another's answer.
        # The patch digest matters: two runs of a dirty working tree share a SHA, and keying on the
        # SHA alone would let the second read the first's -- the very confusion isolation is for.
        token = sha[:12]
        if patch is not None:
            token += "-" + hashlib.sha256(patch.encode()).hexdigest()[:8]
        scratch_root = storage.join_uri(
            root=config.Config.from_dot_env().scratch_root,
            prefix=f"measure-drift/{token:s}",
        )

    clone = materialize_commit(
        repo_root=repo_root,
        sha=sha,
        directory=out / "clones" / sha[:12],
        scratch_root=scratch_root,
        patch=patch,
    )
    command = [
        "uv",
        "run",
        "python",
        f"tools/{TOOL_NAME:s}",
        "--capture-to",
        str(directory),
        "--capture-label",
        str(label),
    ]
    for iso_3166 in iso_3166s or ():
        command += ["--iso", iso_3166]
    logger.info(f"[{label!s}] running {' '.join(command):s} in {clone}")
    subprocess.run(command, check=True, cwd=clone)
    return Manifest.read(directory=directory)


def compare_target(
    out: pathlib.Path, rtol: float, target: Target
) -> tuple[list[Finding], list[Drift], list[SumDrift]]:
    """Every check for one (methodology, country), in the order the report prints them.

    Alignment first, because everything after it assumes the rows line up.
    """
    pair = {
        side: load_capture(directory=out / str(side), slug=target.slug)
        for side in (Side.BASELINE, Side.HEAD)
    }
    left, right = pair[Side.BASELINE][0], pair[Side.HEAD][0]

    print(f"\n=== {target.slug:s}: alignment ===")
    findings = check_alignment(
        baseline=left,
        crop_names=target.crop_names,
        head=right,
        slug=target.slug,
        written=(pair[Side.BASELINE][1], pair[Side.HEAD][1]),
    )

    print(f"\n=== {target.slug:s}: per-value drift ===")
    drifts, drift_findings = check_value_drift(
        baseline=left, head=right, rtol=rtol, target=target
    )
    findings += drift_findings

    print(f"\n=== {target.slug:s}: sum drift ===")
    totals = check_sum_drift(baseline=left, head=right, target=target)

    print(f"\n=== {target.slug:s}: within-side invariants ===")
    broken = {side: check_invariants(df=pair[side][0]) for side in pair}
    for side in (Side.BASELINE, Side.HEAD):
        state = f"{len(broken[side]):d} broken" if broken[side] else "all five hold"
        print(f"  {side!s:<10}{state:s}")
    findings += compare_invariants(
        baseline=broken[Side.BASELINE], head=broken[Side.HEAD], slug=target.slug
    )

    moved = sum(drift.exceeds(rtol=rtol) for drift in drifts) + sum(
        total.exceeds(rtol=rtol) for total in totals
    )
    print(f"\n  {target.slug:s}: {moved:d} measure(s) moved beyond {rtol:g}")
    return findings, drifts, totals


def get_parser() -> argparse.ArgumentParser:
    """The CLI, kept out of main because it is 45 lines of boilerplate around six decisions."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--baseline", help="commit-ish to compare the working tree against"
    )
    parser.add_argument(
        "--iso",
        action="append",
        dest="iso_3166s",
        help="restrict the hardcoded targets to these countries; repeatable",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help="where to write captures; default under $TMPDIR",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="give each side its own SCRATCH_ROOT and recompute from cold; always correct, and "
        "priced as the full pipeline rather than a cache read",
    )
    parser.add_argument("--rtol", default=DEFAULT_RTOL, type=float)
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="re-run the checks on the captures already in --out, without capturing again",
    )
    parser.add_argument(
        "--capture-to",
        type=pathlib.Path,
        help=argparse.SUPPRESS,  # set when this tool re-invokes itself inside a clone
    )
    parser.add_argument(
        "--capture-label", choices=tuple(Side), type=Side, help=argparse.SUPPRESS
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = get_parser().parse_args()

    iso_3166s = tuple(args.iso_3166s) if args.iso_3166s else None
    targets = tuple(iter_targets(iso_3166s=iso_3166s))
    assert targets, f"No hardcoded targets match {iso_3166s=}"

    if args.capture_to is not None:
        assert args.capture_label is not None, "--capture-to needs --capture-label"
        capture(directory=args.capture_to, label=args.capture_label, targets=targets)
        return 0

    assert args.baseline is not None, "--baseline is required"
    repo_root = get_repo_root()
    baseline_sha = resolve_commit(args.baseline, cwd=repo_root)
    head_sha = resolve_commit("HEAD", cwd=repo_root)
    assert baseline_sha != head_sha or get_dirty_paths(cwd=repo_root), (
        f"{args.baseline:s} resolves to HEAD and the tree is clean, so both sides would run "
        "identical code and any verdict would be vacuous"
    )
    out = args.out or (
        pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
        / "measure-drift"
        / baseline_sha[:12]
    )
    logger.info(
        f"Capturing {len(targets):d} target(s) at {baseline_sha[:12]:s} and "
        f"{head_sha[:12]:s} into {out} ({'isolated' if args.isolated else 'shared':s} cache)"
    )

    manifests = {
        label: (
            Manifest.read(directory=out / str(label))
            if args.compare_only
            else capture_side(
                label=label,
                sha=sha,
                repo_root=repo_root,
                out=out,
                targets=targets,
                iso_3166s=iso_3166s,
                isolated=args.isolated,
            )
        )
        for label, sha in ((Side.BASELINE, baseline_sha), (Side.HEAD, head_sha))
    }

    baseline, head = (manifests[label] for label in (Side.BASELINE, Side.HEAD))
    print(f"\n=== captured to {out} ===")
    print(f"  {'target':<34}{'baseline rows':>14}{'head rows':>11}  digest")
    identical = []
    for target in targets:
        same = baseline.parquet_sha256[target.slug] == head.parquet_sha256[target.slug]
        identical += [target.slug] if same else []
        print(
            f"  {target.slug:<34}{baseline.row_counts[target.slug]:>14d}"
            f"{head.row_counts[target.slug]:>11d}  {'identical' if same else 'differs':s}"
        )
    if identical:
        # The likeliest reason for identical bytes is that both sides resolved to one parquet
        print(
            f"\n  {len(identical):d} target(s) wrote byte-identical parquets. If you expected"
            "\n  movement, check whether both sides read the same cached parquet -- compare the"
            "\n  'Loading from' URIs logged above, or rerun with --isolated."
        )

    findings, drifts, totals = (
        list(itertools.chain.from_iterable(kind))
        for kind in zip(
            *(
                compare_target(out=out, rtol=args.rtol, target=target)
                for target in targets
            ),
            strict=True,
        )
    )

    # Cross-target, so it cannot live in compare_target: agreement is a property of a pair of
    # legs, and only shows where one country is covered by both methodologies.
    iso_3166_to_methodologies: dict[str, set[attribute.Methodology]] = (
        collections.defaultdict(set)
    )
    for target in targets:
        iso_3166_to_methodologies[target.iso_3166].add(target.methodology)
    for iso_3166, methodologies in sorted(iso_3166_to_methodologies.items()):
        if len(methodologies) < 2:
            continue
        pair = {
            side: {
                methodology: load_capture(
                    directory=out / str(side),
                    slug=Target(methodology=methodology, iso_3166=iso_3166).slug,
                )[0]
                for methodology in methodologies
            }
            for side in (Side.BASELINE, Side.HEAD)
        }
        findings += compare_leg_agreement(
            baseline=check_leg_agreement(
                statistical=pair[Side.BASELINE][attribute.Methodology.STATISTICAL],
                jurisdictional_direct=pair[Side.BASELINE][
                    attribute.Methodology.JURISDICTIONAL_DIRECT
                ],
            ),
            head=check_leg_agreement(
                statistical=pair[Side.HEAD][attribute.Methodology.STATISTICAL],
                jurisdictional_direct=pair[Side.HEAD][
                    attribute.Methodology.JURISDICTIONAL_DIRECT
                ],
            ),
            iso_3166=iso_3166,
        )

    print_findings(findings=findings)
    print(
        f"\n{sum(drift.exceeds(rtol=args.rtol) for drift in drifts):d} of {len(drifts):d} "
        f"column(s) and {sum(total.exceeds(rtol=args.rtol) for total in totals):d} of "
        f"{len(totals):d} total(s) moved by more than {args.rtol:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
