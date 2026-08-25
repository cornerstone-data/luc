"""Shared vocabulary: what a row means, and where it came from.

Nothing here computes anything. It exists so `prepare` and `report` cannot disagree about what a
column holds.

Two conventions are load-bearing.

Provenance travels in the row, not in a sidecar. Every produced row carries `code_version` and
`source_version`. A merged artifact can hold rows from several runs, since `capture`'s per-ISO merge
keeps rows it did not recompute, so a table spanning more than one `code_version` is a defect and is
reported as one.

`Comparability` records whether a ratio can be read as a level. WRI publishes emissions factors at
the provincial grain but no production there, so a provincial per-kg factor cannot be rebased onto
our denominator and carries shape only. Intensity needs no denominator and carries level. Marking
this per row keeps the two from being averaged together.
"""

import dataclasses
import enum
import hashlib
import pathlib
import subprocess
import typing

# The emissions columns are metric tonnes despite the `_mt` suffix reading as megatonnes:
# `production_kg = production_mt * KG_PER_TONNE` in jdluc/trace.py settles it. Presentation divides
# by TONNES_PER_MEGATONNE once, in the formatter, so no intermediate carries mixed units.
KG_PER_TONNE = 1000
TONNES_PER_MEGATONNE = 1e6
# Uncommitted changes outside these cannot move a number, so they do not belong in `code_version`.
# Same set tools/measure-drift.py uses.
RESULT_BEARING_PATHS = ("jdluc", "pyproject.toml", "uv.lock")
# efs.parquet's own index, and what any other capture is re-keyed onto. admin_id rather than
# jurisdiction_name: a machine identifier, where a display string could be renamed. `methodology`
# is a key level because both legs share one artifact, so it separates sLUC from jdLUC.
CANONICAL_KEY = ("admin_level", "admin_id", "crop_name", "methodology")
NATIONAL = "NATIONAL"
PROVINCIAL = "PROVINCIAL"
# The two legs, as `methodology` records them in efs.parquet.
STATISTICAL = "STATISTICAL"
JURISDICTIONAL_DIRECT = "JURISDICTIONAL_DIRECT"
# A yield ratio outside this is not a disagreement about one quantity, it is two different
# quantities: milling and ginning yields all fall well outside it, and no genuine national yield
# estimate differs from another by half.
PRODUCT_FORM_BOUNDS = (0.5, 2.0)
# Within those bounds, how far a median may sit from parity before it is worth reporting. MapSPAM
# runs about 7% under FAOSTAT, so a tolerance below that would fire on every crop.
YIELD_RATIO_TOLERANCE = 0.15


class Source(enum.StrEnum):
    """Who reports a figure.

    SLUC and JDLUC are the legs under test; the rest are what they are tested against.
    """

    SLUC = enum.auto()
    JDLUC = enum.auto()
    WRI = enum.auto()
    ORBAE = enum.auto()
    EPA = enum.auto()
    FAOSTAT = enum.auto()
    GFW_TCL = enum.auto()
    SPAWN = enum.auto()


class SourceTier(enum.StrEnum):
    """How reproducible a figure is. `Confidence` is derived from this."""

    # Re-downloadable from the pinned revision recorded in sources.lock.json
    PULLED = enum.auto()
    # Transcribed from a publication, with table and page cited where it is transcribed
    TRANSCRIBED = enum.auto()
    # Requires a file on the operator's machine; recorded with a sha256 so the bytes are pinned
    SUPPLIED = enum.auto()
    # Computed here from ingested rasters
    MEASURED = enum.auto()
    # Copied from a working directory with no recorded provenance. Sufficient for ranking a
    # magnitude, not for a claim
    BORROWED = enum.auto()


class EmissionPool(enum.StrEnum):
    """Which pool an emission came from.

    GRASSLAND has no column of its own: it is the remainder after the three named pools are
    subtracted from the total.

    LUC_ONLY is the total excluding peatland occupation. Orbae's LUC figure excludes occupation, so
    a like-for-like comparison against Orbae uses LUC_ONLY rather than TOTAL.
    """

    FOREST = enum.auto()
    GRASSLAND = enum.auto()
    PEATLAND_CONVERSION = enum.auto()
    PEATLAND_OCCUPATION = enum.auto()
    TOTAL = enum.auto()
    LUC_ONLY = enum.auto()
    # Hectares rather than emissions, for the extent probes
    CROPLAND_EXTENT = enum.auto()


class GasScope(enum.StrEnum):
    """Which gases a figure covers.

    WRI publishes both scopes of one product; they differ by CH4 and N2O, about 0.5% for USA maize.
    """

    CO2 = enum.auto()
    CO2E = enum.auto()


class Statistic(enum.StrEnum):
    """What a control's expected number actually is.

    RANK_CORRELATION exists because Orbae's export is permanently three years off ours: level cannot
    cross that offset while rank very nearly can, so those controls hold an ordering rather than a
    magnitude. Declared per control so a correlation is never read, averaged or tolerance-checked as
    though it were a ratio.
    """

    RATIO = enum.auto()
    RANK_CORRELATION = enum.auto()


class Comparability(enum.StrEnum):
    """Whether a ratio can be read as a level, or as a shape only. See the module docstring."""

    LEVEL = enum.auto()
    PATTERN_ONLY = enum.auto()


class Aggregation(enum.StrEnum):
    """Whether a figure is as its source published it, or rolled up here.

    A rollup applies our provincial weights, so it is not the source's own national figure. The
    difference between a rollup and the published figure isolates the activity-data term.
    """

    AS_PUBLISHED = enum.auto()
    ROLLED_UP = enum.auto()


class DiscountBasis(enum.StrEnum):
    """Whether emissions are GHGP linearly discounted or undiscounted.

    sLUC, WRI and Orbae are all GHGP-LSRS discounted and therefore matched. A committed figure is
    not comparable to a discounted one.
    """

    GHGP_LINEAR = enum.auto()
    COMMITTED = enum.auto()


class Confidence(enum.IntEnum):
    """How much weight a finding's magnitude can carry. Ordered, because the register sorts on it.

    Reduced by BORROWED evidence, by having one anchor where a comparison elsewhere has two, and by
    PATTERN_ONLY comparability. Carbon density is permanently LOW: SoilGrids, Harris and Huang are
    the datasets our carbon densities come from, so there is no external anchor for them.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class UnpairedReason(enum.Enum):
    """Why a crop one source publishes has no counterpart in the other.

    The value is the sentence the report prints, so a reason cannot be shown without its
    explanation. None of these is a disagreement: they are all reasons a comparison was never
    attempted, which is what makes them worth naming rather than omitting.
    """

    SPAM_GROUP = (
        "MapSPAM aggregates several FAOSTAT items under one name, and the member list is "
        "MapSPAM's to define, so no FAOSTAT figure can be assigned yet"
    )
    SPAM_SPLIT = (
        "MapSPAM splits one FAOSTAT item in two, so giving either name the figure would invent "
        "the split and giving both would double the total"
    )
    UNMAPPED = (
        "outside the canonical MapSPAM taxonomy the dataset maps, so it has no FAOSTAT item code "
        "at all"
    )
    TOO_FEW_COUNTRIES = "reported by too few countries for a median to describe the crop rather than its reporters"


class Severity(enum.StrEnum):
    """Why a finding is worth reading, independent of its size.

    BLOCKING: a comparison could not be made, and figures depending on it are meaningless.
    DEFECT: the run contradicts itself.
    ADVISORY: worth knowing.
    """

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
            case Severity.ADVISORY:
                return "--"
            case _:
                typing.assert_never(self)


@dataclasses.dataclass(frozen=True)
class YieldAgreement:
    """How one source's yield for a crop compares against another's, over the countries with both.

    Here rather than in `prepare` because the two thresholds below are what "mismatch" means, and
    `report` labels rows by them. A definition both modules read cannot drift between them.
    """

    crop_name: str
    countries: int
    median_ratio: float
    lowest_ratio: float
    highest_ratio: float

    @property
    def is_product_form_mismatch(self) -> bool:
        low, high = PRODUCT_FORM_BOUNDS
        return not low <= self.median_ratio <= high

    @property
    def is_beyond_tolerance(self) -> bool:
        return abs(self.median_ratio - 1.0) > YIELD_RATIO_TOLERANCE


@dataclasses.dataclass(frozen=True)
class Finding:
    """One thing wrong with one target, sized so findings can be ranked against each other.

    Magnitude, breadth and confidence are carried because severity gives no ordering: it does not
    distinguish a gap worth tens of megatonnes across nine countries from one worth a fraction of
    that in a single country.

    `slug` is stable across runs, so the same gap stays recognizable as its magnitude moves. Where a
    gap corresponds to an entry in docs/further_research.md, `slug` is that entry's heading; this
    tool measures the entries in that document rather than maintaining a second list.
    """

    slug: str
    severity: Severity
    message: str
    confidence: Confidence
    magnitude_tonnes: float | None = None
    affected_rows: int = 0
    affected_iso_3166s: tuple[str, ...] = ()


def git(*arguments: str, cwd: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", *arguments], check=True, cwd=cwd, stdout=subprocess.PIPE, text=True
    )
    return result.stdout.strip()


def get_source_version_key(versions: dict[str, str]) -> str:
    """Every external anchor behind a figure, as one comparable string.

    Sorted, so two runs naming the same anchors in either order compare equal. A string rather than
    a mapping because it occupies one column and travels into a parquet.
    """
    return ",".join(f"{name:s}={versions[name]:s}" for name in sorted(versions))


def get_code_version(repo_root: pathlib.Path) -> str:
    """The commit that produced a row, with a digest of any uncommitted result-bearing change.

    A bare SHA does not identify a dirty tree, and two dirty runs at one SHA would be
    indistinguishable, so the diff is hashed in. Returned as a single string because it occupies a
    single column.

    This identifies the code that ran, not the code whose results were used: jdluc's cache keys on
    (module, qualname, version=, args) and never on file contents, so a stale cached layer can
    contradict the version recorded here.
    """
    sha = git("rev-parse", "HEAD", cwd=repo_root)[:12]
    diff = git("diff", "HEAD", "--", *RESULT_BEARING_PATHS, cwd=repo_root)
    if not diff:
        return sha
    return f"{sha:s}+{hashlib.sha256(diff.encode()).hexdigest()[:8]:s}"
