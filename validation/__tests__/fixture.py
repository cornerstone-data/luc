"""Invented rows that exercise the schema, so `report` can be developed before any capture exists.

Deliberately synthetic and deliberately round. The fixture proves plumbing, never numbers, so
nothing here should be mistakable for a result: the countries are fictional ISO codes in the
user-assigned `XA` range, and every ratio is a value no real comparison would land on exactly.

It covers what the report has to survive: a control inside tolerance and one outside it, a rank
control inside its absolute band whose movement would exceed a relative one and a rank control
outside it either way, a conservation overrun, a grassland row that exists only as a remainder, a
`PATTERN_ONLY` row that must not be read as a level, a `ROLLED_UP` row with partial coverage, a
`BORROWED` row whose confidence is degraded, a control frozen against anchors this run did not read,
a pair whose own anchor contradicts itself across two of its rows, and a target with no anchor at all
so the coverage section has something to report.
"""

import pandas

from validation import schema, targets

CODE_VERSION = "0000000fixture"
# No row carries this. A table spanning two code versions is a defect state rather than a fixture
# row, so the test that needs one applies this to a copy; it lives here to keep every invented
# identifier in one place.
SUPERSEDED_CODE_VERSION = "0000000stale00"
# The anchors every row here was measured against, and the one a stale control was frozen against.
# Two distinct values are the whole point: a control frozen against anchors the run did not read
# cannot have its movement attributed.
SOURCE_VERSION = "ORBAE=fixture00000,WRI=fixture00000"
SUPERSEDED_SOURCE_VERSION = "ORBAE=fixture00000,WRI=superseded00"


def get_comparisons() -> pandas.DataFrame:
    """One row per (target, pool, measure) -- the shape `prepare` will emit for real."""
    records = [
        # A control comfortably inside tolerance
        {
            "iso_3166": "XAA",
            "crop_name": "MAIZE",
            "anchor_deforestation_share": 1.250,
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.SLUC_OVER_WRI,
            "statistic": schema.Statistic.RATIO,
            "numerator": 2.0,
            "denominator": 5.0,
            "comparability": schema.Comparability.LEVEL,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": True,
            "baseline": 0.40,
            "tolerance": targets.DEFAULT_TOLERANCE,
            "worst_tier": schema.SourceTier.PULLED,
            "confidence": schema.Confidence.HIGH,
        },
        # A control that has moved well outside tolerance -- must surface above the tables
        {
            "iso_3166": "XAB",
            "crop_name": "SOYBEAN",
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.SLUC_OVER_WRI,
            "statistic": schema.Statistic.RATIO,
            "numerator": 1.0,
            "denominator": 5.0,
            "comparability": schema.Comparability.LEVEL,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": True,
            "baseline": 0.70,
            "tolerance": targets.DEFAULT_TOLERANCE,
            "worst_tier": schema.SourceTier.PULLED,
            "confidence": schema.Confidence.HIGH,
        },
        # Anchor-versus-anchor, and not a control here only because the two frozen rank controls
        # below already cover that path. In targets.json five of these are frozen: both sides are
        # external, so the pair can be baselined before any capture exists.
        {
            "iso_3166": "XAB",
            "crop_name": "SOYBEAN",
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.ORBAE_OVER_WRI,
            "statistic": schema.Statistic.RANK_CORRELATION,
            "numerator": 4.0,
            "denominator": 5.0,
            "comparability": schema.Comparability.PATTERN_ONLY,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": False,
            "baseline": None,
            "tolerance": targets.DEFAULT_TOLERANCE,
            "worst_tier": schema.SourceTier.BORROWED,
            "confidence": schema.Confidence.LOW,
        },
        # Grassland exists only as the remainder, and has no external anchor at all
        {
            "iso_3166": "XAA",
            "crop_name": "MAIZE",
            "anchor_deforestation_share": 1.250,
            "emission_pool": schema.EmissionPool.GRASSLAND,
            "measure": targets.Measure.SLUC_OVER_JDLUC,
            "statistic": schema.Statistic.RATIO,
            "numerator": 30.0,
            "denominator": 12.0,
            "comparability": schema.Comparability.LEVEL,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": False,
            "baseline": None,
            "tolerance": targets.DEFAULT_TOLERANCE,
            "worst_tier": schema.SourceTier.MEASURED,
            "confidence": schema.Confidence.LOW,
        },
        # Provincial per-kg EF: WRI publishes no production at that grain, so shape only
        {
            "iso_3166": "XAC",
            "crop_name": "OILPALM",
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.SLUC_OVER_WRI,
            "statistic": schema.Statistic.RATIO,
            "numerator": 0.5,
            "denominator": 25.0,
            "comparability": schema.Comparability.PATTERN_ONLY,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": False,
            "baseline": None,
            "tolerance": targets.DEFAULT_TOLERANCE,
            "worst_tier": schema.SourceTier.PULLED,
            "confidence": schema.Confidence.MEDIUM,
        },
        # A rollup on our weights over partial coverage -- never a published figure
        {
            "iso_3166": "XAC",
            "crop_name": "OILPALM",
            "emission_pool": schema.EmissionPool.TOTAL,
            "measure": targets.Measure.SLUC_OVER_WRI,
            "statistic": schema.Statistic.RATIO,
            "numerator": 8.0,
            "denominator": 10.0,
            "comparability": schema.Comparability.LEVEL,
            "aggregation": schema.Aggregation.ROLLED_UP,
            "coverage_fraction": 0.62,
            "is_control": False,
            "baseline": None,
            "tolerance": targets.DEFAULT_TOLERANCE,
            "worst_tier": schema.SourceTier.PULLED,
            "confidence": schema.Confidence.MEDIUM,
        },
        # A rank control inside its band, and the row that pins the arithmetic: +0.443 to +0.500 is
        # 0.057 in correlation units, inside the 0.10 band, but 12.9% of the baseline -- so a
        # relative test fires on it and an absolute one must not.
        {
            "iso_3166": "XAF",
            "crop_name": "SOYBEAN",
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.ORBAE_OVER_WRI,
            "ratio": 0.500,
            "statistic": schema.Statistic.RANK_CORRELATION,
            "comparability": schema.Comparability.PATTERN_ONLY,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": True,
            "baseline": 0.443,
            "tolerance": targets.DEFAULT_RANK_TOLERANCE,
            "worst_tier": schema.SourceTier.SUPPLIED,
            "confidence": schema.Confidence.MEDIUM,
        },
        # Frozen against a WRI revision this run did not read. Its movement is large enough to fire
        # on any tolerance, so what it exercises is that a moved anchor is reported as ADVISORY and
        # the movement is not attributed at all.
        {
            "iso_3166": "XAH",
            "crop_name": "SOYBEAN",
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.ORBAE_OVER_WRI,
            "ratio": 0.900,
            "statistic": schema.Statistic.RANK_CORRELATION,
            "comparability": schema.Comparability.PATTERN_ONLY,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": True,
            "baseline": 0.443,
            "baseline_source_version": SUPERSEDED_SOURCE_VERSION,
            "tolerance": targets.DEFAULT_RANK_TOLERANCE,
            "worst_tier": schema.SourceTier.SUPPLIED,
            "confidence": schema.Confidence.MEDIUM,
        },
        # The same baseline moved 0.243 -- outside the band under either arithmetic.
        {
            "iso_3166": "XAG",
            "crop_name": "SOYBEAN",
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.ORBAE_OVER_WRI,
            "ratio": 0.200,
            "statistic": schema.Statistic.RANK_CORRELATION,
            "comparability": schema.Comparability.PATTERN_ONLY,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            "is_control": True,
            "baseline": 0.443,
            "tolerance": targets.DEFAULT_RANK_TOLERANCE,
            "worst_tier": schema.SourceTier.SUPPLIED,
            "confidence": schema.Confidence.MEDIUM,
        },
    ]
    frame = pandas.DataFrame.from_records(data=records)
    # A rank row has no numerator and denominator -- rho is measured, not divided -- so it carries
    # `ratio` directly, and the quotient is derived only where the two parts exist. from_records
    # fills each absent key with NaN, which is what makes the two shapes coexist in one frame.
    frame["ratio"] = frame["ratio"].fillna(frame["numerator"] / frame["denominator"])
    frame["code_version"] = CODE_VERSION
    frame["source_version"] = SOURCE_VERSION
    # Every invented row sits on a chosen pair: the fixture exercises rendering, and a row excluded
    # from the table would exercise nothing.
    frame["is_target"] = True
    # Every control but the superseded one was frozen against the anchors this run read.
    frame["baseline_source_version"] = frame["baseline_source_version"].fillna(
        SOURCE_VERSION
    )
    return frame


def get_forest_pools() -> pandas.DataFrame:
    """Per-country forest-conversion pool against the sum of per-crop forest emissions.

    XAD is over the bound, which is physically impossible and therefore outranks every anchor
    disagreement: no allocation can hand out more than the pool holds.
    """
    return pandas.DataFrame.from_records(
        data=[
            {"iso_3166": "XAA", "attributed_tonnes": 4.0e6, "pool_tonnes": 1.0e7},
            {"iso_3166": "XAB", "attributed_tonnes": 9.0e6, "pool_tonnes": 1.0e7},
            {"iso_3166": "XAD", "attributed_tonnes": 2.5e7, "pool_tonnes": 1.0e7},
        ]
    )


def get_unanchored_targets() -> tuple[targets.Target, ...]:
    """Targets in scope that no anchor covers, so the coverage section has something to say.

    Silence is not agreement, and a report that omits these reads as though it checked them.
    """
    return (
        targets.Target(
            iso_3166="XAE",
            crop_name="WHEAT",
            basis="ranked",
            reason="Invented, to exercise the coverage section's unanchored branch",
        ),
    )
