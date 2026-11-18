"""Read the pulled anchors into the frames `report` renders, and check them against each other.

The anchors are WRI's factors and yield factors, Orbae's export and FAOSTAT's production. Our side is
a `trace` table, which `read_trace_table` reads from the parquet the command line names.

  Transforms only; `python -m validation` runs them and renders the result.
"""

import collections.abc
import csv
import dataclasses
import enum
import io
import json
import logging
import pathlib
import zipfile

import iso3166
import pandas

from jdluc import attribute, statistical, taxonomy
from jdluc.datasets import faostat, ifpri_mapspam
from validation import pull, schema, targets

logger = logging.getLogger(__name__)

# The comparison year: FAOSTAT's harvested area for the deforestation share is read for it, unless
# `--year` names another.
#
# It is also the WRI *reporting* year to read, and that is a match of windows rather than a
# convention: a reporting year names where the LSRS 20-year window sits, so 2020 covers loss years
# 2001-2020, which is the span our own lookback covers. See WRI_REPORTING_YEARS.
REFERENCE_YEAR = 2020
# Below this a median is one or two countries and says more about them than about the crop.
MINIMUM_COUNTRIES = 5
# Below this two sources are not describing the same ordering, applied to the NATIONAL,
# cross-country agreement per crop that `get_anchor_shape_agreement` measures. Set well under WRI's
# +0.961 self-agreement across the same offset, so only a real method difference trips it.
#
# The provincial, within-country correlations `iter_orbae_wri_comparisons` yields are a different
# quantity and are deliberately NOT gated on this -- they are floored on MINIMUM_PROVINCES alone. A
# control measures movement, not agreement, so a weakly-agreeing pair still guards: USA soybean is
# frozen at +0.443 precisely so that a change in how much the two anchors disagree surfaces on every
# run.
MINIMUM_CROSS_COUNTRY_RANK_AGREEMENT = 0.5
# Below this a rank correlation says more about which provinces happen to be shared than about the two
# sources. BOL soybean has three and is deliberately left uncompared.
MINIMUM_PROVINCES = 5
# WRI's two file families use different MapSPAM taxonomies within one release: the emissions-factor
# files use Crop2005's 42 codes, the yield factors Crop2020's 46. Only two codes differ, and losing
# them loses every row for those crops, so the join resolves the rename rather than dropping them.
YIELD_TAXONOMY_YEAR = 2020
# WRI's five columns are one reporting-year series, not five estimates of one quantity. The reporting
# year names the position of the LSRS 20-year window, so 2024 covers loss years 2005-2024: each column
# drops the oldest loss year, adds a real new one, and reweights the rest by the linear discount --
# 9.75% for the reporting year itself, falling 0.5pt/yr to 0.25% nineteen years back. The denominator
# moves with it: SPAM 2020 production rescaled by the FAOSTAT national ratio for that year, and for
# 2024 by a random forest, FAOSTAT not having published 2024 when WRI built it.
#
# That the window slides over new loss data rather than re-discounting a fixed 2001-2020 series is
# measurable in the published files: re-discounting bounds LD_2024/LD_2020 at 31/39 = 0.795 for any
# non-negative series, and 62% of pairs exceed it, 864 of them rising monotonically. So the spread
# across the five is the price of leaving the reporting year unpinned, not evidence that WRI disagrees
# with itself -- and REFERENCE_YEAR selects a column rather than averaging them.
#
# The crop side does not move with the reporting year at all: allocation comes from crop-area
# expansion between SPAM snapshots, and the 2010-2020 snapshot allocates every loss year from 2011 to
# 2024. That is why all 42 crops appear in all five columns; see docs/validation.md.
WRI_REPORTING_YEARS = (2020, 2021, 2022, 2023, 2024)
# Where the ingested FAOSTAT parquet is kept once fetched. 1.3 MiB against the 33 MiB archive it
# replaces, and cached so the report keeps running offline after the first read.
FAOSTAT_CACHE = pull.CACHE / "faostat.parquet"
# How much of a revision or digest identifies an anchor in a `source_version`. Long enough not to
# collide, short enough that a stale-baseline message is readable.
SOURCE_VERSION_LENGTH = 12
# The one part of eligibility that cannot be computed from the pinned anchors: which countries
# intersect a GNW tile, which needs a spatial join against a 93 MiB GeoPackage. 229 ISO codes, so
# `tools/build-tiled-countries.py` commits this and `get_eligible` derives the rest here.
#
# The shortlist itself is deliberately NOT committed. Every other column in it -- WRI's
# deforestation emissions, FAOSTAT's
# production, WRI's provincial unit counts -- is already pinned by sha256 in sources.lock.json, and
# committing a second copy would break the rule the lock rests on: the lock is committed, the bytes are
# not. Recomputing all 864 pairs from those anchors takes about a tenth of a second, and it cannot go
# stale against the pin because it is read from whatever the pin currently names.
TILED_ISO_3166S = pull.DATA / "tiled_iso_3166s.json"
# E4. An emissions factor that cannot move a companywide number is not worth a target.
MINIMUM_PRODUCTION_KG = 100 * 1_000 * 1_000
# E3. Land area, a proxy for cropland weighting. Cropland weighting would need a `trace` table, and
# eligibility is computed from the anchors alone.
MINIMUM_AREA_COVERAGE = 0.5
# The conversion matrix names cropland and pasture as destinations and nothing else, so forest
# cleared for a woody perennial resolves to no destination: the clearance is identified and dated,
# neither destination layer recognises a young plantation, and the carbon is reported in
# `dropped-emissions` rather than charged. MapSPAM keeps the production in the denominator either
# way, so the pipeline attributes close to zero to exactly these crops. They stay eligible on
# purpose: a pair we would rank highly and cannot compute is the finding, not a filtering mistake.
PERENNIAL_CROP_NAMES = frozenset(
    {
        statistical.Crop.ARABICA_COFFEE.value,
        statistical.Crop.ROBUSTA_COFFEE.value,
        statistical.Crop.COCONUT.value,
        statistical.Crop.OILPALM.value,
    }
)
# Three herbaceous perennials are deliberately absent, and they are the test of whether this list
# tracks the canopy rather than the word "perennial". Sugarcane is ratooned; banana and plantain are
# giant herbs with a pseudostem rather than wood. A cropland mask sees all three, so the pipeline
# should resolve them and they are not part of the detection gap.
for _herbaceous in (
    statistical.Crop.SUGARCANE,
    statistical.Crop.BANANA,
    statistical.Crop.PLANTAIN,
):
    assert _herbaceous.value not in PERENNIAL_CROP_NAMES
# docs/further_research.md headings. The register measures that document's entries rather than
# keeping a second list, so a slug here is a heading there, verbatim.
PERENNIAL_RESEARCH_SLUG = "Woody perennial crops resolve to no destination"


def read_wri_yields(grain_name: str) -> pandas.DataFrame:
    """WRI's yield per country and crop, as published.

    `yield_kg` is kg/ha and `yield_factor_kg` its reciprocal; only the former is read, since the
    reciprocal carries no information the comparison needs.

    The crop set is asserted rather than trusted, because this file is the only place a change in what
    WRI publishes would show up cheaply. It carries 46 crops against the factor files' 42 -- the
    Crop2020 taxonomy against Crop2005's -- and the difference has to be exactly the two renames plus
    `pull.WRI_CROPS_WITHOUT_FACTORS`. A release that gave rubber a factor, or dropped one we pull,
    would otherwise leave `WRI_CROP_CODES` quietly describing the wrong release: every count in the
    report would still reconcile, over a crop list no longer matching the anchor.
    """
    path = pull.get_pulled_path(remote=pull.get_wri_yield_remote(grain_name=grain_name))
    frame = pandas.read_csv(path)
    assert len(frame), f"{path} is empty"
    expected = {
        ifpri_mapspam.get_reported_crop_name(
            canonical_crop_name=crop_code, year=YIELD_TAXONOMY_YEAR
        )
        for crop_code in pull.WRI_CROP_CODES
    } | set(pull.WRI_CROPS_WITHOUT_FACTORS)
    published = set(frame["crop"])
    assert published == expected, (
        f"{path} carries a different crop set than {len(pull.WRI_CROP_CODES):d} factor families "
        f"plus {len(pull.WRI_CROPS_WITHOUT_FACTORS):d} factorless crops describe: "
        f"{sorted(published - expected)} unexpected, {sorted(expected - published)} missing. WRI has "
        f"changed what it publishes, so WRI_CROP_CODES and WRI_CROPS_WITHOUT_FACTORS need updating "
        f"against the new revision rather than this assertion relaxing"
    )
    renamed = frame.rename(
        columns={"GID_0": "iso_3166", "GID_1": "gadm_id", "crop": "crop_name"}
    )
    # gadm_id only exists at the provincial grain; callers select what they need.
    return renamed[
        [
            name
            for name in ("iso_3166", "gadm_id", "crop_name", "yield_kg")
            if name in renamed
        ]
    ]


def read_wri_national_emissions(gas_scope: str = "CO2") -> pandas.DataFrame:
    """WRI's national figures, long over its five reporting years.

    Wide-to-long because WRI publishes `{LD,production,EF}_{2020..2024}` as columns, `LD` being its
    name for linearly discounted deforestation emissions. All five are populated, and they are a
    reporting-year series rather than five estimates of one quantity: each column slides the LSRS
    20-year window forward a year over both the loss series and the production denominator.
    """
    frames = []
    for mapspam_code in pull.WRI_CROP_CODES:
        path = pull.get_pulled_path(
            remote=pull.get_wri_factor_remote(
                crop_code=mapspam_code,
                gas_scope=gas_scope,
                grain=pull.get_grain(grain_name="national"),
            )
        )
        wide = pandas.read_csv(path).rename(columns={"GID_0": "iso_3166"})
        for year in WRI_REPORTING_YEARS:
            frames.append(
                pandas.DataFrame(
                    {
                        "iso_3166": wide["iso_3166"],
                        "crop_name": mapspam_code,
                        "year": year,
                        "gas_scope": gas_scope,
                        "deforestation_tonnes": wide[f"LD_{year:d}"],
                        "production_tonnes": wide[f"production_{year:d}"],
                        "ef_kg_per_kg": wide[f"EF_{year:d}"],
                    }
                )
            )
    return pandas.concat(frames, ignore_index=True)


def get_anchor_stability(emissions: pandas.DataFrame) -> pandas.DataFrame:
    """How much WRI's factor moves across its five reporting years, per country and crop.

    This bounds what a comparison against WRI can resolve *while the reporting year is unpinned*. If
    WRI's 2020 and 2024 factors differ by more than a control's tolerance, then a control firing on
    that pair tells us nothing about our pipeline until the year is pinned -- the movement could be
    entirely the window sliding.

    It is not a measure of WRI's self-consistency, and must not be reported as one: the five columns
    are a series, so most of this spread is real temporal signal -- loss years entering and leaving
    the window, and a denominator rescaled to each year's production.

    Spread is max over min rather than a standard deviation: five points is too few for a moment, and
    the ratio is what compares against a tolerance expressed as a fraction.
    """
    usable = emissions[emissions["ef_kg_per_kg"] > 0]
    grouped = usable.groupby(["iso_3166", "crop_name"])["ef_kg_per_kg"]
    stability = grouped.agg(
        highest="max", lowest="min", median="median", reporting_years="size"
    )
    stability = stability[stability["reporting_years"] == len(WRI_REPORTING_YEARS)]
    stability["spread"] = stability["highest"] / stability["lowest"]
    return stability.sort_values("spread", ascending=False)


def get_stability_findings(
    stability: pandas.DataFrame, tolerance: float
) -> list[schema.Finding]:
    """One finding, sized by how much of the anchor outruns a control's tolerance across the years."""
    beyond = stability[stability["spread"] > 1.0 + tolerance]
    if not len(beyond):
        return []
    return [
        schema.Finding(
            affected_rows=len(beyond),
            confidence=schema.Confidence.HIGH,
            message=(
                f"{len(beyond):,d} of {len(stability):,d} (country, crop) pairs "
                f"({len(beyond) / len(stability):.0%}) have a WRI factor that moves more than the "
                f"{tolerance:.0%} control tolerance across its five reporting years; the median "
                f"spread is {stability['spread'].median():.2f}x and the upper decile "
                f"{stability['spread'].quantile(0.9):.2f}x. Most of that is the LSRS window sliding "
                "rather than anchor noise, so a control firing on such a pair cannot distinguish a "
                "pipeline change from a change of reporting year: the year has to be pinned with the "
                "baseline rather than left implicit"
            ),
            severity=schema.Severity.ADVISORY,
            slug="wri-reporting-year-sensitivity",
        )
    ]


def get_deforestation_share_findings(
    deforestation: pandas.DataFrame,
) -> list[schema.Finding]:
    """Pairs where WRI's deforestation-linked area exceeds the crop's whole harvested area.

    Internally inconsistent in WRI's own release rather than a disagreement with us: a crop cannot be
    grown on more land than it is harvested from. Advisory because it does not invalidate the pairs
    that are consistent, but a pair over 1.0 cannot anchor a share comparison.
    """
    over = deforestation[deforestation["deforestation_share"] > 1.0]
    if not len(over):
        return []
    return [
        schema.Finding(
            affected_iso_3166s=tuple(sorted(set(over["iso_3166"]))[:12]),
            affected_rows=len(over),
            confidence=schema.Confidence.MEDIUM,
            message=(
                f"{len(over):,d} of {len(deforestation):,d} (country, crop) pairs "
                f"({len(over) / len(deforestation):.1%}) give WRI more deforestation-linked area "
                f"than FAOSTAT reports harvested, up to {over['deforestation_share'].max():.1f}x. "
                "A crop cannot be grown on more land than it is harvested from, so these pairs "
                "cannot anchor a share comparison whichever side is wrong"
            ),
            severity=schema.Severity.ADVISORY,
            slug="wri-deforestation-area-exceeds-harvested-area",
        )
    ]


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One eligible (country, crop) pair, with the figures a person needs to judge it."""

    iso_3166: str
    crop_name: str
    mapspam_code: str
    deforestation_tonnes: float
    production_kg: float
    is_perennial: bool
    is_decomposed_group_crop: bool
    provincial_units: int
    area_coverage: float


def read_wri_deforestation(mapspam_code: str) -> pandas.DataFrame:
    """WRI's deforestation-linked emissions for one crop, all countries.

    The CO2 scope rather than CO2e: the two are the same product differing by about half a percent,
    and this figure is read to compare pairs against each other, not as a level.
    """
    path = pull.get_pulled_path(
        remote=pull.get_wri_factor_remote(
            crop_code=mapspam_code,
            gas_scope="CO2",
            grain=pull.get_grain(grain_name="national"),
        )
    )
    frame = pandas.read_csv(path).rename(columns={"GID_0": "iso_3166"})
    return frame[["iso_3166", f"LD_{REFERENCE_YEAR:d}"]].rename(
        columns={f"LD_{REFERENCE_YEAR:d}": "deforestation_tonnes"}
    )


def get_provincial_unit_counts(mapspam_code: str) -> dict[str, int]:
    """How many GADM provinces WRI publishes a factor for, per country. E3's first half.

    A country with no provincial row is not eligible at the provincial grain, which is the default
    grain, however large its national figure is.
    """
    path = pull.get_pulled_path(
        remote=pull.get_wri_factor_remote(
            crop_code=mapspam_code,
            gas_scope="CO2",
            grain=pull.get_grain(grain_name="provincial"),
        )
    )
    counts = pandas.read_csv(path).groupby("GID_0")["GID_1"].nunique()
    return {str(iso_3166): int(units) for iso_3166, units in counts.items()}


def iter_candidates(
    area_coverage: dict[str, float],
    production_by_pair: pandas.DataFrame,
    tiled_iso_3166s: set[str],
) -> collections.abc.Iterator[Candidate]:
    """Every pair passing E1 through E4, with its figures attached.

    The filters are applied in E1-to-E4 order, so a rejection is attributable to exactly one of them.

    `statistical.Crop` is a StrEnum whose name is the pipeline vocabulary and whose value is the
    MapSPAM code, so there is no cross-walk to build: WRI and FAOSTAT both key on the code.
    """
    production = production_by_pair.set_index(["iso_3166", "crop_name"])[
        "production_kg"
    ]
    codes = {crop.value: crop.name for crop in statistical.Crop}
    for mapspam_code, crop_name in sorted(codes.items()):
        national = read_wri_deforestation(mapspam_code=mapspam_code)
        provincial_units = get_provincial_unit_counts(mapspam_code=mapspam_code)
        for row in national.to_dict("records"):
            iso_3166 = str(row["iso_3166"])
            if iso_3166 not in tiled_iso_3166s:  # E1
                continue
            units = provincial_units.get(iso_3166, 0)
            if (
                not units or area_coverage.get(iso_3166, 0.0) < MINIMUM_AREA_COVERAGE
            ):  # E3
                continue
            production_kg = float(production.get((iso_3166, mapspam_code), 0.0))
            if production_kg < MINIMUM_PRODUCTION_KG:  # E4
                continue
            yield Candidate(
                area_coverage=area_coverage.get(iso_3166, 0.0),
                crop_name=crop_name,
                deforestation_tonnes=float(row["deforestation_tonnes"]),
                is_decomposed_group_crop=(
                    mapspam_code in ifpri_mapspam.CONSTITUENT_TO_GROUP_NAME
                ),
                is_perennial=mapspam_code in PERENNIAL_CROP_NAMES,
                iso_3166=iso_3166,
                mapspam_code=mapspam_code,
                production_kg=production_kg,
                provincial_units=units,
            )


def get_eligible() -> pandas.DataFrame:
    """The eligible shortlist, derived from the pinned anchors on every run.

    Derived rather than committed. Every figure here already exists in an anchor the lock pins by
    sha256, so a committed copy would be a second, un-pinned record of the same numbers -- and one
    that goes stale silently the moment the WRI pin moves. Reading it from the pin instead costs
    about a tenth of a second and cannot disagree with what it was derived from.

    The single input that is not derivable this way is `tiled_iso_3166s`, which needs a spatial join
    against a GeoPackage; that is committed, and `tools/build-tiled-countries.py` regenerates it.
    """
    tiled = set(json.loads(TILED_ISO_3166S.read_text())["tiled_iso_3166s"])
    assert tiled, f"{TILED_ISO_3166S} names no countries, so every pair would fail E1"
    key_map = json.loads((pull.DATA / "gadm_to_world_bank_admin_1.json").read_text())
    production = read_faostat_production()
    candidates = tuple(
        iter_candidates(
            area_coverage=key_map["area_coverage"],
            production_by_pair=production[production["year"] == REFERENCE_YEAR][
                ["iso_3166", "crop_name", "production_kg"]
            ],
            tiled_iso_3166s=tiled,
        )
    )
    assert candidates, (
        "no pair passed E1-E4, which would mean a filter is mis-specified"
    )
    frame = pandas.DataFrame.from_records(
        [dataclasses.asdict(candidate) for candidate in candidates]
    )
    counted = targets.read_document()["provenance"]["eligible_pairs"]
    assert len(frame) == counted, (
        f"{len(frame):,d} pairs are eligible but {targets.TARGETS} was chosen against "
        f"{counted:,d}; the target set was picked from a different shortlist than this one, so "
        "re-derive it before reading either"
    )
    # Sorted on the anchor's own figure, the one quantity here that is neither ours nor a judgement.
    # Ordering is for reading; the choosing happens in targets.json.
    return frame.sort_values("deforestation_tonnes", ascending=False).reset_index(
        drop=True
    )


def get_perennial_findings(eligible: pandas.DataFrame) -> list[schema.Finding]:
    """What it means that some of the largest eligible pairs are crops the pipeline cannot see.

    Sized from the anchors alone, so it needs no `trace` table: these pairs carry WRI's own deforestation
    attribution while no destination layer resolves what replaced the forest. This survived dropping
    the scorer unchanged, which is the clearest evidence the ranking was never what produced it.

    Nothing here quotes how far the legs land from the anchor. That is read from a `trace` table and
    moves with the emissions model, so it belongs in the comparisons table rather than frozen into
    a finding computed from the anchors alone.

    BLOCKING rather than advisory. For these pairs the pipeline is not disagreeing with an anchor, it
    is structurally unable to produce a comparable number at all.
    """
    perennial = eligible[eligible["is_perennial"]]
    if not len(perennial):
        return []
    at_stake = float(perennial["deforestation_tonnes"].sum())
    largest = perennial.nlargest(1, "deforestation_tonnes").to_dict("records")[0]
    return [
        schema.Finding(
            affected_iso_3166s=tuple(sorted(set(perennial["iso_3166"]))),
            affected_rows=len(perennial),
            confidence=schema.Confidence.HIGH,
            magnitude_tonnes=at_stake,
            message=(
                f"{len(perennial):d} eligible pairs are woody perennials, carrying "
                f"{at_stake / schema.TONNES_PER_MEGATONNE:,.0f} Mt of WRI-attributed deforestation "
                f"between them and led by {largest['iso_3166']!s} {largest['crop_name']!s} at "
                f"{float(largest['deforestation_tonnes']) / schema.TONNES_PER_MEGATONNE:,.0f} Mt. "
                "No destination layer resolves a young plantation, so their clearance is "
                "reported as dropped rather than charged while MapSPAM keeps their "
                "production in the denominator"
            ),
            severity=schema.Severity.BLOCKING,
            slug=PERENNIAL_RESEARCH_SLUG,
        ),
    ]


def get_target_anchor_consistency_findings(
    deforestation: pandas.DataFrame,
) -> list[schema.Finding]:
    """Targets whose own WRI row gives more deforested area than the crop is harvested on.

    Keyed on the target set rather than on the comparisons, which is the whole point. Orbae reaches
    50 pairs and neither affected target is among them, so this check hung off the comparison frame
    would find nothing and its silence would read as agreement -- the failure mode `render_coverage`
    exists to prevent everywhere else.

    `get_deforestation_share_findings` counts the same contradiction across WRI's whole release.
    That count is what decides whether the anchor is broadly usable; this names the rows we chose to
    stand on, and says which of them carries a control. 71 pairs out of 2,349 reads as a rounding
    error right up until one of them is the only armed guard on its row.
    """
    shares = {
        (row["iso_3166"], row["crop_name"]): row["deforestation_share"]
        for row in deforestation.to_dict("records")
    }
    armed = {control.target.slug for control in targets.iter_controls()}
    findings = []
    for target in targets.iter_targets():
        # The deforestation frame is keyed on MapSPAM codes, where a target names the crop.  An
        # unknown name is malformed input rather than a pair to pass over: passing over it would
        # drop the guard and report agreement.
        crop = statistical.Crop[target.crop_name]
        share = shares.get((target.iso_3166, crop.value))
        if share is None or share <= 1.0:
            continue
        findings.append(
            schema.Finding(
                # The same slug as the release-wide count: one mechanism, so the register groups
                # them rather than ranking the same thing twice.
                affected_iso_3166s=(target.iso_3166,),
                affected_rows=1,
                confidence=schema.Confidence.MEDIUM,
                message=(
                    f"{target.slug:s} stands on a WRI row that contradicts itself, giving "
                    f"{float(share):.3f}x more deforestation-linked area than FAOSTAT reports "
                    f"harvested for the crop"
                    + (
                        ", and it carries a control"
                        if target.slug in armed
                        else f", though it is only `{target.basis:s}` and arms nothing"
                    )
                    + ". A crop cannot be grown on more land than it is harvested from, so a "
                    "disagreement on this pair is the anchor's before it is ours"
                ),
                severity=schema.Severity.ADVISORY,
                slug="wri-deforestation-area-exceeds-harvested-area",
            )
        )
    return findings


def get_anchor_shape_agreement() -> pandas.DataFrame:
    """Per crop, whether Orbae and WRI agree on which countries have the highest factor.

    Rank rather than ratio, because Orbae's export is three years off ours and no matching export is
    coming. That offset is affordable for rank and not for level: WRI's own provincial rank holds at
    +0.961 across the same three years while its level spans 1.42x, so a rank disagreement here is
    method rather than vintage.

    National grain only. Both sources key countries on ISO 3166, so this needs no key map, where a
    provincial comparison would need a third fuzzy-matched artifact.
    """
    national = read_orbae()
    national = national[national["admin_level"] == schema.NATIONAL]
    rows = []
    for crop_name, group in national.groupby("crop_name"):
        wri = read_wri_national_emissions()
        wri = wri[(wri["crop_name"] == crop_name) & (wri["year"] == REFERENCE_YEAR)]
        merged = group[["iso_3166", "ef_kg_per_kg"]].merge(
            wri[wri["ef_kg_per_kg"] > 0][["iso_3166", "ef_kg_per_kg"]],
            on="iso_3166",
            suffixes=("_orbae", "_wri"),
        )
        if len(merged) < MINIMUM_COUNTRIES:
            continue
        rows.append(
            {
                "crop_name": crop_name,
                "countries": len(merged),
                "rank_correlation": merged["ef_kg_per_kg_orbae"]
                .rank()
                .corr(merged["ef_kg_per_kg_wri"].rank()),
                "median_level_ratio": (
                    merged["ef_kg_per_kg_orbae"] / merged["ef_kg_per_kg_wri"]
                ).median(),
            }
        )
    return pandas.DataFrame(rows).sort_values("rank_correlation")


def get_anchor_disagreement_findings(
    agreement: pandas.DataFrame,
) -> list[schema.Finding]:
    """Crops where the two external anchors disagree with each other more than usefully.

    This is an anchor-quality row, not a pipeline gap: it says the two sources cannot corroborate each
    other for these crops, so neither can be treated as independent confirmation of the other. It is
    measurable from the anchors alone, which is why it is seeded rather than waiting.
    """
    weak = agreement[
        agreement["rank_correlation"] < MINIMUM_CROSS_COUNTRY_RANK_AGREEMENT
    ]
    if not len(weak):
        return []
    named = ", ".join(
        f"{row['crop_name']!s} {float(row['rank_correlation']):+.2f} over "
        f"{int(row['countries']):d} countries"
        for row in weak.to_dict("records")
    )
    return [
        schema.Finding(
            affected_rows=int(weak["countries"].sum()),
            confidence=schema.Confidence.MEDIUM,
            message=(
                f"Orbae and WRI barely agree on which countries carry the highest factor for "
                f"{len(weak):d} of {len(agreement):d} shared crops: {named:s}. WRI's own provincial "
                "rank holds at +0.961 across the same three-year offset, so this is method rather "
                "than vintage. For these crops the two anchors cannot corroborate each other, so an "
                "agreement with either is not independent confirmation, and the Orbae control "
                "measures are rank controls rather than level ones, and `ORBAE_OVER_WRI` is where "
                "this disagreement gets measured on every run"
            ),
            severity=schema.Severity.ADVISORY,
            slug="external-anchors-disagree-on-shape",
        )
    ]


def get_orbae_admin_ids() -> dict[tuple[str, str], str]:
    """(iso_3166, Orbae province name) to the World Bank `admin_id` a `trace` table is keyed on.

    Orbae's provincial ids are opaque strings, so its provinces reach anything else by name through a
    committed, reviewed map -- a strict lookup, with no matching at runtime.
    """
    matched = json.loads((pull.DATA / "orbae_to_world_bank_admin_1.json").read_text())[
        "matched"
    ]
    return {
        (key.split(":", 1)[0], key.split(":", 1)[1]): admin_id
        for key, admin_id in matched.items()
    }


def get_orbae_gadm_ids() -> dict[tuple[str, str], str]:
    """The same provinces carried one hop further, to GADM `GID_1`, for joining against WRI.

    WRI keys on GADM where our own rows never do, so this is the World Bank map above composed with
    the GADM one. A province the second map cannot reach is dropped rather than guessed at; the map's
    own `unmatched` block records which, so the gap is a number and not a surprise.
    """
    keys = json.loads((pull.DATA / "gadm_to_world_bank_admin_1.json").read_text())[
        "matched"
    ]
    world_bank_to_gadm = {
        world_bank_id: gadm_id for gadm_id, world_bank_id in keys.items()
    }
    return {
        key: world_bank_to_gadm[admin_id]
        for key, admin_id in get_orbae_admin_ids().items()
        if admin_id in world_bank_to_gadm
    }


def iter_orbae_wri_comparisons(
    minimum_provinces: int = MINIMUM_PROVINCES,
) -> collections.abc.Iterator[dict[str, object]]:
    """Orbae against WRI, one row per (country, crop), as a provincial rank correlation.

    The only comparison with no term of ours, because both sides are external. Rank
    rather than ratio: Orbae's export is permanently three years off ours and WRI's own level spans
    1.42x over that gap while its provincial rank holds at +0.961.

    FOREST pool throughout, since WRI is forest-only -- comparing its factor against Orbae's total
    would be the category mismatch that makes the URY 35x and CAN 109x rows meaningless.
    """
    gadm_ids = get_orbae_gadm_ids()
    provincial = read_orbae()
    provincial = provincial[provincial["admin_level"] == schema.PROVINCIAL]
    for (iso_3166, crop_name), group in provincial.groupby(["iso_3166", "crop_name"]):
        # A province with no committed key is dropped here rather than joined loosely; the map's
        # own `unmatched` block records which, so the gap is a number and not a surprise.
        resolved = pandas.Series(
            [
                gadm_ids.get((str(iso_3166), name))
                for name in group["jurisdiction_name"]
            ],
            dtype="object",
            index=group.index,
        )
        left = group.assign(gadm_id=resolved)
        left = left[resolved.notna() & (left["forest_kg_per_kg"] > 0)]
        path = pull.get_pulled_path(
            remote=pull.get_wri_factor_remote(
                crop_code=str(crop_name),
                gas_scope="CO2",
                grain=pull.get_grain(grain_name="provincial"),
            )
        )
        wri = pandas.read_csv(path).rename(columns={"GID_1": "gadm_id"})
        merged = left[["gadm_id", "forest_kg_per_kg"]].merge(
            wri[wri[f"EF_{REFERENCE_YEAR:d}"] > 0][
                ["gadm_id", f"EF_{REFERENCE_YEAR:d}"]
            ],
            on="gadm_id",
        )
        if len(merged) < minimum_provinces:
            continue
        yield {
            "iso_3166": str(iso_3166),
            "crop_name": str(crop_name),
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.ORBAE_OVER_WRI,
            "statistic": schema.Statistic.RANK_CORRELATION,
            "ratio": float(
                merged["forest_kg_per_kg"]
                .rank()
                .corr(merged[f"EF_{REFERENCE_YEAR:d}"].rank())
            ),
            "provinces": len(merged),
            "comparability": schema.Comparability.PATTERN_ONLY,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": len(merged) / len(group),
            # Both sides external, so no term of ours is in it; MEDIUM rather than HIGH because rank
            # is a weaker claim than level and Orbae is a supplied file rather than a pinned pull.
            "confidence": schema.Confidence.MEDIUM,
            "worst_tier": schema.SourceTier.SUPPLIED,
        }


def get_orbae_export_version() -> str:
    """The digest of the export the Orbae figures came from, read from the committed lock.

    The bytes rather than the filename: Orbae is `SUPPLIED`, so a new export can arrive under the
    same name, and the digest is the only thing tying a frozen baseline to the data it was set
    against.
    """
    recorded = pull.read_lock().get("orbae", {}).get(ORBAE_EXPORT.name, {})
    digest = recorded.get("sha256")
    assert digest, (
        f"{ORBAE_EXPORT.name} has no digest in {pull.LOCK}; run `--stage pull` to pin it. The "
        "frozen Orbae baselines are identified by these bytes, so a comparison against an "
        "unpinned export cannot say which release it agreed with"
    )
    return str(digest)[:SOURCE_VERSION_LENGTH]


def get_anchor_versions(measure: targets.Measure) -> dict[str, str]:
    """The identity of every external anchor a measure divides, keyed by `schema.Source` name.

    Our own legs are absent rather than null, since the `trace` table identifies them. That
    makes ORBAE_OVER_WRI the case this exists for: it has no term of ours, so these versions are its
    whole identity, and a change in them is the only thing that can move it.
    """
    versions = {}
    for source in (measure.numerator, measure.denominator):
        match source:
            case schema.Source.WRI:
                versions[source.name] = pull.WRI_REVISION[:SOURCE_VERSION_LENGTH]
            case schema.Source.ORBAE:
                versions[source.name] = get_orbae_export_version()
            case schema.Source.SLUC | schema.Source.JDLUC | schema.Source.JDLUC_PROXY:
                continue
            case _:
                raise NotImplementedError(
                    f"{measure.name:s} divides {source.name:s}, which records no version here; "
                    "add one before a control can be frozen against it"
                )
    return versions


def read_trace_table(path: pathlib.Path) -> pandas.DataFrame:
    """A `trace` table's rows for `schema.ASSESSMENT_YEAR`, every leg it holds.

    A release table, or a branch's `trace` runs merged into one parquet. One table from one run
    cannot mix code versions, so nothing here checks for that. Only the one year is kept, since the
    comparisons below pivot and sum over the other levels and would add years together.
    """
    table = pandas.read_parquet(path)
    assert table.index.names == list(schema.CANONICAL_KEY), (
        f"{path} is indexed by {table.index.names}, not {schema.CANONICAL_KEY}"
    )
    table = table[
        table.index.get_level_values("assessment_year") == schema.ASSESSMENT_YEAR
    ]
    assert len(table), f"{path} holds no rows for {schema.ASSESSMENT_YEAR:d}"
    return table


# The MapSPAM code each `trace` crop row is compared under. `trace` keys on FAOSTAT items, and
# FAOSTAT has one item each for coffee and millet where MapSPAM 2005 onward has two, so those two
# take MapSPAM 2000's combined codes.
COMMODITY_ID_TO_CROP_CODE = {
    486: statistical.Crop.BANANA.value,
    44: statistical.Crop.BARLEY.value,
    176: statistical.Crop.BEAN.value,
    125: statistical.Crop.CASSAVA.value,
    191: statistical.Crop.CHICKPEA.value,
    249: statistical.Crop.COCONUT.value,
    656: ifpri_mapspam.Crop2000.COFF.name,
    195: statistical.Crop.COWPEA.value,
    242: statistical.Crop.GROUNDNUT.value,
    201: statistical.Crop.LENTIL.value,
    56: statistical.Crop.MAIZE.value,
    79: ifpri_mapspam.Crop2000.MILL.name,
    254: statistical.Crop.OILPALM.value,
    taxonomy.InventedCommodityId.OTHER_OILCROPS: statistical.Crop.OTHER_OILCROPS.value,
    taxonomy.InventedCommodityId.OTHER_PULSES: statistical.Crop.OTHER_PULSES.value,
    197: statistical.Crop.PIGEONPEA.value,
    489: statistical.Crop.PLANTAIN.value,
    116: statistical.Crop.POTATO.value,
    270: statistical.Crop.RAPESEED.value,
    27: statistical.Crop.RICE.value,
    328: statistical.Crop.COTTON.value,
    289: statistical.Crop.SESAME_SEED.value,
    83: statistical.Crop.SORGHUM.value,
    236: statistical.Crop.SOYBEAN.value,
    157: statistical.Crop.SUGARBEET.value,
    156: statistical.Crop.SUGARCANE.value,
    267: statistical.Crop.SUNFLOWER.value,
    122: statistical.Crop.SWEET_POTATO.value,
    15: statistical.Crop.WHEAT.value,
    137: statistical.Crop.YAM.value,
}
assert set(COMMODITY_ID_TO_CROP_CODE) == {
    taxonomy.METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
        attribute.Methodology.STATISTICAL
    ][crop.name]
    for crop in statistical.Crop
}
# WRI publishes the two coffees and the two millets apart, as MapSPAM 2005 onward does
SPLIT_CROP_CODE_TO_COMBINED_CROP_CODE = {
    statistical.Crop.ARABICA_COFFEE.value: ifpri_mapspam.Crop2000.COFF.name,
    statistical.Crop.PEARL_MILLET.value: ifpri_mapspam.Crop2000.MILL.name,
    statistical.Crop.ROBUSTA_COFFEE.value: ifpri_mapspam.Crop2000.COFF.name,
    statistical.Crop.SMALL_MILLET.value: ifpri_mapspam.Crop2000.MILL.name,
}
# The MapSPAM crops FAOSTAT names one-for-one, cocoa, tea and tobacco among them. Coffee and millet
# are left out, since WRI's yields call arabica COFF and small millet MILL.
FAOSTAT_ITEM_CODE_TO_CROP_CODE = {
    15: ifpri_mapspam.CANONICAL_CROP_CLS.WHEA.name,
    27: ifpri_mapspam.CANONICAL_CROP_CLS.RICE.name,
    44: ifpri_mapspam.CANONICAL_CROP_CLS.BARL.name,
    56: ifpri_mapspam.CANONICAL_CROP_CLS.MAIZ.name,
    83: ifpri_mapspam.CANONICAL_CROP_CLS.SORG.name,
    116: ifpri_mapspam.CANONICAL_CROP_CLS.POTA.name,
    122: ifpri_mapspam.CANONICAL_CROP_CLS.SWPO.name,
    125: ifpri_mapspam.CANONICAL_CROP_CLS.CASS.name,
    137: ifpri_mapspam.CANONICAL_CROP_CLS.YAMS.name,
    156: ifpri_mapspam.CANONICAL_CROP_CLS.SUGC.name,
    157: ifpri_mapspam.CANONICAL_CROP_CLS.SUGB.name,
    176: ifpri_mapspam.CANONICAL_CROP_CLS.BEAN.name,
    191: ifpri_mapspam.CANONICAL_CROP_CLS.CHIC.name,
    195: ifpri_mapspam.CANONICAL_CROP_CLS.COWP.name,
    197: ifpri_mapspam.CANONICAL_CROP_CLS.PIGE.name,
    201: ifpri_mapspam.CANONICAL_CROP_CLS.LENT.name,
    236: ifpri_mapspam.CANONICAL_CROP_CLS.SOYB.name,
    242: ifpri_mapspam.CANONICAL_CROP_CLS.GROU.name,
    249: ifpri_mapspam.CANONICAL_CROP_CLS.CNUT.name,
    254: ifpri_mapspam.CANONICAL_CROP_CLS.OILP.name,
    267: ifpri_mapspam.CANONICAL_CROP_CLS.SUNF.name,
    270: ifpri_mapspam.CANONICAL_CROP_CLS.RAPE.name,
    289: ifpri_mapspam.CANONICAL_CROP_CLS.SESA.name,
    328: ifpri_mapspam.CANONICAL_CROP_CLS.COTT.name,
    486: ifpri_mapspam.CANONICAL_CROP_CLS.BANA.name,
    489: ifpri_mapspam.CANONICAL_CROP_CLS.PLNT.name,
    661: ifpri_mapspam.CANONICAL_CROP_CLS.COCO.name,
    667: ifpri_mapspam.CANONICAL_CROP_CLS.TEAS.name,
    826: ifpri_mapspam.CANONICAL_CROP_CLS.TOBA.name,
}
# MapSPAM groups of several FAOSTAT items
SPAM_GROUP_CROP_CODES = {
    ifpri_mapspam.CANONICAL_CROP_CLS.OCER.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.OFIB.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.OOIL.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.OPUL.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.ORTS.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.REST.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.TEMF.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.TROF.name,
    ifpri_mapspam.CANONICAL_CROP_CLS.VEGE.name,
}
# Every canonical crop is mapped, split or grouped
assert set(FAOSTAT_ITEM_CODE_TO_CROP_CODE.values()) | set(
    SPLIT_CROP_CODE_TO_COMBINED_CROP_CODE
) | SPAM_GROUP_CROP_CODES == {crop.name for crop in ifpri_mapspam.CANONICAL_CROP_CLS}
assert len(set(FAOSTAT_ITEM_CODE_TO_CROP_CODE.values())) == len(
    FAOSTAT_ITEM_CODE_TO_CROP_CODE
)
assert not set(FAOSTAT_ITEM_CODE_TO_CROP_CODE.values()) & (
    set(SPLIT_CROP_CODE_TO_COMBINED_CROP_CODE) | SPAM_GROUP_CROP_CODES
)


def with_crop_codes(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Only the crop rows, each with the MapSPAM code it is compared under in `crop_code`.

    Some rows are not crops: the livestock rows, and the dropped and unattributed rows (see
    `emit.NonCommodity`). None has a MapSPAM code to key a join on, and neither WRI nor Orbae
    publishes an anchor for any of them.
    Every comparison below is therefore about crops, and this is the one place that is said --
    stated as what is kept rather than what is removed, so a row added later is excluded by
    default rather than by being remembered here.
    """
    crops = frame[frame["commodity_id"].isin(COMMODITY_ID_TO_CROP_CODE)].copy()
    crops["crop_code"] = [
        COMMODITY_ID_TO_CROP_CODE[int(commodity_id)]
        for commodity_id in crops["commodity_id"]
    ]
    return crops


def iter_sluc_jdluc_comparisons(
    table: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, object]]:
    """The statistical and direct legs' factors against each other, nationally, per crop both model.

    With no external anchor in it, a disagreement proves one of our own legs wrong rather than
    raising a question about a yardstick. It is a filter on `methodology` within one table rather
    than a join between two, which is what `trace.CANONICAL_KEY` carrying `methodology` buys.

    The published emissions factor is the quantity compared, because that is what a control on this
    pair guards. It is not a pure emissions ratio: the legs use different production denominators --
    NASS for jdLUC, MapSPAM for sLUC -- so a movement here is either leg's numerator or either leg's
    denominator. Localizing it further would need the two legs' production compared directly, which
    nothing here does.
    """
    national = with_crop_codes(
        frame=table[
            table.index.get_level_values("admin_level") == schema.NATIONAL
        ].reset_index()
    )
    by_methodology = national.pivot_table(
        columns="methodology",
        index=["admin_id", "crop_code"],
        values="emissions_factor_kgco2e_per_kg",
    )
    # NB: the jurisdictional-direct leg is US-only, so a table of countries it does not cover has
    # no column for it at all. A head-to-head needs both legs, and holding only one is
    # not a disagreement about anything -- there is simply nothing here to compare.
    if not {schema.STATISTICAL, schema.JURISDICTIONAL_DIRECT}.issubset(
        by_methodology.columns
    ):
        return
    # Pastureland needs no filtering here: jdLUC emits no such row, so requiring both legs drops it
    both = by_methodology.dropna(
        subset=[schema.STATISTICAL, schema.JURISDICTIONAL_DIRECT]
    ).reset_index()
    for row in both.to_dict("records"):
        jdluc = float(row[schema.JURISDICTIONAL_DIRECT])
        if jdluc <= 0:
            # A zero denominator is not a disagreement, it is a leg with nothing to say here.
            # jdLUC carries a NASS yield for every crop it models, so what reaches this is an
            # admin unit where NASS suppressed or never surveyed one, leaving hectares and
            # emissions with no production behind them.
            continue
        yield {
            "iso_3166": str(row["admin_id"]),
            # The MapSPAM code, because `get_comparisons` renames back to the canonical name once.
            "crop_name": str(row["crop_code"]),
            "emission_pool": schema.EmissionPool.TOTAL,
            "measure": targets.Measure.SLUC_OVER_JDLUC,
            "statistic": schema.Statistic.RATIO,
            "ratio": float(row[schema.STATISTICAL]) / jdluc,
            "comparability": schema.Comparability.LEVEL,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            # Both sides are ours, measured from ingested rasters, and the comparison carries a level
            # rather than a shape -- the strongest evidence in the design.
            "confidence": schema.Confidence.HIGH,
            "worst_tier": schema.SourceTier.MEASURED,
        }


def get_emissions_by_methodology(
    admin_level: str, table: pandas.DataFrame
) -> pandas.DataFrame:
    """Each crop row's emissions in tonnes, one column per leg. A leg the table does not hold gets
    a column of NaN, so a pair including it compares nothing rather than failing.
    """
    rows = with_crop_codes(
        frame=table[
            table.index.get_level_values("admin_level") == admin_level
        ].reset_index()
    )
    return rows.pivot_table(
        aggfunc="sum",
        columns="methodology",
        index=["admin_id", "crop_code"],
        values="emissions_mt",
    ).reindex(
        columns=[
            schema.JURISDICTIONAL_DIRECT,
            schema.JURISDICTIONAL_DIRECT_PROXY,
            schema.STATISTICAL,
        ]
    )


def iter_leg_emissions_comparisons(
    table: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, object]]:
    """Each pair of legs against each other in tonnes, nationally, per crop both model.

    Tonnes leave production out, so a disagreement here is in how a leg splits a country's
    emissions among its crops. A crop the denominator leg gives nothing is left out.
    """
    national = get_emissions_by_methodology(admin_level=schema.NATIONAL, table=table)
    for numerator, denominator, measure in (
        (
            schema.STATISTICAL,
            schema.JURISDICTIONAL_DIRECT,
            targets.Measure.SLUC_OVER_JDLUC_EMISSIONS,
        ),
        (
            schema.JURISDICTIONAL_DIRECT_PROXY,
            schema.STATISTICAL,
            targets.Measure.JDLUC_PROXY_OVER_SLUC_EMISSIONS,
        ),
        (
            schema.JURISDICTIONAL_DIRECT,
            schema.JURISDICTIONAL_DIRECT_PROXY,
            targets.Measure.JDLUC_OVER_JDLUC_PROXY_EMISSIONS,
        ),
    ):
        both = national[[numerator, denominator]].dropna()
        both = both[both[denominator] > 0].reset_index()
        for row in both.to_dict("records"):
            yield {
                "iso_3166": str(row["admin_id"]),
                "crop_name": str(row["crop_code"]),
                "emission_pool": schema.EmissionPool.TOTAL,
                "measure": measure,
                "statistic": schema.Statistic.RATIO,
                "ratio": float(row[numerator]) / float(row[denominator]),
                "provinces": 0,
                "comparability": schema.Comparability.LEVEL,
                "aggregation": schema.Aggregation.AS_PUBLISHED,
                "coverage_fraction": 1.0,
                "confidence": schema.Confidence.HIGH,
                "worst_tier": schema.SourceTier.MEASURED,
            }


def iter_leg_province_comparisons(
    table: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, object]]:
    """The direct leg against each of the other two, ranking the US states by emissions per crop.

    The direct leg places crops by CDL pixels, so it is the reference for where within a country
    the emissions sit. A state either leg gives nothing is left out.
    """
    provincial = get_emissions_by_methodology(
        admin_level=schema.PROVINCIAL, table=table
    )
    provincial["iso_3166"] = provincial.index.get_level_values("admin_id").str[:3]
    for numerator, denominator, measure in (
        (
            schema.STATISTICAL,
            schema.JURISDICTIONAL_DIRECT,
            targets.Measure.SLUC_OVER_JDLUC_EMISSIONS,
        ),
        (
            schema.JURISDICTIONAL_DIRECT,
            schema.JURISDICTIONAL_DIRECT_PROXY,
            targets.Measure.JDLUC_OVER_JDLUC_PROXY_EMISSIONS,
        ),
    ):
        either = provincial[(provincial[numerator] > 0) | (provincial[denominator] > 0)]
        both = either[(either[numerator] > 0) & (either[denominator] > 0)]
        available = (
            either.groupby(["iso_3166", either.index.get_level_values("crop_code")])
            .size()
            .to_dict()
        )
        groups = [
            (iso_3166, crop_code, group)
            for (iso_3166, crop_code), group in both.groupby(
                ["iso_3166", both.index.get_level_values("crop_code")]
            )
            if len(group) >= MINIMUM_PROVINCES
        ]
        for iso_3166, crop_code, group in groups:
            yield {
                "iso_3166": str(iso_3166),
                "crop_name": str(crop_code),
                "emission_pool": schema.EmissionPool.TOTAL,
                "measure": measure,
                "statistic": schema.Statistic.RANK_CORRELATION,
                "ratio": float(group[numerator].rank().corr(group[denominator].rank())),
                "provinces": len(group),
                "comparability": schema.Comparability.PATTERN_ONLY,
                "aggregation": schema.Aggregation.AS_PUBLISHED,
                "coverage_fraction": len(group) / available[(iso_3166, crop_code)],
                "confidence": schema.Confidence.MEDIUM,
                "worst_tier": schema.SourceTier.MEASURED,
            }


def iter_trace_wri_comparisons(
    table: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, object]]:
    """The statistical and proxy legs' forest emissions against WRI's published deforestation
    emissions, tonnes against tonnes.

    The direct leg is left out: it is US-only, and WRI's US crops sit in its national rows, which
    the statistical and proxy legs already compare.

    Nationally, with no yield, area or production term on either side. A provincial rollup is ruled
    out by arithmetic rather than by preference. WRI's provincial
    factor is a ratio over deforestation-linked production while its yield factor is a
    whole-jurisdiction yield, so the intensity built from the two is emissions over WRI's
    deforestation-linked *area* rather than over the crop's:

        intensity x area == (LD / production_defor) x (production_total / area_total) x area
                         == LD / deforestation_share

    Carrying that on our crop area therefore compares against WRI's own figure multiplied by
    1 / deforestation_share -- 6.55x at the median, 1.02x for CIV COCO but 18.96x for USA MAIZ -- and
    the factor does not cancel in the ratio, because our denominator is all of our production where
    WRI's is the deforested part. A control tolerance frozen on that would be frozen mostly on the
    share. `LD` carries no such basis: it is published in tonnes, and our forest total is the same
    quantity in the same units.

    **The cost is the provincial grain, and WRI's release cannot avoid it.** Provincially it publishes
    a factor and no production, so no provincial `LD` can be recovered -- the intensity was the only
    route to one, and the intensity is what carries the share. The provincial factors keep the one job
    where that does not bite: `iter_orbae_wri_comparisons`, which is anchor against anchor, so the
    share sits on both sides of the ratio and cancels.

    CO2e on both sides, which the national grain is what buys: WRI publishes CO2e nationally and CO2
    only provincially, so only a national comparison matches our own CO2e.

    FOREST throughout, because WRI is forest-only. Against our total this would be the category
    mismatch that makes a grassland-dominated country read as a 35x disagreement.
    """
    wri = read_wri_national_emissions(gas_scope="CO2e")
    wri = wri[(wri["year"] == REFERENCE_YEAR) & (wri["deforestation_tonnes"] > 0)]
    # Tonnes add, so WRI's two coffees and two millets sum to the one FAOSTAT item each that our
    # rows carry
    deforestation = (
        wri.assign(
            crop_name=wri["crop_name"].replace(SPLIT_CROP_CODE_TO_COMBINED_CROP_CODE)
        )
        .groupby(["iso_3166", "crop_name"], as_index=False)["deforestation_tonnes"]
        .sum()
    )
    levels = table.index.get_level_values
    national = with_crop_codes(
        frame=table[levels("admin_level") == schema.NATIONAL].reset_index()
    )
    # A leg with no production of a crop in a country attributes nothing to it, and the ratio is a
    # hard zero. That is an absence rather than a disagreement, and would drag any statistic over
    # these rows. A zero ratio where production *is* present stays: attributing nothing to a crop we
    # grow is a finding.
    compared = national[national["production_kg"] > 0].merge(
        deforestation,
        left_on=["admin_id", "crop_code"],
        right_on=["iso_3166", "crop_name"],
    )
    for methodology, measure in (
        (schema.STATISTICAL, targets.Measure.SLUC_OVER_WRI),
        (schema.JURISDICTIONAL_DIRECT_PROXY, targets.Measure.JDLUC_PROXY_OVER_WRI),
    ):
        for row in (
            compared[compared["methodology"] == methodology]
            .sort_values(["admin_id", "crop_code"])
            .to_dict("records")
        ):
            yield {
                "iso_3166": row["admin_id"],
                "crop_name": row["crop_code"],
                "emission_pool": schema.EmissionPool.FOREST,
                "measure": measure,
                "statistic": schema.Statistic.RATIO,
                "ratio": float(row["forest_emissions_mt"])
                / float(row["deforestation_tonnes"]),
                # Nothing is joined provincially, so nothing can be dropped: both sides are the
                # whole national figure, and a zero here means "no provinces used" rather than "a
                # thin join".
                "provinces": 0,
                "comparability": schema.Comparability.LEVEL,
                "aggregation": schema.Aggregation.AS_PUBLISHED,
                "coverage_fraction": 1.0,
                # One anchor. It shares the statistical leg's MapSPAM expansion-share family, so for
                # that leg it corroborates the forest pool and carbon density rather than the
                # allocation.
                "confidence": schema.Confidence.MEDIUM,
                "worst_tier": schema.SourceTier.PULLED,
            }


def iter_orbae_trace_comparisons(
    table: pandas.DataFrame,
) -> collections.abc.Iterator[dict[str, object]]:
    """Each of our legs against Orbae, province by province, as a rank correlation of the factor
    and, over the same provinces, of the emissions. The two disagree: in v202611 the proxy leg ranks
    Brazilian soy's emissions closer to Orbae than the statistical leg, and its factor less close.

    Rank rather than ratio, permanently. Every Orbae row is assessment year 2023 against our 2020,
    and no 2020 export is coming, so level cannot cross the offset while rank very nearly can --
    WRI's own factors span 1.42x at the median over the same three years while its provincial rank
    holds at +0.961.

    LUC_ONLY on both sides, which is what makes the two comparable at all: Orbae's published factor
    and emissions exclude peatland occupation, so ours have to as well. Orbae's own `ef_kg_per_kg`
    is taken as that figure rather than the sum of its pool columns. The pools do not reconcile against the total for
    a large minority of rows -- 47% of the schema 2.2.0 rows against 14% of the 2.0.0 ones -- which is
    the mixed-schema defect `get_orbae_findings` already reports, showing up in the decomposition. The
    total is the quantity Orbae publishes; the breakdown is the part that disagrees with itself.

    Every Orbae row is `Method = jdLUC`, the family the direct and proxy legs belong to. The direct
    leg is US-only; the proxy leg runs wherever Orbae does.
    """
    admin_ids = get_orbae_admin_ids()
    orbae = read_orbae()
    orbae = orbae[
        (orbae["admin_level"] == schema.PROVINCIAL) & (orbae["ef_kg_per_kg"] > 0)
    ].copy()
    orbae["admin_id"] = [
        admin_ids.get((str(iso_3166), str(name)))
        for iso_3166, name in zip(
            orbae["iso_3166"], orbae["jurisdiction_name"], strict=True
        )
    ]
    orbae = orbae.dropna(subset=["admin_id"])

    levels = table.index.get_level_values
    provincial = with_crop_codes(
        frame=table[levels("admin_level") == schema.PROVINCIAL].reset_index()
    )
    # `admin_id` leads with the ISO, which is the only place the country appears on our rows.
    provincial["iso_3166"] = provincial["admin_id"].str[:3]
    provincial["luc_only_tonnes"] = (
        provincial["emissions_mt"] - provincial["peatland_occupation_emissions_mt"]
    )
    provincial["luc_only_kg_per_kg"] = (
        provincial["luc_only_tonnes"]
        * schema.KG_PER_TONNE
        / provincial["production_kg"]
    )
    provincial = provincial[
        (provincial["production_kg"] > 0) & (provincial["luc_only_kg_per_kg"] > 0)
    ]

    available = orbae.groupby(["iso_3166", "crop_name"]).size().to_dict()
    for methodology, factor_measure, emissions_measure in (
        (
            schema.STATISTICAL,
            targets.Measure.SLUC_OVER_ORBAE,
            targets.Measure.SLUC_OVER_ORBAE_EMISSIONS,
        ),
        (
            schema.JURISDICTIONAL_DIRECT,
            targets.Measure.JDLUC_OVER_ORBAE,
            targets.Measure.JDLUC_OVER_ORBAE_EMISSIONS,
        ),
        (
            schema.JURISDICTIONAL_DIRECT_PROXY,
            targets.Measure.JDLUC_PROXY_OVER_ORBAE,
            targets.Measure.JDLUC_PROXY_OVER_ORBAE_EMISSIONS,
        ),
    ):
        merged = provincial[provincial["methodology"] == methodology].merge(
            orbae[["admin_id", "crop_name", "ef_kg_per_kg", "luc_emissions_tonnes"]],
            left_on=["admin_id", "crop_code"],
            right_on=["admin_id", "crop_name"],
            suffixes=("", "_orbae"),
        )
        groups = [
            (iso_3166, crop_code, group)
            for (iso_3166, crop_code), group in merged.groupby(
                ["iso_3166", "crop_code"]
            )
            if len(group) >= MINIMUM_PROVINCES
        ]
        for iso_3166, crop_code, group in groups:
            for measure, ours_column, orbae_column in (
                (factor_measure, "luc_only_kg_per_kg", "ef_kg_per_kg"),
                (emissions_measure, "luc_only_tonnes", "luc_emissions_tonnes"),
            ):
                yield {
                    "iso_3166": str(iso_3166),
                    "crop_name": str(crop_code),
                    "emission_pool": schema.EmissionPool.LUC_ONLY,
                    "measure": measure,
                    "statistic": schema.Statistic.RANK_CORRELATION,
                    "ratio": float(
                        group[ours_column].rank().corr(group[orbae_column].rank())
                    ),
                    "provinces": len(group),
                    "comparability": schema.Comparability.PATTERN_ONLY,
                    "aggregation": schema.Aggregation.AS_PUBLISHED,
                    "coverage_fraction": len(group) / available[(iso_3166, crop_code)],
                    # Rank is a weaker claim than level, and Orbae is a supplied file rather than a
                    # pinned pull, so this cannot carry what a LEVEL comparison would.
                    "confidence": schema.Confidence.MEDIUM,
                    "worst_tier": schema.SourceTier.SUPPLIED,
                }


def get_comparisons(
    deforestation: pandas.DataFrame, table: pandas.DataFrame
) -> pandas.DataFrame:
    """Every comparison this run can make, with its control expectation attached.

    One row per (target, pool, measure, statistic). Orbae-against-WRI reads nothing from `table`, both sides
    being external; every other measure reads it. A measure with no row is absent rather than null,
    so `render_coverage` reports the targets no comparison reached.

    `deforestation` rides along so each row carries whether its own anchor is self-consistent. That
    fact belongs on the row for the same reason `comparability` does: a comparison against a WRI row
    claiming more deforested area than the crop is harvested from looks identical to a sound one once
    it is a number, and it is the row's own anchor that is wrong rather than us.
    """
    frame = pandas.DataFrame.from_records(
        list(iter_orbae_wri_comparisons())
        + list(iter_sluc_jdluc_comparisons(table=table))
        + list(iter_leg_emissions_comparisons(table=table))
        + list(iter_leg_province_comparisons(table=table))
        + list(iter_trace_wri_comparisons(table=table))
        + list(iter_orbae_trace_comparisons(table=table))
    )
    assert len(frame), "no comparison could be made, which means a join broke"
    # Keyed on the statistic too: one measure can carry a national ratio and a provincial rank for
    # the same pair, and a control guards only the one it was frozen on.
    expectations = {
        (
            control.target.iso_3166,
            control.target.crop_name,
            control.measure,
            control.statistic,
        ): control
        for control in targets.iter_controls()
    }
    # The MapSPAM code back to the `statistical.Crop` name targets use. WRI's combined coffee and
    # millet codes have no such name, so they keep their code.
    crop_to_name = {code: code for code in frame["crop_name"]} | {
        statistical.Crop[name].value: name for name in statistical.Crop.__members__
    }
    controls = [
        expectations.get(
            (
                row["iso_3166"],
                crop_to_name[row["crop_name"]],
                row["measure"],
                row["statistic"],
            )
        )
        for row in frame.to_dict("records")
    ]
    # Joined on the MapSPAM code, before the rename below puts both sides in different vocabularies.
    shares = {
        (row["iso_3166"], row["crop_name"]): row["deforestation_share"]
        for row in deforestation.to_dict("records")
    }
    frame["anchor_deforestation_share"] = [
        shares.get((row["iso_3166"], row["crop_name"]))
        for row in frame.to_dict("records")
    ]
    frame["crop_name"] = [crop_to_name[code] for code in frame["crop_name"]]
    frame["is_control"] = [control is not None for control in controls]
    # A target pair is what the report is about; everything else is context the comparison happens
    # to reach. `is_control` is narrower -- a target can be compared without carrying an expectation.
    chosen = {(target.iso_3166, target.crop_name) for target in targets.iter_targets()}
    frame["is_target"] = [
        (row["iso_3166"], row["crop_name"]) in chosen
        for row in frame.to_dict("records")
    ]
    frame["baseline"] = [control.baseline if control else None for control in controls]
    frame["tolerance"] = [
        control.tolerance if control else targets.DEFAULT_TOLERANCE
        for control in controls
    ]
    frame["baseline_source_version"] = [
        control.baseline_source_version if control else None for control in controls
    ]
    # Every anchor the row divides, not just WRI: an ORBAE_OVER_WRI ratio moves when either side is
    # revised, and recording one of the two would let an Orbae re-export pass as a pipeline change.
    frame["source_version"] = [
        schema.get_source_version_key(versions=get_anchor_versions(measure=measure))
        for measure in frame["measure"]
    ]
    return frame


def get_uncompared_targets(
    comparisons: pandas.DataFrame, table: pandas.DataFrame
) -> targets.UncomparedTargets:
    """Targets no comparison reached, split by whether `table` holds their country.

    Two silences that read alike and mean opposite things. A target whose country the table holds,
    and which still has no comparison, has no anchor -- a fact about WRI and Orbae that running more
    of our own pipeline cannot change. A target whose country the table lacks says nothing about
    anchors at all, and is answered by a `trace` run that includes it. Reporting the second as the
    first is how a table of a few countries comes to look like an absent yardstick.
    """
    compared = set(zip(comparisons["iso_3166"], comparisons["crop_name"], strict=True))
    traced = {
        str(admin_id)[:3] for admin_id in table.index.get_level_values("admin_id")
    }
    uncompared = [
        target
        for target in targets.iter_targets()
        if (target.iso_3166, target.crop_name) not in compared
    ]
    return targets.UncomparedTargets(
        not_in_table=tuple(t for t in uncompared if t.iso_3166 not in traced),
        unanchored=tuple(t for t in uncompared if t.iso_3166 in traced),
    )


def get_scope_difference() -> pandas.DataFrame:
    """CO2e against CO2 for the same country, crop and year.

    WRI publishes both as separate directories of the same product, differing by CH4 and N2O. Having
    the ratio means a comparison never has to guess which scope an outside figure came from.
    """
    merged = read_wri_national_emissions(gas_scope="CO2").merge(
        read_wri_national_emissions(gas_scope="CO2e"),
        on=["iso_3166", "crop_name", "year"],
        suffixes=("_co2", "_co2e"),
    )
    merged = merged[merged["ef_kg_per_kg_co2"] > 0]
    merged["scope_ratio"] = merged["ef_kg_per_kg_co2e"] / merged["ef_kg_per_kg_co2"]
    return merged[["iso_3166", "crop_name", "year", "scope_ratio"]]


def get_deforestation_share(
    emissions: pandas.DataFrame, faostat_areas: pandas.DataFrame, year: int
) -> pandas.DataFrame:
    """The share of a crop's harvested area WRI treats as deforestation-linked.

    WRI's `production` column is deforestation-linked production, not national, so dividing it by the
    yield gives the deforestation-linked area. Against FAOSTAT's total harvested area that becomes a
    share, which is the quantity our own expansion share should be compared against -- a share
    against a share, with no emissions term in either.
    """
    wri_yields = read_wri_yields(grain_name="national")
    for_year = emissions[
        (emissions["year"] == year) & (emissions["production_tonnes"] > 0)
    ]
    merged = for_year.merge(wri_yields, on=["iso_3166", "crop_name"]).merge(
        faostat_areas, on=["iso_3166", "crop_name"]
    )
    merged = merged[(merged["yield_kg"] > 0) & (merged["area_hectares"] > 0)]
    merged["deforestation_hectares"] = (
        merged["production_tonnes"] * schema.KG_PER_TONNE / merged["yield_kg"]
    )
    merged["deforestation_share"] = (
        merged["deforestation_hectares"] / merged["area_hectares"]
    )
    return merged[
        [
            "iso_3166",
            "crop_name",
            "deforestation_hectares",
            "area_hectares",
            "deforestation_share",
        ]
    ]


# --- Orbae -----------------------------------------------------------------------------
# Three source-specific decisions stand between the export and any join: a commodity crosswalk, a
# product-form rebasing, and a country code whose length depends on the grain.
# **The rebasing is the one that would silently ruin a comparison.** Orbae publishes two of its fifteen
# commodities against a *processed* product -- palm as crude palm oil, sugarcane as cane sugar -- and its
# per-kg factors are per kg of that product. WRI and FAOSTAT are both on the raw commodity, verified at
# 0.993 for oil palm fruit. So Orbae's palm factor reads 3.94x high and its sugarcane factor 8.91x high
# against them until divided by the conversion factor the export helpfully carries. Five of the thirty
# targets are affected.
# **Everything here is PATTERN_ONLY, permanently.** Every row is `Assessment year = 2023` against our
# 2020 comparison year, and no 2020 export is coming, so the offset is a property of the anchor rather
# than a temporary gap.
# That is a limit, not a disqualification, and the difference is measured rather than assumed. WRI
# publishes all five of its own reporting years, so it can be asked what a three-year offset costs.
# Level does not survive it: its own 2020-to-2024 factors span 1.42x at the median and 2.90x at the
# upper decile.
# Provincial *shape* very nearly does: rank correlation between its 2020 and 2023 provincial factors is
# +0.961 at the median over 3,186 (country, crop) pairs, 95% of them above +0.7 and none below +0.3, and
# every Orbae-covered control pair sits between +0.964 and +0.995. So a rank comparison against Orbae
# survives the offset and a ratio does not.
# The export mixes schema versions -- 2.0.0, 2.1.0 and 2.2.0 in one file -- so a cross-country
# comparison drawn from it spans anchor versions; that is recorded rather than fixed. Provincial ids
# are opaque Orbae strings (`AUS-20230119-1`), so its provinces reach a World Bank `admin_id` by name,
# through the committed `orbae_to_world_bank_admin_1.json` that `get_orbae_admin_ids` reads.


ORBAE_EXPORT = pull.RAW / "orbae" / "20260807_orbae_export.zip"
ORBAE_MEMBER_NAME = "20260807_orbae_export.csv"
# The year every row carries. Named so the PATTERN_ONLY reason is checkable rather than remembered,
# and read against the module's REFERENCE_YEAR above -- the offset between the two is what makes
# every Orbae row PATTERN_ONLY.
ORBAE_ASSESSMENT_YEAR = 2023


class OrbaeCommodity(enum.Enum):
    """Orbae's commodity name to the MapSPAM code it corresponds to one-for-one.

    Member name is the MapSPAM code; value is Orbae's spelling. Twelve of fifteen map; the three that
    do not are in `ORBAE_UNMAPPED_COMMODITIES` with the reason.
    """

    BARL = "Barley"
    COCO = "Cocoa"
    MAIZ = "Corn"
    COTT = "Cotton"
    OILP = "Palm"
    GROU = "Peanut"
    POTA = "Potato"
    RAPE = "Rapeseed"
    SOYB = "Soy"
    SUGC = "Sugarcane"
    SUNF = "Sunflower"
    WHEA = "Wheat"


# Why the other three are absent, so a reader is not left wondering whether they were forgotten.
ORBAE_UNMAPPED_COMMODITIES = {
    "Beef cattle": "livestock, and this pipeline models crops",
    "Coffee": (
        "MapSPAM splits arabica and robusta where Orbae publishes one green-coffee figure, so "
        "assigning it to either would invent the split"
    ),
    "Oats": "a member of MapSPAM's OCER group rather than a code of its own",
}
ORBAE_COMMODITY_TO_CROP_NAME = {member.value: member.name for member in OrbaeCommodity}
assert not set(ORBAE_COMMODITY_TO_CROP_NAME) & set(ORBAE_UNMAPPED_COMMODITIES)
assert all(
    crop_name in ifpri_mapspam.CANONICAL_CROP_CLS.__members__
    for crop_name in ORBAE_COMMODITY_TO_CROP_NAME.values()
)

ORBAE_TRACEABILITY_TO_ADMIN_LEVEL = {
    "0. Jurisdiction - Country": schema.NATIONAL,
    "1. State (ADM1)-level": schema.PROVINCIAL,
}
ORBAE_LUC_FACTOR_COLUMN = "LUC emission factor [kg CO2e / kg product]"
ORBAE_LUC_INTENSITY_COLUMN = "LUC emission factor [kg CO2e / hectare]"
# The LUC factor times production: no peatland occupation, and tonnes, so no conversion factor
ORBAE_LUC_EMISSIONS_COLUMN = "Total LUC emissions in assessment year [tons CO2e]"
ORBAE_CONVERSION_FACTOR_COLUMN = (
    "Product conversion factor [kg input product per kg of output product]"
)
ORBAE_POOL_TO_COLUMN = {
    schema.EmissionPool.FOREST: "Forest conversion emission factor [kg CO2e / kg product]",
    schema.EmissionPool.PEATLAND_CONVERSION: (
        "Peatland conversion emission factor [kg CO2e / kg product]"
    ),
    schema.EmissionPool.PEATLAND_OCCUPATION: (
        "Peatland occupation emission factor [kg CO2e / kg product]"
    ),
}
# Orbae splits grassland where we derive a single remainder, so both parts are read and summed rather
# than one being chosen. Ours is a single derived remainder with no such split, so the comparison is
# undefined and is not attempted.
ORBAE_GRASSLAND_COLUMNS = (
    "Natural grassland conversion emission factor [kg CO2e / kg product]",
    "Pastureland conversion emission factor [kg CO2e / kg product]",
)


@dataclasses.dataclass(frozen=True)
class OrbaeRow:
    """One Orbae figure, rebased and keyed the way the rest of the tool keys things."""

    iso_3166: str
    admin_level: str
    country_name: str
    jurisdiction_name: str
    crop_name: str
    schema_version: str
    ef_kg_per_kg: float
    intensity_tonnes_per_ha: float
    luc_emissions_tonnes: float
    forest_kg_per_kg: float
    grassland_kg_per_kg: float
    peatland_conversion_kg_per_kg: float
    peatland_occupation_kg_per_kg: float


def get_orbae_country(admin_level: str, jurisdiction_id: str) -> iso3166.Country | None:
    """The country an Orbae jurisdiction id names, or None where it names none.

    The grains use different code lengths: a country row's id is ISO alpha-2 (`AR`), a provincial row's
    is prefixed with alpha-3 (`AUS-20230119-1`). Reading both as alpha-2 mis-files seven countries,
    because one alpha-3's first two letters are another country's alpha-2 -- CHN reads as Switzerland,
    MEX as Montenegro, PRY as Puerto Rico -- and drops POL and URY entirely.
    """
    if admin_level == schema.NATIONAL:
        return iso3166.countries_by_alpha2.get(jurisdiction_id[:2])
    return iso3166.countries_by_alpha3.get(jurisdiction_id.split("-")[0])


def get_float(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    return float(value) if value else 0.0


def iter_orbae_rows(path_to_zip: pathlib.Path) -> collections.abc.Iterator[OrbaeRow]:
    """Every national or provincial row carrying a factor, rebased onto the commodity.

    Rows without a factor are skipped: 41% of the export carries one, the rest being jurisdictions
    listed for completeness. ADM2 and ADM3 rows are skipped too -- nothing here joins below the
    provincial grain.
    """
    with (
        zipfile.ZipFile(file=path_to_zip) as archive,
        archive.open(ORBAE_MEMBER_NAME) as member,
    ):
        for row in csv.DictReader(
            io.TextIOWrapper(member, encoding="utf8", errors="replace")
        ):
            if not row[ORBAE_LUC_FACTOR_COLUMN]:
                continue
            admin_level = ORBAE_TRACEABILITY_TO_ADMIN_LEVEL.get(
                row["Traceability level"]
            )
            if admin_level is None:
                continue
            crop_name = ORBAE_COMMODITY_TO_CROP_NAME.get(row["Commodity"])
            if crop_name is None:
                continue
            country = get_orbae_country(
                admin_level=admin_level, jurisdiction_id=row["Jurisdiction ID"]
            )
            if country is None:
                continue
            # kg of input commodity per kg of published product, so dividing returns a per-kg factor
            # on the commodity basis. 1 for thirteen commodities, 3.9405 for palm, 8.9108 for cane.
            conversion = (
                get_float(column=ORBAE_CONVERSION_FACTOR_COLUMN, row=row) or 1.0
            )
            yield OrbaeRow(
                admin_level=admin_level,
                country_name=row["Administrative level 0"],
                crop_name=crop_name,
                ef_kg_per_kg=get_float(column=ORBAE_LUC_FACTOR_COLUMN, row=row)
                / conversion,
                forest_kg_per_kg=(
                    get_float(
                        column=ORBAE_POOL_TO_COLUMN[schema.EmissionPool.FOREST], row=row
                    )
                    / conversion
                ),
                grassland_kg_per_kg=sum(
                    get_float(column=column, row=row)
                    for column in ORBAE_GRASSLAND_COLUMNS
                )
                / conversion,
                intensity_tonnes_per_ha=(
                    get_float(column=ORBAE_LUC_INTENSITY_COLUMN, row=row)
                    / schema.KG_PER_TONNE
                ),
                iso_3166=country.alpha3,
                jurisdiction_name=row["Administrative level 1"]
                or row["Administrative level 0"],
                luc_emissions_tonnes=get_float(
                    column=ORBAE_LUC_EMISSIONS_COLUMN, row=row
                ),
                peatland_conversion_kg_per_kg=(
                    get_float(
                        column=ORBAE_POOL_TO_COLUMN[
                            schema.EmissionPool.PEATLAND_CONVERSION
                        ],
                        row=row,
                    )
                    / conversion
                ),
                peatland_occupation_kg_per_kg=(
                    get_float(
                        column=ORBAE_POOL_TO_COLUMN[
                            schema.EmissionPool.PEATLAND_OCCUPATION
                        ],
                        row=row,
                    )
                    / conversion
                ),
                schema_version=row["Version"],
            )


def pin_orbae_export(path_to_zip: pathlib.Path = ORBAE_EXPORT) -> None:
    """Record the export's digest, because five frozen baselines derive from these bytes.

    Orbae is `SUPPLIED` rather than `PULLED` -- a file placed on the operator's machine, with no URL to
    re-retrieve it from -- so the digest is the only thing tying a baseline to the data it was set
    against. Swap the export and the Orbae/WRI baselines silently describe a different release.
    """
    # No `origin`: a supplied file was never retrieved from anywhere, and its absence says so.
    pull.record_digest(key=path_to_zip.name, path=path_to_zip, source="orbae")


def read_orbae(path_to_zip: pathlib.Path = ORBAE_EXPORT) -> pandas.DataFrame:
    """The export as a frame, with the comparability it is limited to attached to every row.

    `comparability` is a column rather than a caveat because a `PATTERN_ONLY` ratio and a `LEVEL` one
    look identical once they are numbers, and averaging the two together is the mistake this prevents.
    """
    frame = pandas.DataFrame.from_records(
        [dataclasses.asdict(row) for row in iter_orbae_rows(path_to_zip=path_to_zip)]
    )
    assert len(frame), f"{path_to_zip} yielded no usable rows"
    # One ISO per Orbae country name and one name per ISO. This is the check that would have caught
    # the alpha-2 truncation: the mis-mapped rows kept Orbae's own country name while their resolved
    # ISO said something else, so the pairing was two-to-one.
    pairs = frame.groupby("country_name")["iso_3166"].nunique()
    ambiguous = sorted(pairs[pairs > 1].index)
    assert not ambiguous, (
        f"country names resolving to several ISOs: {', '.join(ambiguous)}"
    )
    reverse = frame.groupby("iso_3166")["country_name"].nunique()
    shared = sorted(reverse[reverse > 1].index)
    assert not shared, f"ISOs claimed by several country names: {', '.join(shared)}"

    frame["source"] = schema.Source.ORBAE
    frame["comparability"] = schema.Comparability.PATTERN_ONLY
    frame["discount_basis"] = schema.DiscountBasis.GHGP_LINEAR
    frame["gas_scope"] = schema.GasScope.CO2E
    frame["reporting_year"] = ORBAE_ASSESSMENT_YEAR
    return frame


def get_orbae_findings(frame: pandas.DataFrame) -> list[schema.Finding]:
    """The two things about this export that limit every comparison drawn from it."""
    versions = sorted(set(frame["schema_version"]))
    return [
        schema.Finding(
            affected_iso_3166s=tuple(sorted(set(frame["iso_3166"]))),
            affected_rows=len(frame),
            confidence=schema.Confidence.HIGH,
            message=(
                f"Every row is assessment year {ORBAE_ASSESSMENT_YEAR:d} against our "
                f"{REFERENCE_YEAR:d} comparison year, and no {REFERENCE_YEAR:d} export is coming, "
                "so every Orbae measure is permanently PATTERN_ONLY. WRI's own reporting years "
                "span 1.42x at the median across the same offset, against a provincial rank "
                "correlation of "
                "+0.961, so rank survives it and ratio does not. Every sLUC/Orbae and Orbae/WRI "
                "control therefore holds a rank expectation, with a tolerance in correlation units"
            ),
            severity=schema.Severity.BLOCKING,
            slug="orbae-vintage-offset",
        ),
        schema.Finding(
            affected_rows=len(frame),
            confidence=schema.Confidence.MEDIUM,
            message=(
                f"The export mixes {len(versions):d} schema versions "
                f"({', '.join(versions)}) in one file, so a cross-country comparison drawn from it "
                "spans anchor versions"
            ),
            severity=schema.Severity.ADVISORY,
            slug="orbae-mixed-schema-version",
        ),
    ]


def read_faostat_production() -> pandas.DataFrame:
    """FAOSTAT's ingested production and area, cached locally and pinned by digest.

    Read from the ingest rather than from a copy of the upstream archive: the dataset is the single
    record of what this data is, and duplicating it here duplicated its provenance too. The first read
    fetches from `ingest_root`; every read after it is local, which is what keeps the report fast.
    """
    if not FAOSTAT_CACHE.exists():
        logger.info(f"Fetching the ingested FAOSTAT parquet to {FAOSTAT_CACHE}")
        FAOSTAT_CACHE.parent.mkdir(exist_ok=True, parents=True)
        faostat.load(dataset=faostat.CROP_DATASET).to_parquet(FAOSTAT_CACHE)
        pull.record_digest(
            key="faostat.parquet",
            origin=faostat.CROP_DATASET.get_prefix(tile_id="world"),
            path=FAOSTAT_CACHE,
            source="faostat",
        )
    frame = pandas.read_parquet(FAOSTAT_CACHE)
    if frame.index.names != [None]:
        frame = frame.reset_index()
    frame = frame[frame["commodity_id"].isin(FAOSTAT_ITEM_CODE_TO_CROP_CODE)].copy()
    frame["crop_name"] = frame["commodity_id"].map(FAOSTAT_ITEM_CODE_TO_CROP_CODE)
    return frame.rename(columns={"admin_id": "iso_3166"})


def read_faostat_areas(year: int) -> pandas.DataFrame:
    """FAOSTAT's harvested area per country and crop for one year."""
    frame = read_faostat_production()
    return frame[frame["year"] == year][["iso_3166", "crop_name", "area_hectares"]]
