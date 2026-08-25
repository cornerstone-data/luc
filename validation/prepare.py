"""Read the pulled anchors into the frames `report` renders, and check them against each other.

Anchors only, for now: WRI's factors and yield factors, and FAOSTAT's production. The capture side
-- our own `efs.parquet` -- joins here once it exists, which runs the pipeline and so is separate.

The yield cross-check is the first thing this does, and it runs before any emissions comparison for
a reason. WRI's factor is an intensity over a yield:

    EF [kgCO2/kg] = intensity [kgCO2/ha] x yield_factor_kg [ha/kg]

so a yield disagreement moves every factor built on it. If WRI and FAOSTAT are on different product
forms for a crop -- fruit bunches against palm oil, seed cotton against ginned lint -- the ratio is
off by the milling yield, several-fold, and every emissions comparison for that crop inherits it
while looking like a methodology gap. This finds that before a capture is spent on it.

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
import statistics
import zipfile

import iso3166
import pandas

from jdluc import statistical
from jdluc.datasets import faostat_production, ifpri_mapspam
from validation import pull, schema, targets

logger = logging.getLogger(__name__)

# WRI publishes one yield per (country, crop) with no year on it, so which year it is drawn from has
# to be inferred rather than read. 2020 is the comparison year for everything else here, and
# `--year` sweeps the neighbourhood to see which fits best.
#
# It is also the WRI *reporting* year to read, and that is a match of windows rather than a
# convention: a reporting year names where the LSRS 20-year window sits, so 2020 covers loss years
# 2001-2020, which is the span our own GLAD epochs cover. See WRI_REPORTING_YEARS.
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
FAOSTAT_CACHE = pull.CACHE / "faostat_production.parquet"
# How much of a revision or digest identifies an anchor in a `source_version`. Long enough not to
# collide, short enough that a stale-baseline message is readable.
SOURCE_VERSION_LENGTH = 12
# The one part of eligibility that cannot be computed from the pinned anchors: which countries
# intersect a GFW tile, which needs a spatial join against a 93 MiB GeoPackage. 228 ISO codes, so
# `tools/build-tiled-countries.py` commits this and `get_eligible` derives the rest here.
#
# The shortlist itself is deliberately NOT committed. Every other column in it -- WRI's
# deforestation emissions, FAOSTAT's
# production, WRI's provincial unit counts -- is already pinned by sha256 in sources.lock.json, and
# committing a second copy would break the rule the lock rests on: the lock is committed, the bytes are
# not. Recomputing all 864 pairs from those anchors takes about a tenth of a second, and it cannot go
# stale against the pin because it is read from whatever the pin currently names.
TILED_ISO_3166S = pull.DATA / "tiled_iso_3166s.json"
# E4. An emissions factor that cannot move a companywide number is not worth capture time.
MINIMUM_PRODUCTION_KG = 100 * 1_000 * 1_000
# E3. Land area, a proxy for the cropland weighting that would need a capture -- and the capture needs
# the target set, which needs E3, which is why the proxy stands.
MINIMUM_AREA_COVERAGE = 0.5
# GLAD GLCLUC's cropland class excludes perennial *woody* crops by construction -- it covers "annual
# and perennial herbaceous crops", and "perennial woody crops, permanent pastures and shifting
# cultivation are excluded" (Potapov et al. 2022, Nature Food 3, 19-28,
# doi:10.1038/s43016-021-00429-z). Those appear as tree cover instead, which
# `emit.get_land_class` maps to FOREST. Both legs then restrict the emissions numerator to 2020 GLAD
# cropland while MapSPAM keeps the production in the denominator, so the pipeline attributes close to
# zero to exactly these crops. They stay eligible on purpose: a pair we would rank highly and cannot
# compute is the finding, not a filtering mistake.
PERENNIAL_CROP_NAMES = frozenset(
    {
        statistical.Crop.ARABICA_COFFEE.value,
        statistical.Crop.ROBUSTA_COFFEE.value,
        statistical.Crop.COCONUT.value,
        statistical.Crop.OILPALM.value,
    }
)
# Three herbaceous perennials are deliberately absent, and they are the test of whether this list
# tracks the definition rather than the word "perennial". Sugarcane is ratooned; banana and plantain
# are giant herbs with a pseudostem rather than wood. GLAD's definition admits all three, so the
# pipeline should see them and they are not part of the detection gap.
for _herbaceous in (
    statistical.Crop.SUGARCANE,
    statistical.Crop.BANANA,
    statistical.Crop.PLANTAIN,
):
    assert _herbaceous.value not in PERENNIAL_CROP_NAMES
# docs/further_research.md headings. The register measures that document's entries rather than
# keeping a second list, so a slug here is a heading there, verbatim.
PERENNIAL_RESEARCH_SLUG = "Woody perennial crops fall outside GLAD's cropland class"
OIL_PALM_RESEARCH_SLUG = "The statistical leg under-allocates oil palm relative to WRI"


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
        reporting_years="size", lowest="min", highest="max", median="median"
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
            slug="wri-reporting-year-sensitivity",
            severity=schema.Severity.ADVISORY,
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
            confidence=schema.Confidence.HIGH,
            affected_rows=len(beyond),
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
            slug="wri-deforestation-area-exceeds-harvested-area",
            severity=schema.Severity.ADVISORY,
            message=(
                f"{len(over):,d} of {len(deforestation):,d} (country, crop) pairs "
                f"({len(over) / len(deforestation):.1%}) give WRI more deforestation-linked area "
                f"than FAOSTAT reports harvested, up to {over['deforestation_share'].max():.1f}x. "
                "A crop cannot be grown on more land than it is harvested from, so these pairs "
                "cannot anchor a share comparison whichever side is wrong"
            ),
            confidence=schema.Confidence.MEDIUM,
            affected_rows=len(over),
            affected_iso_3166s=tuple(sorted(set(over["iso_3166"]))[:12]),
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
                iso_3166=iso_3166,
                crop_name=crop_name,
                mapspam_code=mapspam_code,
                deforestation_tonnes=float(row["deforestation_tonnes"]),
                production_kg=production_kg,
                is_perennial=mapspam_code in PERENNIAL_CROP_NAMES,
                is_decomposed_group_crop=(
                    mapspam_code in ifpri_mapspam.CONSTITUENT_TO_GROUP_NAME
                ),
                provincial_units=units,
                area_coverage=area_coverage.get(iso_3166, 0.0),
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
    faostat = read_faostat_production()
    candidates = tuple(
        iter_candidates(
            area_coverage=key_map["area_coverage"],
            production_by_pair=faostat[faostat["year"] == REFERENCE_YEAR][
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

    Sized from the anchors alone, so it needs no capture: these pairs carry WRI's own deforestation
    attribution while both legs attribute close to nothing to them. This survived dropping the
    scorer unchanged, which is the clearest evidence the ranking was never what produced it.

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
            slug=PERENNIAL_RESEARCH_SLUG,
            severity=schema.Severity.BLOCKING,
            message=(
                f"{len(perennial):d} eligible pairs are woody perennials, carrying "
                f"{at_stake / schema.TONNES_PER_MEGATONNE:,.0f} Mt of WRI-attributed deforestation "
                f"between them and led by {largest['iso_3166']!s} {largest['crop_name']!s} at "
                f"{float(largest['deforestation_tonnes']) / schema.TONNES_PER_MEGATONNE:,.0f} Mt. "
                "GLAD's cropland class excludes perennial woody crops, so these appear as tree cover and "
                "both legs exclude them from the emissions numerator while MapSPAM keeps their "
                "production in the denominator. The pipeline attributes close to zero where the "
                "anchor does not"
            ),
            confidence=schema.Confidence.HIGH,
            magnitude_tonnes=at_stake,
            affected_rows=len(perennial),
            affected_iso_3166s=tuple(sorted(set(perennial["iso_3166"]))),
        ),
        schema.Finding(
            slug=OIL_PALM_RESEARCH_SLUG,
            severity=schema.Severity.ADVISORY,
            message=(
                "Oil palm is the measured case of the entry above: the sLUC-to-WRI ratio for "
                "Indonesian oil palm is 0.008, decomposing as a 0.020 detection term and a 0.38 "
                "allocation term. The detection term belongs to the perennial entry"
            ),
            confidence=schema.Confidence.HIGH,
            affected_rows=int(
                (eligible["mapspam_code"] == statistical.Crop.OILPALM.value).sum()
            ),
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
    armed = {target.slug for target in targets.iter_control_targets()}
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
                slug="wri-deforestation-area-exceeds-harvested-area",
                severity=schema.Severity.ADVISORY,
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
                confidence=schema.Confidence.MEDIUM,
                affected_rows=1,
                affected_iso_3166s=(target.iso_3166,),
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
    measurable with no capture, which is why it is seeded rather than waiting.
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
            slug="external-anchors-disagree-on-shape",
            severity=schema.Severity.ADVISORY,
            message=(
                f"Orbae and WRI barely agree on which countries carry the highest factor for "
                f"{len(weak):d} of {len(agreement):d} shared crops: {named:s}. WRI's own provincial "
                "rank holds at +0.961 across the same three-year offset, so this is method rather "
                "than vintage. For these crops the two anchors cannot corroborate each other, so an "
                "agreement with either is not independent confirmation, and the Orbae control "
                "measures are rank controls rather than level ones, and `ORBAE_OVER_WRI` is where "
                "this disagreement gets measured on every run"
            ),
            confidence=schema.Confidence.MEDIUM,
            affected_rows=int(weak["countries"].sum()),
        )
    ]


def get_orbae_admin_ids() -> dict[tuple[str, str], str]:
    """(iso_3166, Orbae province name) to the World Bank `admin_id` our capture is keyed on.

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

    The only comparison that can be made before a capture, because both sides are external. Rank
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

    Our own legs are absent rather than null, since `code_version` already identifies them. That
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
            case schema.Source.SLUC | schema.Source.JDLUC:
                continue
            case _:
                raise NotImplementedError(
                    f"{measure.name:s} divides {source.name:s}, which records no version here; "
                    "add one before a control can be frozen against it"
                )
    return versions


def read_efs() -> pandas.DataFrame | None:
    """The captured emissions factors, or None before a capture has run.

    Read here rather than through `validation.capture` so the reporting path never imports the
    module that runs the pipeline -- the same division `read_forest_pools` uses.
    """
    if not pull.EFS.exists():
        return None
    return pandas.read_parquet(pull.EFS)


def iter_sluc_jdluc_comparisons() -> collections.abc.Iterator[dict[str, object]]:
    """Our two legs against each other, nationally, per crop both of them model.

    The only comparison in the design with no external anchor in it, and so the only one where a
    disagreement proves one of our own legs wrong rather than raising a question about a yardstick.
    It is a filter on `methodology` within one artifact rather than a join between two, which is what
    `trace.CANONICAL_KEY` carrying `methodology` buys.

    The published emissions factor is the quantity compared, because that is what a control on this
    pair guards. It is not a pure emissions ratio: the legs use different production denominators --
    NASS for jdLUC, MapSPAM for sLUC -- so a movement here is either leg's numerator or either leg's
    denominator. Localizing it further would need the two legs' production compared directly, which
    nothing here does.
    """
    efs = read_efs()
    if efs is None:
        return
    national = efs[
        efs.index.get_level_values("admin_level") == schema.NATIONAL
    ].reset_index()
    by_methodology = national.pivot_table(
        index=["admin_id", "crop_name"],
        columns="methodology",
        values="emissions_factor_kgco2e_per_kg",
    )
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
            "crop_name": statistical.Crop[str(row["crop_name"])].value,
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


def iter_sluc_wri_comparisons() -> collections.abc.Iterator[dict[str, object]]:
    """Our forest emissions against WRI's published deforestation emissions, tonnes against tonnes.

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
    efs = read_efs()
    if efs is None:
        return
    deforestation = {
        (row["iso_3166"], row["crop_name"]): float(row["deforestation_tonnes"])
        for row in read_wri_national_emissions(gas_scope="CO2e").to_dict("records")
        if row["year"] == REFERENCE_YEAR and row["deforestation_tonnes"] > 0
    }
    levels = efs.index.get_level_values
    national = efs[
        (levels("admin_level") == schema.NATIONAL)
        & (levels("methodology") == schema.STATISTICAL)
    ].reset_index()
    national["crop_code"] = [
        statistical.Crop[name].value for name in national["crop_name"]
    ]
    for row in sorted(
        national.to_dict("records"),
        key=lambda record: (record["admin_id"], record["crop_code"]),
    ):
        anchor_tonnes = deforestation.get((row["admin_id"], row["crop_code"]))
        if not anchor_tonnes:
            continue
        if row["production_kg"] <= 0:
            # MapSPAM puts no production of this crop in this country, so we attribute nothing and
            # the ratio is a hard zero. That is an absence rather than a disagreement, and seven of
            # them would drag any statistic over these rows. A zero ratio where production *is*
            # present stays: attributing nothing to a crop we do grow is a finding.
            continue
        yield {
            "iso_3166": row["admin_id"],
            "crop_name": row["crop_code"],
            "emission_pool": schema.EmissionPool.FOREST,
            "measure": targets.Measure.SLUC_OVER_WRI,
            "statistic": schema.Statistic.RATIO,
            "ratio": float(row["forest_emissions_mt"]) / anchor_tonnes,
            # Nothing is joined provincially, so nothing can be dropped: both sides are the whole
            # national figure, and a zero here means "no provinces used" rather than "a thin join".
            "provinces": 0,
            "comparability": schema.Comparability.LEVEL,
            "aggregation": schema.Aggregation.AS_PUBLISHED,
            "coverage_fraction": 1.0,
            # One anchor, and it shares this leg's MapSPAM expansion-share family, so it
            # corroborates the forest pool and carbon density rather than the allocation.
            "confidence": schema.Confidence.MEDIUM,
            "worst_tier": schema.SourceTier.PULLED,
        }


def iter_orbae_capture_comparisons() -> collections.abc.Iterator[dict[str, object]]:
    """Each of our legs against Orbae, province by province, as a rank correlation.

    Rank rather than ratio, permanently. Every Orbae row is assessment year 2023 against our 2020,
    and no 2020 export is coming, so level cannot cross the offset while rank very nearly can --
    WRI's own factors span 1.42x at the median over the same three years while its provincial rank
    holds at +0.961.

    LUC_ONLY on both sides, which is what makes the two comparable at all: Orbae's published factor
    excludes peatland occupation, so ours has to as well. Orbae's own `ef_kg_per_kg` is taken as that
    figure rather than the sum of its pool columns. The pools do not reconcile against the total for
    a large minority of rows -- 47% of the schema 2.2.0 rows against 14% of the 2.0.0 ones -- which is
    the mixed-schema defect `get_orbae_findings` already reports, showing up in the decomposition. The
    total is the quantity Orbae publishes; the breakdown is the part that disagrees with itself.

    JDLUC_OVER_ORBAE is the only method-family-matched comparison in the design: every Orbae row is
    `Method = jdLUC`, and the USA is the one jurisdiction where we run a jdLUC leg too.
    """
    efs = read_efs()
    if efs is None:
        return
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

    levels = efs.index.get_level_values
    provincial = efs[levels("admin_level") == schema.PROVINCIAL].reset_index()
    provincial["crop_code"] = [
        statistical.Crop[name].value for name in provincial["crop_name"]
    ]
    # `admin_id` leads with the ISO, which is the only place the country appears on our rows.
    provincial["iso_3166"] = provincial["admin_id"].str[:3]
    provincial["luc_only_kg_per_kg"] = (
        (provincial["emissions_mt"] - provincial["peatland_occupation_emissions_mt"])
        * schema.KG_PER_TONNE
        / provincial["production_kg"]
    )
    provincial = provincial[
        (provincial["production_kg"] > 0) & (provincial["luc_only_kg_per_kg"] > 0)
    ]

    for methodology, measure in (
        (schema.STATISTICAL, targets.Measure.SLUC_OVER_ORBAE),
        (schema.JURISDICTIONAL_DIRECT, targets.Measure.JDLUC_OVER_ORBAE),
    ):
        ours = provincial[provincial["methodology"] == methodology]
        merged = ours.merge(
            orbae[["admin_id", "crop_name", "ef_kg_per_kg"]],
            left_on=["admin_id", "crop_code"],
            right_on=["admin_id", "crop_name"],
            suffixes=("", "_orbae"),
        )
        for (iso_3166, crop_code), group in merged.groupby(["iso_3166", "crop_code"]):
            if len(group) < MINIMUM_PROVINCES:
                continue
            available = len(
                orbae[
                    (orbae["iso_3166"] == iso_3166) & (orbae["crop_name"] == crop_code)
                ]
            )
            yield {
                "iso_3166": str(iso_3166),
                "crop_name": str(crop_code),
                "emission_pool": schema.EmissionPool.LUC_ONLY,
                "measure": measure,
                "statistic": schema.Statistic.RANK_CORRELATION,
                "ratio": float(
                    group["luc_only_kg_per_kg"]
                    .rank()
                    .corr(group["ef_kg_per_kg"].rank())
                ),
                "provinces": len(group),
                "comparability": schema.Comparability.PATTERN_ONLY,
                "aggregation": schema.Aggregation.AS_PUBLISHED,
                "coverage_fraction": len(group) / available if available else 0.0,
                # Rank is a weaker claim than level, and Orbae is a supplied file rather than a
                # pinned pull, so this cannot carry what a LEVEL comparison would.
                "confidence": schema.Confidence.MEDIUM,
                "worst_tier": schema.SourceTier.SUPPLIED,
            }


def read_forest_pools() -> pandas.DataFrame | None:
    """The conservation bound's inputs from the last capture, or None before one has run.

    None rather than an empty frame, because `report.render` distinguishes them: a missing frame
    renders as "not run", where an empty one would render as a table with no country over its bound
    and read as a pass. This is the check that outranks every anchor, so its silence must not.

    Read here rather than through `validation.capture` so the reporting path never imports the
    module that runs the pipeline.
    """
    if not pull.FOREST_POOLS.exists():
        return None
    return pandas.read_parquet(pull.FOREST_POOLS)


def get_comparisons(
    repo_root: pathlib.Path, deforestation: pandas.DataFrame
) -> pandas.DataFrame:
    """Every comparison this run can make, with its control expectation attached.

    One row per (target, pool, measure). Orbae-against-WRI needs no capture, both sides being
    external; sLUC-against-jdLUC needs one and yields nothing until it has run. A measure with no row
    is absent rather than null, so `render_coverage` reports the targets no comparison reached.

    `deforestation` rides along so each row carries whether its own anchor is self-consistent. That
    fact belongs on the row for the same reason `comparability` does: a comparison against a WRI row
    claiming more deforested area than the crop is harvested from looks identical to a sound one once
    it is a number, and it is the row's own anchor that is wrong rather than us.
    """
    frame = pandas.DataFrame.from_records(
        list(iter_orbae_wri_comparisons())
        + list(iter_sluc_jdluc_comparisons())
        + list(iter_sluc_wri_comparisons())
        + list(iter_orbae_capture_comparisons())
    )
    assert len(frame), "no comparison could be made, which means a join broke"
    expectations = {
        (control.target.iso_3166, control.target.crop_name, control.measure): control
        for control in targets.iter_controls()
    }
    crop_to_name = {
        statistical.Crop[name].value: name for name in statistical.Crop.__members__
    }
    controls = [
        expectations.get(
            (row["iso_3166"], crop_to_name.get(row["crop_name"], ""), row["measure"])
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
    frame["crop_name"] = [crop_to_name.get(code, code) for code in frame["crop_name"]]
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
    frame["code_version"] = schema.get_code_version(repo_root=repo_root)
    # Every anchor the row divides, not just WRI: an ORBAE_OVER_WRI ratio moves when either side is
    # revised, and recording one of the two would let an Orbae re-export pass as a pipeline change.
    frame["source_version"] = [
        schema.get_source_version_key(versions=get_anchor_versions(measure=measure))
        for measure in frame["measure"]
    ]
    return frame


def get_unanchored_targets(
    comparisons: pandas.DataFrame,
) -> tuple[targets.Target, ...]:
    """Targets no comparison reached, so their silence is not read as agreement.

    Today that is most of them: only Orbae-against-WRI can be computed before a capture, and Orbae
    covers 12 of the 30 targets.
    """
    compared = set(zip(comparisons["iso_3166"], comparisons["crop_name"], strict=True))
    return tuple(
        target
        for target in targets.iter_targets()
        if (target.iso_3166, target.crop_name) not in compared
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
# Two further hazards are recorded rather than fixed. The export mixes schema versions -- 2.0.0, 2.1.0
# and 2.2.0 in one file -- which is the same defect `code_version` guards against in our own
# artifact. And provincial ids are opaque Orbae strings (`AUS-20230119-1`), so a provincial join needs
# name matching like the GADM-to-World-Bank map, which is not built.


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
                get_float(row=row, column=ORBAE_CONVERSION_FACTOR_COLUMN) or 1.0
            )
            yield OrbaeRow(
                iso_3166=country.alpha3,
                admin_level=admin_level,
                country_name=row["Administrative level 0"],
                jurisdiction_name=row["Administrative level 1"]
                or row["Administrative level 0"],
                crop_name=crop_name,
                schema_version=row["Version"],
                ef_kg_per_kg=get_float(row=row, column=ORBAE_LUC_FACTOR_COLUMN)
                / conversion,
                intensity_tonnes_per_ha=(
                    get_float(row=row, column=ORBAE_LUC_INTENSITY_COLUMN)
                    / schema.KG_PER_TONNE
                ),
                forest_kg_per_kg=(
                    get_float(
                        row=row, column=ORBAE_POOL_TO_COLUMN[schema.EmissionPool.FOREST]
                    )
                    / conversion
                ),
                grassland_kg_per_kg=sum(
                    get_float(row=row, column=column)
                    for column in ORBAE_GRASSLAND_COLUMNS
                )
                / conversion,
                peatland_conversion_kg_per_kg=(
                    get_float(
                        row=row,
                        column=ORBAE_POOL_TO_COLUMN[
                            schema.EmissionPool.PEATLAND_CONVERSION
                        ],
                    )
                    / conversion
                ),
                peatland_occupation_kg_per_kg=(
                    get_float(
                        row=row,
                        column=ORBAE_POOL_TO_COLUMN[
                            schema.EmissionPool.PEATLAND_OCCUPATION
                        ],
                    )
                    / conversion
                ),
            )


def pin_orbae_export(path_to_zip: pathlib.Path = ORBAE_EXPORT) -> None:
    """Record the export's digest, because five frozen baselines derive from these bytes.

    Orbae is `SUPPLIED` rather than `PULLED` -- a file placed on the operator's machine, with no URL to
    re-retrieve it from -- so the digest is the only thing tying a baseline to the data it was set
    against. Swap the export and the Orbae/WRI baselines silently describe a different release.
    """
    # No `origin`: a supplied file was never retrieved from anywhere, and its absence says so.
    pull.record_digest(key=path_to_zip.name, source="orbae", path=path_to_zip)


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
            slug="orbae-vintage-offset",
            severity=schema.Severity.BLOCKING,
            message=(
                f"Every row is assessment year {ORBAE_ASSESSMENT_YEAR:d} against our "
                f"{REFERENCE_YEAR:d} comparison year, and no {REFERENCE_YEAR:d} export is coming, "
                "so every Orbae measure is permanently PATTERN_ONLY. WRI's own reporting years "
                "span 1.42x at the median across the same offset, against a provincial rank "
                "correlation of "
                "+0.961, so rank survives it and ratio does not. Every sLUC/Orbae and Orbae/WRI "
                "control therefore holds a rank expectation, with a tolerance in correlation units"
            ),
            confidence=schema.Confidence.HIGH,
            affected_rows=len(frame),
            affected_iso_3166s=tuple(sorted(set(frame["iso_3166"]))),
        ),
        schema.Finding(
            slug="orbae-mixed-schema-version",
            severity=schema.Severity.ADVISORY,
            message=(
                f"The export mixes {len(versions):d} schema versions "
                f"({', '.join(versions)}) in one file, so a cross-country comparison drawn from it "
                "spans anchor versions. This is the defect a `code_version` column exists to catch "
                "in our own artifact, here on the anchor side"
            ),
            confidence=schema.Confidence.MEDIUM,
            affected_rows=len(frame),
        ),
    ]


# The one EPA figure this tool uses, hardcoded rather than parsed: a transcription pipeline for a single
# number is not worth the dependency, and the number is checkable by hand in a minute.
#
# EPA GHG Inventory 1990-2022, published April 2024, Chapter 6, Table 6-40 on page 6-75, "Net CO2 Flux
# from Soil, Dead Organic Matter and Biomass Carbon Stock Changes in Land Converted to Cropland by
# Land-Use Change Category (MMT CO2 Eq.)", row "Grassland Converted to Cropland".
#
# To reproduce, from a machine with network access:
#
#   uv run --with pdfplumber python -c "
#   import pdfplumber, urllib.request, pathlib
#   u = ('https://www.epa.gov/system/files/documents/2024-04/'
#        'us-ghg-inventory-2024-chapter-6-land-use-land-use-change-and-forestry_0.pdf')
#   r = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
#   pathlib.Path('ch6.pdf').write_bytes(urllib.request.urlopen(r).read())
#   print([l for p in pdfplumber.open('ch6.pdf').pages if 'Table 6-40:' in (p.extract_text() or '')
#          for l in p.extract_text().split(chr(10)) if l.startswith('Grassland Converted')])"
#
# That PDF is sha256 a9914b673a61b3bba926253f9d68168dde2ef1487c388455fd78922a69263d5d, 5.5 MiB, 194
# pages. It is not in `sources.lock.json` because nothing here retrieves it, so there is nothing to pin.
#
# The comparison year is 2020, where EPA reports 10.6, putting sLUC's 30.9 Mt US grassland figure at
# about 2.9x -- the same comparison `docs/further_research.md` draws. The 2022 column is the easy
# mistake to make here: at 16.3 it is the row's most recent number and it is not our year.
#
# Only one mismatch remains, and it is not the temporal basis. The two sides are both annual: the
# GHGP per-year weights integrate to exactly 1.0 over the 20-year window, so `SPAN_TO_LINEAR_DISCOUNT_WEIGHT`
# allocates a conversion's emissions to one sourcing year rather than accumulating twenty, and the
# production denominator is a weighted mean over the same spans. What does differ is scope -- EPA's row
# covers all cropland where sLUC's 30.9 Mt is the corn-soy-wheat subset -- and closing that gap can only
# raise our side, so the disagreement is a floor rather than an artifact.
EPA_RELEASE = "1990-2022, published 2024"
EPA_GRASSLAND_TO_CROPLAND_MMT_BY_YEAR = {
    1990: 27.3,
    2005: 17.2,
    2018: 13.7,
    2019: 13.0,
    2020: 10.6,
    2021: 16.1,
    2022: 16.3,
}


def read_faostat_production() -> pandas.DataFrame:
    """FAOSTAT's ingested production and area, cached locally and pinned by digest.

    Read from the ingest rather than from a copy of the upstream archive: the dataset is the single
    record of what this data is, and duplicating it here duplicated its provenance too. The first read
    fetches from `ingest_root`; every read after it is local, which is what keeps the report fast.
    """
    if not FAOSTAT_CACHE.exists():
        logger.info(f"Fetching the ingested FAOSTAT parquet to {FAOSTAT_CACHE}")
        FAOSTAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        faostat_production.load().to_parquet(FAOSTAT_CACHE)
        pull.record_digest(
            key="faostat_production.parquet",
            source="faostat",
            path=FAOSTAT_CACHE,
            origin=faostat_production.DATASET.get_prefix(tile_id="world"),
        )
    frame = pandas.read_parquet(FAOSTAT_CACHE)
    if frame.index.names != [None]:
        frame = frame.reset_index()
    return frame.rename(columns={"admin_id": "iso_3166"})


def get_faostat_yields() -> pandas.DataFrame:
    """FAOSTAT's yield per country, crop and year, derived rather than read.

    `jdluc.datasets.faostat_production` carries area and production and deliberately not yield,
    because a MapSPAM group crop's yield is not the sum of its constituents'. Dividing here
    reproduces FAOSTAT's own published yield exactly for the one-to-one crops, which are all this
    compares.

    Read through `read_faostat_production`, so this comes from the ingested parquet rather than
    from a second copy of the upstream archive. Its digest is recorded in sources.lock.json, so
    the bytes are pinned either way.
    """
    frame = read_faostat_production()
    frame["yield_kg"] = frame["production_kg"] / frame["area_hectares"]
    return frame[frame["area_hectares"] > 0][
        ["iso_3166", "crop_name", "year", "yield_kg"]
    ]


def read_faostat_areas(year: int) -> pandas.DataFrame:
    """FAOSTAT's harvested area per country and crop for one year."""
    frame = read_faostat_production()
    return frame[frame["year"] == year][["iso_3166", "crop_name", "area_hectares"]]


def get_yield_comparison(
    faostat_yields: pandas.DataFrame, wri_yields: pandas.DataFrame, year: int
) -> pandas.DataFrame:
    """One row per country and crop the two sources both report, with their ratio.

    An inner join, so a crop WRI publishes and FAOSTAT does not simply does not appear. That is the
    intended behavior and `get_unpaired_crop_names` reports what it dropped, since a crop silently
    absent from a comparison reads as a crop that agreed.
    """
    for_year = faostat_yields[faostat_yields["year"] == year]
    assert len(for_year), f"FAOSTAT has no rows for {year:d}"
    merged = wri_yields.merge(
        for_year, on=["iso_3166", "crop_name"], suffixes=("_wri", "_faostat")
    )
    merged = merged[(merged["yield_kg_wri"] > 0) & (merged["yield_kg_faostat"] > 0)]
    merged["ratio"] = merged["yield_kg_wri"] / merged["yield_kg_faostat"]
    return merged


def iter_yield_agreements(
    comparison: pandas.DataFrame,
) -> collections.abc.Iterator[schema.YieldAgreement]:
    """Per crop, the median ratio over countries and the spread around it.

    Median rather than mean, and spread rather than a standard deviation: a handful of countries
    with a near-zero denominator produce ratios in the hundreds, which would move a mean and tell
    us nothing about the product form.
    """
    for crop_name, group in comparison.groupby("crop_name"):
        ratios = sorted(group["ratio"])
        if len(ratios) < MINIMUM_COUNTRIES:
            continue
        yield schema.YieldAgreement(
            crop_name=str(crop_name),
            countries=len(ratios),
            median_ratio=statistics.median(ratios),
            lowest_ratio=ratios[0],
            highest_ratio=ratios[-1],
        )


def get_unpaired_crop_names(
    comparison: pandas.DataFrame, wri_yields: pandas.DataFrame
) -> dict[schema.UnpairedReason, tuple[str, ...]]:
    """WRI's crops the comparison could not reach, grouped by why.

    Grouped rather than listed, because the reasons are not interchangeable and a single list
    invites one explanation to be read over all of them. A group crop cannot be compared until
    someone writes down its member items; a crop outside the canonical taxonomy is simply not
    mapped yet; and one short of MINIMUM_COUNTRIES has a median that would say more about its
    reporting countries than about the crop.
    """
    unpaired = set(wri_yields["crop_name"]) - set(comparison["crop_name"])
    compared = set(comparison["crop_name"])
    mapped = {item.name for item in faostat_production.ItemCode}
    reasons: dict[schema.UnpairedReason, tuple[str, ...]] = {}
    for reason, names in (
        (
            schema.UnpairedReason.SPAM_GROUP,
            unpaired & faostat_production.SPAM_GROUP_CROP_NAMES,
        ),
        (
            schema.UnpairedReason.SPAM_SPLIT,
            unpaired & faostat_production.SPLIT_CROP_NAMES,
        ),
        (schema.UnpairedReason.TOO_FEW_COUNTRIES, (unpaired & mapped) - compared),
        (
            schema.UnpairedReason.UNMAPPED,
            unpaired
            - mapped
            - faostat_production.SPAM_GROUP_CROP_NAMES
            - faostat_production.SPLIT_CROP_NAMES,
        ),
    ):
        if names:
            reasons[reason] = tuple(sorted(names))
    assert sum(map(len, reasons.values())) == len(unpaired), (
        f"{len(unpaired):d} unpaired crops but {sum(map(len, reasons.values())):d} explained"
    )
    return reasons


def get_yield_findings(
    agreements: tuple[schema.YieldAgreement, ...], year: int
) -> list[schema.Finding]:
    """A finding per crop whose two yields do not agree, product-form mismatches first.

    A mismatch is BLOCKING rather than a disagreement: it means the comparison was never
    like-for-like, so the emissions factors built on that yield cannot be read at all until the
    product form is settled.
    """
    findings = []
    for agreement in agreements:
        if agreement.is_product_form_mismatch:
            findings.append(
                schema.Finding(
                    slug=f"yield-product-form-{agreement.crop_name.lower():s}",
                    severity=schema.Severity.BLOCKING,
                    message=(
                        f"{agreement.crop_name:s}: WRI's yield is {agreement.median_ratio:.2f}x "
                        f"FAOSTAT's across {agreement.countries:d} countries in {year:d}. That is "
                        "a different product form, not a disagreement -- a milling or ginning "
                        "yield -- so every emissions factor built on this yield is off by the same "
                        "factor and cannot be compared until the form is settled"
                    ),
                    confidence=schema.Confidence.HIGH,
                    affected_rows=agreement.countries,
                )
            )
        elif agreement.is_beyond_tolerance:
            findings.append(
                schema.Finding(
                    slug=f"yield-disagreement-{agreement.crop_name.lower():s}",
                    severity=schema.Severity.ADVISORY,
                    message=(
                        f"{agreement.crop_name:s}: WRI's yield is {agreement.median_ratio:.3f}x "
                        f"FAOSTAT's across {agreement.countries:d} countries in {year:d}, beyond "
                        f"the {schema.YIELD_RATIO_TOLERANCE:.0%} tolerance. The two are on the same "
                        "product form, so this is a denominator difference and it moves every "
                        "factor for this crop proportionally"
                    ),
                    confidence=schema.Confidence.MEDIUM,
                    affected_rows=agreement.countries,
                )
            )
    return findings
