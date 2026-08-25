# Validating the pipeline against external datasets

A supplement to [`methodology.md`](methodology.md) and [`architecture.md`](architecture.md). The methodology says how emissions are quantified and attributed; this describes the tooling in `validation/` that measures the result against datasets produced by other people, and — more importantly — **why it is shaped the way it is**.

It is internal tooling with two jobs: to put measured magnitudes on the methodology gaps we already know about, and to surface ones we do not. It is not auditor-facing and it is not CI.

**It extends [`further_research.md`](further_research.md) rather than forking it.** That document is already a gap register: entries ordered by estimated impact, each with a prose *Potential impact* and *Potential improvement path*. What it lacks is measurement. So a finding's `slug` is one of that document's headings, verbatim, and this tooling's contribution is a magnitude, a breadth and a confidence per entry — turning "potentially substantial in other geographies" into a number that can be ranked against the others. A gap with no entry there gets a new slug and should be proposed back to that document.

______________________________________________________________________

## 1. What it answers

Three questions, each narrowing the next.

1. **Where do we disagree with the outside world?** Every target against every anchor.
2. **Which term carries the disagreement?** Intensity, yield, activity data, detection, extent, allocation, carbon density or discount. §4 answers most of this without touching a raster.
3. **What should we fix first?** Ranked by magnitude and confidence.

These are questions, not components. A stage says *when code runs*; a question says *what is being asked*, and one stage serves several.

Two constraints shape everything below. **Changing the methodology is out of scope** — every probe is report-only. And **the report's exit code never depends on its findings**: it exits 0 whenever it ran, however bad the news, because a report that fails on a bad result cannot be used to characterize bad results. It fails only on unreadable input or a missing join key. The generators under `tools/` are the deliberate exception — a build tool that emits a broken committed artifact exits nonzero, because a bad committed mapping is not a finding, it is a defect.

## 2. Layout, and why it is split this way

```
tools/                          offline generators; the output is the artifact
  build-national-mappings.py    GADM<->World Bank and Orbae<->World Bank maps, self-checking
  build-tiled-countries.py      the 220 countries E1 admits, self-checking
validation/
  schema.py     vocabulary: enums, Finding, code_version, source identity
  targets.py    reads data/targets.json; Measure, Target, Control
  pull.py       anchor retrieval, the data-directory layout, sources.lock.json
  prepare.py    all transforms, eligibility, and the source-specific readers
  report.py     Markdown renderers; reads no files and computes no quantities
  capture.py    runs both pipeline legs and writes efs.parquet    [runs the pipeline]
  __main__.py   the reporting entrypoint
  data/         committed JSONs; .cache/ is gitignored
```

The split is **by what the code is, not by what it computes**.

`tools/` needs geopandas and reads a 93 MiB GeoPackage. Nothing imports it — the hyphenated filenames make that literal — so its cost stays off the reporting path. It runs rarely, and what it produces is a committed artifact rather than console output, because a generator whose result evaporates cannot be read by anything downstream.

`capture.py` is the only module that runs the pipeline itself, which is why it is run by hand rather than reached from `python -m validation`. Keeping it separate is what lets the report finish in seconds.

`report.py` reads only the frames it is handed, and computes nothing. That constraint is load-bearing: it forces every filter, rescale, factorization and cross-walk into `prepare.py`, where each is named and testable, and leaves a report section as a pivot plus a caption.

### `pull` is not `ingest`

`jdluc.ingest` mirrors a registered dataset into the pipeline's own managed storage, normalized and tile-partitioned, so the pipeline can compute over it. `pull` retrieves an anchor exactly as published, to a local cache, pinned by revision and sha256 in a committed lock, so the report can compare against it.

**The lock is committed; the bytes are not.** 10 MiB across the 128 WRI files reproduces from a 66 KiB lock, and the digests prove a fresh checkout got what the lock describes. The same rule decides what else gets committed: anything derivable from a pinned anchor is derived on each run rather than stored beside it. The eligible shortlist is the worked example — 864 pairs recomputed in about a tenth of a second from anchors the lock already pins, where a committed copy would be a second, unpinned record of the same numbers that goes stale the moment the pin moves.

**One artifact does not reproduce from the lock: the capture.** `efs.parquet` and `forest_pools.parquet` are re-keyed extracts of the pipeline's own parquet outputs, so they are neither committed here nor re-fetchable; producing them means re-running `validation.capture`, which takes hours of pipeline. Everything upstream of a capture reproduces in seconds — eligibility, yields, the anchor tables, and `ORBAE_OVER_WRI`, whose two sides are both external. Only the FAOSTAT half of that needs more than the lock: `prepare.read_faostat_production` reads the ingested parquet rather than a second copy of the upstream archive, so a checkout that has never run the ingest fetches it once and caches it beside the pulled anchors, with `pull.record_digest` pinning what it got. Every measure with a term of ours does not, and neither does the conservation bound. A report carrying those numbers is reproducible only alongside the run that produced them, and `code_version` and `source_version` on every row name which run that is.

FAOSTAT **is** an ingested dataset, because it is a plausible pipeline input: national tabular production statistics keyed on jdluc's own `admin_id` and `crop_name`, the global analogue of `usda_nass_quickstats`. WRI, EPA and Orbae stay outside `jdluc/`. Every pipeline module imports `jdluc.datasets`, so putting the yardstick there would let `statistical.py` consume the number it is measured against, with no contract preventing it.

## 3. Choosing what to validate

A target is a **(country, crop) pair**, not a triple with a methodology: the legs available are derivable. `jurisdictional_direct.workflow` asserts `iso_3166 == "USA"`, and all 11 jdLUC crops are a subset of sLUC's 32 — so a US target yields three comparisons, including **sLUC against jdLUC, which needs no external anchor at all**. That internal head-to-head is the cheapest comparison in the design and the only one where a disagreement proves one of our own legs wrong without an outside arbiter.

Eligibility is mechanical, four filters, and re-derived on every run:

|     | Filter                                                                                 | Why                                                                      |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| E1  | the crop is modeled, and the country is one of the 220 in `coverage.md`                | a pair we cannot compute is not a target                                 |
| E2  | at least one anchor, for at least one measure                                          | nothing to compare against, nothing to learn                             |
| E3  | WRI publishes provincial rows, and the key map covers >=50% of the country's land area | the provincial grain is the default                                      |
| E4  | national production >= 100 kt                                                          | a factor that cannot move a companywide number is not worth capture time |

**E2 is deliberately weak.** Requiring a WRI *emissions* anchor would exclude every grassland-dominated pair, since WRI is forest-only — exactly the regime with the least external validation and the largest unanchored pool.

**E3 weights by land area because counting units misleads in both directions.** Brazil matches 87% of its provinces but 100% of its area, the four it misses totaling 12 km²; Algeria matches 96% of its provinces and only 74% of its area, missing two Saharan wilayas of 607,000 km². Counting would pass Algeria and fail Brazil. Area is still a proxy — the weight that matters is cropland — but cropland weighting needs a capture, which needs the target set, which needs E3.

**E1 keeps woody perennials on purpose.** Oil palm, coconut and coffee rank at the top on WRI's deforestation while the pipeline attributes close to zero to them, because GLAD's cropland class excludes perennial *woody* crops by construction — it covers "annual and perennial herbaceous crops", and "perennial woody crops, permanent pastures and shifting cultivation are excluded from the definition" (Potapov et al., 2022, *Nature Food* 3, 19–28, [doi:10.1038/s43016-021-00429-z](https://doi.org/10.1038/s43016-021-00429-z)). The distinction is canopy, not lifespan: sugarcane is perennial and ratooned but herbaceous, and GLAD sees it.

Keeping those pairs eligible is the finding, not a filtering mistake, and it is the single largest thing this tooling has measured: **54 eligible pairs hold 415 Mt of WRI-attributed deforestation between them**, led by Indonesian oil palm at 218 Mt. It needs no capture, because both sides come from anchors.

The list is woody crops only, and that is a narrower thing than "perennial": sugarcane, banana and plantain are all perennial and all herbaceous, so GLAD's definition admits them and they are not part of this gap. An assertion in `prepare.py` names all three, because reading the list as "perennial" rather than "woody" is the mistake it invites.

### What E3 costs, and why most of it cannot be recovered

E3 rests on a key map between GADM's `GID_1` and the World Bank's `ADM1CD_c`, built offline by `tools/build-national-mappings.py` from WRI's own key file and the World Bank admin-1 GeoPackage. Roughly 63% of units match, plus a handful of hand-written overrides. Of the 244 countries in the layer, 53 are fully covered, 87 are partial but still clear E3, and 104 fall below it.

Weighed by the anchor's own figure rather than by country count, the loss is small and concentrated. Against 2,662 Mt of WRI 2020 deforestation over all GFW-tiled countries, the countries E3 excludes hold **45 Mt — 1.7%**. The 87 partial countries keep their pairs and lose provinces at the margin; among the chosen targets the worst are Mozambique at 27% of provincial units unmapped, Malaysia 13%, Mexico 13% and Thailand 12%, while Brazil, Argentina, the USA, Bolivia and the DRC lose none.

**Most of the excluded 45 Mt is not a matching failure and cannot be fixed by a better matcher.** Diagnosing each excluded country by whether its two sources share a vocabulary splits them cleanly:

- **Different administrative levels — 41 Mt across 29 countries, 92% of the loss.** The two sources describe different things, so no name matching can succeed. The Philippines is the largest at 24.8 Mt: the World Bank gives it 17 *regions* where GADM gives 81 *provinces*, which nest inside them — **0 of the 81 names match, even fuzzily**. The Central African Republic (7 regions against 17 prefectures, 8.6 Mt), Sri Lanka (9 against 25, 4.0 Mt) and Malawi (4 against 28, 2.5 Mt) are the same failure at smaller scale.
- **Names partly overlap — 3.7 Mt across 61 countries, 8% of the loss.** Here a better matcher might help, and it is not worth building: the largest is South Sudan at 1.1 Mt, Guyana fuzzy-matches 9 of its 10 units for 0.4 Mt, and Russia — 83 provincial units, almost all unmapped — carries so little deforestation-linked production that its ~68 unmatched province names are not worth overriding.

So the recoverable share of E3's cost is **8% of 1.7%**, and the rest needs a different mechanism rather than a better matcher.

A country in the first group is carried as a **national ratio only** rather than dropped. `CAF MAIZE`, `PHL COCONUT` and `PHL RICE` are in the target set on that basis, off the eligible shortlist entirely. It keeps the anchor comparison available at the grain where it is defined, and records why the provincial grain is not.

Carrying two pairs for one country buys something a single pair cannot. `PHL COCONUT` is a `gap` entry, expected to fail on the woody-perennial detection gap; `PHL RICE` is `ranked`, and rice is herbaceous, so GLAD's cropland class does see it. Holding the country fixed across the two separates "the Philippines is under-detected because its crops are perennial" from "the Philippines is under-detected because it is the Philippines" — the same construction as `IDN MAIZE` sitting beside `IDN OILPALM`. Between them they cover 12.9 of the country's 24.8 Mt.

**The map is deliberately injective, and that is load-bearing rather than hygiene.** A many-to-one map is representable in the stored direction — 81 GADM provinces to 17 World Bank regions is a perfectly good `gadm_id -> world_bank_id` mapping — but `prepare.get_orbae_gadm_ids` inverts it to resolve Orbae's provinces onto GADM keys, and a dict inversion under a many-to-one map silently keeps whichever entry comes last. Eighty of the Philippines' eighty-one provinces would vanish without a word. So admitting many-to-one countries at the provincial grain is not a matter of relaxing a check: it needs the inversion to handle one-to-many, and the comparison to go through the intensity-times-area rollup, since an emissions factor is a ratio and cannot be summed up an administrative hierarchy.

Finally, `area_coverage` weights by land area, which is a proxy for the cropland weighting that would actually matter — so these figures overstate the loss in arid countries, Algeria's missing Sahara being the standing example, and could understate it where the unmapped provinces are the agricultural ones. Correcting that needs a capture, which needs the target set, which needs E3.

### There is no scoring

Ranking the eligible set by a blend of weighted, percentile-ranked criteria does not work here, and the arithmetic is worth stating. One criterion of the four takes only three values, so percentile-ranking it turns a near-boolean into a swing that outweighs its nominal weight: such a blend's top 21 has **90% overlap with sorting on that one flag alone, and 24% with sorting on the anchor's own deforestation figure**. It drops `BRA SOYBEAN`, the largest pair in the set, in favor of pairs under 8 Mt.

The formalism is not merely redundant, it conceals that one binary flag decides everything. So the set is chosen by a person and recorded in `data/targets.json` with a written reason per pair. Every pair can be asked "why are you here?", and the answer is a sentence rather than a score.

A target is never dropped merely because it now agrees. That would bias the set toward disagreement, destroy the controls, and make every run look worse than the last. **Agreement is a result.**

## 4. WRI's factor factorizes, which localizes most disagreements without a raster

```
EF [kgCO2/kg] = intensity [kgCO2/ha] x yield_factor_kg [ha/kg]
```

**The identity is checkable nationally, and only there.** Nationally WRI publishes deforestation emissions, production and the factor, so `EF = LD / production` has three independent terms — `LD` being WRI's column name for the emissions, short for *linearly discounted*, since they carry the GHGP 20-year discount already applied. It holds across 4,442 rows over 42 crops, worst relative error 0.05%, which is rounding in the published precision. Provincially WRI publishes only the factor, so `intensity = EF ÷ yield` has no third term that could contradict it. Nothing provincial can fail this check, and it must not be claimed as one.

What *can* fail provincially is the join. It was measured when the provincial path was built — **54,655 of 55,000 provincial factors (99.37%) convert to an intensity**, once the taxonomy rename below is resolved — and the report does not recompute it: `ORBAE_OVER_WRI` is the only measure the provincial factors reach, and each of its rows carries the province count its correlation was taken over, so a join that degraded would show there as a thinner pair or none at all.

Three consequences follow.

**Intensity, not the per-kg factor, is what a provincial row supports.** Provincial per-kg factors span 0.004 to 8,993 because they are ratios over deforestation-linked production, so a province holding a sliver of it beside a large clearing yields a number meaningless on its own. Intensity removes the *production* denominator, and that is all it removes: because WRI's yield is a whole-jurisdiction figure while its factor's denominator is deforestation-linked production, `EF × yield` is emissions over WRI's deforestation-linked *area* rather than over the crop's. The identity is exact — `intensity × area == LD ÷ deforestation_share` — and it holds to a median 0.01% across the 2,312 pairs carrying deforestation emissions, a yield and a FAOSTAT area.

**That identity is why intensity anchors no comparison of ours.** Carried on our crop area it compares us against WRI's own figure inflated by 1 ÷ `deforestation_share`: 6.55× at the median, 1.02× for `CIV COCO` and 1.23× for `IDN OILP`, but 1.78× for `BRA SOYB` and 18.96× for `USA MAIZ`. Smallest where deforestation dominates the footprint and largest in the temperate pairs, so it is neither a wash nor a constant a tolerance could absorb. Intensity therefore serves the one job where it is not a level against us — `ORBAE_OVER_WRI`, anchor against anchor — and `SLUC_OVER_WRI` compares tonnes against tonnes instead.

**The split is exact and additive in logs**: `log(EF ratio) = log(intensity ratio) + log(yield ratio)`. An all-yield disagreement means a denominator problem; an all-intensity one means carbon density or destination-class treatment. Which it is decides whether raster work is worth running.

**Provincial intensity is dominated by slivers, so nothing reads it as a level.** Derived intensity has a median of 2.2 tCO₂/ha, but 1.68% of rows exceed 1,000 tCO₂/ha — more than any biome holds — topping out at 4.1 million for a province with essentially no production of that crop. They are spread across nearly every crop rather than concentrated, so this is the sliver case and not a data defect. No distribution statistic over provincial intensity is reported for that reason, and `ORBAE_OVER_WRI`, the one measure built on it, is a rank correlation: a sliver holds a single position in an ordering however extreme its value, so it cannot pull a correlation the way it pulls a mean.

### `SLUC_OVER_WRI` is national, and compares WRI's deforestation emissions against ours

An emissions factor is a ratio, so provincial factors cannot be averaged up, and WRI publishes no provincial weights. A rollup on our own weights — `Σ(WRI_intensity × our_area) / Σ(our_production)` — was the obvious way round that, and it is the construction the identity above rules out: it compares us against `LD ÷ deforestation_share` rather than against the emissions themselves.

So the measure is our forest emissions over WRI's published deforestation emissions, nationally, in tonnes, with no yield, area or production term on either side. Both are the same physical quantity in the same units, which is the strongest form this comparison can take, and it is the construction `further_research.md`'s oil-palm decomposition already used.

**Six of seven control priors come back within their quoted precision.** The `inherited` figures in `data/targets.json` are from earlier work, and the first capture reproduces them under this construction: `BRA SOYBEAN` 0.190 against a prior of 0.19, `PRY SOYBEAN` 0.067 against 0.07, `ARG SOYBEAN` 0.423 against 0.43, `ARG MAIZE` 0.441 against 0.46, `COD MAIZE` 0.319 against 0.32 and `AGO MAIZE` 0.571 against 0.58. `CAF MAIZE` is the exception at 0.357 against 0.40, 11% under. That is independent of the algebra above and points the same way, since the priors were computed as emissions against emissions.

Two consequences, one paid and one earned. **The provincial grain is lost for this measure, and WRI's release cannot give it back**: provincially it publishes a factor and no production, so no provincial emissions figure exists to compare against — intensity was the only route to one, and intensity is what carries the share. **The gas scope is now matched**: WRI publishes CO2e nationally and CO2 only provincially, so a provincial construction was pinned to CO2 against our CO2e, and this is CO2e on both sides. Every row is therefore `AS_PUBLISHED` with `coverage_fraction` 1.0 — nothing is joined, so nothing can be dropped.

Note this narrows what E3 is buying. It still gates the provincial factors `ORBAE_OVER_WRI` needs, but a country whose key map is too thin for E3 can now be compared against WRI nationally — the 104 countries E3 excludes hold 45 Mt of WRI deforestation, and none of that is out of reach any more. Whether E3 should stay an eligibility filter or become a reported attribute is an open question, deliberately not settled here.

### The reporting year is the window, not a vintage

WRI publishes `{LD,production,EF}_{2020..2024}`, and the year is not a data vintage: it names where the LSRS 20-year assessment window sits. Reporting year 2020 covers loss years 2001–2020, 2024 covers 2005–2024, and each loss year carries the linear discounting factor for its age — 9.75% for the reporting year itself, falling 0.5 pt/yr to 0.25% nineteen years back, summing to 100% (Fitts et al., 2025a, Table 4, which takes the schedule from the draft LSRG). Those are the weights in `emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT`: WRI's five-year block sums are exactly 5× ours, because ours multiplies a span *total* by a span-average factor where WRI multiplies annual emissions by annual factors.

**The window slides over real new loss data, so the five columns are a series.** Re-discounting a fixed 2001–2020 series would bound `LD_2024 / LD_2020` at 31/39 = 0.795 for any non-negative series; 62% of the 4,423 populated pairs exceed that bound, 864 rise monotonically and hold 49% of global `LD_2024`, and the global total is flat at 2,664 → 2,669 Mt where re-discounting would have to fall. The 19 pairs that decay to exactly zero are the kernel's fingerprint rather than a counterexample: `GEO LENT` and `Z07 OCER` both run 7 : 5 : 3 : 1 : 0, one loss year in 2004 and nothing since. The denominator slides too — `production_Y` is SPAM 2020 production rescaled by the FAOSTAT national ratio for year *Y* (Fitts et al., 2025a, step 5), which reproduces to a median 0.02% (2021), 0.05% (2022) and 0.24% (2023), then 6.2% for 2024, because WRI predicted 2024 with a per-(country, crop) random forest before FAOSTAT published it. It is not national production: over 1,133 pairs above 100 kt it is a median 12% of FAOSTAT's national figure — 52% for `BRA SOYB`, 5% for `USA MAIZ`, 93% for `CIV COCO` — which is the deforestation-linked denominator this section relies on, measured.

So `REFERENCE_YEAR = 2020` matches windows rather than picking a convention, and the spread across the five is the price of leaving the year unpinned, not WRI disagreeing with itself.

**The crop side does not move with the reporting year at all.** Crops enter only through the product allocation factor: per 10 km cell, a crop's physical-area expansion over the expansion of all agricultural land, with pasture from Global Pasture Watch and SPAM's subsistence area in the denominator only (Fitts et al., 2025a, Equation 1 and endnote 21). SPAM exists for 2000, 2005, 2010 and 2020, so the PAF is a step function — 2000→2005 allocates loss years 2001–2005, 2005→2010 allocates 2006–2010, and 2010→2020 allocates 2011 through **2024** (ibid., Table 3 and endnote 22), the same freeze as our `GLAD_TO_MAPSPAM_SPAN` reusing 2010→2020 for two spans. Weighted by the discount, that puts 6.25% / 18.75% / 75.00% of a 2020 factor on the three snapshots, and 0.25% / 8.75% / 91.00% of a 2024 one.

Because the PAF is identical in every column, crop coverage is a property of the SPAM snapshots rather than of the reporting year: all 42 crops appear in all five columns, and only 27 of the 4,654 rows the lock pins flip between zero and non-zero — every one of those is loss timing, not crop coverage.

**Crops missing from a snapshot get two different treatments, and neither is ours.** SPAM 2000 carries 21 crops against 2020 v2's 46, so WRI gap-fills the older years, and the instruction is worth quoting because it is the whole of the method: “to extrapolate or interpolate SPAM crop data for missing years (e.g., 2000), use a linear regression to estimate crop area in each jurisdiction and ensure that the total crop area expected for 2000 remains consistent”, then “proportionally divide the predicted SPAM crop area for disaggregated crop categories to missing data years using linear extrapolation” — preserving each data year's total area and staying “consistent with per-product physical area estimates at the national country level provided by FAOSTAT annually” (Fitts et al., 2025a, pp. 36–37, “Additional details for calculating the PAF for product expansion”). Their worked example is arabica and robusta coffee, one category in SPAM 2000 and two later. Crops that exist only in the newest snapshot are dropped instead, the published set being the 42 present in SPAM 2005, 2010 *and* 2020. So `CITR`, `ONIO`, `RUBB` and `TOMA` have yield factors but no emission factor — there is no `EF_ADM0_RUBB_*.csv` at the revision `pull.WRI_REVISION` pins, meaning **WRI publishes no factor for rubber** — and the factor files keep the 2005-era `ACOF`/`SMIL` codes rather than 2020's `COFF`/`MILL`.

Ours decomposes rather than gap-fills, per pixel and by within-group shares pooled over the later snapshots — described in [`methodology.md`](methodology.md) and bounded in [`further_research.md`](further_research.md), so it is not restated here. Both hold a group's total and redistribute inside it, so the two agree on a group aggregate; what differs is the evidence each split runs on and the grain it runs at. Ours reads the pixel's own composition in the later snapshots. WRI's is a jurisdiction-level regression through the year series, held consistent with FAOSTAT's per-product national area, and the guidebook does not say how that estimate is spatialized. So an individual sibling — banana against plantain, arabica against robusta, coconut against oil palm — need not agree in any given cell, and a sibling-level disagreement can be laid at the 2000 crop split. **What bounds it is weight rather than agreement**: the 2000 snapshot carries the same 6.25% of a 2020 factor's numerator for both methods, and neither treatment touches a crop SPAM 2000 maps individually, which is 28 of the 35 pairs in the target set. The seven it does touch are the `OOIL` constituents oil palm and coconut, and `SWPY`'s yam, the one of the three whose group has no catch-all.

What is worth carrying is where the two stop, because they stop in the same place. Gap-filling only ever bridges a snapshot that is *coarser*; it cannot invent a crop's first appearance. WRI's 42 are the crops SPAM maps in 2005, 2010 *and* 2020, ours are the 32 nameable in 2000 after decomposition, and a crop first mapped in 2020 falls outside both. So rubber has no external anchor rather than merely no model of ours, and the four 2020-only crops need no decision here: `YEAR_TO_UNRECOVERABLE_CROP_NAMES` already excludes them, and E2 could not have offered them an anchor.

**Sources.** Everything above is WRI's own construction rather than reverse-engineering, with two exceptions flagged as measurements in the text — that the window slides over new loss data, and that the denominator follows FAOSTAT — which are inferred from the published files because no document states them.

- **Fitts et al., 2025a** — the method. *Geospatial Methods for Corporate GHG Accounting of Deforestation and Land Occupation*, guidebook, version 1, December 2025, World Resources Institute, [doi:10.46830/wrigb.22.00158](https://doi.org/10.46830/wrigb.22.00158). Table 4 is the discounting schedule, Equation 1 the PAF, Table 3 and endnote 22 which loss years each SPAM snapshot allocates, step 5 the production denominator, and the note under Equation 1 the gap-filling of older SPAM releases.
- **Fitts et al., 2025b** — the numbers. *Statistical Land Use Change Emissions from Deforestation and Land Occupation for 42 Agricultural Crop Categories*, technical note, World Resources Institute and Quantis, published as [`wri/GCSC`](https://github.com/wri/GCSC). `pull.WRI_REVISION` pins it by commit and `sources.lock.json` by digest, so every count in this section is reproducible from the lock.
- **The inputs it composes** are cited in the guidebook's own reference list rather than repeated here — tree cover loss (Hansen et al., 2013), the GFW forest carbon flux model (Harris et al., 2021, revised in Gibbs et al., 2025), 1 km loss drivers (Sims et al., 2025), SPAM 2020 v2 (IFPRI, 2024) and Global Pasture Watch (Parente et al., 2024, [doi:10.1038/s41597-024-04139-6](https://doi.org/10.1038/s41597-024-04139-6)) — the last because the pasture term in WRI's allocation denominator has no counterpart in ours.

## 5. What the report checks, in order

**The conservation bound comes first, ahead of every anchor.** The sum of forest emissions across crops must not exceed the country's forest-conversion pool — that pool being forest conversion over all 2020 cropland with no crop share applied, so it bounds what any allocation can hand out. It earns first place because it needs no anchor, no baseline and no target selection, and because it is *decisive* where an anchor comparison is merely suggestive: a country over 100% has an attribution bug, and every anchor comparison for it is moot until that is fixed.

**The pool and the attribution must come from the same run.** A pool from one code version compared against a capture from another compares a numerator and a denominator computed by different code, which is enough to make the result indicative rather than decisive.

The pipeline emits no pool column, so `validation.capture` derives it: it reads back the cached per-tile emissions layer the capture attributed from — `statistical.get_downscaled_luc_emissions` is keyed on `(skip_glad_crop_filter, tile_id)` alone, so this is the same bands rather than a parallel derivation — clips it to the national geometry, and sums the discounted forest bands with no crop share applied. The discount weights and the pixel areas come from `emit` rather than being restated, so only the assembly lives in two places.

What that buys is provenance, not merely a number: the pool is computed in the same process, at the same commit, as the capture it is written beside, and carries the same `code_version`. It is therefore freezable. Note this is a *same-run* guarantee rather than a same-function one — the pool and the attribution are computed by different code paths over identical inputs, so a change to how emissions are allocated moves one without the other, which is the point, while a change to the underlying layer moves both, which is also the point.

The pool is re-derived only for the countries a run recomputed, for the same reason: carrying one forward beside freshly captured emissions would reintroduce the mismatch this is guarding against.

**Findings come before tables**, sorted by severity then magnitude, because a table invites the reader to draw a conclusion the findings may have already disqualified. **Coverage comes last but is not optional**: a report that silently omits the targets no anchor covered reads as though it checked them, and an empty cell is indistinguishable from agreement unless something says so.

Severity is deliberately independent of size. `BLOCKING` means a comparison could not be made and figures depending on it are meaningless; `DEFECT` means the run contradicts itself; `ADVISORY` means worth knowing. Magnitude, breadth and confidence are carried separately, because severity gives no ordering — it does not distinguish a gap worth tens of megatonnes across nine countries from one worth a fraction of that in a single country.

## 6. Controls, and what makes one trustworthy

Controls are rows where a *stable* relationship is expected, so a change means something broke rather than something was learned. Without them, "biased on palm and peat" and "biased everywhere" look identical.

**`Orbae / WRI` contains no term of ours.** If it moves, an anchor was revised — distinguishable from a pipeline change, which moves `sLUC/WRI` and `sLUC/Orbae` together while leaving `Orbae/WRI` still. That is what stops a single anchor carrying the whole guard, and it is the only measure that can be frozen before the pipeline has ever run, since both sides are external.

Four properties of a control are worth stating because each was arrived at by getting it wrong first.

**A control fires against `baseline`, never against `inherited`.** An inherited figure is a prior number used once, as a sanity check when the baseline is frozen. Tolerance means "how much movement deserves a flag", not a noise band — the pipeline is deterministic, so movement comes only from a code or data change.

**A rank correlation is not tolerance-checked as though it were a ratio.** Orbae's export is permanently three years off ours, and level cannot cross that offset while rank very nearly can: WRI's own factors span 1.42× at the median across the same three years while its provincial rank holds at +0.961. So those controls hold an ordering, and their tolerance is absolute, in correlation units. A relative band would make the weakest-agreeing control the twitchiest — at ρ +0.443 a relative 10% fires on a move of 0.044 where +0.886 tolerates 0.089 — which is backwards.

**A baseline is meaningless without the anchors it was frozen against.** Each carries the identity of every external anchor behind it. If they differ from what a run read, the control reports a *stale baseline* rather than a moved one: reading the movement would book an anchor revision as a change in our own pipeline, which is precisely the confusion `Orbae/WRI` exists to prevent.

**A control cannot rest on a row its own anchor contradicts.** Where WRI attributes more deforestation-linked area to a crop than FAOSTAT reports harvested for it, the pair cannot anchor a comparison whichever side is wrong. This is reported per target rather than only as a release-wide count, because a count of pairs out of thousands reads as a rounding error right up until one of them is the only armed guard on its row.

**A row whose mechanism is known to be broken cannot guard anything.** Oil palm is a target and a register row, never a control: the annuals-only gap is active in exactly that crop, so its apparent agreement may well be coincidental.

## 7. Provenance travels in the row

Timestamps are ambiguous across concurrent branches, and a sidecar goes stale against the artifact it describes. So every produced row carries `code_version` — the commit that produced it, with a digest of any uncommitted result-bearing change — and `source_version`, naming every external anchor behind the figure.

This is needed because a capture scoped to some countries keeps rows for the ones it did not recompute, so an artifact's modification time says nothing about whether a country's rows are current. A `code_version` column makes such a row self-identifying, and a table spanning more than one value raises a `DEFECT`. The check runs at read time, so it holds however the artifact was assembled.

**What `code_version` does not cover.** `storage.cache_to_parquet` keys on `(module, qualname, version=, args)` and never on file contents, so it identifies the code that ran rather than the code every cached layer was built from. Clearing our artifact cannot clear jdluc's cache; that needs a `version=` bump on the changed function and on every cached function downstream, or a different `SCRATCH_ROOT`. Read those versions from source rather than from notes — they move, and a module with no cache version of its own is invisible to the key entirely.

## 8. Limits on what this can conclude

- **Grassland and peat can be sized but not confirmed.** No external anchor exists. WRI is forest-only, Orbae's grassland carbon is biome-invariant and its peat inverted, and SoilGrids, Harris and Huang *are* our carbon densities. The only non-circular route is transcription work: Spawn et al. (2019) and IPCC for biome grassland carbon, and the IPCC 2013 Wetlands Supplement for drained organic soils.
- **WRI cannot arbitrate allocation.** It shares this leg's MapSPAM expansion-share family, so an agreement may reflect shared method rather than shared answer. It stays independent on the forest *pool* and on carbon density, which is what the controls rest on. Genuinely independent evidence about allocation comes from three places only: Orbae, the US sLUC-versus-jdLUC head-to-head, and the conservation bound, which needs no anchor at all.
- **Where anchors disagree with each other more than they disagree with us, that is an anchor-quality row rather than a pipeline gap.** Brazil is why magnitude is measured against the *nearest* anchor: WRI runs about 5× high there while Orbae sits near our value, so measuring to the mean would book someone else's disagreement as our gap. Orbae and WRI barely agree on shape for the two crops that matter most — cross-country rank +0.018 for oil palm and +0.135 for soy — so for those neither corroborates the other.
- **The target set is partly WRI-shaped.** Materiality is read off WRI's deforestation figure, so a regime WRI is blind to — grassland conversion above all — is under-represented by construction.
- **Never size a share-denominator effect by expansion weighting.** Weighting by expansion predicted −1.4% for Indonesia against an actual −91%, three times running. The pathological cells carry near-zero *net* expansion with large *gross* churn, so they are low-expansion by construction and an expansion-weighted mean suppresses exactly what matters. Weight by the crop's area or by emissions.

## 9. Anchor quirks that have bitten

Recorded because each one silently produces a plausible wrong answer rather than an error.

- **WRI uses two MapSPAM taxonomies within one release.** The emissions-factor files use Crop2005's 42 codes, the yield factors Crop2020's 46. Only two codes differ, and both are pure renames — but joining without resolving them loses 100% of those two crops' rows.
- **WRI's yields are FAOSTAT-derived, not MapSPAM-derived.** They agree with FAOSTAT across all 29 comparable crops, medians 0.947–1.024, including the two product-form risks that would have shown up several-fold: oil palm at 0.993 over 44 countries and seed cotton at 1.017 over 86. MapSPAM runs about 7% under FAOSTAT, so our MapSPAM-based production will show a systematic ~7% gap against WRI's yields that is a **source difference, not a methodology gap**.
- **WRI's five columns are a reporting-year series, not five vintages of one quantity.** The year names where the LSRS 20-year window sits, so each column drops a loss year, adds a real new one, and rescales the denominator to that year's production — §4 has the measurements that rule out a re-discounted fixed series. Factors span 1.42× at the median and 2.90× at the upper decile across the five, and 89% of pairs exceed a ±10% tolerance on that spread alone. That is the ceiling on what an *unpinned* comparison against WRI can resolve, and it is why the reporting year is pinned rather than left implicit. Reading the spread as WRI's own inconsistency overstates it: most of it is real temporal signal.
- **Orbae publishes two commodities on a processed basis** — palm as crude palm oil, sugarcane as cane sugar — so its factors must be divided by the conversion factor the export carries, or they read 3.9× and 8.9× high.
- **Orbae uses different code lengths per grain** — alpha-2 at country level, alpha-3-prefixed at provincial. Reading both as alpha-2 mis-filed 162 rows across 7 countries and dropped 63; there is now a two-way name/ISO injectivity check.
- **Emissions attributed to no crop are discarded.** Shares sum to at most one and the shortfall is charged to nobody — 12.6% of conversion and 13.9% of occupation globally, rising to 23.6% of Indonesia's occupation. Summing the factor table therefore does not give a jurisdiction's total, and a national-totals table without that row misleads.

## 10. Unverified assumptions

- **The temporal bases — settled.** WRI's half is answered, and in our favor: a numerator discounted over the 20 loss years ending in the reporting year, over a denominator that is production in the reporting year alone (Fitts et al., 2025a, Equation 4 and step 5). Ours is the same shape, so `SLUC_OVER_WRI` needs no matched-window correction, and comparing deforestation emissions against ours carries no extent term that would need one. The bases agree inside our own artifact too: `crop_hectares` carries the same linear discounting over 2000–2020 that production and emissions do, so every per-hectare figure divides quantities measured over one window. Measured on Ireland and Indonesia, that basis puts yields within ~4% of FAO's for the same country and crop, against ~25% for a 2020-snapshot denominator.
- **Provincial intensity is shape-only, and its shape is contaminated.** `ORBAE_OVER_WRI` compares two anchors province by province, so `LD ÷ deforestation_share` cancels only where the share is constant across a country's provinces, which it is not. A rank correlation survives that better than a level does, which is why those rows are rank rows — but the residual is unmeasured, and it is the reason a strengthening or weakening of an Orbae/WRI correlation cannot yet be read as a method change.
- **WRI's 2000 gap-fill carries three constraints with no stated precedence.** The guidebook sets a jurisdiction-level linear regression per crop, preservation of that data year's total crop area, and proportional division within a disaggregated crop category, all held consistent with FAOSTAT's per-product national physical area (Fitts et al., 2025a, pp. 36–37). Which binds when they conflict is not stated, so §4's reading — that a group's total survives the gap-fill, which is why the two methods agree on a group aggregate and can differ only between siblings — is an inference from the quoted instruction rather than something the guidebook asserts. It is silent too on how a jurisdiction-grain estimate is spatialized to SPAM's 10 km cells, the larger gap of the two: it leaves a sibling-level comparison undefined at the pixel rather than merely uncertain. Both are bounded by the 6.25% of a 2020 factor's numerator the 2000 snapshot carries, and reach only the seven target pairs whose crop is a 2000 group constituent.
- **Orbae's grassland split** — `natgrass` against `pasture` — versus our single derived remainder. Moot while only its forest term is used, and undefined otherwise.
- **The international cropland-extent factor**, which does not exist. CDL is US-only, so the ×1.10 CDL∩GLAD correction has no international counterpart and every international total carries an unquantified factor.
