# GLAD GLC vs CDL row crop comparison

**Historical.** The restriction this document measures no longer exists: the jurisdictional-direct leg is no longer masked to GLAD-identified cropland, and the emissions core no longer reads the GLAD land-cover series at all. The confusion matrix below is kept as a record of what that cropland class did and did not see, which is the same question the current cropland layer has to answer.

A supplement to [`methodology.md`](methodology.md), assessing whether restricting the jurisdictional-direct leg to GLAD-identified cropland biases the resulting emissions factors.

The methodology calculates emissions for pixels identified by GLAD GLC as cropland, using CDL to allocate among crops. This means that any CDL pixels not identified as cropland by GLAD GLC 2020 are excluded from both emissions and production.

To test whether this exclusion likely biases our EFs, we computed confusion matrices crossing CDL row crop classification against GLAD GLC cropland (pixel value 244) for all 48 CONUS states + DC, comparing CDL 2020 × GLAD GLC 2020.

|                      | GLAD GLC cropland | GLAD GLC non-cropland |          Total |
| -------------------- | ----------------: | --------------------: | -------------: |
| **CDL row crop**     |     99,475,866 ha |         10,248,055 ha | 109,723,921 ha |
| **CDL not row crop** |     37,238,049 ha |        631,957,357 ha | 669,195,406 ha |
| **Total**            |    136,713,915 ha |        642,205,412 ha | 778,919,327 ha |

90.7% of CDL row crop pixels are also identified as crops by GLAD.

The 9.3% of "lost" CDL row crops are concentrated in regions with smaller, more fragmented fields:

| Region                         | GLAD GLC confirmation rate |
| ------------------------------ | -------------------------- |
| Corn Belt (IA, IL, IN, NE, OH) | 91–95%                     |
| Great Plains (ND, SD, KS, MT)  | 91–95%                     |
| Southeast (AL, FL, GA, SC)     | 75–82%                     |
| Northeast (CT, MA, RI, PA)     | 39–77%                     |

These pixels fall into two cases, which we analyzed for 10 key agricultural states, giving what we'd expect to be a representative picture for CONUS overall:

**GLC changed 2000→2020** (24% of excluded pixels)

| GLC 2000 source → GLC 2020 destination | Area (ha) | % of disagreement | Emissions relevance            |
| -------------------------------------- | --------: | ----------------: | ------------------------------ |
| Cropland → non-cropland                | 1,130,028 |               21% | None                           |
| Short veg → non-cropland               |    99,962 |                2% | Grassland conversion emissions |
| Forest → non-cropland                  |    32,392 |              \<1% | Forest conversion emissions    |
| Wetland → non-cropland                 |    16,720 |              \<1% | Grassland/forest               |
| Water/bare/other → non-cropland        |     6,790 |              \<1% | Negligible                     |

**GLC stable 2000=2020** (76% of excluded pixels)

| GLC 2020 class (stable since 2000) | Area (ha) | % of disagreement | Emissions relevance                                                |
| ---------------------------------- | --------: | ----------------: | ------------------------------------------------------------------ |
| Built-up                           | 2,419,875 |               44% | None — farmsteads, grain bins, rural infrastructure. Zero biomass. |
| Short vegetation                   | 1,622,525 |               30% | Would generate grassland→cropland emissions if misclassified.      |
| Forest                             |    77,956 |                1% | Would generate forest emissions if misclassified.                  |
| Wetland                            |    66,092 |                1% | Would generate forest or grassland emissions.                      |
| Water/bare/other                   |     8,429 |              \<1% | Negligible                                                         |

Because lost pixels are excluded from both the emissions numerator and the production denominator, what matters for EF accuracy is whether they are systematically different from the in-scope population. If so, it appears that it's in a way that biases the EF slightly upward rather than downward: only ~3% of lost pixels have a GLC history showing conversion from a non-crop source, compared with the ~6.5% conversion-from-non-crop rate observed among the in-scope CDL row crop population. The dropped pixels are, if anything, biased toward stable, non-converting land.

## The statistical leg

The comparison above concerns the jurisdictional-direct leg, where the GLAD-cropland restriction it measures was symmetric: the same per-pixel crop mask gated both the emissions numerator and the crop-area denominator, so a lost pixel dropped out of both together.

The statistical leg applies no cropland mask to its emissions numerator (`get_downscaled_luc_emissions` in `statistical.py`), and needs none to keep non-agricultural conversions out of it. `emit` charges a conversion's carbon only where a destination layer claims the pixel as cropland or pasture at the assessment year (see [`methodology.md`](methodology.md)). Carbon whose pixel no layer claims is reported on its own as `dropped-emissions`, so it never enters the pool that is spread across crops by expansion share: a town's expansion is not charged to soybeans. The denominator is MapSPAM crop production — a crop quantity, and therefore zero wherever there is no cropland — so it is cropland-restricted by construction, to MapSPAM's delineation of cropland.

The residual inconsistency is that the destination layers and MapSPAM do not delineate cropland identically: a cell MapSPAM credits with crop area, but where no destination layer claims the converted pixels, sits in the denominator with none of their emissions in the numerator. This is the same dataset-delineation disagreement quantified above for CDL, here between the destination layers and MapSPAM. At the ~10 km MapSPAM resolution the two footprints cannot be co-registered any more tightly.

See [`analyses/cdl_glad_glc_comparison.ipynb`](../analyses/cdl_glad_glc_comparison.ipynb) for the full analysis.
