# GLAD GLC vs CDL row crop comparison

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

The comparison above concerns the jurisdictional-direct leg, where the GLAD-cropland restriction is symmetric: the same per-pixel crop mask gates both the emissions numerator and the crop-area denominator, so a lost pixel drops out of both together.

The statistical leg applies the same GLAD-2020-cropland restriction, but only to the emissions numerator (`get_downscaled_luc_emissions` in `statistical.py`). This is deliberate, not an oversight. Its denominator is MapSPAM crop production — a crop quantity, and therefore zero wherever there is no cropland — so the denominator is already cropland-restricted by construction, to MapSPAM's delineation of cropland rather than GLAD's. Masking the numerator to GLAD-2020 cropland is what brings the two into alignment. Without it, conversion emissions whose destination is non-agricultural (e.g. forest or grassland lost to built-up) would enter the emissions pool and be spread across crops by expansion share — charging, say, a town's expansion to soybeans.

The only residual inconsistency is that GLAD's and MapSPAM's cropland footprints do not perfectly coincide: a pixel MapSPAM credits with crop area but GLAD does not call cropland in 2020 sits in the denominator yet is excluded from the numerator. This is the same dataset-delineation disagreement quantified above for CDL, now between GLAD and MapSPAM. At the ~10 km MapSPAM resolution the two footprints cannot be co-registered any more tightly, and the mismatch is not removable by dropping the mask — that would only add the non-agricultural emissions described above.

See [`analyses/cdl_glad_glc_comparison.ipynb`](../analyses/cdl_glad_glc_comparison.ipynb) for the full analysis.
