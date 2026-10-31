# Crop-Independent Emissions (CIE)

`emit` writes a second set of bands, prefixed `cie-`, beside its destination-gated output (see [`data.md`](data.md#2-per-pixel-emissions-zarr)). They describe the same land conversions and carbon pools, but stop **before** anything that depends on what the land became:

- **No destination gate.** A pixel is charged as soon as it lost a source class (forest, natural grassland or pasture) inside the 20-year window, whether or not a destination dataset claims it in the assessment year.
- **No destination-specific factor.** Mineral soil is the full stock at risk. The cropland loss fraction is not applied.
- **No discount.** Each pool is a single undiscounted value. The span split and the 20-year linear discount are left to the consumer.

This makes the `cie-` bands a common input from which any destination- or crop-specific accounting can be derived: a consumer that wants a different soil factor per crop, a different discount, or the emissions a conversion would carry under a destination other than the one observed, can compute it from these bands without rerunning `emit`. The existing bands are one such derivation (see [Deriving the existing bands](#deriving-the-existing-bands-from-cie) below). The code is `emit.get_crop_independent_emissions`.

## Side by side

| quantity                | existing band(s)                                                                                                                                                                                                      | CIE band                                                                                                                                                                                | difference                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| what happened           | `conversion`: source × destination, 0–5 (`emit.Conversion`)                                                                                                                                                           | `cie-conversion-source`: 0 none · 1 forest · 2 natural grassland · 3 pasture (`emit.ConversionSource`)                                                                                  | source only; set whether or not a destination resolved                                 |
| when                    | `conversion-year`                                                                                                                                                                                                     | `cie-conversion-year`                                                                                                                                                                   | copy: the year is already 0 wherever there is no source                                |
| vegetation              | `vegetation-emissions:tco2e-per-ha:{span}` ×4                                                                                                                                                                         | `cie-vegetation-emissions-undiscounted:tco2e-per-ha`                                                                                                                                    | one band rather than one per span; not gated on destination                            |
| soil                    | `soil-emissions:tco2e-per-ha:{span}` ×4, one band holding either term: on peat, the 621 tCO2e/ha pulse; off peat, SOC × 44/12 × the cropland loss fraction for a cropland destination, or 0 for a pasture destination | two bands: `cie-mineral-soil-carbon-at-risk:tco2e-per-ha` (SOC × 44/12 off peat, 0 on peat) and `cie-peat-transformation-emissions-undiscounted:tco2e-per-ha` (621 on peat, 0 off peat) | mineral and peat kept apart; no loss fraction on mineral; neither gated on destination |
| peat occupation         | `cropland-peatland-occupation:tco2e-per-ha`, `pastureland-peatland-occupation:tco2e-per-ha`: 37.3 on peat under that destination                                                                                      | `cie-peat-occupation-emissions:tco2e-per-ha-per-year`: 37.3 on all peat                                                                                                                 | the potential, whatever the destination and whether or not the pixel was converted     |
| discounted total        | `emissions-per-hectare:tco2e-per-ha`                                                                                                                                                                                  | —                                                                                                                                                                                       | the discount is applied downstream                                                     |
| unclaimed source carbon | `dropped-emissions:tco2e-per-ha`                                                                                                                                                                                      | —                                                                                                                                                                                       | these pixels are carried in the CIE pools themselves                                   |
| destination evidence    | `destination-dataset`: bitmask, 1 Descals oil palm · 2 GACED30 cropland · 4 GPW cultivated grassland                                                                                                                  | `cie-destination-dataset`                                                                                                                                                               | copy                                                                                   |
| climate zone            | — (input only)                                                                                                                                                                                                        | `cie-climate-zone`: `ipcc_climate_zones.Zone`, 1–10                                                                                                                                     | new: needed to look up destination-specific soil factors                               |
| pixel area              | `hectares-per-pixel:ha`                                                                                                                                                                                               | `cie-hectares-per-pixel:ha`                                                                                                                                                             | copy                                                                                   |

`{span}` is one of `2000-2005`, `2005-2010`, `2010-2015` and `2015-2020`. A conversion falls in span `(a, b)` when `a < year ≤ b`. The copied bands are duplicated so that the `cie-` set is complete on its own.

## Deriving the existing bands from CIE

Every existing band can be computed from the `cie-` bands. `emit.derive_from_cie` does so, though `jurisdictional_direct` and `statistical` still read the existing bands directly. The only extra inputs are two lookup tables already in `emit`: the soil loss fraction and the discount weights. Below, `ConversionSource` and `Conversion` refer to the band values in the table above.

**Destination.** Cropland outranks pasture:

- `to_cropland = cie-destination-dataset & (1 | 2)`
- `to_pasture = ¬to_cropland ∧ cie-destination-dataset & 4`

**Conversion.** It fired where `cie-conversion-source ≠ NONE ∧ (to_cropland ∨ to_pasture)`, except pasture to pasture, which is not a conversion. The `conversion` value is the pair of the two: forest to cropland, forest to pasture, natural grassland to cropland, natural grassland to pasture, and pasture to cropland. A pasture-to-pasture pixel appears in neither the per-span pools nor `dropped-emissions`.

**Per-span pools.** Each pool is set only where a conversion fired and `cie-conversion-year` is in the span:

- `vegetation-emissions:{span}` = `cie-vegetation-emissions-undiscounted`
- `soil-emissions:{span}` = `cie-peat-transformation-emissions-undiscounted` + `f(cie-climate-zone)` × `cie-mineral-soil-carbon-at-risk` for a cropland destination
- `soil-emissions:{span}` = `cie-peat-transformation-emissions-undiscounted` for a pasture destination
- the per-span total, `e(span)`, is vegetation + soil (not itself a band)

Here `f` is the cropland soil loss fraction, 1 − retention (IPCC 2019, Vol 4, Table 5.5; `emit.CLIMATE_ZONE_TO_SOC_RETENTION_FRACTION`).

**Occupation.** Each band is gated on its destination only:

- `cropland-peatland-occupation` = `cie-peat-occupation-emissions` where `to_cropland`
- `pastureland-peatland-occupation` = `cie-peat-occupation-emissions` where `to_pasture`

**Discounted values.** Both use the span weights `w` = 0.0125, 0.0375, 0.0625 and 0.0875, which sum to 0.2 (`emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT`):

- `emissions-per-hectare` = Σ<sub>span</sub> `w` × `e(span)` + both occupation bands
- `dropped-emissions` = Σ<sub>span</sub> `w` × `cie-vegetation-emissions-undiscounted`, where there is a source, the year is in the span, and neither `to_cropland` nor `to_pasture` holds

The vegetation, soil and year identities are unit-tested in `jdluc/__tests__/emit_test.py` (`test_cie_*`), and `test_derive_from_cie_reproduces_every_band_it_replaces` checks every existing band against `derive_from_cie` on a grid covering each event, destination and soil case. `tools/compare-emit-layers.py` checks the vegetation and year identities on a real tile, and the largest difference between each existing band and its derived counterpart. It also totals the old and new pools by source, and by whether a conversion fired, which shows how much carbon the CIE bands carry that the existing bands leave uncharged.

## The crop-class layer

`jdluc/derive.py` is a second derivation, without a destination gate: for every pixel and each crop class (annual, perennial, paddy rice, pasture) it answers *what if this crop class occupied this pixel?* It applies the three things CIE leaves out:

- **Mineral soil.** `cie-mineral-soil-carbon-at-risk` × (1 − F_LU), with F_LU by crop class and climate zone (IPCC 2019 Refinement Vol 4 Table 5.5 for cropland, Table 6.2 for grassland), capped at 1. Annual crops take `emit`'s cropland retention fractions, so the two agree.
- **Peat occupation.** An annual emission factor by crop class and peat climate regime (IPCC 2013 Wetlands Supplement, Ch 2), in place of the single `cie-peat-occupation-emissions` value.
- **Discounting.** A weight by conversion year over the 20-year window: `linear` (the year-by-year form of `emit`'s span weights), `equal`, or `five-year-spans` (`emit`'s own).

Its fields carry a `crop_class` dimension where they depend on a factor and a `discounting` dimension where they are discounted. Nothing is stored: a *product* names the fields, crop classes and discountings to write, and `export.write_tile` writes them per tile from `emit`'s cached output, computing only those.

With five-year-span discounting, annual crops reproduce `emit`'s cropland bands, and pasture its pasture bands, wherever the destination matches, except for peat occupation; the tests in `jdluc/__tests__/derive_test.py` check both.

## Encoding

- Every band is float32 with NaN as no-data, including the enum and bitmask bands (`geo.unify_dtype_and_no_data`).
- Bands share the tile's 30 m grid. The units follow the band name after the first `:`.
- Adding the `cie-` bands takes each tile from 16 bands to 25. The `emit` cache version is 5.

## Not yet covered

- **Assessment year.** It is fixed at 2020 (`emit.ASSESSMENT_YEAR`), which sets the lookback window and the destination predicates.
- **Continent.** Some destination-specific factors vary by continent, which no ingested dataset provides yet.
- **Land cover at the start of the window.** There is no band for it. A forest bit would need a year-2000 tree-cover extent, which is not currently ingested.
