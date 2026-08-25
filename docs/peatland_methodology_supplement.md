# Peatland emissions: a simplified model for GHGP/LSRS reporting

## Introduction

Drained peatlands are among the largest land-based sources of greenhouse gas emissions, but are challenging to model within existing corporate and product-level accounting standards — peat emissions do not fit neatly into either "land use change" or "land management" categories within the existing frameworks. The recently released GHGP Land Sector and Removals Standard (LSRS), for all its detailed guidance on other topics, provides no peatland-specific methodology.

EF providers have addressed this problem in various ways. Some treat peat oxidation as pure land management (LM) emission: flat annual emission factors are applied to every hectare-year of crop occupation. This is what IPCC recommends for Tier 1 national accounting. But this approach fails to reflect the time-dependent nature of peat carbon loss, which is a particularly unsatisfying outcome for the corporate use case because it dramatically under-incentivizes care around new peatland drainage, while overly penalizing occupation of areas drained multiple decades in the past.

Other providers instead fold peat drainage into their land use change (LUC) frameworks. This is better, but it means they attribute no peatland emissions to land drained more than 20 years ago, which is contrary to physical reality, and results in significantly undercounting overall emissions on agricultural land.

At least one provider has built a hybrid approach, where peatland drained within the last 20 years has emissions treated as LUC, while older peatland is treated as LM. This seems a promising variation, but without further refinement, it results in attributing _higher_ emissions to long-drained peatland than more recently drained areas under the GHGP discounting framework, a clearly erroneous outcome.

This document presents an alternative two-part model for estimating peatland drainage emissions within the GHGP/LSRS framework as a _combination_ of an initial pulse of land use change emissions for the 20 years after conversion, followed by steady state land management emissions that persist across decades. This can be seen as an evolution of the current state-of-the-art hybrid approach. However *both* types of emissions are attributed to all peatlands, with their relative importance shifting as time passes from initial drainage.

The resulting model is grounded in the physical evidence of how peat carbon behaves after drainage, while integrating directly into the GHGP accounting framework.

## What happens when peatlands are drained

### Peat drainage releases large volumes of GHGs through multiple pathways

Peatlands accumulate organic matter over millennia under waterlogged, anaerobic conditions that suppress decomposition. The result is deep deposits of partially decomposed plant material — peat — with very high carbon density. Although peatlands cover less than 3% of the Earth's land surface, they store approximately 600 Gt of carbon, roughly one-third of the global soil carbon pool and twice the carbon in all the world's forest biomass (Page et al., 2011a; Yu et al., 2010).

This carbon store is maintained by high water levels. When the water table is at or near the surface, oxygen cannot penetrate the peat, and decomposition proceeds slowly — limited to anaerobic pathways that produce methane but preserve the bulk of the organic matter. When peatland is converted to agricultural usage, it's systematically drained, setting off a number of GHG release pathways:

- **Gaseous CO₂, CH₄, and N₂O from peat oxidation** is the dominant effect: lowering the water table exposes the upper peat profile to oxygen, activating aerobic microbial decomposition and converting stored carbon in the drained zone above the water table to CO₂ in the air. Nitrogen locked in peat organic matter is likewise broken down and made available for nitrification and denitrification.
- **Dissolved organic carbon exported via drainage water** is the second major release pathway. Lowering peatland water tables is not a one-time event: drainage is maintained through a complex series of ditches. As rainwater repeatedly flows through the peat matrix, it flushes carbon rich organic compounds into the water system. The majority subsequently decays to CO₂ in downstream waterways. Evans et al., 2016 estimated that dissolved carbon contributes up to 25% of total peatland carbon fluxes.
- **CH₄ emissions from drainage ditches** are a third source. While peatland drainage reduces CH₄ emissions from the peat surface (by eliminating anaerobic conditions), the ditch network creates new anaerobic zones that emit CH₄ at significant rates (IPCC, 2014).
- **Finally, peatland fires** are an episodic but catastrophic pathway, particularly in tropical peatlands. Drained peat is highly flammable, and fire events can release more carbon in days than annual oxidative losses produce in years. Peatland fires are excluded from the current methodology, and are an area for future improvement.

### Complex carbon structure leads to extended decay

Peat organic matter is not homogeneous. It consists of a complex mix of organic components with widely varying susceptibilities to decomposition. Clymo (1984, 1992) first demonstrated that although approximately 90% of original plant biomass is lost in the upper peat layer within 100 years after drainage, selective decay continues in deeper layers even thereafter as more resistant compounds persist.

Bader et al. (2018) showed that the chemical composition of peat shifts progressively as drainage proceeds: the relative abundance of labile compounds (polysaccharides, simple carbohydrates) decreases while recalcitrant compounds (lignin, polyphenols) become proportionally more abundant. This selective depletion of labile carbon is the primary physical mechanism that produces declining emission rates over time — as the most easily decomposed components are consumed, the remaining peat becomes increasingly resistant to further oxidation.

More recently, McCalmont et al. (2021) carefully measured net ecosystem CO₂ exchange at two oil palm plantations of different ages on tropical peat in Malaysian Borneo, reconstructing a 12-year time series. They found that mean annual net emission for the newer plantation (137.8 Mg CO₂ ha⁻¹ yr⁻¹) was an order of magnitude higher than at the mature plantation (17.5 Mg CO₂ ha⁻¹ yr⁻¹), confirming that emissions were front-loaded, but that some emissions persist after long periods.

## Estimating emissions values

Although the scientific literature is clear on the general pathways and time-series for peatland emissions, estimates for the actual emissions value vary widely. One reason is different measurement approaches and different carbon boundaries. The subsequent sections provide a short summary, but the International Council on Clean Transportation has a much more detailed review in an excellent, if slightly dated, whitepaper (Page et al., 2011b).

### Measurement approaches

Two principal methods are used to quantify carbon losses from drained peatlands, and they systematically produce different estimates.

**Flux measurements** (eddy covariance towers and closed chambers) directly measure gaseous exchange between the peat surface and the atmosphere. Eddy covariance provides continuous, landscape-scale data; chambers provide spatially targeted but temporally sparse measurements. Both include root respiration (not a peat carbon loss, must be somehow estimated and subtracted out) alongside the peat decomposition signal and exclude waterborne carbon losses.

**Subsidence monitoring** measures the physical lowering of the peat surface over time and converts it to carbon loss using measured bulk density and carbon content. It integrates oxidative decomposition and dissolved organic carbon export, but can be confounded by physical compaction which lowers the peat surface without exporting CO₂, especially in the first years after drainage.

The result of the different boundaries from the two basic measurement approaches are results that vary significantly based on local circumstances: on established plantations where compaction has stabilized and root respiration is significant, subsidence-based estimates tend to exceed flux measurements because they capture DOC that flux instruments miss; on bare or recently drained peat where compaction is ongoing and roots are absent, this ordering can reverse (Mos et al., 2023). It is possible to try to reconcile measurements, but it inevitably requires some inference or modeling.

### IPCC Tier 1 emission factors

The IPCC 2013 Wetlands Supplement made an attempt to consolidate all the best evidence on peatland emissions to that point, and provided Tier 1 default emission factors for drained organic soils, stratified by climate zone and post-drainage land use.

The table below shows these emissions factors.

| Climate zone  | Land use | CO₂ oxidation | Dissolved carbon | CH₄ | N₂O | **Total** |
| ------------- | -------- | ------------- | ---------------- | --- | --- | --------- |
| **Tropical**  | Acacia   | 73.3          | 3.0              | 1.3 | 1.0 | **78.6**  |
|               | Cropland | 51.3          | 3.0              | 1.4 | 2.1 | **57.9**  |
|               | Oil palm | 40.3          | 3.0              | 1.2 | 0.5 | **45.0**  |
| **Temperate** | Cropland | 29.0          | 1.1              | 1.6 | 5.6 | **37.3**  |
|               | Pasture  | 22.4          | 1.1              | 2.0 | 3.5 | **29.0**  |
| **Boreal**    | Cropland | 29.0          | 0.4              | 1.6 | 5.6 | **36.6**  |
|               | Pasture  | 20.9          | 0.4              | 1.6 | 4.1 | **27.0**  |

_All values in t CO₂-eq ha⁻¹ yr⁻¹, converted from t C using × 3.667, and using AR6 GWP values_

Critically, these IPCC Tier 1 factors are based on calibration data from very differently aged plantations. The tropical plantations are under 10 years at median. In contrast, the temperate and boreal peatlands are decades to centuries old. This approach may make sense for national inventories, since it reflects a real difference in the typical ages of peatland drainage in those regions, but it must be parsed out for time-dependent models.

The appendix below shows a review of the studies used to generate the IPCC values, and the time periods of the sites used for each emissions factor.

## Modeling time-dependence mathematically

Two studies have attempted to model peatland emission rate declines over multi-decade timescales based on the various empirical measurements.

**Swails et al. (2022)** used the DNDC process-based biogeochemical model, validated against field measurements from oil palm plantations in Central Kalimantan, to simulate 30 years of peat decomposition. They used first-order kinetics to describe carbon flows in the system, i.e. modeling decay speeds as proportional to the reserves of organic matter in selected carbon pools, taking into account the influence of factors such as temperature, humidity, and soil texture. They partitioned soil organic matter into four pools for this modeling exercise: microbial biomass, labile humus, litter, and passive humus, each with its own decay rate constant. That is, DNDC modeled CO₂ emission at any time *t* as:

```
E_DNDC(t) = Σᵢ Aᵢ × exp(-kᵢ × t)    for i = each of 4 pools
```

**Qiu et al. (2021)** used the ORCHIDEE-PEAT land surface model to simulate historical carbon emissions from cultivated northern peatlands. ORCHIDEE also models soil carbon decomposition based on first-order kinetics and turnover times, however, using three carbon pools instead of four (active, slow, passive).

## A simplified model for GHGP reporting

Building on the robust scientific literature, but simplifying from the more complex models in Swails et al. (2022) and Qiu et al. (2021), our approach decomposes peatland drainage emissions into two components that map onto the LSRS reporting categories:

**Land use change (LUC):** A transient pulse of excess emissions above steady state, driven by rapid oxidation of labile carbon in the years following drainage. This is a rapid carbon stock change attributable to the conversion event. It is linearly discounted over the LSRS 20-year assessment period, approximating the initial decay and release period.

**Land management (LM):** The ongoing baseline rate of recalcitrant peat oxidation, plus non-CO₂ emissions (ditch CH₄, soil N₂O), that persists every year drainage is maintained. This is reported annually, indefinitely, as a land management emission.

Set up in this way, the model requires just two parameters: total land use change emissions, and annual land management emissions. We choose values of these two parameters for each climate zone and current land use, following the IPCC approach. From there, the shape of the simplified emissions decay profile is set by the LSRS linear discounting rule.

To set the parameters, we start by building an average reference curve for time-based emissions from the Qiu et al. (2021) and Swails et al. (2022) process model outputs and the IPCC emissions factors, taking into account the age of the sites on which the IPCC values were based. We then set the two GHGP parameters (P_LUC, E_LM) to approximate this reference curve.

### Reference curve

#### CO₂ from peat oxidation

We start by fitting CO₂ emissions from each of the two process models with a double exponential plus constant:

```
E_CO₂(t) = A_fast × exp(-k_fast × t) + A_slow × exp(-k_slow × t) + C
```

This equation approximates the three pool model from Qiu et al. (2021) (active, slow, passive). We fit the double exponential plus constant to Figure S11B from the paper, which plots their modeled emissions decay curve for peatlands converted in 1900. We then repeat this process for Fig. 6 from Swails et al. (2022), which shows modeled CO₂ for the first 30 years of tropical palm plantations.

Notably, the Swails et al. modeled tropical emissions curve is *lower* than the Qiu et al. temperate emissions curve at all years. This is surprising: there is a theoretical basis to suspect that emissions rates from peat should be higher in tropical regions than temperate and boreal regions — decay is highly temperature dependent, with most models suggesting organic compounds should degrade to CO₂ at least 2x faster for every 10 degrees of average temperature increase. The reversal of the expected relationship in the Qiu and Swails studies is partially explained by the fact that the Qiu curve represents gross peat decomposition, while the Swails curve represents net emissions (after vegetation carbon offsets). But this provides only a partial explanation.

The IPCC reference values for CO₂ emissions show a similar pattern. Although the absolute IPCC factors are higher for tropical regions, the difference is no larger than the gap that would be expected from age differences alone, if the general shape of the Swails and Qiu curves is correct. In other words, the IPCC data points also fail to show higher tropical emissions, once corrected for age of the measured sites.

Given these observations, we do not attempt to build regional or crop specific curves. Rather we simply use a blended version of the Swails model, the Qiu model, and the IPCC values to construct a single reference CO₂ emissions curve for all climate zones and land use types. We generate this curve by:

- adjusting the Swails value to match IPCC tropical palm EF at year 10 (the approximate midpoint of the sites used to generate that value); we leave the Qiu curve as is, since it's already reasonably well calibrated to the IPCC values
- build a new dataset that includes the averages of the Qiu and recalibrated Swails curves for years 1 through 20, and the IPCC temperate/boreal cropland CO₂ value (29.0 t CO₂ ha⁻¹ yr⁻¹) for steady state in the out years,
- fitting a new double exponential to the resulting data points

We use the cropland value for the steady state because recent literature has undermined the evidence that long-drained peatland has lower emissions on grassland (Holzknecht et al. 2025, Keck et al. 2024); as between the cropland and grassland values, the more recent studies better support the IPCC's cropland EF (Tiemeyer et al., 2020); and in any event this approach ensures the model errs on the side of conservatism. Admittedly, the final single, cross-region, cross-crop curve may be an oversimplification, but we do not see sufficient data to support a more varied approach at this time. This could be an area for future iteration.

Relative to the IPCC values, our final blended curve closely matches the IPCC value for oil palm in the relevant time period, is slightly below the IPCC values for tropical acacia and cropland, and then matches the long-term cropland values for boreal and temperate locations. Notably, the IPCC acacia value — the largest outlier — has been questioned in the more recent literature because it is based on subsidence measurements in extremely young plantations, which are disproportionately inflated by compaction. (Deshmukh et al., 2023).

The figure below shows the Swails and Qiu curves, the IPCC emissions factors along with the approximate ages of the sites they represent (based on a partial review of the citations in the Wetlands Supplement), and the final blended reference curve.

![peat-fig-1](figures/peat-fig-1.png)

#### Adding in non-CO₂ emissions

Next, we add non-CO₂ pathways — dissolved organic carbon (DOC), ditch CH₄, and soil N₂O — from IPCC Tables 2.2–2.5.

| Climate zone | Land use | Non-CO₂ (t CO₂-eq ha⁻¹ yr⁻¹) |
| ------------ | -------- | ---------------------------- |
| Tropical     | Oil palm | 4.7                          |
| Tropical     | Cropland | 6.5                          |
| Tropical     | Acacia   | 5.3                          |
| Temperate    | Cropland | 8.3                          |
| Temperate    | Pasture  | 6.6                          |
| Boreal       | Cropland | 7.6                          |
| Boreal       | Pasture  | 6.1                          |

As in the previous section, we construct a time-varying non-CO₂ curve. In this case, we just use the IPCC sites directly as a smooth blend between the early acacia values and the late cropland values.

- **Years 1–10:** Non-CO₂ ≈ 5.3 t CO₂-eq ha⁻¹ yr⁻¹ (tropical acacia value, reflecting young drainage)
- **Years 40+:** Non-CO₂ ≈ 8.3 t CO₂-eq ha⁻¹ yr⁻¹ (temperate cropland value, reflecting mature drainage)
- **Years 10–40:** Smooth logistic transition between the two anchors

This curve captures the physical observations that soil N₂O from progressive nitrogen mineralization intensifies as drainage matures, following the opposite trajectory of CO₂ (Swails et al., 2022), and is consistent with the idea that other pathways, such as dissolved carbon and ditch methane, remain relatively stable. The resulting all-GHG reference curve is the sum of the time-dependent CO₂ curve and this time-varying non-CO₂ component.

![peat-fig-2](figures/peat-fig-2.png)

Note that the IPCC N₂O emission factors are intended to capture N₂O from mineralization of peat nitrogen, not from fertilizer application. Fertilizer-related N₂O should be captured separately under standard agricultural land management accounting.

### GHGP parameterization

Finally we use the updated reference curve — now including both CO₂ and non-CO₂ emissions — to set the two parameters of our GHGP model.

First we set **E_LM** equal to the steady-state of the all-GHG reference curve: the CO₂ floor plus the long-run non-CO₂ asymptote (temperate cropland value). That is **37.3 t CO₂-eq ha⁻¹ yr⁻¹**.

Second, we set **P_LUC** by least-squares fit of the GHGP linear ramp to the all-GHG reference curve over years 1 to 20, with E_LM fixed from the previous step. That comes to **621 t CO₂ ha⁻¹**.

The linear ramp undershoots the reference curve in year 1 and slightly overshoots in years 4–20, but matches total emissions very closely over the 20 year period. The graphs below show (a) annual emissions for the reference curve and the GHGP approximation, alongside IPCC EFs; and (b) cumulative emissions for the reference curve and the GHGP approximation, alongside cumulative emissions for different regions and crops if calculated with time-invariant IPCC EFs.

![peat-fig-3](figures/peat-fig-3.png)

![peat-fig-4](figures/peat-fig-4.png)

## Calculations

See the [accompanying python notebook](../analyses/peatland_emissions_modeling.ipynb) for curve fits and related calculations.

## Appendix: IPCC Literature Review

This catalogues a partial review of IPCC citations. We believe these to be representative (the IPCC Wetlands Supplement acknowledges the age difference between sites used for the published EFs), but completing this review is an area for refinement.

| IPCC Category             | Papers reviewed                                                         | Sites with age data                      | Est. drainage age range          |
| ------------------------- | ----------------------------------------------------------------------- | ---------------------------------------- | -------------------------------- |
| Tropical Acacia           | Hooijer 2012, Jauhiainen 2012a, Basuki 2012, Hergoualc'h & Verchot 2011 | 125+ sites (Hooijer) + 12 plots (Basuki) | **5–10 yr**                      |
| Tropical Oil Palm         | Hooijer 2012, Hergoualc'h & Verchot 2011, Melling 2005a/2007            | 42 sites (Hooijer) + 1 site (Melling)    | **5–18 yr**                      |
| Tropical Cropland         | Hergoualc'h & Verchot 2011 (indirect)                                   | few                                      | **8–15 yr**                      |
| Boreal/Temperate Cropland | Qiu et al. 2021 Table S2 (summarizing all 39 sites)                     | 24 of 39 with known ages                 | **26–300 yr**                    |
| Temperate Grassland       | Schrier-Uijl et al. 2014                                                | 2 Dutch sites                            | **~200–800+ yr** (Dutch polders) |
| Boreal Grassland          | -                                                                       | -                                        | **unknown**                      |

## References

- Bader, C., Müller, M., Schulin, R. & Leifeld, J. (2018). Peat decomposability in managed organic soils in relation to land use, organic matter composition and temperature. *Biogeosciences*, 15, 703–719.

- Clymo, R.S. (1984). The limits to peat bog growth. *Philosophical Transactions of the Royal Society of London B*, 303, 605–654.

- Clymo, R.S. (1992). Models of peat growth. *Suo*, 43, 127–136.

- Deshmukh, C.S., Susanto, A.P., Nardi, N., et al. (2023). Net greenhouse gas balance of fibre wood plantation on peat in Indonesia. *Nature*, 616, 740–746.

- Evans, C.D., Renou-Wilson, F. & Strack, M. (2016). The role of waterborne carbon in the greenhouse gas balance of drained and re-wetted peatlands. *Aquatic Sciences*, 78, 573–590.

- Holzknecht A, Land M, Dessureault-Rompré J, Elsgaard L, Lång K, Berglund Ö. Effects of converting cropland to grassland on greenhouse gas emissions from peat and organic-rich soils in temperate and boreal climates: a systematic review. Environ Evid. 2025 Jan 19;14(1):1.

- IPCC (2014). 2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands. Hiraishi, T., et al. (eds). IPCC, Switzerland.

- Keck H, Meurer KHE, Jordan S, Kätterer T, Hadden D and Grelle A (2024) Setting-aside cropland did not reduce greenhouse gas emissions from a drained peat soil in Sweden. Front. Environ. Sci. 12:1386134.

- McCalmont, J., Kho, L.K., Teh, Y.A., et al. (2021). Short- and long-term carbon emissions from oil palm plantations converted from logged tropical peat swamp forest. *Global Change Biology*, 27, 2361–2376.

- Mos, H., Harun, M.H., Jantan, N.M., Hashim, Z., Ibrahim, A.S. & Yusup, Y. (2023). Differences in CO₂ emissions on a bare-drained peat area in Sarawak, Malaysia, based on different measurement techniques. *Agriculture*, 13, 622.

- Page, S.E., Rieley, J.O. & Banks, C.J. (2011a). Global and regional importance of the tropical peatland carbon pool. *Global Change Biology*, 17, 798–818.

- Page, S.E., Morrison, R., Malins, C., Hooijer, A., Rieley, J.O. & Jauhiainen, J. (2011b). Review of peat surface greenhouse gas emissions from oil palm plantations in Southeast Asia. ICCT White Paper 15.

- Qiu, C., Ciais, P., Zhu, D., et al. (2021). Large historical carbon emissions from cultivated northern peatlands. *Science Advances*, 7, eabf1332.

- Schrier-Uijl, A.P., Kroon, P.S., Hendriks, D.M.D., et al. (2014). Agricultural peatlands: towards a greenhouse gas sink — a synthesis of a Dutch landscape study. *Biogeosciences*, 11, 4559–4576.

- Swails, E., Hergoualc'h, K., Deng, J., Frolking, S. & Novita, N. (2022). How can process-based modeling improve peat CO₂ and N₂O emission factors for oil palm plantations? *Science of the Total Environment*, 839, 156153.

- Tiemeyer, B., Freibauer, A., Albiac Borraz, E., et al. (2020). A new methodology for organic soils in national greenhouse gas inventories: Data synthesis, derivation and application. *Ecological Indicators*, 109, 105838.

- Truskavetskii, R.S. (2014). Carbon budget of drained peat bogs in Ukrainian Polesie. *Eurasian Soil Science*, 47, 687–693.

- Yu, Z.C., Loisel, J., Brosseau, D.P., Beilman, D.W. & Hunt, S.J. (2010). Global peatland dynamics since the last glacial maximum. *Geophysical Research Letters*, 37, L13402.
