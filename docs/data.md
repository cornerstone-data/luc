# Cornerstone LUC: data products

This document is the reference for the artifacts the pipeline produces: how to get them, grids, and full schemas. For *what* the numbers mean see `methodology.md`; for *how* they are produced see `architecture.md`; to reproduce them yourself see the ["Running the pipeline"](../README.md#running-the-pipeline) section of the README.

The data is licensed [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/); please follow the attribution guidance in `ATTRIBUTION.md`.

## Access

Of the three artifacts below, one is published and two are not.

**The emissions-factor table is published as an archived dataset.** Each data version is deposited as `emissions-factors.parquet` in a public data archive that mints a DOI per version, and the matching git tag names the commit behind it. The archive is still being set up; cite a version by its DOI, or follow the fallback in `ATTRIBUTION.md`.

**The two raster artifacts are pipeline outputs.** They are zarr stores far too large to deposit, so they exist only where a run writes them: under the `SCRATCH_ROOT` of whoever ran the pipeline, keyed by the cache hash `storage.py` derives (see [`architecture.md`](architecture.md#caching-versioning-and-provenance)). Sharing the underlying maps in a form that can be read remotely is open work — the schemas below are documented so a run's own output is legible in the meantime.

## Conventions

Unless noted otherwise, every raster artifact shares these conventions:

| Property          | Value                                                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CRS               | `EPSG:4326`                                                                                                                                  |
| Grid / resolution | GLAD native — `0.00025°` (~30 m). The statistical leg additionally works on a coarser ~10 km MapSPAM grid (`0.0833°`); see `methodology.md`. |
| Dimensions        | `(y, x)` per variable, one variable per band                                                                                                 |
| Dtype             | `float32` (all variables, including categorical codes)                                                                                       |
| No-data           | `NaN`                                                                                                                                        |
| Format            | zarr (rasters) / parquet (tables), chunked for out-of-core reads                                                                             |

Emission quantities use two unit conventions: `t CO₂e/ha` (tonnes CO₂e per hectare, on the raster layers) and `t CO₂e` (metric tonnes, on the tabular rollups — the `*_mt` columns). Areas are hectares (`ha`), production is kilograms (`kg`).

## Artifacts

### 1. Harmonized inputs (zarr)

Written by the `harmonize` stage; a pipeline output, not a published artifact (see [Access](#access)).

Every source raster reprojected and warped onto the common grid — the input to the emissions core. One variable per fully-qualified source band, named `{source}:{product}:{band}`.

| Variable                                                        | Type    | Units       | Description                                                                          |
| --------------------------------------------------------------- | ------- | ----------- | ------------------------------------------------------------------------------------ |
| `glad:glcluc:year=2000` … `glad:glcluc:year=2020`               | float32 | GLCLUC code | GLAD land-cover/land-use class, one variable per year (2000, 2005, 2010, 2015, 2020) |
| `gfw:harris-agb:aboveground-biomass-mg-per-ha`                  | float32 | Mg/ha       | Forest above-ground woody biomass (2000)                                             |
| `huang:bgb:belowground-biomass-mg-per-ha`                       | float32 | Mg/ha       | Forest below-ground (root) biomass                                                   |
| `soilgrids:organic-carbon-stocks:organic-soil-carbon-mg-per-ha` | float32 | Mg/ha       | Soil organic carbon stock, 0–30 cm                                                   |
| `gfw:global-peatlands:is-peatland`                              | float32 | 0/1         | Binary peatland mask                                                                 |
| `ipcc:climate-zones:climate-zone`                               | float32 | zone code   | IPCC climate domain per pixel                                                        |

```python
import xarray
# written by the harmonize stage
harmonized = xarray.open_zarr("<scratch-root>/<cache-key>.zarr", consolidated=False)
```

### 2. Per-pixel emissions (zarr)

Written by the `emit` stage; a pipeline output, not a published artifact (see [Access](#access)).

The crop-agnostic, per-pixel, per-span LUC emissions produced by the `emit` stage. Spans are the four consecutive GLAD intervals `2000-2005`, `2005-2010`, `2010-2015`, `2015-2020`; per-year layers cover `2000, 2005, 2010, 2015, 2020`.

| Variable                                             | Type    | Units          | Description                                                                                                 |
| ---------------------------------------------------- | ------- | -------------- | ----------------------------------------------------------------------------------------------------------- |
| `land-class:{year}`                                  | float32 | LandClass code | Per-year land class (Forest, Grassland, Cropland, Built-up, Water, Snow/ice, Ocean), stored as a float code |
| `vegetation-emissions:tco2e-per-ha:{before}-{after}` | float32 | t CO₂e/ha      | Per-span vegetation-carbon loss (above-ground, below-ground, dead organic matter, grassland)                |
| `soil-emissions:tco2e-per-ha:{before}-{after}`       | float32 | t CO₂e/ha      | Per-span soil-carbon loss (mineral stock change + peatland drainage pulse)                                  |
| `emissions:tco2e-per-ha:{before}-{after}`            | float32 | t CO₂e/ha      | Per-span total (vegetation + soil), before temporal discounting                                             |
| `peatland-occupation:tco2e-per-ha`                   | float32 | t CO₂e/ha      | Current-year annual peatland-occupation emissions                                                           |
| `emissions-per-hectare:tco2e-per-ha`                 | float32 | t CO₂e/ha      | 20-year linearly-discounted LUC emissions + peatland occupation — the headline per-hectare layer            |
| `hectares-per-pixel:ha`                              | float32 | ha             | Pixel area (varies with latitude)                                                                           |

```python
import xarray
# written by the emit stage
emissions = xarray.open_zarr("<scratch-root>/<cache-key>.zarr", consolidated=False)
```

### 3. Emissions factors (parquet)

Deposited per data version as `emissions-factors.parquet` (see [Access](#access)).

The final per-(jurisdiction, crop) table produced by the `trace` stage. Indexed by `(admin_level, admin_id, crop_name, methodology)` — all strings — where `admin_level` is `PROVINCIAL` (World Bank admin-1) or `NATIONAL` (admin-0, summed from provincial rows) and `methodology` is `JURISDICTIONAL_DIRECT` or `STATISTICAL`.

| Column                             | Type    | Units        | Description                                                                                  |
| ---------------------------------- | ------- | ------------ | -------------------------------------------------------------------------------------------- |
| `jurisdiction_name`                | string  | —            | Display name (e.g. `United States of America \| Iowa`)                                       |
| `crop_hectares`                    | float64 | ha           | Crop area in the jurisdiction (land occupation), on production's discounted 2000–2020 window |
| `peatland_crop_hectares`           | float64 | ha           | Crop area on peatland                                                                        |
| `emissions_mt`                     | float64 | t CO₂e       | Total allocated LUC emissions                                                                |
| `peatland_occupation_emissions_mt` | float64 | t CO₂e       | Annual peatland-occupation emissions on crop pixels                                          |
| `forest_emissions_mt`              | float64 | t CO₂e       | Forest-conversion emissions                                                                  |
| `peatland_conversion_emissions_mt` | float64 | t CO₂e       | Peatland-conversion (drainage-pulse) emissions                                               |
| `production_kg`                    | float64 | kg           | Crop production (NASS yield × area for the direct leg; MapSPAM for the statistical leg)      |
| `yield_kg_per_ha`                  | float64 | kg/ha        | `production_kg / crop_hectares`                                                              |
| `emissions_factor_kgco2e_per_kg`   | float64 | kg CO₂e / kg | `1000 × emissions_mt / production_kg` — the headline factor                                  |
| `peatland_occupation_fraction`     | float64 | ratio (0–1)  | `peatland_occupation_emissions_mt / emissions_mt`                                            |

Both attribution legs emit the same columns.

```python
import pandas
emission_factors = pandas.read_parquet("emissions-factors.parquet")
```

The `attribute` stage writes an intermediate rollup parquet with the same index and the non-derived subset of these columns (`crop_hectares`, `peatland_crop_hectares`, `emissions_mt`, `peatland_occupation_emissions_mt`, `forest_emissions_mt`, `peatland_conversion_emissions_mt`, and — statistical only — `production_mt`); `trace` joins production and appends the ratio columns (`production_kg`, `yield_kg_per_ha`, `emissions_factor_kgco2e_per_kg`, `peatland_occupation_fraction`).
