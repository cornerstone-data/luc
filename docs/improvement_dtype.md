# A shared standard for data types and missing values

The objectives are:

- **Storage:** store each layer in the smallest type that holds its values, without changing them.
- **Accuracy:** never change a value silently, and never lose track of missing data during a calculation.
- **Meaning:** for every cell, tell apart a **value** (zero included), a **null** (the source says it doesn't know), a cell **outside the source's coverage**, and a cell that is **not available** because processing failed or hasn't run yet.

The proposal is split into three contracts, which can be adopted and tested one after the other:

1. **Values:** what is stored in each cell, in which type, and how it was transformed.
2. **Availability:** where a source has data, and whether processing of a tile, variable and year is complete.
3. **Calculations:** which inputs each result needs, what happens when one is missing, and what each total covers.

## How Zarr stores data

A Zarr array is a grid of cells, all of the same type, cut into blocks and saved as separate objects in a bucket:

```text
 one 1° tile = one shard (one object)          one chunk = 1,000 × 1,000 cells
 ┌──────┬──────┬──────┬──────┐                 each cell: 1 byte in uint8
 │chunk │chunk │chunk │chunk │                 → 1 MB before compression,
 ├──────┼──────┼──────┼──────┤                   much less after
 │chunk │chunk │empty │empty │
 ├──────┼──────┼──────┼──────┤                 the index at the end of the shard
 │chunk │empty │empty │empty │                 says where each chunk starts,
 ├──────┼──────┼──────┼──────┤                 or that it is empty
 │empty │empty │empty │empty │
 └──────┴──────┴──────┴──────┘
       + index
```

- **Chunks.** The array is cut into chunks of 0.25° (1,000 × 1,000 cells on the 30 m grid). A chunk is the smallest piece that is read or written, and it is compressed before it is saved.
- **Shards.** The 16 chunks of one 1° tile are saved together in one object, the shard, with a small index at its end that says where each chunk is.
- **Size of a cell.** Every cell takes the size of the array's type, whatever its value: 1 byte in `uint8`, 2 in `uint16`, 4 in `float32`. Storing 0, 2 or 255 costs the same.
- **Metadata.** Next to the data, a small `zarr.json` file gives the type, the shape, the chunk size, the `fill_value` and free-form attributes (units, meaning of codes, source).

**Empty costs almost nothing.** If every cell of a chunk equals the `fill_value`, the chunk is not saved: the shard index just marks it as empty (16 bytes). If all 16 chunks of a tile are empty, the shard itself is not saved, and the tile costs nothing. Reading an empty chunk or tile gives back the `fill_value` for every cell. For example, one year of GPW grassland on the grid is stored in 16,962 shards, out of 64,800 1° tiles over the globe: the other tiles, mostly ocean, are not stored at all.

**Partly empty chunks are cheap too.** A chunk with some data is saved whole, so each of its cells needs a value: the cells without data hold the `fill_value`. A long run of the same value compresses to almost nothing, so these cells add very little.

**What costs space is variety, not the values themselves.** A chunk full of different values compresses less than a uniform one. Reserving codes for missing data (e.g. 254 and 255 in `uint8`) therefore doesn't make an array bigger. It only decides what those bytes mean.

### Folder structure and where the metadata is written

A Zarr v3 store is a folder (or a prefix in a bucket). Each **group** and each **array** is a sub-folder with its own `zarr.json`. The data of an array is under its `c/` folder, with **one file per shard**. Chunks don't have their own files: they are inside their shard.

```text
raw.zarr/
├── zarr.json                                   ① root group: the grid
├── gpw/                                        one source = one group
│   ├── zarr.json                               ② source group: where the data comes from
│   ├── grassland_class/                        one variable = one array
│   │   ├── zarr.json                           ③ array: type, shape, shards, fill value, codes, mapping
│   │   └── c/                                  data: one file per shard
│   │       ├── 10/82/279                       year index 10 (2010), tile x279y082
│   │       ├── 10/82/280                       the next tile to the east
│   │       └── …                               empty tiles (ocean): no file at all
│   ├── grassland_class_coverage/
│   │   ├── zarr.json                           ④ coverage array: where the source applies
│   │   └── c/…
│   ├── year/   zarr.json + c/0                 coordinates: one small array each
│   ├── y/      zarr.json + c/0
│   ├── x/      zarr.json + c/0
│   └── _done/                                  ⑤ completion markers (plain files, not Zarr)
│       └── grassland_class/2010/x279y082.json
├── gnw_tcl/                                    same layout for every source
│   └── …
└── …
```

The name of a shard file is its position in the array: `c/{year}/{row}/{column}`, counted in shards. With 1° shards, row 82 and column 279 is tile `x279y082`. A reader goes straight to the file it needs, without listing the folder.

Each level holds a different part of the metadata:

| File                                       | Level        | What it holds                                                                                                                                                                                               |
| ------------------------------------------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ① `raw.zarr/zarr.json`                     | root group   | the grid shared by every source: coordinate system, cell size, origin, grid name and version                                                                                                                |
| ② `gpw/zarr.json`                          | source group | the source: product, release, download URL, ingest date, code version                                                                                                                                       |
| ③ `gpw/grassland_class/zarr.json`          | array        | the Zarr fields (type, shape, shard and chunk size, compression, `fill_value`) and the attributes that describe the values: codes and their meanings, units, null code, mapping from the source, resampling |
| ④ `gpw/grassland_class_coverage/zarr.json` | array        | the same for the coverage array: one per variable, with the same dimensions (year, y, x)                                                                                                                    |
| ⑤ `gpw/_done/…/x279y082.json`              | plain file   | completion of one tile, variable and year: `complete` or `failed`, with the source files used and the reason                                                                                                |

Zarr readers ignore `_done/`, because it has no `zarr.json`. Only the project reader uses it.

The array's `zarr.json` (③) in full. The first part is defined by Zarr; `attributes` is ours:

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [25, 720000, 1440000],
  "data_type": "uint8",
  "dimension_names": ["year", "y", "x"],
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 4000, 4000]}},
  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
  "fill_value": 255,
  "codecs": [{
    "name": "sharding_indexed",
    "configuration": {
      "chunk_shape": [1, 1000, 1000],
      "codecs": [
        {"name": "bytes"},
        {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle", "typesize": 1, "blocksize": 0}}
      ],
      "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
      "index_location": "end"
    }
  }],
  "attributes": {
    "_FillValue": 255,
    "missing_value": [254],
    "flag_values": [0, 1, 2],
    "flag_meanings": "not_grassland cultivated_grassland natural_grassland",
    "state_codes": {"null": 254, "nothing_written": 255},
    "mapping": "identity",
    "resampling": "none"
  }
}
```

- `shape`: 25 years × 720,000 rows × 1,440,000 columns, i.e. the whole globe at 1/4000°.
- `chunk_grid`: the size of one **shard**, one year of one 1° tile (4,000 × 4,000 cells). This is what becomes one file.
- `sharding_indexed` → `chunk_shape`: the size of one **chunk** inside the shard, 0.25° (1,000 × 1,000 cells), and how each chunk is compressed.
- `index_location: end`: the shard's index (where each chunk starts, or that it is empty) is at the end of the file.
- `fill_value`: what a reader gets for a chunk or shard that was never written.

`run.zarr` follows the same layout: one group per step (e.g. `emit/`), one array per result layer, with the same root grid and the same kind of metadata.

## Motivation

### Everything becomes float32 today

`geo.unify_dtype_and_no_data` turns every band's nodata into NaN and casts the band to float32 before harmonize writes it. Most source bands are small integers:

| Dataset                                       | Source type | Content                                    |
| --------------------------------------------- | ----------- | ------------------------------------------ |
| `gnw_tcl`                                     | `uint8`     | year of tree-cover loss                    |
| `gpw_grassland`                               | `uint8`     | grassland class                            |
| `liao_gaced30`                                | `uint8`     | cropland class                             |
| `gnw_global_peatlands`                        | `uint8`     | peat flag                                  |
| `ipcc_climate_zones`                          | `uint8`     | climate zone code                          |
| `usda_nass_cdl`                               | `uint8`     | crop class                                 |
| `gnw_harris_agb`                              | `uint16`    | aboveground biomass                        |
| `descals_oil_palm`                            | `uint16`    | planting year                              |
| `soilgrids_ocs`                               | `int16`     | soil organic carbon stock                  |
| `huang_bgb`, `gpw_livestock`, `ifpri_mapspam` | `float32`   | belowground biomass, livestock, crop areas |

Casting a `uint8` band to float32 makes it 4× bigger before compression, and floats usually compress worse than small integers. Harmonize is 92% of the cache (1.2 TiB on the continent-scale benchmark), and this cast is expected to be a large part of it. Class codes also become floats (`3` becomes `3.0`), so a reader can no longer tell a code from a measurement. For large IDs, float32 is not just bigger but wrong: above 16,777,216, two different integers can become the same float.

### One NaN for several meanings

After harmonize, a NaN can mean any of these:

- the source does not cover this place (outside its published extent, ocean);
- the source covers it but has no value there (cloud, sensor gap, water or rock in SoilGrids);
- the source file was missing or corrupted, so nothing was ingested;
- the layer has not been processed yet for this tile.

These call for different responses. The first is a methodology question (what to do where a source doesn't apply). The second is a scientific gap (keep it missing, or apply a declared assumption). The last two are processing failures: the work is incomplete and should be retried, not published as a result. The pipeline cannot tell them apart today.

### Missing data is silently turned into zero

Because NaN is ambiguous, emit replaces it with zero to keep the arithmetic going:

- `get_mineral_soil_emissions` uses `soil_organic_carbon.fillna(0)`, so a SoilGrids gap counts as "no soil carbon". The code comment explains why: otherwise the NaN would also discard the cell's vegetation emissions.
- `get_grassland_carbon` and `get_mineral_soil_emissions` use `climate_zones.fillna(0)`. Zone 0 is not in the lookup table, so a cell with no climate zone gets zero carbon.

Each choice may be right, but it is hidden in the code, applies to every reason a value can be missing, and is not recorded in the results. A total computed this way looks complete even when part of its area had no input.

### No-data conventions differ by dataset

The `no_data` field of each dataset uses a different convention: 255 for most `uint8` layers, 0 for CDL (where 0 is CDL's "background"), 65535 for Harris AGB, 32767 for SoilGrids, −1 or NaN for MapSPAM, and `None` (no nodata at all) for TCL, Descals, Huang BGB and GPW livestock. Nothing defines which value means what.

### The Zarr fill-value trap

In Zarr, a chunk that was never written is not an error: reading it returns the array's `fill_value`. This is useful (empty ocean costs nothing), but it hides the reason. With `fill_value: 0`, a chunk that was never written and a chunk measured as "not grassland" (0) read back exactly the same. The Xu peat layer rasterized on the grid shows this: 0 means "not peat", so a cell outside the source's coverage and a non-peat cell look the same.

Even with a fill value that can't be confused with data, **an unwritten chunk can't say why it is empty**: the source doesn't cover it, the file failed, or ingest hasn't run yet. That answer has to come from a separate record.

## Contract 1: values

### 1.1 Keep the source type, losslessly

Each band is stored in its source type, with its values unchanged:

| Kind of data         | Stored as                                   | Example                                     |
| -------------------- | ------------------------------------------- | ------------------------------------------- |
| masks and flags      | `uint8`: 0, 1                               | peat, protected area                        |
| classes              | `uint8` (or `uint16`), codes unchanged      | GPW grassland, GACED30, climate zones, CDL  |
| years                | source type (`uint8` or `uint16`)           | TCL loss year, Descals planting year        |
| IDs                  | `uint16` or `uint32`                        | admin units                                 |
| integer measurements | source type                                 | Harris AGB (`uint16`), SoilGrids (`int16`)  |
| float measurements   | `float32`                                   | Huang BGB, livestock density, MapSPAM areas |
| per-cell results     | `float32` by default; `float64` if declared | t CO₂e/ha                                   |

Rules:

- **Values are bit-equal to the source** for every band that isn't transformed. This is a test at ingest.
- **Class codes, flags, IDs and years are never cast to float**, in storage or when read.
- **float32 is the default for measurements and results**, not an absolute rule. A layer may use float64 when its declared error budget requires it.
- **`bool` is not used for stored layers.** It has no room for a missing marker, and `uint8` costs about the same once compressed.

### 1.2 The metadata describes the values and the mapping from the source

Each array's metadata (its `zarr.json`) says two things:

1. **What each stored value means:** units for a measurement, or the list of codes and their meanings for classes and flags.
2. **The mapping from the source:** which source value became which stored value. For most layers it is the identity (every value is kept as it is); for some, a few codes are changed.

The metadata is the definition: the stored data must follow it, and a reader needs nothing else to interpret a cell.

Identity, e.g. GPW grassland, whose source codes are kept as they are:

```json
"attributes": {
  "flag_values": [0, 1, 2],
  "flag_meanings": "not_grassland cultivated_grassland natural_grassland",
  "mapping": "identity"
}
```

A few codes changed, e.g. CDL, whose 0 ("background", outside the United States) becomes our standard 255:

```json
"attributes": {
  "flag_values": [1, 5, 24, ...],
  "flag_meanings": "corn soybeans winter_wheat ...",
  "mapping": {"0": 255, "other values": "unchanged"}
}
```

Ingest checks that the data follows the metadata: with an identity mapping, every cell is exactly the source's value; with a table, applying the table to the source gives exactly the stored data.

**Moving a source onto the grid is declared too.** A source on another grid (e.g. GACED30, at 1/3710.65° instead of 1/4000°) has to be resampled: each cell of our grid takes the value of the closest source cell. The values are the same codes, but some source cells are used twice and others not at all. The metadata records the method (`"resampling": "nearest"`), and ingest checks that each 1° tile keeps the same share of each class as the source, within an agreed tolerance.

**A scale factor and an offset are allowed, as a declared mapping.** A measurement can be stored as an integer, with two numbers in the metadata that turn it back into the real value:

```text
stored = round((value − add_offset) / scale_factor)
value  = stored × scale_factor + add_offset
```

For example, biomass 85.37 Mg/ha with `scale_factor: 0.1` and `add_offset: 0` is stored as `854` in `uint16`, and read back as 85.4. xarray, GDAL and QGIS apply the two numbers by themselves when they read.

```json
"attributes": {
  "units": "Mg/ha",
  "scale_factor": 0.1,
  "add_offset": 0,
  "mapping": {"scale": "round to nearest", "max_error": 0.05}
}
```

- **An offset alone is lossless.** Years from 2000 to 2253 fit in `uint8` with `add_offset: 2000` and `scale_factor: 1`: 2015 is stored as 15, and nothing is rounded.
- **A scale factor rounds the value**, after the division. The rounding is always to the **nearest** integer, never a cut (85.37 → 854, not 853), because cutting always goes down and would make every total slightly too small. The largest error is half a step: 0.05 with a scale of 0.1.
- **The whole range must fit** below the reserved codes. With `uint16` and a scale of 0.1, the largest value is 6,553.3. Ingest checks the source's maximum, and refuses the layer if it doesn't fit, rather than cutting the largest values.
- **Missing values are mapped first.** A source NaN or nodata becomes the null code before the scale is applied, so it is never rounded into a number.
- **Ingest checks the result.** It reads the stored integers back, compares them with the source, and records the largest difference in the metadata. It must be at most half a step.

Rounding changes values, and a small change can move a cell across a threshold (e.g. 25.04 rounded to 25.0 under a threshold of 25). So for now, **no layer uses a scale factor**. One may be used later, layer by layer, when the storage gain is measured and the emissions results are shown not to change. An offset alone, being lossless, can be used at any time.

### 1.3 Accumulate totals in float64

Per-cell results are float32, but a jurisdiction total adds up millions of cells, and float32 accumulation loses digits. Rollups use float64 accumulators.

## Contract 2: availability

### 2.1 Four answers for every cell

| State                | Meaning                                                 | Example                                          | Where it comes from                       |
| -------------------- | ------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------- |
| **value**            | the source gives a value (zero included)                | class 2 = natural grassland; 0 = not grassland   | the data                                  |
| **null**             | the source covers the cell but says it doesn't know     | cloud, sensor gap, SoilGrids water/rock/ice      | the data, declared per dataset            |
| **outside coverage** | the source does not apply here, by its published extent | CDL outside the United States; ocean             | the coverage record of the source release |
| **not available**    | processing failed or hasn't run for this tile           | file corrupted or missing; tile not ingested yet | the completion record                     |

Zero is an ordinary value. Each of the other three needs a different response:

- **null:** keep it missing, or apply an assumption that the calculation declares (contract 3);
- **outside coverage:** apply the methodology's coverage rule (e.g. another layer is used outside the United States);
- **not available:** the work is incomplete. It is retried, and by default blocks publication of the results that need it. It never becomes an ordinary partial result silently.

### 2.2 How a cell is stored

The fill value is never a data value, and it only means "nothing was written here". Two designs can mark null and "nothing written" in the data:

| Design                             | How                                                                                                                    | Benefit                                                         | Cost                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **A. Reserved codes**              | integers keep the top two codes (`uint8`: 254 = null, 255 = fill); floats use NaN plus a state array                   | the data alone shows which cells are missing                    | different handling for integers and floats; a source using 254 or 255 must be widened or remapped |
| **B. Native values + state array** | the data keeps the full source range; a companion `uint8` state array per variable says value, null or nothing written | the full source range is kept; the same handling for every type | one more array to store and read                                                                  |

Neither design is clearly better on paper. **Decide by benchmarking both on the same tiles**, with identical data, compression, chunks and shards, and compare compressed size, number of requests per read, and implementation complexity. In both designs:

- the stored marks are value, null and nothing written;
- **nothing written** is resolved into "outside coverage" or "not available" with the records of section 2.3, never from the data alone.

### 2.3 Coverage and completion records

Two records answer why a cell has nothing written:

- **Coverage** says where a source release applies. It is a property of the source, and it is stored with the same dimensions as the data it describes: per variable and per year. If 2010 grassland was ingested and 2011 failed, these are two different answers. One mask may be shared by several variables or years only when their coverage is checked and found identical.
- **Completion** says, for each tile, variable and year, whether processing finished: `complete`, `failed` (with the reason and the source files involved) or absent (not run yet). A `complete` tile with nothing written in a cell means the cell is outside coverage. Anything else means the cell is not available.

### 2.4 Publishing a tile

Data, coverage and completion are separate writes, so a reader could see a mix of old and new. The rule is:

1. write the data;
2. write its coverage (and state array, in design B);
3. write the completion marker **last**, with an identifier of what was written (e.g. the run name and source release).

A reader only trusts a tile whose completion marker exists and matches what it expects. A rerun overwrites the marker last, after the data. A run is published only when every tile it needs is complete, or explicitly as **partial**, with the list of incomplete tiles.

### 2.5 What is written in `zarr.json`

An integer layer, design A:

```json
{
  "data_type": "uint8",
  "fill_value": 255,
  "attributes": {
    "_FillValue": 255,
    "missing_value": [254],
    "flag_values": [0, 1, 2],
    "flag_meanings": "not_grassland cultivated_grassland natural_grassland",
    "state_codes": {"null": 254, "nothing_written": 255},
    "source": "gpw/grassland@v2_beta",
    "mapping": "identity",
    "resampling": "none"
  }
}
```

A float result:

```json
{
  "data_type": "float32",
  "fill_value": "NaN",
  "attributes": {
    "_FillValue": "NaN",
    "units": "t CO2e/ha",
    "state_array": "emit_state"
  }
}
```

There are two kinds of reader, with two expectations:

- **Generic readers** (xarray's default decoding, GDAL, QGIS) must at least **mask every missing code**. That's why both 254 (`missing_value`) and 255 (`_FillValue`) are declared with their usual CF names: a generic reader then shows both as empty, and never draws or scales 254 as a measurement. It can't tell null from nothing written, and isn't expected to.
- **The project reader** keeps all the states. It opens arrays without automatic masking and decodes them with one function (e.g. `jdluc.values.decode`), which returns the **values in their own type** (integers stay integers) and a **state array** (value, null, outside coverage, not available), using the completion and coverage records. Measurements are converted to float only when a calculation needs it. The project reader doesn't rely on xarray's fill-value masking, whose behaviour for Zarr depends on the version.

`state_codes`, `state_array`, `mapping` and `resampling` are our own attributes; generic tools ignore them.

## Contract 3: calculations

### 3.1 Missing values follow the inputs actually used

A result is missing only if an input **that its calculation actually needs** is missing. Which inputs are needed often depends on the cell. Today's emissions code already has these cases:

- **Peat replaces the mineral soil term** (`emit.get_conversion_emissions`). On a peat cell, missing mineral soil carbon doesn't make the soil emissions missing.
- **Forest biomass is only needed for a forest conversion.** A missing biomass value doesn't affect a grassland conversion.
- **Destination layers have a priority.** If a higher-priority layer has already set the destination, a missing lower-priority layer doesn't matter. If a higher-priority layer is null, the destination is unknown, because it might have claimed the cell.

So each calculation declares, for each branch, its **required inputs**, and the result's state comes from those inputs only:

| Required inputs of the branch used | Result                                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| all values                         | value                                                                                         |
| any not available                  | not available                                                                                 |
| any null (none not available)      | null                                                                                          |
| any outside coverage               | the coverage rule of that input (e.g. another layer, a declared default, or outside coverage) |

The rules also cover operations other than arithmetic:

- **comparisons** (`x > threshold`, `class == CROPLAND`): a null input gives a null result, not `false`;
- **selections** (first match wins): a null before the first match gives null; a null after it is ignored;
- **time series** (last departure year): a null year makes the result null only if it could change the answer.

### 3.2 Exceptions are declared, not hidden

A calculation can declare, for a given input, that null is treated as a value, e.g. that a SoilGrids gap (water, rock, ice) counts as zero soil carbon. This replaces today's `fillna(0)` with a setting in the run config, recorded in the run's manifest:

```yaml
emit:
  inputs:
    soilgrids_ocs/ocs: {null_as: 0}            # gaps are water, rock or ice: no soil carbon
    ipcc_climate_zones/zone: {null_as: error}
```

`null_as` applies to null only. "Not available" is never replaced by a value: the tile is retried, or the result is published as partial.

### 3.3 Totals say what they cover

Every rollup (attribution, country totals) reports, next to the total, the share of its area that is value, null, outside coverage and not available. A total with a not-available share is flagged as incomplete.

## Worked example: one corrupted input file

A layer is ingested from several source files. Shard `x279y082` (1°, 4 × 4 chunks of 0.25°) sits across two of them: file A covers its western half, file B its eastern half. File A fails its checksum at ingest; file B is fine.

The calculation is deliberately simple: **grassland area in the shard**, i.e. the area of the cells whose class is 1 (cultivated) or 2 (natural grassland). Each square below is one 0.25° block of cells (design A codes): `255` = nothing written, `254` = null, `0` = not grassland, `1` and `2` = grassland.

```text
 1. source files              2. stored shard                  3. availability
                                 gpw/grassland_class (uint8)      (data + records)

 ┌─────────┬─────────┐         ┌───┬───┬───┬───┐               ┌───┬───┬───┬───┐
 │ file A  │ file B  │         │255│255│ 0 │ 2 │               │ ✗ │ ✗ │ v │ v │
 │         │         │         ├───┼───┼───┼───┤               ├───┼───┼───┼───┤
 │ corrupt │  valid  │ ingest  │255│255│ 1 │254│    decode     │ ✗ │ ✗ │ v │ n │
 │    ✗    │    ✓    │ ──────► ├───┼───┼───┼───┤   ──────►     ├───┼───┼───┼───┤
 │         │         │         │255│255│ 1 │254│               │ ✗ │ ✗ │ v │ n │
 │         │         │         ├───┼───┼───┼───┤               ├───┼───┼───┼───┤
 │         │         │         │255│255│ 0 │ 0 │               │ ✗ │ ✗ │ v │ v │
 └─────────┴─────────┘         └───┴───┴───┴───┘               └───┴───┴───┴───┘
                                 west: never written              v = value   n = null
                                 east: copied from B              ✗ = not available
                                                                  (completion: failed, file A)
```

1. **Ingest.** File A is rejected, and nothing is written for it. File B is written as it is: its classes, its zeros and its nulls (254, a cloud gap in the source). The completion record for this tile, variable and year says `failed: file A, checksum`, and no completion marker is written.
2. **Decode.** On the east, the data alone gives values and nulls. On the west, the data only says "nothing written". The completion record says why: the tile failed, so these cells are **not available**, not outside coverage. If file A were simply outside the source's extent and the tile were complete, the same 255 would decode as outside coverage.
3. **Result.** Grassland area over the eastern half is computed, with 2 null blocks reported separately. The shard's result is marked **incomplete** (8 of 16 blocks not available), so it is not published as an ordinary result: the tile is retried, or the run is published explicitly as partial, listing this tile.
4. **Retry.** When file A is fixed and re-ingested, its blocks are written, then the completion marker. The tile becomes complete, and the result is published.

## Expected gains, to be measured

- **Storage.** `uint8` bands are 4× smaller than float32 before compression. The actual gain after compression must be **measured**: write the same tile as float32 and in its native types, with identical data, compression, chunks, shards and coverage arrays, and compare. The same benchmark decides between designs A and B.
- **Accuracy.** No stored value changes except through a mapping or resampling declared in the metadata and checked at ingest; integers stay integers when read; totals are added up in float64.
- **Meaning.** A missing or corrupted file shows up as an incomplete tile, not as zero emissions or a smaller total. Every total says which part of its area had values, nulls, no coverage, or no data yet.

## Implementation and acceptance

Implement in stages, one contract at a time:

1. **Values.** Stop the float32 cast; store each band in its source type; the reader keeps integer types. Measure the storage gain on one tile.
2. **Availability.** Benchmark designs A and B on the same tiles and choose. Declare null codes and coverage for every dataset; write coverage per variable and year; write completion markers last; add the project decoder.
3. **Calculations.** Declare the required inputs of each branch in emit; replace each `fillna` with a declared `null_as`; add the four shares to every rollup; block or flag publication when a needed tile is not complete.

Acceptance requires evidence that:

- Bands stored in their source type are bit-equal to the source, and integers (including IDs above 16,777,216) are read back unchanged.
- A failed file gives "not available" cells, an incomplete tile and a result that is not published as complete; a cell outside a complete tile's coverage gives "outside coverage".
- Coverage that differs between two years of the same variable is recorded separately.
- A reader never treats a tile as complete before its data and coverage are written.
- Generic readers mask both null and nothing-written codes.
- On a peat cell, missing mineral soil carbon doesn't make the soil result missing; on a grassland conversion, missing forest biomass doesn't make the result missing.
- Totals are the same as today's where no needed input is null, outside coverage or not available (apart from float64 accumulation).

The decisions for discussion are: the four states and who resolves them, design A or B (after the benchmark), coverage per variable and year, the publication rule, required inputs per calculation branch, and `null_as` replacing `fillna`.

## References

- [Zarr v3 core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html), for `data_type` and `fill_value`.
- [CF conventions](https://cfconventions.org/), for `_FillValue`, `missing_value`, `scale_factor`, `add_offset`, `flag_values` and `flag_meanings`.
- [xarray `open_zarr`](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html), for its masking options and their version-dependent behaviour.
