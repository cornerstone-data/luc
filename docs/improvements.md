# Improvements

Proposed changes to the pipeline, to be taken up one at a time. Each entry has a **description** of the topic and how it works today, the **motivation** for changing it, and the **proposed change**. An entry is removed once it lands.

They're listed in dependency order: the lattice (1) comes first because the tiling (2) and chunking (3) are defined in terms of it. The raw store (5), the run store (6) and the execution model (7) assume all three. The vector format (8) is independent; the rasterized vector layers (9) build on the raw store, and the config file (10) on the run store. Bucket access (18) is independent, but pays off most once chunks are sharded (3) and the work is spread over many machines (7).

## Summary

| #   | Name                          | Description                                                                     | Motivation                                                                        |
| --- | ----------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 1   | Define the grid explicitly    | Write down, in one place, the shared global grid every dataset sits on.         | A single shared grid makes combining data simple, reliable and checkable.         |
| 2   | Smaller work units            | Process the world in small 1° squares instead of large 10° squares.             | Smaller jobs are faster, cheaper to retry and spread better across machines.      |
| 3   | Tidy storage blocks           | Store data in blocks that line up with the work squares.                        | Clean boundaries avoid conflicts between workers and cut storage costs.           |
| 4   | Keep original data types      | Store each layer in its original compact form instead of converting it.         | Converting everything inflates storage several times over for no benefit.         |
| 5   | One shared source store       | Load each source once into a single shared store already on the grid.           | Avoids keeping duplicate copies and redoing the same alignment every run.         |
| 6   | One named folder per run      | Give each run a readable name and keep all its results together.                | Results become easy to find, share, compare and trace back to their settings.     |
| 7   | Run on many cloud machines    | Spread the work across many small cloud machines instead of one big one.        | One machine is too slow to process the whole world in reasonable time.            |
| 8   | Modern vector format          | Store map shapes (e.g. borders) in the same format as the other tables.         | It is faster to read and keeps one format for all non-grid data.                  |
| 9   | Turn shapes into grid layers  | Convert every map shape layer into a grid layer when it is loaded.              | Shapes then combine with other data the same simple way, without repeated work.   |
| 10  | Rules in a config file        | Move the methodology rules and thresholds out of the code into a settings file. | What makes one run differ from another is visible and recorded in one place.      |
| 11  | Say what "no value" means     | Record what an empty cell means for each layer.                                 | "Not covered", "no data" and "zero" are currently mixed up and indistinguishable. |
| 12  | Read only what's needed       | Load only the layers and areas a step actually uses.                            | Reading unused layers and empty ocean wastes time and money.                      |
| 13  | Track what produced a result  | Identify results by the exact data and code that produced them.                 | Today an outdated result can be reused silently if someone forgets a manual step. |
| 14  | Reuse the processing engine   | Start the internal processing engine once instead of for every save.            | Restarting it each time is slow and forces saves to happen one at a time.         |
| 15  | One command per run           | Replace the five manual steps with one command, plus a status command.          | It's easy to forget a step or mix settings, and hard to know what's done.         |
| 16  | Check settings up front       | Validate the settings file before the run starts.                               | Mistakes should fail in seconds, not hours into a run.                            |
| 17  | Settings from the environment | Read settings from the environment, with the local file as a fallback.          | Cloud machines usually receive settings that way, not as a file.                  |
| 18  | Faster bucket access          | Talk to the cloud buckets through obstore, a Rust library, instead of gcsfs.    | Lower overhead per request, fewer dependencies, and native support in Zarr v3.    |

## 1. Define the lattice explicitly

**Description.** The analysis grid is EPSG:4326 at 1/4000° (~30 m), anchored at the graticule origin, but the code never names it. It's implied in several places that have to agree with each other:

- `tiling.TileResolution.GLAD` sets 40,000 pixels per 10° tile.
- `harmonize.Grid` hardcodes EPSG:4326 and builds a float geotransform from a tile ID's north-west corner.
- `liao_gaced30` derives its own `PIXELS_PER_DEGREE`.
- `emit.get_hectares_per_pixel` recovers the cell size by averaging coordinate differences.

**Motivation.**

- **A shared standard makes every other step simpler.** If every layer, every ingest and every output uses the same fixed lattice, then joining two layers is a lookup by cell index, with no resampling and no alignment check at each step. Tiles, chunks and shards (changes 2 and 3), the raw store (5), the run store (6) and the cloud tasks (7) are all defined from it. Most of the other changes in this list become simple because of this one.
- **Nothing checks alignment.** No code verifies that an ingested raster lands on the grid. Alignment holds only because every ingest happens to get it right, and a source that's off by a fraction of a cell would be resampled silently with no error.
- **The anchor and conventions aren't written down.** The pixel convention and cell indexing are undocumented, so a collaborator, or GeoCollision, has to reverse-engineer them to join data.
- **Cell area is approximate.** The spherical `cos(lat)` estimate is off by −0.75% to +0.5% against the WGS 84 ellipsoid, depending on latitude, and that error carries into every hectare and emissions total.
- **Changing resolution is a migration.** Moving to a finer grid would mean finding and editing every implicit copy of the cell size.
- **Coarse grids don't fit.** MapSPAM (1/12°) and GPW livestock (1/120°) don't divide evenly into 1/4000°, so they're stretched by a fractional factor today.

**Proposed change.** Add `jdluc/lattice.py` as the only module where a cell size appears. It would define:

- a `Lattice` class: EPSG:4326, integer `cells_per_degree`, cell size as the exact rational `1/N`, anchored at (0°, 0°) so cell edges sit at `lon = i/N`, `lat = j/N`;
- the pixel convention: `AREA_OR_POINT=Area`, north-up, with west and north edges owned;
- global integer indices `(I, J)` from the north-west corner `(-180°, 90°)`, with exact index-to-coordinate functions and no accumulated floating-point arithmetic;
- a conformance check (CRS, step, origin) that ingest runs before writing; anything off the lattice is refused unless it was warped on purpose;
- the ellipsoidal (WGS 84) cell area per row.

Two instances would be declared:

| Name     | Cells per degree | Cell size                     | Role                                                          |
| -------- | ---------------- | ----------------------------- | ------------------------------------------------------------- |
| `L4000`  | 4,000            | 1/4000° = 0.00025° (~30 m)    | Today's grid: GLAD, Hansen TCL, GPW grassland, GACED30        |
| `L12000` | 12,000           | 1/12000° ≈ 0.0000833° (~10 m) | Finer sibling for 10 m sources, such as Sentinel-2-based maps |

`L12000` is an exact 3× refinement of `L4000`: every 30 m cell is 3 × 3 cells of 10 m, with no edge moved, so moving between the two grids is replication rather than resampling. It's also the first of these grids that the coarse inputs fit into evenly: one MapSPAM cell is 1,000 cells of `L12000` and one GPW livestock cell is 100.

**A predefined grid, like H3.** H3 is a global grid that is defined once for everyone: each cell has a fixed ID, so any two datasets indexed in H3 can be joined by cell ID, without discussing projections or pixel origins. The lattice should play the same role for this project, and for related ones such as GeoCollision: a published standard, with a fixed origin, a fixed cell size and fixed cell IDs, that every process uses. Each cell has a stable ID (`cell_id = J × width + I`), each tile has a stable name (e.g. `x279y082`), and a cell means the same place in every store and every run. Two differences with H3 work in our favour. The cells are the pixels of the main sources (Hansen, GPW), so those sources are used as they are, with no resampling. And the levels nest exactly: one `L4000` cell is exactly 3 × 3 `L12000` cells, while an H3 cell's children only approximately cover it.

`TileResolution`, `harmonize.Grid` and the per-dataset pixel constants would then be derived from a `Lattice` instead of restating it. This matches GeoCollision's `grid.py` and `grid-spec.md`, which define the same 1/4000° lattice, so cell IDs stay interchangeable between the two projects.

## 2. Process in 1° work tiles instead of 10°

**Description.** The pipeline's unit of work, cache and retry is the 10° tile (e.g. `20N_090W`), the footprint of a Hansen GFC file. Harmonize, emit, downscale and attribute all run once per 10° tile. `attribute.workflow` gives each 10° tile its own thread.

**Motivation.**

- **Tasks are huge.** A 10° tile is 1.6 × 10⁹ pixels per band. The harmonize stack of ~56 bands is about 358 GB uncompressed as float32, so a task needs a large machine and takes hours. The continent-scale harmonize benchmark took ~53 h.
- **Failures are expensive.** If a spot VM is preempted, the whole tile's work is lost.
- **Load is uneven.** A full-land tile and a coastal sliver cost very different amounts, but both are one task, so a run waits on its largest tiles.
- **Too few tasks.** A few hundred tiles can't keep a cloud scheduler (Cloud Batch, GKE) busy across many small workers.
- **Coarse pixels don't fit 10°, and 0.25° is too small.** One degree is exactly 4,000 GLAD pixels, 120 GPW livestock pixels and 12 MapSPAM pixels, so a 1° tile edge never cuts through a coarse source pixel. Every 10° tile is also whole in coarse pixels, but it's far too big to be a task. Every 0.25° tile is whole too, but it's too small.

**Proposed change.**

- **Make the 1° tile the unit of processing, caching and retry.** A 1° tile is 1.6 × 10⁷ pixels per band, about 3.6 GB for the full float32 stack, which fits a small worker and finishes in minutes.
- **Keep 10° only for ingest.** It's the native footprint of the Hansen and GLAD sources. Ingested COGs are internally tiled, so a 1° window read from them is a cheap range request.
- **Use resolution-independent tile IDs.** Name tiles by index, e.g. `x279y082` as in GeoCollision, so the same ID means the same ground on `L4000` and `L12000`.
- **Skip empty tiles.** Most of the lattice is ocean, so a cheap land-mask check should skip a tile before any band is read.
- **Update the tile lookup.** `worldbank_jurisdictions` country-to-tile lookups would return 1° tiles. Attribute's one-thread-per-tile rule carries over unchanged.

## 3. Align chunks to degree lines and shard by tile

**Description.** `geo.get_chunk_size` picks the largest power-of-two chunk that keeps all bands under 1 GiB. For the ~56-band harmonize stack that's 2048 × 2048 pixels. Each chunk is stored as its own object.

**Motivation.**

- **Chunks fall in arbitrary places.** 2048 doesn't divide 40,000 (or 4,000), so chunk edges don't line up with tiles or degree lines. The last chunk in each row and column is ragged.
- **Chunk size depends on band count.** Adding a band can change the chunking of every store.
- **Writers can collide.** When several workers write neighbouring tiles, a chunk that straddles a tile edge has two writers. Today that's avoided only by `storage.COMPUTE_LOCK`, which serializes writes.
- **Too many objects.** At 30 m and 0.25° chunks, an unsharded global variable is about 1 million objects per year in GCS, and request count is what drives cost and latency.

**Proposed change.**

- **Fix the chunk at 0.25° (1,000 px on `L4000`, 3,000 on `L12000`).** Size it in degrees so the chunk count per year is the same at any resolution, and so it always lands on degree lines.
- **Use Zarr v3 sharding with one shard per 1° tile (4 × 4 chunks).** A global variable then drops to about 65,000 objects per year. Each read costs two requests: one for the shard index and one for the data.
- **One writer per shard.** Because a shard is exactly one work tile, parallel workers never write the same object, and `COMPUTE_LOCK` can go.
- **For multi-year arrays, put time first in the shard.** GeoCollision's benchmark measured a point time-series at 2 requests with time-major shards against 48 with per-year shards.

## 4. Keep native dtypes in harmonize

**Description.** `geo.unify_dtype_and_no_data` casts every harmonized band to float32 and turns nodata into NaN before the zarr is written. Most source bands are categorical `uint8` (land classes, CDL, masks, loss year) or small integers.

**Motivation.**

- **Harmonize dominates storage.** It's 92% of the cache, 1.2 TiB for the continent-scale benchmark.
- **The cast inflates it.** Storing a `uint8` band as float32 quadruples its uncompressed size. Floats also compress worse than small integers.
- **Codes and measurements get blurred.** Class codes become floats, so readers can't tell a code from a measured value.

**Proposed change.**

- **Store each band in its source dtype** with an explicit integer fill value declared per dataset (the `no_data` field already exists).
- **Convert in emit.** Emit would cast to float32 and turn fill into NaN when it reads, so the arithmetic doesn't change.
- **Expect a 2–4× smaller cache.** That's the estimate for harmonize's cache from the band mix. Measure it on one tile before adopting.

## 5. Ingest straight into a raw zarr on the lattice

**Description.**

- **Ingest writes COGs.** Each source is downloaded, checked and converted to a Cloud-Optimized GeoTIFF under `INGEST_ROOT`, at its own resolution and on its own grid, with provenance tags (`watershed-*`).
- **Harmonize does the alignment later.** For each tile, harmonize builds one GDAL VRT per band that puts the COG on the tile grid, resampling when the source is not at 30 m. It then casts the result to float32 and saves it as a new zarr under `SCRATCH_ROOT`.
- **Emit, downscale and attribute read the harmonize zarr**, never the COGs.

**Motivation.**

- **The same values are stored twice.** For 30 m sources (GLAD, TCL, GPW grassland, GACED30), the harmonize zarr holds the same values as the COG, only in float32. This copy is most of harmonize's 1.2 TiB.
- **The alignment is redone again and again.** Harmonize's cache key includes the list of datasets in the stack. Adding one dataset, or bumping the version, recomputes every band of every tile: all the VRT reads and resampling run again, even for sources that did not change.
- **The COG's main reason is inspection, not processing.** `architecture.md` chose COGs so each ingested file can be opened in QGIS or with `gdalinfo` and checked by hand. That matters while a dataset is being added, but every run after that pays the processing cost.
- **COGs are the slow path for processing.** A COG is good for viewing in QGIS or GDAL, but every read goes through GDAL and a VRT. A zarr on the lattice is read directly by xarray and dask, chunk by chunk, with the chunks and shards of change 3.
- **Sources are used by every run.** They change rarely (a new release of a dataset), while runs change often (new thresholds, new processes). Work done once at ingest is saved for every run after it.

**Proposed change.**

- **Keep today's ingest code, change only its last step.** The per-dataset `save_tile_id_to_local_path` functions already build a GeoTIFF in a temporary folder, and they stay as they are. Only the end of `RasterDataset.ingest_a_tile` changes:

  ```
  today:     source → temp GeoTIFF → validate → convert to COG → upload COG
  proposed:  source → temp GeoTIFF → validate → lattice check (change 1) → write region in raw.zarr → delete temp
  ```

  The lattice check is new: today nothing verifies that an ingested raster sits on the grid.

- **Ingest writes into one shared `raw.zarr`.** Each source becomes a group, and each band an array (e.g. `gnw_tcl/lossyear`), on the lattice of change 1, with the chunks and shards of change 3 and the source dtype of change 4.

- **Do the alignment once, at ingest.**

  - **Sources already on the lattice** (GLAD, TCL, GPW grassland): copied as they are, with no resampling. The values must be bit-equal to the source, which is a test.
  - **30 m sources on another grid** (GACED30): warped once onto the lattice at ingest, and the method is recorded in the array's attributes.
  - **Coarse sources** (SoilGrids, Harris AGB, Huang BGB, MapSPAM, GPW livestock): stored at their own resolution, on their own lattice, in their own group. Bringing them to 30 m is a methodology choice (copy each value or interpolate), so it belongs to the run (change 6), not to ingest.

- **Emit reads `raw.zarr` directly.** There is no harmonize step and no harmonize copy for 30 m sources.

- **Don't keep the COGs by default.** The temporary GeoTIFF is deleted once its region is written. To look at a layer in QGIS, export a COG from `raw.zarr` (`tools/dump-zarr-to-cogs.py` already does this for zarr stores); this export should be one simple command, since easy inspection was the main reason for COGs. For the few layers the team inspects often, the COG can be kept next to the zarr as an option.

- **Move the provenance into the zarr.** The `watershed-*` tags (data version, git SHA, processing time, source and product, remote URL) become attributes of each group in `raw.zarr`, together with the source cell size and any resampling method.

- **A new source release is a new group or a new version.** `raw.zarr` is never changed in place. A new release of a dataset gets its own group (e.g. `gnw_tcl@v1.12`), and each run records in its manifest which groups it read.

**Measured.** We compared one year (2010) of GPW grassland, stored as the source GeoTIFFs and as a zarr on the 1/4000° lattice (from GeoCollision). The zarr is 1.9× smaller:

|             | Zarr (`adastra-zarr/gcg4000.zarr/gpw/grassland_class/c/10`) | GeoTIFF (`adastra-input-public-data/…/global_pasture_watch/v2_beta/2010/raw`) |
| ----------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Size        | **8.46 GiB**                                                | **16.14 GiB**                                                                 |
| Objects     | 16,962 shards of 1°                                         | 35 files                                                                      |
| Compression | blosc zstd-5 + bitshuffle                                   | deflate                                                                       |
| Overviews   | none                                                        | 9 levels                                                                      |

Part of the gap comes from the GeoTIFF overviews, which add about one third. Without them, the zarr is still about 1.4× smaller (estimated, not measured). The rest comes from better compression and from empty 1° tiles (ocean) not being written at all. The cost is that the zarr has no overviews, so a viewer needs an exported COG or a pyramid built from the zarr.

## 6. One named store per run

**Description.**

- **Every cached call is its own store.** The decorators in `storage.py` write each cached result as its own store, named by a hash of the function, its `version` number and its arguments: `{SCRATCH_ROOT}/{hash}.zarr` or `.parquet`.
- **Harmonize, emit and downscale therefore write one zarr per tile.** A global run leaves hundreds of unrelated hash-named stores in one flat folder.

**Motivation.**

- **Outputs can't be found or browsed.** A hash name is deterministic but unreadable: it doesn't say which step, tile or country produced it, so a store can only be found by recomputing the hash or searching the logs.
- **A run has no identity.** Nothing groups together what one run produced, so a run can't be shared, archived or deleted as a whole. Nothing records which code and settings made a result.
- **Nothing can be opened as one dataset.** To look at emit over a whole region, you open many per-tile stores and merge them.
- **Completion is coarse.** "Done" means a tile's whole store exists, so a partial write is either complete or invisible.

**Proposed change.**

- **Name the run when it starts.** The user gives the run a name, and everything the run produces goes under `runs/{name}/`. The name identifies a version of the emissions layers, not a crop or a country: a run is global, and one name means one set of methods and settings (thresholds, datasets, processing choices). For example `v1`, then `v2` when a threshold or a process changes. The exact settings behind each name are written in `manifest.json`.

  ```
  SCRATCH_ROOT/
    raw.zarr/                      ← ingested sources, shared by all runs (change 5)
    runs/
      v1/
        run.zarr/                  ← every layer the run computes
          emit/…
          downscale/…
        attribution.parquet
        emissions-factors.parquet
        manifest.json              ← git SHA, settings and thresholds, dates, layer sources
      v2/                          ← same layout, different thresholds or processes
        …
  ```

- **One zarr for the whole run, on the lattice.** `run.zarr` covers the whole globe on the lattice of change 1, with all layers and all dimensions (x, y, year). One `open_zarr` gives everything. A layer on another grid (e.g. at MapSPAM resolution) goes in its own group with its own coordinates.

- **Read sources from `raw.zarr`.** Sources are not copied into the run; the run reads them from the shared `raw.zarr` (change 5), which is on the same lattice, so a cell (I, J) is the same place in both. The only source-derived layers stored in the run are the coarse sources brought to 30 m, because how to do that is a run setting; the manifest records the method.

- **Each tile writes its own region.** The run creates `run.zarr` once with the global shape and no data, then each worker writes its tile with `to_zarr(region=...)`, aligned to its shard (change 3). No two workers write the same object.

- **Track completion per tile with a marker**, written last (e.g. `runs/{name}/_done/emit/x279y082`). A rerun with the same name skips tiles that have one. A run covers the globe but can be filled step by step: computing more regions later just fills new parts of the same store.

- **Reuse earlier runs.** A run can start from another one (e.g. `v2` started with `--from-run v1`). Layers whose code and inputs didn't change are copied or linked from it, and only the rest is recomputed. This keeps the time savings of today's cache (e.g. not redoing the ~21 h emit), and the manifest records where each layer came from.

- **Keep tables as parquet.** Attribution rollups and the emissions-factor table aren't on the grid, so they stay as parquet files in the run folder, with readable names.

## 7. Two-level execution model for cloud runs (proposed by Claude)

**Description.**

- **Each zarr write starts its own dask cluster.** `storage.write_dask_dataset_to_zarr` starts an in-process `distributed.LocalCluster` sized by `NUMBER_OF_DASK_WORKERS`.
- **Writes are serialized.** Every write holds a global lock (`COMPUTE_LOCK`).
- **Stages fan out with threads on one host.** Attribute and ingest use thread pools over tiles.
- **The pipeline is single-host by design** (see `architecture.md`).

**Motivation.**

- **One host can't run the global corpus quickly.** A continent-scale harmonize alone takes ~53 h on one machine.
- **The lock blocks more parallelism.** Spreading tiles across more threads doesn't help while every write holds `COMPUTE_LOCK`.
- **A single global dask graph won't work either.** The graph alone would take gigabytes in the client. The whole job would also become one retry unit, and "skip an empty tile" couldn't be decided before every read was already in the graph.

**Proposed change.**

- **Outer level.** A scheduler (Cloud Batch, GKE Jobs, or Dask on Kubernetes) hands out 1° tiles as small, serializable task descriptions carrying only identifiers. The tile is the unit of retry and of failure.
- **Inner level.** Inside each task, a small dask graph runs with an explicit local scheduler, never the cluster's, to avoid a worker submitting work to itself. GeoCollision's benchmark found processes faster than threads (3.3× vs 2.3×) and lighter on memory.
- **Infrastructure:**
  - compute in the same region as the buckets;
  - spot VMs of about 4–8 vCPU and 16–32 GB;
  - writes made safe to retry: write the region first and the completion marker last;
  - a service account on the VMs instead of user application-default credentials.
- **GDAL settings for `/vsigs`:** keep `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`, and add `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES`, `GDAL_HTTP_MULTIPLEX=YES`, `VSI_CACHE=TRUE` and a per-worker `GDAL_CACHEMAX`.
- **Coarse layers.** MapSPAM and GPW livestock stay on their own coarse grids in their own stores. They're joined to the 30 m data at 0.25° or 1° boundaries, where all the grids meet exactly, rather than resampled to 30 m.

## 8. Store vectors as GeoParquet

**Description.**

- **Vectors are FlatGeobuf.** `VectorDataset.ingest_a_tile` converts each vector source with `geo.convert_vector_to_flatgeobuf` and writes `vector/{source}/{product}/{version}/whole-world/world.fgb`, with a spatial index (`SPATIAL_INDEX=True`).
- **The only vectors are the World Bank boundaries** (admin 0, 1 and 2), world-wide.
- **They are always read in full.** `worldbank_jurisdictions` loads them with `geopandas.read_file(path_to_fgb)`, then filters by country or tile in memory.

**Motivation.**

- **The spatial index is never used.** FlatGeobuf's strength is reading a small area over HTTP with its index. Here the whole file is read every time, so that strength brings nothing.
- **Full reads are slow in FlatGeobuf.** It is read row by row through GDAL and is not compressed. GeoParquet is columnar, compressed and read through Arrow, which is usually much faster for a full read, and can read only the needed columns.
- **Two tabular formats.** Every other table in the pipeline (ingested tables, attribution, emissions factors) is already parquet. GeoParquet keeps one format for everything that isn't a raster.

**Proposed change.**

- **Write vectors as GeoParquet.** In `convert_vector_to_flatgeobuf` (renamed), `to_file(..., driver="FlatGeobuf")` becomes `to_parquet(...)`, and in `worldbank_jurisdictions`, `read_file` becomes `geopandas.read_parquet`.
- **Use GeoParquet 1.1 with a bbox column and rows sorted by space**, so a small-area read stays possible if it's ever needed.
- **Nothing is lost for inspection.** GeoParquet still opens in QGIS and GDAL (≥ 3.5).

## 9. Rasterize every vector layer onto the lattice

**Description.**

- **Vectors are combined with rasters by clipping.** Today the only vector layers are the World Bank boundaries (admin 0, 1 and 2). For each tile and each jurisdiction that touches it, attribution clips the raster stack with the polygon (`geo.clip_dset`, which uses `rio.clip_box` then `rio.clip`), then sums the emissions inside it. See `jurisdictional_direct.py` and `statistical.py`.
- **There is no general rule for vectors.** `VectorDataset` only stores the polygons (change 8). How a polygon layer is turned into cells is decided by the code that uses it, at the moment it uses it.
- **More vector layers are likely.** Protected areas, concessions, farm or supply-shed boundaries, and peatland polygons are typical inputs for this kind of work. Each would today need its own clipping code.

**Motivation.**

- **Vectors are the only data that doesn't join by cell.** Every raster layer joins with the others by cell on the lattice (change 1). Vectors are the exception: they use a second, slower way of combining data, and each new vector layer adds more of it.
- **Clipping is repeated work.** Each run, each tile and each polygon clips the same geometry again, although vector sources change as rarely as raster sources.
- **Cell membership is decided silently.** Which 30 m cells belong to a polygon is decided inside `rio.clip` each time (a cell belongs to the polygon that contains its centre), and never stored or checked. Where the vector data has overlaps or gaps, a cell can be counted in two polygons, or in none, and nothing reports it.
- **It doesn't fit the raw and run stores.** With `raw.zarr` and `run.zarr` (changes 5 and 6), every input is an array on the lattice, except the vectors.

**Proposed change.**

- **Rasterize every vector layer at ingest.** Each vector source is burned into `raw.zarr` as an array on the lattice, in its own group, as GeoCollision does with `gdal_rasterize`. Examples: `worldbank/admin_0_id`, `worldbank/admin_1_id`, `worldbank/admin_2_id`, and later `protected_areas/id` or `concessions/id`. After ingest, a vector layer is used exactly like a raster layer.
- **Declare how each layer is burned.** The vector dataset definition states what the cell holds:
  - **an ID** (default): each cell carries the integer ID of the polygon that contains its centre, for layers that partition space, such as jurisdictions;
  - **a class or a flag**: each cell carries a value taken from an attribute (e.g. a protection category), for thematic layers;
  - **a priority rule for overlaps**: when polygons overlap (e.g. protected areas), the definition says which one wins, or stores a flag per polygon type instead of one ID.
- **Keep a lookup table per layer.** A small parquet table maps each integer ID to the polygon's attributes (`id`, `name`, level, …).
- **Keep the vectors.** The GeoParquet (change 8) keeps the exact geometry, names and areas, for reports and checks. The array is what processing uses.
- **Processing becomes a group-by or a mask.** Attribution groups the emissions by the value of the jurisdiction array; a thematic layer becomes a mask or a class in emit. Both are exact joins by cell, like every other layer, with no clipping.
- **Check every rasterization.** A cell is assigned to one polygon only, so edges are approximate at 30 m, and polygons smaller than a cell can disappear. At ingest, compare each polygon's area from the array with its vector area, list polygons that got no cell, and record both in the array's attributes.

**Measured.** We compared the Xu et al. global peatland map, stored as the source GeoPackage and as the rasterized layer on the 1/4000° lattice (from GeoCollision). The zarr is about 5.7× smaller:

|         | Zarr (`adastra-zarr/gcg4000.zarr/xu_peat`) | GeoPackage (`adastra-input-public-data/…/xu_peat/v1/_/raw`) |
| ------- | ------------------------------------------ | ----------------------------------------------------------- |
| Size    | **176 MiB**                                | **1,009 MiB**                                               |
| Objects | 6,557 shards of 1°                         | 1 file                                                      |
| Content | `uint8` per cell: 1 = peat, 0 = not peat   | 2.6 million polygons, with geometry and attributes          |

The two don't hold the same information. The GeoPackage keeps the exact polygon edges and their attributes; the zarr keeps only "peat or not" per 30 m cell, so edges are rounded to the cell and polygons smaller than a cell can disappear. But that one value per cell is all processing needs, and it compresses very well. Shards with no peat are not written at all, because the fill value (0) is the "not peat" value. This also shows why each layer must say what its fill value means: here 0 means "measured, not peat", not "no data", so a cell outside the source's coverage and a non-peat cell look the same.

## 10. Move the methodology rules to a config file

**Description.** The rules of the emissions core are written in Python. For example, `emit.DestinationDataset` and `DESTINATION_DATASET_TO_PREDICATE` say which layer claims a pixel as cropland or pasture (Descals oil palm, GACED30, GPW), in which year, and with which priority. Thresholds and constants (assessment year, lookback, carbon fractions) are module constants.

**Motivation.**

- **The rules are parameters, not logic.** Each destination is "layer + year + condition → flag", then grouped with a priority. Adding one means editing an enum, a dict and an assert in `emit.py`.
- **They are what makes one run different from another.** A new threshold or a new destination layer is a new run (`v2`, change 6), but today those settings are spread through the code instead of in one place the manifest can record.

**Proposed change.**

- **A config file for the rules**, read at the start of a run and copied into its manifest:

  ```yaml
  assessment_year: 2024
  destinations:            # order = bit order in `destination-dataset` and priority
    - {name: descals_oil_palm, group: cropland, layer: descals/planting_year,
       condition: {operator: between, min: 1, max: "{assessment_year}"}}
    - {name: liao_gaced30, group: cropland, layer: liao_gaced30/cropland,
       year: "{assessment_year}", condition: {operator: "==", value: CROPLAND}}
    - {name: gpw_grassland, group: pasture, layer: gpw/grassland_class,
       year: "{assessment_year}", condition: {operator: "==", value: CULTIVATED}}
  group_priority: [cropland, pasture]
  ```

- **Small generic functions in code**, called from the config, as in GeoCollision: condition → flag (`AttributeValueMapper`), first match wins (`AttributeSelector`), last departure year (`find_transitions`), lookup tables such as IPCC carbon by climate zone (`AttributeReclassifier`), group-by for attribution (`AggregatedParquet`).

- **Keep the real methodology in code.** Discounting, peatland occupation, dropped emissions and the statistical downscaling stay Python, with their tests. Only rules, thresholds and layer choices move to config.

- **Don't build a second generic engine.** GeoCollision is already the generic engine (lattice, store, transformers, config). cornerstone_luc should be one methodology on top of it, and other uses become new configs there, not changes here.

- **The destination order is part of the run.** It sets which bit each layer gets in `destination-dataset`, so it is recorded in the manifest, and changing it means a new run.

## 11. Say what "no value" means

**Description.** Harmonize turns every band's nodata into NaN (`geo.unify_dtype_and_no_data`), and emit treats NaN as "no data".

**Motivation.** "No value" can mean three different things: outside the source's coverage, measured but no data, or not ingested yet. Some layers also use their fill as a real value (in the Xu peat layer, 0 means "not peat"). With one NaN for all of them, a result can't tell them apart.

**Proposed change.** Each array in `raw.zarr` and `run.zarr` declares what its fill value means (e.g. `fill_is_a_value: true`), and each step declares how nulls spread (e.g. `null_policy: propagate` or `as_value`), as GeoCollision does. Coverage is recorded per layer, so "outside coverage" is a separate answer.

## 12. Read only what's needed

**Description.** Emit opens the whole harmonize stack for each tile, and every tile is read, including ocean.

**Motivation.** Most of the lattice is ocean, and many steps use only a few of the ~56 bands. Reading the rest costs time and requests for nothing.

**Proposed change.**

- **Open only the arrays a step uses.** With `raw.zarr` (change 5), each step lists its inputs and opens only those.
- **Walk back from the outputs**, as GeoCollision does: an array no output depends on is never read.
- **Skip empty tiles with a gate layer.** Every step first reads a land or coverage mask, and skips the tile if it holds nothing, before reading anything else.

## 13. Provenance by content

**Description.** The cache key is the function, its arguments and a hand-written `version=` number (`storage.get_cache_decorator`). It ignores the input data and the code itself.

**Motivation.** If someone changes the code or an input and forgets to bump `version`, the old result is returned with no error. `tools/measure-drift.py` warns about it: "a clean result can be a lie".

**Proposed change.** Key each result on its content: the identity of each input (the `raw.zarr` group and its release) and the identity of the code (as GeoCollision's `code_identity` does). Record both in the run manifest (change 6) for each layer. The manual `version=` goes away.

## 14. One dask scheduler per process (from Claude)

**Description.** `storage.write_dask_dataset_to_zarr` starts a new `distributed.LocalCluster` for every write, under a global lock (`COMPUTE_LOCK`).

**Motivation.** Starting a cluster for every write is slow, and the lock allows only one write at a time.

**Proposed change.** Start one local scheduler per process, and reuse it for every write. Once each tile is one shard (change 3), drop `COMPUTE_LOCK`.

## 15. One command per run

**Description.** A run is five commands (ingest, harmonize, emit, attribute, trace) that must be called in the right order. The only way to know what's done is to read the logs or recompute hashes.

**Motivation.** It's easy to forget a step, or to run one with different settings from the others. There's no simple answer to "what's done?".

**Proposed change.** One CLI with a few commands:

- `jdluc run v2 --config v2.yaml --countries BRA`: runs every step needed, in order;
- `jdluc status v2`: tiles done, missing and failed, per step;
- `jdluc export v2 emit --bbox …`: writes a COG for QGIS.

## 16. Check the config before running

**Description.** Errors in settings appear during the run, from `assert` statements, sometimes hours in.

**Motivation.** A typo or a wrong value should fail in seconds, not after a long run.

**Proposed change.** A `jdluc check v2.yaml` command validates the config from change 10 without opening any store: unknown keys, wrong types, unknown layers or classes, unreachable rules. `run` calls it first. This is the same idea as GeoCollision's `process --check`.

## 17. Read settings from environment variables

**Description.** `config.Config.from_dot_env` stops if there's no `.env` file (`raise_error_if_not_found=True`).

**Motivation.** Cloud containers usually get their settings as environment variables, not as a file.

**Proposed change.** Read environment variables first, and use `.env` only as a fallback for local work.

## 18. Use obstore for bucket access

**Description.** Every read and write to the GCS buckets goes through `fsspec` and `gcsfs` (a direct dependency in `pyproject.toml`), apart from GDAL:

- `storage.path_exists` and `storage.put_file` call `fsspec.url_to_fs` to check for and upload ingested COGs, FlatGeobufs and parquet files (`datasets/base.py`).
- `storage.open_zarr_to_dask_dataset` and `storage.write_dask_dataset_to_zarr` pass a bare `gs://…` string to `xarray.open_zarr` / `to_zarr`, which zarr resolves through fsspec and gcsfs.
- `pandas.read_parquet` / `to_parquet` on `gs://` URIs (the `cache_to_parquet` cacher, FAOSTAT, USDA NASS) also go through fsspec and gcsfs.
- GDAL and rasterio don't use fsspec: `storage.to_gdal_path` rewrites `gs://` to `/vsigs/`, and GDAL reads the bucket itself.

**Motivation.**

- **gcsfs is a slow layer for chunked I/O.** gcsfs is pure Python on top of `aiohttp`, and each request goes through fsspec's async-to-sync bridge. obstore is a thin Python binding over the Rust `object_store` crate (the one used by Apache Arrow, DataFusion and Delta Lake), which makes requests natively, with less overhead per request and real parallelism outside the GIL.
- **The request count is about to grow.** With 0.25° chunks and 1° shards (change 3), each read is a shard-index request plus a range request, and a run makes many of them over many small workers (change 7). Per-request overhead is what limits throughput there, more than bandwidth.
- **Zarr v3 supports it natively.** `zarr.storage.ObjectStore` wraps an obstore store directly, with no fsspec in between, and does concurrent range reads for sharded arrays.
- **Fewer and smaller dependencies.** gcsfs pulls in `aiohttp`, `google-auth`, `google-cloud-storage` and their dependency trees; obstore is one wheel with no Python dependencies. That's a smaller container image for cloud workers (change 7).
- **One way to talk to the bucket.** Today the bucket is reached through fsspec in some places and implicitly through zarr's own URL handling in others. A single `store` object, built once from the config, makes it explicit which bucket, prefix and credentials a run uses.

**Proposed change.**

- **Build the store once in `storage.py`.** A helper turns a root URI into an obstore store (`obstore.store.from_url("gs://bucket/prefix")`, or `GCSStore` for `gs://` and `LocalStore` for local paths, so local runs and tests keep working). Credentials come from application-default credentials or the VM's service account, as today.
- **Zarr through `zarr.storage.ObjectStore`.** `open_zarr_to_dask_dataset` and `write_dask_dataset_to_zarr` pass `store=zarr.storage.ObjectStore(store)` instead of a URI string. The same applies to `raw.zarr` and `run.zarr` (changes 5 and 6) once they exist.
- **Replace the fsspec helpers.** `path_exists` becomes `obstore.head` (catching "not found"), and `put_file` becomes `obstore.put` from the local file; there are no directories to create in an object store.
- **Parquet through the fsspec adapter, at first.** pandas and pyarrow still expect an fsspec filesystem. obstore ships one (`obstore.fsspec.FsspecStore`), which can be registered for `gs://` so `read_parquet` / `to_parquet` keep their current calls. Later, parquet files can be read and written as bytes with `obstore.get` / `put`, or through pyarrow's own GCS filesystem.
- **Leave GDAL alone.** GDAL reads `/vsigs/` with its own HTTP client, so `to_gdal_path` and the GDAL settings of change 7 don't change.
- **Then drop `gcsfs` from the dependencies**, and the `gcsfs` logger line in `write_dask_dataset_to_zarr`.
- **Measure before switching.** Time a full read of one harmonize zarr and one emit write on the same machine, with gcsfs and with obstore, and keep the numbers here, as for changes 5 and 9.
