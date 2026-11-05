# Running the pipeline on a cluster

The pipeline runs on a single host by design (see [`architecture.md`](architecture.md#single-host-by-design)), and nothing here changes that. The `orchestration` package is an optional layer for running it on Kubernetes with [Argo Workflows](https://argoproj.github.io/workflows/), one 10° tile per pod, so that a country or a global backfill takes hours instead of days. `jdluc` runs unmodified inside the pods. Argo takes over only the driver's job: sequencing, fan-out, retries and visibility.

## Design

**One tile per pod, and no Dask cluster.** The pipeline processes each 10° tile independently, and one tile's working set fits one high-memory node. So the unit of distribution is a whole tile: a pod runs a tile's stages, each with its own in-process `LocalCluster`, exactly as a laptop run does. Nothing crosses between tiles except trace's final merge, over kilobyte-scale parquets. A shared Dask scheduler would add version skew between driver and workers and a scheduler that a heavy tile can destabilise, and a Dask failure would then look like a pipeline failure. Tasks are hour-long with tens of GB working sets, and they're already parallel inside the pod, so a distributed Dask cluster is the wrong granularity anyway.

**The cache is the resume mechanism.** Every stage is wrapped in `@storage.cache_to_zarr` / `@storage.cache_to_parquet`, writing idempotently to `SCRATCH_ROOT`. Retrying anything skips finished work, so the orchestration can stay simple. A zarr counts as cached only once its `.complete` marker is written, so a pod killed mid-write leaves nothing a later pod would mistake for a result.

**`orchestration` may not change a result.** It is one subcommand per workflow task, and every cache-key argument goes straight through to `jdluc`, with `jdluc`'s own defaults. An import-linter contract keeps `jdluc` from importing it.

**Ingest is separate from compute.** They are different workloads:

|                       | Ingest                            | Compute                   |
| --------------------- | --------------------------------- | ------------------------- |
| Bound by              | network and disk                  | memory and CPU            |
| Needs source-API keys | **yes** (USDA, Harvard Dataverse) | **no** (reads warm cache) |
| Cadence               | once per country × source version | every code change         |

So only the ingest pods carry the keys, a flaky third-party source fails an ingest task rather than a compute run, and one warm `INGEST_ROOT` serves many compute runs. A laptop run ingests lazily, from inside harmonize; a workflow ingests up front instead, and `compute` runs harmonize with `skip_ingest=True`. That needs the list of datasets a run reads, which no `jdluc` constant declares, so `orchestration/plan.py` keeps one by hand. When `jdluc` starts reading a new dataset, add it there.

## The DAG

[`orchestration/workflows/cornerstone.example.yaml`](../orchestration/workflows/cornerstone.example.yaml) is an Argo `WorkflowTemplate`, with placeholders for the image, service account and storage class. Each run covers its `scope`: one country, or `world` for every country and GNW tile, as `--backfill` does locally.

```
plan -+-> ingest-world[dataset] -+-> tile[tile] --> trace
      +--------------------------+

tile:  ingest[dataset] -------------------+-> compute
       ingest-allowed-failures[dataset] --+
```

| Task                      | Pods                 | Runs                                                                                                      | Keys |
| ------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------- | ---- |
| `plan`                    | 1                    | `python -m orchestration plan`: ingests the boundaries if absent, then writes the JSON lists to loop over | no   |
| `ingest-world`            | 1 per dataset        | `jdluc/ingest.py NAME --backfill`, for each whole-world dataset                                           | yes  |
| `ingest`                  | 1 per dataset × tile | `python -m orchestration ingest-tile`: one dataset on one tile                                            | yes  |
| `ingest-allowed-failures` | 1 per dataset × tile | The same, for the datasets in `allow-failures`                                                            | yes  |
| `compute`                 | 1 per tile           | `python -m orchestration compute`: harmonize → emit → attribute, for each country the tile touches        | no   |
| `trace`                   | 1                    | `python -m orchestration trace`: the cross-tile merge, written as the emissions-factor table              | no   |

- **The tile list is derived, never typed in.** `plan` lists the tiles the scope's boundaries touch, the same derivation the `jdluc` CLIs use.
- **A failed tile fails only itself.** Each tile is a DAG of its own. A failed required ingest leaves that tile's compute unscheduled, and the other tiles carry on.
- **Ingest fails loudly, with exceptions.** A missing tile would otherwise reach harmonize's `ignore_missing_tiles` and become an all-nodata band without complaint. The datasets in `allow-failures` are the exception: by default only `DESCALS_OIL_PALM`, which is published only where oil palm was found, so most tiles have none. Compute runs whether or not those ingests fail.
- **`compute` checks its inputs.** With `--check-ingests` it raises before harmonize if a required ingest is missing from the ingest root, as a second line of defence behind the DAG.
- **`trace` needs every tile.** A table missing tiles would pass for a whole one, so with any tile failed `trace` doesn't run, and the run ends Failed. Once the failed tiles are fixed, `argo retry` reruns them, then `trace`.
- **Tiles wait for `ingest-world`.** That takes minutes once the whole-world datasets are cached.

The flat alternative, one fan-out per stage across every tile, is simpler to read in the Argo UI, but one failed (dataset, tile) ingest then stops compute on every tile. `plan` writes the flat lists too (`ingest-tiles`, `tiles`, …), so a flat workflow needs no further code.

## Setup

**The image.** The repo-root `Dockerfile` builds it. Build from a full clone, not a git worktree, since the image carries `.git` for provenance (`utils.get_git_version`). Tag the image with the commit's SHA, and set the template's `image` to it.

**Identity.** The pods need read/write access to `INGEST_ROOT`, `SCRATCH_ROOT` and `EXPORT_ROOT`. On a cloud provider, bind the template's Kubernetes service account to a cloud identity (on GKE, Workload Identity), so that no storage keys go in the image or the Secrets.

**The two Secrets.** `Config.from_dot_env()` reads a `.env` *file*, so each pod mounts a Secret at `/app/.env`. Start from `.env.example`, keep the real file outside the repo, and fill in every field. Then make a copy with the two API-key fields present but blank, so that only the ingest pods get the keys:

```bash
kubectl -n argo create secret generic cornerstone-env --from-file=.env=cloud.env
kubectl -n argo create secret generic cornerstone-env-nokeys --from-file=.env=cloud-nokeys.env
```

**Disk is what limits speed.** Ingest writes tens of GB of uncompressed GeoTIFF locally before converting it to a COG. On a node's shared boot disk, those writes can throttle the whole node to a crawl. So each pod gets its own SSD-backed generic ephemeral volume at `/localtmp`, with `TMPDIR` and `CPL_TMPDIR` pointed at it, created and deleted with the pod (on GKE, the built-in `premium-rwo` class). Don't shrink these volumes: SSD throughput scales with size. Each container also has a small `ephemeral-storage` request and limit, which covers only its own overlay, so a stray writer evicts its own pod and not the node.

**Pod sizing.** `compute` requests 56 GiB, with peaks near 40 GiB observed: one tile per 64 GiB node, with room for system pods. The other tasks fit in 24 GiB. Spot or preemptible nodes work well, since a reclaimed node only fails its pod and `argo retry` resumes from the cache. Add their `nodeSelector` and `tolerations` under `templateDefaults`.

## Running

```bash
argo template create orchestration/workflows/cornerstone.example.yaml
argo submit --from workflowtemplate/cornerstone -p scope=HND --generate-name cornerstone-hnd-
```

Parameters: `scope` (one ISO 3166 alpha-3 code, or `world`), `methodology` (`STATISTICAL`, the default, or `JURISDICTIONAL_DIRECT`, which is US-only) and `allow-failures` (comma-separated tiled datasets, by default `DESCALS_OIL_PALM`). The template's `parallelism` and `activeDeadlineSeconds` are sized for one country; raise them for `world`.

`trace` writes `{EXPORT_ROOT}/emissions-factors/{scope}-{methodology}.parquet`. Intermediate results stay under `INGEST_ROOT` and `SCRATCH_ROOT`.

**Reruns are cheap.** Rerunning a country, or `argo retry` after a failure, skips everything already cached. A pod killed mid-write leaves a zarr without its `.complete` marker, and the next attempt to write it raises rather than reading back missing chunks. Once no pod is writing that zarr, delete it by hand to let it be recomputed.

**Gotchas:**

- **`compute` and `trace` must use the same methodology.** It is a cache-key argument, so a mismatch doesn't fail: it silently recomputes every tile inside the single `trace` pod. The template passes one `methodology` to both, so this bites only when running `python -m orchestration` by hand.
- **Storage, not compute, is the cost that recurs.** Per-tile intermediates in `SCRATCH_ROOT` are hundreds of GiB. Before a large run, decide their retention, for example with bucket lifecycle rules.
- **A warm cache can fake a fast run.** When measuring timings, make sure the run does real work.

## Extending it

To add tasks of your own, such as a per-tile export after `compute`, wrap the package rather than editing it:

- **Subcommands.** `orchestration.__main__.build_parser()` returns the parser and its subparsers. Add a subcommand with `add_subparser` and the `add_scope` / `add_methodology` / `add_tile_id` helpers, then call `run(parser)`. A subcommand's flags are exactly its function's parameters.
- **Plan lists.** `orchestration.plan.get_task_to_items()` returns the lists without writing them. Add lists of your own, or keys to each `tile-units` item, then write them with `write_task_to_items()`. Call `check_allow_failures()` and `ingest_boundaries()` first, as `write_plan` does.
- **The workflow.** Call the template's tasks with `templateRef: {name: cornerstone, template: compute}` (and likewise `ingest-tile`, `trace`, …) from DAGs of your own, instead of copying the file.
