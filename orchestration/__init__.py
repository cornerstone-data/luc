"""The glue between a workflow engine (Argo) and jdluc: one subcommand per workflow task.

What may live here: listing the units of work a run fans out over, and running exactly one
unit per call. What must not: anything that changes a result. Every argument that reaches a
cache key is passed straight through to jdluc, with jdluc's own defaults.

  python -m orchestration plan HND --methodology=STATISTICAL --output-dir=/plan
  python -m orchestration ingest-tile --dataset=GNW_TREE_COVER_LOSS --tile-id=10N_090W
  python -m orchestration compute HND --methodology=STATISTICAL --tile-id=10N_090W
  python -m orchestration trace HND --methodology=STATISTICAL

Each subcommand that takes countries takes `world` in their place, for every country and GNW
tile, as jdluc's `--backfill` covers.

Whole-world ingest needs no wrapper: the workflow runs jdluc's own CLI for it. A tile's ingest
has one, `ingest-tile`, so that its pod names the tile rather than the run's countries.

To add tasks of your own, wrap rather than edit: `__main__.build_parser` returns the parser and
its subparsers to add subcommands to, and `plan.get_task_to_items` the lists to add to before
`plan.write_task_to_items`. The example in `workflows/` shows how a workflow calls each
subcommand. See docs/orchestration.md.
"""
