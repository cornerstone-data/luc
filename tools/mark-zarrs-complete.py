"""Mark zarrs written before the `complete` marker existed as complete, so they are not recomputed.

`ZarrCacher` counts a zarr as cached only once its `complete` marker exists, and zarrs written
before the marker have none. This trusts every one of them, including any that are actually
partial: delete a zarr you suspect instead. Only zarrs directly under the root are checked, which
is where `ZarrCacher` writes them.

Run it before the marker change merges, and once more after deploy to catch zarrs the old code
wrote in between. Run it only while nothing is writing to the root, since a zarr mid-write would be
marked complete. It lists what it would mark unless given `--apply`, and is idempotent.

  uv run python tools/mark-zarrs-complete.py
  uv run python tools/mark-zarrs-complete.py --apply
"""

import argparse
import logging
import os

import fsspec  # type: ignore[import-untyped]

from jdluc import config, storage

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", help="defaults to SCRATCH_ROOT from .env")
    parser.add_argument(
        "--apply", action="store_true", help="write the markers, not just list them"
    )
    args = parser.parse_args()
    root = args.root or config.Config.from_dot_env().scratch_root

    fs, path = fsspec.url_to_fs(url=root)
    cachers = [
        storage.ZarrCacher(
            hash_key=os.path.basename(os.path.dirname(zarr_json)).removesuffix(".zarr"),
            root=root,
        )
        for zarr_json in sorted(fs.glob(f"{path.rstrip('/'):s}/*.zarr/zarr.json"))
    ]
    unmarked = [c for c in cachers if not storage.path_exists(uri=c.complete_uri)]
    for cacher in unmarked:
        logger.info(
            f"{'Marking' if args.apply else 'Would mark'} {cacher.uri:s} complete"
        )
        if args.apply:
            fs.touch(fsspec.url_to_fs(url=cacher.complete_uri)[1])
    logger.info(
        f"{len(unmarked):d} of {len(cachers):d} zarrs {'marked' if args.apply else 'to mark'}"
    )


if __name__ == "__main__":
    main()
