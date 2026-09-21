import argparse
import curses
import logging
import os
import tempfile

import dask.diagnostics
import rio_cogeo.cogeo
import rio_cogeo.profiles
import xarray

from jdluc import storage

logger = logging.getLogger(__name__)


def select_variable_names(variable_names: list[str]) -> list[str]:
    def _run(stdscr: curses.window) -> list[str]:
        curses.curs_set(0)
        idx, top, chosen = 0, 0, dict.fromkeys(variable_names, False)
        header = "↑/↓ move · space toggle · a all · enter confirm · q quit"
        while True:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            stdscr.addnstr(0, 0, header, max_x - 1)
            rows = max_y - 2  # lines available for the list
            if idx < top:
                top = idx  # scroll up
            elif idx >= top + rows:
                top = idx - rows + 1  # scroll down
            for screen_row, i in enumerate(
                range(top, min(top + rows, len(variable_names)))
            ):
                n = variable_names[i]
                mark = "x" if chosen[n] else " "
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addnstr(screen_row + 2, 0, f"[{mark}] {n}", max_x - 1, attr)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(variable_names)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(variable_names)
            elif key == curses.KEY_NPAGE:
                idx = (idx + 5) % len(variable_names)
            elif key == curses.KEY_PPAGE:
                idx = (idx - 5) % len(variable_names)
            elif key == curses.KEY_HOME:
                idx = 0
            elif key == curses.KEY_END:
                idx = len(variable_names) - 1
            elif key == ord(" "):
                chosen[variable_names[idx]] = not chosen[variable_names[idx]]
            elif key == ord("a"):
                v = not all(chosen.values())
                chosen = dict.fromkeys(variable_names, v)
            elif key in (curses.KEY_ENTER, 10, 13):
                return [n for n in variable_names if chosen[n]]
            elif key in (27, ord("q")):
                return []

    return curses.wrapper(_run)


def write_variable_to_cog(
    darray: xarray.DataArray, overview_resampling: str, path_to_cog: str
) -> None:
    import rioxarray  # noqa

    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(p=path_to_cog), suffix=".tif"
    ) as tif:
        with dask.diagnostics.ProgressBar(dt=5, minimum=1):
            logger.info("Writing to GeoTIFF")
            darray.rio.to_raster(
                tif.name,
                BIGTIFF="IF_SAFER",
                blockxsize=512,
                blockysize=512,
                compress="ZSTD",
                driver="GTiff",
                dtype="float32",
                lock=True,
                num_threads="all_cpus",
                tiled=True,
            )
        logger.info("Converting to COG")
        rio_cogeo.cogeo.cog_translate(
            allow_intermediate_compression=True,
            config={
                "GDAL_NUM_THREADS": "ALL_CPUS",
                "GDAL_TIFF_INTERNAL_MASK": True,
            },
            dst_kwargs=dict(rio_cogeo.profiles.DEFLATEProfile())
            | {"BIGTIFF": "IF_SAFER", "predictor": 3},
            dst_path=path_to_cog,
            in_memory=False,
            overview_resampling=overview_resampling,
            quiet=False,
            source=tif.name,
            use_cog_driver=True,
        )
        logger.info(f"Finished writing to {path_to_cog=:s}")


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_zarr", type=os.path.expanduser)
    parser.add_argument("output_dir", type=os.path.expanduser)
    parser.add_argument("--dump-all-bands", action="store_true")
    parser.add_argument(
        "--overview-resampling",
        choices=("average", "mode", "nearest"),
        default="average",
        help="how overviews reduce; use mode or nearest for class codes",
    )
    args = parser.parse_args()

    os.makedirs(exist_ok=True, name=str(args.output_dir))
    dset = storage.open_zarr_to_dask_dataset(path_to_zarr=str(args.path_to_zarr))

    for variable_name in select_variable_names(sorted(map(str, dset))):
        logger.info(f"Dumping {variable_name=:s}")
        write_variable_to_cog(
            darray=dset[variable_name],
            overview_resampling=str(args.overview_resampling),
            path_to_cog=os.path.join(str(args.output_dir), f"{variable_name:s}.tif"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
