"""Contrast emit's existing bands with its `cie-` bands (Crop-Independent Emissions) on one tile.

Pixels are grouped by CIE conversion source and by whether the old record fired a conversion
(i.e. a destination claimed the pixel). For each group: pixel count, hectares, and the hectare-
weighted totals of the old undiscounted pools (summed over spans) beside CIE's. Where a
conversion fired, the old vegetation must equal CIE's and the year must match; the residuals
are printed so any disagreement is visible. Where none fired, CIE carries emissions the old
bands charge to nobody (and report, discounted, as `dropped-emissions`).

Last, every existing band is rebuilt from the `cie-` bands by `emit.derive_from_cie`, and the
largest absolute difference from the stored band is printed; each should be ~0.

Must run from the repo root, so Config finds .env.

  uv run python tools/compare-emit-layers.py 10N_010W
  uv run python tools/compare-emit-layers.py --zarr /path/to/emit.zarr
"""

import argparse

import dask
import xarray

from jdluc import emit, storage

SPANS = [
    f"{before:d}-{after:d}" for before, after in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT
]
TCO2E = "tco2e-per-ha"


def get_rows_checks(
    dset: xarray.Dataset,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    source = dset["cie-conversion-source"]
    fired = dset["conversion"] != emit.Conversion.NONE
    ha = dset["hectares-per-pixel:ha"]
    old_vegetation = sum(dset[f"vegetation-emissions:{TCO2E}:{span}"] for span in SPANS)
    old_soil = sum(dset[f"soil-emissions:{TCO2E}:{span}"] for span in SPANS)
    new_vegetation = dset[f"cie-vegetation-emissions-undiscounted:{TCO2E}"]
    new_mineral = dset[f"cie-mineral-soil-carbon-at-risk:{TCO2E}"]
    new_peat = dset[f"cie-peat-transformation-emissions-undiscounted:{TCO2E}"]

    lazy: list[dict[str, object]] = []
    for member in emit.ConversionSource:
        for is_fired in (True, False):
            mask = (source == member) & (fired if is_fired else ~fired)
            lazy.append(
                {
                    "source": member.name.lower(),
                    "fired": is_fired,
                    "pixels": mask.sum(),
                    "ha": ha.where(mask).sum(),
                    "old veg t": (ha * old_vegetation).where(mask).sum(),
                    "cie veg t": (ha * new_vegetation).where(mask).sum(),
                    "old soil t": (ha * old_soil).where(mask).sum(),
                    "cie mineral t": (ha * new_mineral).where(mask).sum(),
                    "cie peat t": (ha * new_peat).where(mask).sum(),
                }
            )
    checks = {
        "max |old veg - cie veg| where fired": abs(old_vegetation - new_vegetation)
        .where(fired)
        .max(),
        "year mismatches where fired": (
            dset["conversion-year"] != dset["cie-conversion-year"]
        )
        .where(fired, other=False)
        .sum(),
        "fired pixels with no CIE source": (
            fired & (source == emit.ConversionSource.NONE)
        ).sum(),
    }
    derived = emit.derive_from_cie(dset=dset)
    checks |= {
        f"max |derived - existing| {name:s}": abs(derived[name] - dset[name]).max()
        for name in map(str, derived.data_vars)
    }
    rows, checks = dask.compute(lazy, checks)
    return rows, {name: float(value) for name, value in checks.items()}


def print_rows_checks(rows: list[dict[str, object]], checks: dict[str, float]) -> None:
    print("| " + " | ".join(rows[0]) + " |")
    print("|" + "---|" * len(rows[0]))
    for row in rows:
        cells = (v if isinstance(v, str | bool) else f"{v:,.0f}" for v in row.values())
        print("| " + " | ".join(map(str, cells)) + " |")
    print()
    for name, value in checks.items():
        print(f"- {name}: {value:g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("tile_id", nargs="?", help="run (or read the cache of) emit")
    group.add_argument("--zarr", help="an emit zarr already on disk")
    args = parser.parse_args()
    dset = (
        storage.open_zarr_to_dask_dataset(path_to_zarr=args.zarr)
        if args.zarr
        else emit.workflow(tile_id=args.tile_id)
    )
    print_rows_checks(*get_rows_checks(dset=dset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
