"""Fit FDaP thresholds for chosen countries to FAOSTAT area harvested.

For every source file touching the countries, read the pixels whose centres fall in the file's own
degree, as ingest does, and add their hectares to one of 1,000 probability bins for the country and
year they fall in. Summed from the top, the bins give the area above every threshold. Each year's
threshold is where that area meets FAOSTAT's area harvested, interpolating log area against log
threshold between `LOWEST_THRESHOLD` and `fdap.HIGHEST_THRESHOLD`. Where the area can't reach
FAOSTAT's within that range, the threshold is the end of the range, marked OVER (the model maps
more than FAOSTAT even at the top) or UNDER.

Prints each country and year's FAOSTAT area, its area at the commodity's floor and the fitted
threshold, and writes every curve to `<commodity>-<countries>-curves.csv` in the working directory.

Run from the repo root. The source bucket is requester-pays and in us-central1, so run it on a VM
there; a country's files are a few hundred megabytes each:

  uv run python tools/fit-fdap-thresholds.py cocoa GIN SLE TGO
  uv run python tools/fit-fdap-thresholds.py coffee BDI RWA --threads 16
"""

import argparse
import collections
import concurrent.futures
import logging
import math

import geopandas
import numpy
import pandas
import rasterio
import rasterio.features
import rasterio.windows
import shapely
import xarray

from jdluc import emit, storage
from jdluc.datasets import faostat, fdap, ifpri_mapspam, worldbank_jurisdictions

logger = logging.getLogger(__name__)

BINS = 1000
# The bottom of the threshold sweep; its top is `fdap.HIGHEST_THRESHOLD`
LOWEST_THRESHOLD = 0.0025
COMMODITY_TO_FLOOR = {
    fdap.Commodity.COCOA: fdap.COCOA_PRECISION_RECALL_CROSSOVER,
    fdap.Commodity.COFFEE: fdap.COFFEE_PRECISION_RECALL_CROSSOVER,
}
COMMODITY_TO_FAOSTAT_NAME = {
    fdap.Commodity.COCOA: faostat.Crop.COCO.name,
    fdap.Commodity.COFFEE: ifpri_mapspam.Crop2000.COFF.name,
}
# Ad Astra fitted the 2024 band to FAOSTAT 2023, before FAOSTAT published 2024, and so does this
YEAR_TO_FAOSTAT_YEAR = {2018: 2018, 2020: 2020, 2022: 2022, 2024: 2023}


def get_iso_3166_to_hectares_per_bin(
    iso_3166s: list[str],
    latitude: int,
    longitude: int,
    national: geopandas.GeoSeries,
    uri: str,
) -> dict[str, numpy.ndarray]:
    """Each country's hectares in each probability bin, within one source file's own degree."""
    columns = fdap.get_pixel_range(start_degrees=longitude, stop_degrees=longitude + 1)
    rows = fdap.get_pixel_range(start_degrees=latitude, stop_degrees=latitude + 1)
    with (
        rasterio.Env(
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_NUM_THREADS="ALL_CPUS",
            GS_USER_PROJECT=fdap.get_billing_project(),
        ),
        rasterio.open(fp=storage.to_gdal_path(uri=uri)) as source,
    ):
        left = round(source.transform.c / fdap.PIXEL_DEGREES)
        top = round(source.transform.f / fdap.PIXEL_DEGREES)
        window = rasterio.windows.Window(
            col_off=columns.start - left,
            height=len(rows),
            row_off=top - rows.stop,
            width=len(columns),
        )
        # NaN pixels compare false at ingest, so they count as 0 here too
        probability = numpy.nan_to_num(source.read(1, window=window), nan=0.0)
        transform = rasterio.windows.transform(
            transform=source.transform, window=window
        )
    labels = rasterio.features.rasterize(
        dtype="uint8",
        fill=0,
        out_shape=probability.shape,
        shapes=[
            (national[iso_3166], idx) for idx, iso_3166 in enumerate(iso_3166s, start=1)
        ],
        transform=transform,
    )
    # Two columns, since the pixel width comes from the spacing of the x coordinates
    row_hectares = emit.get_hectares_per_pixel(
        darray=xarray.DataArray(
            coords={
                "x": transform.c + (numpy.arange(2) + 0.5) * transform.a,
                "y": transform.f
                + (numpy.arange(probability.shape[0]) + 0.5) * transform.e,
            },
            data=numpy.zeros(dtype="float32", shape=(probability.shape[0], 2)),
            dims=("y", "x"),
        )
    ).values[:, 0]
    hectares_per_bin = numpy.zeros((len(iso_3166s) + 1) * BINS)
    # In strips of rows, so the 64-bit bin and weight arrays stay small
    for start in range(0, probability.shape[0], 1024):
        strip = slice(start, start + 1024)
        bins = numpy.clip((probability[strip] * BINS).astype(numpy.int64), 0, BINS - 1)
        hectares_per_bin += numpy.bincount(
            (labels[strip].astype(numpy.int64) * BINS + bins).ravel(),
            minlength=hectares_per_bin.size,
            weights=numpy.broadcast_to(row_hectares[strip, None], bins.shape).ravel(),
        )
    return {
        iso_3166: hectares_per_bin[idx * BINS : (idx + 1) * BINS]
        for idx, iso_3166 in enumerate(iso_3166s, start=1)
    }


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "commodity", choices=sorted(commodity.value for commodity in fdap.Commodity)
    )
    parser.add_argument(
        "iso_3166s", nargs="+", type=worldbank_jurisdictions.iso_3166_str
    )
    parser.add_argument("--threads", default=8, type=int)
    args = parser.parse_args()
    commodity = fdap.Commodity(args.commodity)

    national = worldbank_jurisdictions.get_jurisdiction_for_admin_level(
        admin_level=worldbank_jurisdictions.AdminLevel.NATIONAL
    ).geometry
    index = fdap.get_index()
    sources = index[index["commodity"] == commodity]
    lat_lon_to_iso_3166s = {
        (latitude, longitude): [
            iso_3166
            for iso_3166 in args.iso_3166s
            if national[iso_3166].intersects(
                shapely.box(longitude, latitude, longitude + 1, latitude + 1)
            )
        ]
        for latitude, longitude in set(
            zip(sources["lat"].tolist(), sources["lng"].tolist(), strict=True)
        )
    }
    tasks = [
        (latitude, longitude, year, uri)
        for latitude, longitude, year, uri in zip(
            sources["lat"].tolist(),
            sources["lng"].tolist(),
            sources["year"].tolist(),
            sources["path"].tolist(),
            strict=True,
        )
        if lat_lon_to_iso_3166s[(latitude, longitude)]
    ]
    logger.info(f"Reading {len(tasks):d} source files for {args.iso_3166s}")

    iso_3166_year_to_hectares_per_bin: collections.defaultdict[
        tuple[str, int], numpy.ndarray
    ] = collections.defaultdict(lambda: numpy.zeros(BINS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_year = {
            executor.submit(
                get_iso_3166_to_hectares_per_bin,
                iso_3166s=lat_lon_to_iso_3166s[(latitude, longitude)],
                latitude=latitude,
                longitude=longitude,
                national=national,
                uri=uri,
            ): year
            for latitude, longitude, year, uri in tasks
        }
        for n, future in enumerate(
            concurrent.futures.as_completed(future_to_year), start=1
        ):
            for iso_3166, hectares_per_bin in future.result().items():
                iso_3166_year_to_hectares_per_bin[
                    (iso_3166, future_to_year[future])
                ] += hectares_per_bin
            if n % 20 == 0:
                logger.info(f"Read {n:d} of {len(tasks):d} source files")

    crops = faostat.load(dataset=faostat.CROP_DATASET).reset_index()
    area_harvested = crops[
        crops["commodity_name"] == COMMODITY_TO_FAOSTAT_NAME[commodity]
    ].set_index(["admin_id", "year"])["area_hectares"]
    thresholds = numpy.arange(BINS) / BINS
    in_range = (thresholds >= LOWEST_THRESHOLD) & (thresholds <= fdap.HIGHEST_THRESHOLD)
    fits: list[dict[str, float | int | str]] = []
    curves: list[pandas.DataFrame] = []
    for (iso_3166, year), hectares_per_bin in sorted(
        iso_3166_year_to_hectares_per_bin.items()
    ):
        # Each bin's lower edge, as a threshold, keeps that bin and every one above it
        hectares_above = hectares_per_bin[::-1].cumsum()[::-1]
        faostat_year = YEAR_TO_FAOSTAT_YEAR[year]
        faostat_hectares = (
            float(area_harvested[(iso_3166, faostat_year)])
            if (iso_3166, faostat_year) in area_harvested.index
            else 0.0
        )
        candidates, areas = thresholds[in_range], hectares_above[in_range]
        if faostat_hectares == 0:
            threshold, status = math.nan, "no FAOSTAT area"
        elif areas[-1] > faostat_hectares:
            threshold, status = fdap.HIGHEST_THRESHOLD, "OVER"
        elif areas[0] < faostat_hectares:
            threshold, status = LOWEST_THRESHOLD, "UNDER"
        else:
            # Area falls as the threshold rises, so interpolate over the reversed, positive part
            positive = areas > 0
            threshold = float(
                numpy.exp(
                    numpy.interp(
                        numpy.log(faostat_hectares),
                        numpy.log(areas[positive][::-1]),
                        numpy.log(candidates[positive][::-1]),
                    )
                )
            )
            status = "fitted"
        floor_hectares = float(
            hectares_above[round(COMMODITY_TO_FLOOR[commodity] * BINS)]
        )
        fits.append(
            {
                "iso_3166": iso_3166,
                "year": year,
                "faostat_year": faostat_year,
                "faostat_kha": faostat_hectares / 1e3,
                "floor_kha": floor_hectares / 1e3,
                "floor_ratio": (
                    floor_hectares / faostat_hectares
                    if faostat_hectares > 0
                    else math.nan
                ),
                "threshold": threshold,
                "status": status,
            }
        )
        curves.append(
            pandas.DataFrame(
                {
                    "iso_3166": iso_3166,
                    "year": year,
                    "threshold": thresholds,
                    "hectares_above": hectares_above,
                }
            )
        )
    pandas.concat(curves).to_csv(
        f"{commodity}-{'-'.join(args.iso_3166s)}-curves.csv", index=False
    )
    print(
        pandas.DataFrame(fits).to_string(
            float_format=lambda v: f"{v:,.3f}", index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
