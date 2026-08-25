"""
Zarr XYZ tile server for QGIS — FastAPI + lru_cache edition.

Install deps:
    pip install fastapi uvicorn[standard] zarr xarray mercantile matplotlib pillow numpy

Usage:
    python tile_server.py --zarr /path/to/data.zarr --var temperature --vmin 0 --vmax 30

Then in QGIS: Layer > Add Layer > Add XYZ Tile Layer
    URL: http://localhost:8000/tiles/{z}/{x}/{y}.png
"""

import argparse
import io
import logging
from functools import lru_cache

import mercantile
import numpy as np
import uvicorn
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 8000
TILE_SIZE = 256
COLORMAP = "viridis"

log = logging.getLogger(__name__)

# ── Module state (populated by main() before the server starts) ──────────────

_da: xr.DataArray | None = None
_vmin: float = 0.0
_vmax: float = 1.0
_resampling: int = Image.BILINEAR


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_dataset(zarr_path: str, variable: str) -> xr.DataArray:
    ds = xr.open_zarr(zarr_path, consolidated=False)
    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found. Available: {list(ds.data_vars)}"
        )
    return ds[variable].squeeze(drop=True)


# ── Rendering ─────────────────────────────────────────────────────────────────


def _find_dim(da: xr.DataArray, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in da.dims:
            return name
    raise ValueError(f"None of {candidates} found in dims {list(da.dims)}")


def _transparent_tile(tile_size: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((tile_size, tile_size, 4), dtype=np.uint8), "RGBA").save(
        buf, format="PNG"
    )
    return buf.getvalue()


def render_tile(
    da: xr.DataArray,
    west: float,
    south: float,
    east: float,
    north: float,
    tile_size: int,
    vmin: float,
    vmax: float,
    colormap: str,
    resampling: int,
) -> bytes:
    lat_dim = _find_dim(da, ("lat", "latitude", "y"))
    lon_dim = _find_dim(da, ("lon", "longitude", "x"))

    subset = da.sel({lat_dim: slice(north, south), lon_dim: slice(west, east)})
    if subset.sizes[lat_dim] == 0:  # ascending lat axis
        subset = da.sel({lat_dim: slice(south, north), lon_dim: slice(west, east)})

    data = subset.values.astype(float)  # ← zarr I/O happens here

    if data.size == 0 or np.all(np.isnan(data)):
        return _transparent_tile(tile_size)

    normed = Normalize(vmin=vmin, vmax=vmax, clip=True)(data)

    lat_vals = subset[lat_dim].values
    if len(lat_vals) > 1 and lat_vals[0] < lat_vals[1]:  # ascending → flip
        normed = np.flipud(normed)
        data = np.flipud(data)

    rgba = (colormaps[colormap](normed) * 255).astype(np.uint8)
    rgba[np.isnan(data), 3] = 0

    img = Image.fromarray(rgba, "RGBA").resize((tile_size, tile_size), resampling)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


# ── Cache ─────────────────────────────────────────────────────────────────────


@lru_cache
def _get_tile(z: int, x: int, y: int) -> bytes:
    log.debug("Cache miss — rendering %d/%d/%d", z, x, y)
    bounds = mercantile.bounds(mercantile.Tile(x, y, z))
    return render_tile(
        da=_da,
        west=bounds.west,
        south=bounds.south,
        east=bounds.east,
        north=bounds.north,
        tile_size=TILE_SIZE,
        vmin=_vmin,
        vmax=_vmax,
        colormap=COLORMAP,
        resampling=_resampling,
    )


# ── FastAPI app & routes ──────────────────────────────────────────────────────

app = FastAPI(title="Zarr tile server")


@app.get("/tiles/{z}/{x}/{y}.png", response_class=Response)
async def tile(z: int, x: int, y: int):
    try:
        png = _get_tile(z, x, y)
    except Exception as exc:
        log.exception("Tile render failed z=%d x=%d y=%d", z, x, y)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/cache/invalidate")
async def cache_invalidate():
    _get_tile.cache_clear()
    log.info("Tile cache cleared")
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    global _da, _vmin, _vmax, _resampling

    parser = argparse.ArgumentParser(description="Zarr XYZ tile server (FastAPI)")
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--var", required=True)
    parser.add_argument("--vmin", required=True, type=float)
    parser.add_argument("--vmax", required=True, type=float)
    parser.add_argument(
        "--nearest",
        action="store_true",
        help="Use nearest-neighbor resampling instead of bilinear",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    _da = load_dataset(args.zarr, args.var)

    lat_dim = _find_dim(_da, ("lat", "latitude", "y"))
    lon_dim = _find_dim(_da, ("lon", "longitude", "x"))
    y_max = float(_da[lat_dim].max())
    x_min = float(_da[lon_dim].min())

    log.info(
        "Loaded '%s' — shape %s, dtype %s, y_max %g, x_min %g",
        args.var,
        dict(_da.sizes),
        _da.dtype,
        y_max,
        x_min,
    )

    _vmin = args.vmin
    _vmax = args.vmax
    log.info("Color range: [%g, %g]", _vmin, _vmax)

    _resampling = Image.NEAREST if args.nearest else Image.BILINEAR
    log.info("Resampling: %s", "nearest" if args.nearest else "bilinear")

    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
