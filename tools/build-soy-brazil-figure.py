"""Render the methodology-walkthrough figure for soy in Brazil from local COGs.

Writes one file, ``soy-brazil-matopiba-methodology.png``: the four-panel hero that
``README.md`` and ``docs/methodology.md`` both carry.  What the land became and the year it
changed sit above soybean area before and after, with emissions as a full-height anchor.

The COGs come from ``tools/dump-zarr-to-cogs.py``: EPSG:4326, float32, NaN nodata, with
overviews.  Only a windowed, decimated view over the AOI is read, so a tile-sized raster
renders in seconds.

The Matopiba AOI straddles two ten-degree tiles, `00N_050W` and `10S_050W`, and the figure needs
two zarrs from each: `emit.workflow` for the conversion bands and `harmonize.workflow` at MAPSPAM
resolution for the soybean areas.  Every step below runs from the repo root.

1.  Produce the zarrs for both tiles, if the scratch root does not already hold them.  Each run
    logs the URI it wrote as ``Finished writing to path_to_zarr=...`` -- that URI is what step 2
    takes, and reading it off the log beats rebuilding the cache key by hand.

2.  Dump the bands to COGs, one directory per tile.  ``dump-zarr-to-cogs.py`` opens one zarr and
    offers its variables in a curses picker -- space to toggle, enter to confirm.

    ``conversion`` and ``conversion-year`` are a class code and a year, so they need
    ``--overview-resampling mode``: the default averages, the decimated read below pulls from
    those overviews, and one converted pixel among a hundred averages to a fraction, which
    survives the "is it NONE" test and lands in the first class -- painting the whole frame.
    ``emissions-per-hectare:tco2e-per-ha`` and the two soybean bands are continuous, so they take
    the default.  Three runs per tile::

        DUMP="uv run python tools/dump-zarr-to-cogs.py"
        $DUMP --overview-resampling mode <emit zarr for 00N_050W>     cogs/00N_050W
        $DUMP                           <emit zarr for 00N_050W>      cogs/00N_050W
        $DUMP                           <mapspam zarr for 00N_050W>   cogs/00N_050W

    and the same three for `10S_050W` into `cogs/10S_050W`.

3.  Stitch each band across the two tiles.  `cog()` below accepts a `.tif.vrt` in place of a
    `.tif`, so nothing has to be renamed::

        cd cogs
        for band in conversion conversion-year emissions-per-hectare:tco2e-per-ha \
                    ifpri:mapspam-physical-area-2000:soybean:ha \
                    ifpri:mapspam-physical-area-2020:soybean:ha; do
            gdalbuildvrt "$band.tif.vrt" 00N_050W/"$band".tif 10S_050W/"$band".tif
        done
        cd ..

4.  Render, straight into the directory the docs read from::

        uv run --with matplotlib --with pillow python tools/build-soy-brazil-figure.py \
            --data-dir cogs --outdir docs/figures

`emit` publishes no land class per year, so the categorical panel is the `conversion` band
-- which of the five conversions fired on each pixel -- and the panel beside it is
`conversion-year`, the year that pixel's source class ended.  Both the conversion codes and the
lookback are imported rather than mirrored, so a change to either cannot leave this behind.
"""

import argparse
import dataclasses
import logging
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import rasterio
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds

from jdluc import emit

logger = logging.getLogger(__name__)

Conversion = emit.Conversion
# The lookback opens at the assessment year less twenty, and nothing can have departed in that
# first year -- a departure needs the year before it to compare against
LOOKBACK_YEARS = emit.LOOKBACK_YEARS_RANGE[1:]
# A conversion is charged to the span holding its year, and each span carries its own discount
# weight, so colouring by span says what a continuous ramp could not: how heavily that pixel
# counts. Sorted oldest first, and shaded light to dark with recency.
SPANS = sorted(emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT)


# Hotter reads as more carbon released: forest sources red and orange, grassland tan and olive
CONVERSION_COLOR = {
    Conversion.FOREST_TO_CROPLAND: "#a11d33",
    Conversion.FOREST_TO_PASTURE: "#e2622b",
    Conversion.RANGELAND_TO_CROPLAND: "#f4a52a",
    Conversion.RANGELAND_TO_PASTURE: "#c9b072",
    Conversion.PASTURE_TO_CROPLAND: "#8aa14f",
}
# Fully transparent RGBA, for the pixels no conversion reached
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
EMISSIONS_LABEL = "Land-use-change emissions (tonnes CO₂-equivalent per hectare)"
SOYBEAN_LABEL = "Soybean area (hectares per 10-kilometre cell)"

CONVERSION_LABEL = {
    Conversion.FOREST_TO_CROPLAND: "Forest → cropland",
    Conversion.FOREST_TO_PASTURE: "Forest → pasture",
    Conversion.RANGELAND_TO_CROPLAND: "Rangeland → cropland",
    Conversion.RANGELAND_TO_PASTURE: "Rangeland → pasture",
    Conversion.PASTURE_TO_CROPLAND: "Pasture → cropland",
}


# Named soy-region AOIs as (min_lon, min_lat, max_lon, max_lat).
# Matopiba, the newest soy frontier and the most in-window Cerrado conversion, as
# (min_lon, min_lat, max_lon, max_lat).  It straddles the two tiles named above.
MATOPIBA = (-49.0, -14.0, -43.0, -6.0)
# The docs carry this file, so its bytes are a repo cost.  Rendering large and downsampling beats
# rendering small -- matplotlib's text resamples better than it hints at a low dpi -- and the
# 8-bit palette is where the saving is: 4532 px RGBA is 4.90 MB, 2000 px paletted 0.72 MB, for a
# mean error of 0.8/255.  Three continuous ramps share the 256 entries, so do not go below them.
FIGURE_WIDTH_PIXELS = 2000
FIGURE_COLOURS = 256
FIGURE_NAME = "soy-brazil-matopiba-methodology.png"


@dataclasses.dataclass
class Layer:
    """A windowed, decimated read of one COG, ready for imshow."""

    data: numpy.ndarray
    extent: tuple[float, float, float, float]  # (left, right, bottom, top)


def read_window(
    path: str, bbox: tuple[float, float, float, float], width: int
) -> Layer:
    """Read `path` over `bbox` decimated to roughly `width` px wide.

    Decimated reads pull from the COG's overviews, so this is cheap regardless
    of the source raster's full size. `bbox` is (min_lon, min_lat, max_lon, max_lat).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    with rasterio.open(path) as ds:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=ds.transform)
        scale = width / window.width
        out_shape = (
            max(1, round(window.height * scale)),
            max(1, round(window.width * scale)),
        )
        data = ds.read(
            1, out_shape=out_shape, resampling=Resampling.nearest, window=window
        )
        left, bottom, right, top = window_bounds(window, ds.transform)
    logger.info("read %s -> %s", os.path.basename(path), data.shape)
    return Layer(data=data, extent=(left, right, bottom, top))


def conversion_artists() -> tuple[
    matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm, list[Line2D]
]:
    """Discrete colormap + norm keyed to Conversion codes, plus legend handles."""
    conversions = sorted(CONVERSION_COLOR, key=lambda c: c.value)
    cmap = matplotlib.colors.ListedColormap(
        [CONVERSION_COLOR[c] for c in conversions]
    ).with_extremes(bad=TRANSPARENT)
    boundaries = [c.value - 0.5 for c in conversions] + [conversions[-1].value + 0.5]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N)
    handles = [
        Line2D(
            [],
            [],
            label=CONVERSION_LABEL[c],
            linestyle="",
            marker="s",
            markeredgecolor="none",
            markerfacecolor=CONVERSION_COLOR[c],
            markersize=10,
        )
        for c in conversions
    ]
    return cmap, norm, handles


def draw_conversion(ax: plt.Axes, layer: Layer, title: str) -> None:
    """Draw which conversion fired on each pixel; the pixels none reached stay transparent."""
    cmap, norm, _ = conversion_artists()
    data = numpy.where(layer.data == Conversion.NONE, numpy.nan, layer.data)
    ax.imshow(
        data,
        cmap=cmap,
        extent=layer.extent,
        interpolation="nearest",
        norm=norm,
        origin="upper",
    )
    _style_geo_axes(ax, title, layer.extent)


def span_artists() -> tuple[
    matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm, list[Line2D]
]:
    """Discrete colormap keyed to the span index, plus legend handles reading oldest first."""
    colours = matplotlib.colormaps["magma"](numpy.linspace(0.75, 0.2, len(SPANS)))
    cmap = matplotlib.colors.ListedColormap(colours).with_extremes(bad=TRANSPARENT)
    norm = matplotlib.colors.BoundaryNorm(
        [index - 0.5 for index in range(len(SPANS) + 1)], cmap.N
    )
    handles = [
        Line2D(
            [],
            [],
            label=f"{before + 1:d}-{after:d}",
            linestyle="",
            marker="s",
            markeredgecolor="none",
            markerfacecolor=colours[index],
            markersize=10,
        )
        for index, (before, after) in enumerate(SPANS)
    ]
    return cmap, norm, handles


def draw_conversion_span(ax: plt.Axes, layer: Layer, title: str) -> None:
    """Which five-year span each converted pixel was charged to; zero means it never converted."""
    cmap, norm, _ = span_artists()
    data = numpy.full(layer.data.shape, numpy.nan)
    for index, (before, after) in enumerate(SPANS):
        # Open below and closed above, exactly as `emit.get_span_to_charge` bins them
        data = numpy.where((layer.data > before) & (layer.data <= after), index, data)
    ax.imshow(
        data,
        cmap=cmap,
        extent=layer.extent,
        interpolation="nearest",
        norm=norm,
        origin="upper",
    )
    _style_geo_axes(ax, title, layer.extent)


def draw_emissions(
    ax: plt.Axes, layer: Layer, title: str, vmax: float
) -> matplotlib.image.AxesImage:
    cmap = matplotlib.colormaps["magma"].with_extremes(bad=TRANSPARENT)
    im = ax.imshow(
        layer.data,
        cmap=cmap,
        extent=layer.extent,
        interpolation="nearest",
        origin="upper",
        vmax=vmax,
        vmin=0.0,
    )
    _style_geo_axes(ax, title, layer.extent)
    return im


def _style_geo_axes(
    ax: plt.Axes, title: str, extent: tuple[float, float, float, float]
) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.tick_params(labelsize=8)
    # At-scale in plate carrée: a degree of longitude is cos(lat) shorter on the
    # ground than a degree of latitude, so stretch the y-axis by 1/cos(mean_lat).
    mean_lat = (extent[2] + extent[3]) / 2.0
    ax.set_aspect(1.0 / math.cos(math.radians(mean_lat)))


def shrink_in_place(path: str) -> None:
    """Downsample and palettise the rendered PNG, reporting what it cost."""
    from PIL import Image

    before = os.path.getsize(path)
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
    height = round(rgb.height * FIGURE_WIDTH_PIXELS / rgb.width)
    resized = rgb.resize((FIGURE_WIDTH_PIXELS, height), Image.LANCZOS)
    resized.quantize(colors=FIGURE_COLOURS, method=Image.MEDIANCUT).save(
        path, optimize=True
    )
    logger.info(
        f"Shrank {path:s} from {before / 1e6:.2f} MB to "
        f"{os.path.getsize(path) / 1e6:.2f} MB "
        f"({FIGURE_WIDTH_PIXELS:d} px wide, {FIGURE_COLOURS:d} colours)"
    )


def robust_vmax(arr: numpy.ndarray, percentile: float = 98.0) -> float:
    finite = arr[numpy.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return 1.0
    return float(numpy.nanpercentile(finite, percentile))


def main() -> None:
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", default=".", help="dir holding the COG .tif files"
    )
    parser.add_argument("--outdir", default="figures", help="dir to write the PNG into")
    parser.add_argument(
        "--bbox",
        help="override the named AOI with an explicit bbox",
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        nargs=4,
        type=float,
    )
    parser.add_argument(
        "--width", default=2000, help="target output width in px", type=int
    )
    parser.add_argument("--dpi", default=300, type=int)
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else MATOPIBA
    os.makedirs(args.outdir, exist_ok=True)

    def cog(name: str) -> str:
        # An AOI straddling two tiles is stitched with `gdalbuildvrt`, so accept a VRT too
        for candidate in (name, f"{name:s}.vrt"):
            path = os.path.join(args.data_dir, candidate)
            if os.path.exists(path):
                return path
        raise SystemExit(f"neither {name:s} nor {name:s}.vrt is in {args.data_dir:s}")

    def out(name: str) -> str:
        return os.path.join(args.outdir, name)

    logger.info("bbox=%s width=%s", bbox, args.width)

    # --- read layers (windowed, decimated) ---
    conversion = read_window(cog("conversion.tif"), bbox, args.width)
    conversion_year = read_window(cog("conversion-year.tif"), bbox, args.width)
    emissions = read_window(
        cog("emissions-per-hectare:tco2e-per-ha.tif"), bbox, args.width
    )
    # MapSPAM is ~10 km, so give it a coarser read at the same extent
    soy_pair = {
        y: read_window(
            cog(f"ifpri:mapspam-physical-area-{y}:soybean:ha.tif"),
            bbox,
            min(args.width, 400),
        )
        for y in (2000, 2020)
    }

    vmax = robust_vmax(emissions.data)
    _, _, legend_handles = conversion_artists()

    # --- 1. methodology strip: what converted and when (top), soy before/after (bottom),
    #        emissions as a full-height anchor column on the right. ---
    soy_pair_cmap = matplotlib.colormaps["YlGn"].with_extremes(bad=TRANSPARENT)
    soy_pair_vmax = robust_vmax(
        numpy.concatenate([soy_pair[2000].data.ravel(), soy_pair[2020].data.ravel()])
    )

    fig = plt.figure(figsize=(15, 11), layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=(2.0, 1.2))
    left = outer[0, 0].subgridspec(2, 2)
    ax_lc0, ax_lc1 = fig.add_subplot(left[0, 0]), fig.add_subplot(left[0, 1])
    ax_sy0, ax_sy1 = fig.add_subplot(left[1, 0]), fig.add_subplot(left[1, 1])
    ax_em = fig.add_subplot(outer[0, 1])

    draw_conversion_span(ax_lc0, conversion_year, "")
    _, _, span_handles = span_artists()
    ax_lc0.legend(
        fontsize=7, framealpha=0.85, handles=span_handles, loc="lower left", ncol=2
    )
    draw_conversion(ax_lc1, conversion, "")
    ax_lc1.legend(fontsize=7, framealpha=0.85, handles=legend_handles, loc="lower left")

    for ax, y in ((ax_sy0, 2000), (ax_sy1, 2020)):
        im_soy = ax.imshow(
            soy_pair[y].data,
            cmap=soy_pair_cmap,
            extent=soy_pair[y].extent,
            interpolation="nearest",
            origin="upper",
            vmax=soy_pair_vmax,
            vmin=0.0,
        )
        _style_geo_axes(ax, "", soy_pair[y].extent)
    fig.colorbar(
        im_soy,
        ax=(ax_sy0, ax_sy1),
        label=SOYBEAN_LABEL,
        location="bottom",
        shrink=0.6,
    )

    im = draw_emissions(ax_em, emissions, "", vmax)
    fig.colorbar(im, ax=ax_em, label=EMISSIONS_LABEL, location="bottom", shrink=0.7)

    # Row labels name the quantity, column titles the question or the year it answers.
    ax_lc0.set_title("When it occurred", fontsize=15, fontweight="bold")
    ax_lc1.set_title("What the land became", fontsize=15, fontweight="bold")
    ax_em.set_title(
        "Emissions released by\nthose changes, 2000 to 2020",
        fontsize=13,
        fontweight="bold",
    )
    ax_lc0.set_ylabel("Land conversion", fontsize=14, fontweight="bold")
    ax_sy0.set_ylabel("Soybean cultivation", fontsize=14, fontweight="bold")
    ax_sy0.set_title("2000", fontsize=13)
    ax_sy1.set_title("2020", fontsize=13)
    for ax in (ax_lc1, ax_sy1):
        ax.set_ylabel("")

    fig.suptitle(
        "Soybean expanded into cleared forest and Cerrado in Matopiba, Brazil,\n"
        "and the carbon those clearances released is charged to the crops that followed",
        fontsize=16,
    )
    fig.savefig(out(FIGURE_NAME), bbox_inches="tight", dpi=args.dpi)
    shrink_in_place(path=out(FIGURE_NAME))
    plt.close(fig)

    logger.info("wrote %s to %s", FIGURE_NAME, args.outdir)


if __name__ == "__main__":
    main()
