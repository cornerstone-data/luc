import collections.abc
import dataclasses
import logging
import math
import typing

import numpy
import rasterio
import rasterio.enums
import rasterio.transform
import shapely
import xarray

logger = logging.getLogger(__name__)


def set_band_names_for_geotiff(
    band_names: collections.abc.Sequence[str], path_to_geotiff: str
) -> None:
    logger.info(f"Setting {band_names=:} for {path_to_geotiff=:s}")
    with rasterio.open(fp=path_to_geotiff, mode="r+") as dataset:
        for idx, band_name in enumerate(band_names, start=1):
            dataset.set_band_description(idx, band_name)


def get_overview_level(height: int, width: int, minimum_pixels: int = 64) -> int:
    if (pixel_ratio := min(height, width) // minimum_pixels) > 0:
        return math.floor(math.log2(pixel_ratio))
    else:
        return 0


def convert_geotiff_to_cog(
    metadata: dict[str, str],
    no_data: float | int | None,
    path_to_cog: str,
    path_to_geotiff: str,
) -> None:
    import rio_cogeo.cogeo
    import rio_cogeo.profiles

    logger.info(f"Getting overview level from {path_to_geotiff=:s}")
    with rasterio.open(fp=path_to_geotiff) as dataset:
        overview_level = get_overview_level(
            height=dataset.height,
            width=dataset.width,
        )
    logger.info(f"Converting from {path_to_geotiff=:s} to {path_to_cog=:s}")
    rio_cogeo.cogeo.cog_translate(
        additional_cog_metadata=metadata,
        config={
            "GDAL_NUM_THREADS": "ALL_CPUS",
            "GDAL_TIFF_INTERNAL_MASK": True,
        },
        dst_kwargs=dict(rio_cogeo.profiles.DEFLATEProfile()) | {"BIGTIFF": "IF_SAFER"},
        dst_path=path_to_cog,
        in_memory=False,
        nodata=no_data,
        overview_level=overview_level,
        quiet=False,
        source=path_to_geotiff,
        use_cog_driver=True,
    )


# NB: an equal-area projection, so that parts of one jurisdiction can be ranked by size
EQUAL_AREA_CRS = 6933


def convert_vector_to_flatgeobuf(
    id_column_names: tuple[str, ...],
    name_column_names: tuple[str, ...],
    path_to_flatgeobuf: str,
    path_to_vector: str,
) -> None:
    import geopandas

    def maybe_repair_mojibake(name: str) -> str:
        try:
            return name.encode("latin-1").decode("utf-8")
        except UnicodeError:
            return name

    logger.info(
        f"Opening {path_to_vector=:s} for {id_column_names=:} and {name_column_names=:}"
    )
    gdf: geopandas.GeoDataFrame = geopandas.read_file(
        columns=id_column_names + name_column_names, filename=path_to_vector
    )
    logger.info(
        f"Forming {id_column_names=:} and {name_column_names=:}, dissolving, and saving to {path_to_flatgeobuf=:s}"
    )
    (
        geopandas.GeoDataFrame(
            crs=gdf.crs,
            data={
                "id": gdf[list(id_column_names)].astype(str).agg(" | ".join, axis=1),
                "name": gdf[list(name_column_names)]
                .astype(str)
                .map(maybe_repair_mojibake)
                .agg(" | ".join, axis=1),
                # NB: so that dissolve can inherit values from the largest regions
                "square_meters": gdf.geometry.to_crs(crs=EQUAL_AREA_CRS).area,
            },
            geometry=gdf.geometry,
        )
        .sort_values(ascending=False, by="square_meters")
        .dissolve(by="id")
        .drop(columns="square_meters")
        .reset_index(drop=False)
        .to_file(path_to_flatgeobuf, SPATIAL_INDEX=True, driver="FlatGeobuf")
    )


def validate_geotiff(dtype: str, path_to_geotiff: str) -> None:
    logger.info(f"Validating integrity of {path_to_geotiff=:s}")
    with rasterio.open(path_to_geotiff) as dataset:
        assert isinstance(dataset, rasterio.DatasetReader)
        assert dataset.crs is not None
        assert dataset.crs.to_epsg() == 4326
        assert set(dataset.dtypes) == {dtype}
        assert dataset.height > 0
        assert dataset.width > 0
        assert dataset.transform.a > 0
        assert dataset.transform.e < 0


def get_chunk_size(
    dtypes: collections.abc.Iterable[numpy.dtype],
    # 1 GiB
    max_bytes_per_chunk: int = 1 << 30,
    number_of_dimensions: int = 2,
) -> int:
    # Sum over all the variables
    bytes_per_pixel = sum(dtype.itemsize for dtype in dtypes)
    pixels_per_chunk = max_bytes_per_chunk / bytes_per_pixel
    # Round down to nearest power of two
    return 1 << int(math.log2(pixels_per_chunk) / number_of_dimensions)


ATTRS_TO_MOVE = ("_FillValue", "scale_factor", "add_offset", "dtype", "missing_value")


def unify_dtype_and_no_data(darray: xarray.DataArray) -> xarray.DataArray:
    import rioxarray  # noqa

    if (no_data := darray.rio.nodata) is not None:
        darray = darray.where(darray != no_data, other=numpy.nan)
    darray = darray.astype(numpy.float32).rio.write_nodata(numpy.nan)
    assert isinstance(darray, xarray.DataArray)
    for key in ATTRS_TO_MOVE:
        if key in darray.attrs:
            darray.encoding[key] = darray.attrs.pop(key)
    return darray.transpose(..., "y", "x")


def clip_dset(dset: xarray.Dataset, geometry: shapely.Geometry) -> xarray.Dataset:
    assert isinstance(geometry, shapely.Polygon | shapely.MultiPolygon)
    import rioxarray  # noqa

    overlap = geometry.intersection(shapely.box(*dset.rio.bounds()))
    if overlap.area == 0:
        logger.warning("Geometry and Dataset don't overlap; returning an empty array")
        assert set(dset.dims) == {"x", "y"}
        return dset.isel(x=slice(0, 0), y=slice(0, 0))
    else:
        ret = dset.rio.clip_box(
            *overlap.bounds, allow_one_dimensional_raster=True
        ).rio.clip([overlap], drop=True)
    return typing.cast(xarray.Dataset, ret)


def exact_merge(*dsets: xarray.Dataset) -> xarray.Dataset:
    return xarray.merge(
        list(dsets),
        combine_attrs="identical",
        compat="identical",
        join="exact",
    ).unify_chunks()


@dataclasses.dataclass
class AxisOverlaps:
    coarse_cells: slice
    weights: numpy.ndarray


def get_overlap_weights(
    coarse_size: int, fine_size: int, fine_start: int, fine_stop: int
) -> AxisOverlaps:
    """Along one axis, the share of each fine cell that falls inside each coarse cell.

    The axis is cut into `coarse_size` equal coarse cells and, separately, into `fine_size` equal
    fine cells. Only the fine cells from `fine_start` up to `fine_stop` are looked at. The result
    names the coarse cells those fine cells fall in, and for every pair of coarse and fine cell
    gives the share of the fine cell inside the coarse cell: 1 when it lies wholly inside, 0 when
    it lies outside, and a fraction when the coarse cell's edge cuts through it.
    """
    first = fine_start * coarse_size // fine_size
    # fine_stop is exclusive, so when the last fine cell ends exactly on a coarse cell edge, the
    # coarse cell that starts at that edge isn't included
    last = (fine_stop * coarse_size - 1) // fine_size
    edges = numpy.arange(first, last + 2) * (fine_size / coarse_size)
    fine = numpy.arange(fine_start, fine_stop)[numpy.newaxis, :]
    overlaps = numpy.minimum(fine + 1, edges[1:, numpy.newaxis]) - numpy.maximum(
        fine, edges[:-1, numpy.newaxis]
    )
    return AxisOverlaps(
        coarse_cells=slice(first, last + 1), weights=overlaps.clip(min=0)
    )


def get_weighted_sums(
    block: numpy.ndarray,
    column_weights: numpy.ndarray,
    row_weights: numpy.ndarray,
    valid: numpy.ndarray,
) -> numpy.ndarray:
    """The weighted sum of the valid values in `block` and the weighted count of them, as a
    (2, coarse rows, coarse columns) array, each fine cell counted by its share of the coarse cell.
    """
    values_and_counts = numpy.stack([numpy.where(valid, block, 0), valid]).astype(
        numpy.float64
    )
    return row_weights @ values_and_counts @ column_weights.T


def downscale_darray(
    darray: xarray.DataArray,
    epsg: int,
    height: int,
    resampling: rasterio.enums.Resampling,
    transform: tuple[float, float, float, float, float, float],
    width: int,
) -> xarray.DataArray:
    """Average `darray` onto a coarser grid that covers the same area.

    `transform`, `height` and `width` describe the coarser grid. Each coarse cell gets the mean of
    the fine cells inside it, skipping missing (NaN) ones. A fine cell that the edge of a coarse
    cell cuts through counts only by the share of it inside that coarse cell. A coarse cell with
    no non-missing fine cell inside it is NaN. This matches GDAL's average resampling.

    The fine grid is read one chunk at a time. Each chunk adds its weighted sum and weighted count
    to the coarse cells it overlaps, and the mean is taken once every chunk is added in. So only a
    few chunks are in memory at once, never the whole fine grid.
    """
    from rioxarray.rioxarray import affine_to_coords

    assert resampling == rasterio.enums.Resampling.average
    assert darray.rio.crs.to_epsg() == epsg
    assert darray.dims == ("y", "x")
    affine = rasterio.transform.Affine.from_gdal(*transform)
    # Weights by row and column index need both grids to run north to south and west to east.
    assert all(grid.a > 0 and grid.e < 0 for grid in (darray.rio.transform(), affine))
    assert numpy.allclose(
        darray.rio.bounds(),
        rasterio.transform.array_bounds(height=height, transform=affine, width=width),
    )

    logger.info(f"Downscaling {darray.name=:s}")
    fine_height, fine_width = darray.shape
    source_nodata = darray.rio.nodata

    def get_partial_sums(
        block: numpy.ndarray, block_info: dict[int | None, typing.Any]
    ) -> numpy.ndarray:
        (row_start, row_stop), (column_start, column_stop) = block_info[0][
            "array-location"
        ]
        rows = get_overlap_weights(
            coarse_size=height,
            fine_size=fine_height,
            fine_start=row_start,
            fine_stop=row_stop,
        )
        columns = get_overlap_weights(
            coarse_size=width,
            fine_size=fine_width,
            fine_start=column_start,
            fine_stop=column_stop,
        )
        valid = numpy.isfinite(block)
        if source_nodata is not None:
            valid &= block != source_nodata
        partial_sums = numpy.zeros(shape=(1, 1, 2, height, width))
        partial_sums[0, 0, :, rows.coarse_cells, columns.coarse_cells] = (
            get_weighted_sums(
                block=block,
                column_weights=columns.weights,
                row_weights=rows.weights,
                valid=valid,
            )
        )
        return partial_sums

    data = darray.chunk(chunks=1 << 11).data  # 2k
    numerator, denominator = (
        data.map_blocks(
            get_partial_sums,
            chunks=(
                (1,) * data.numblocks[0],
                (1,) * data.numblocks[1],
                (2,),
                (height,),
                (width,),
            ),
            dtype=numpy.float64,
            new_axis=[2, 3, 4],
        )
        .sum(axis=(0, 1), split_every=4)
        .compute()
    )
    mean = numpy.full(fill_value=numpy.nan, shape=(height, width))
    numpy.divide(numerator, denominator, out=mean, where=denominator > 0)
    return unify_dtype_and_no_data(
        darray=xarray.DataArray(
            coords=affine_to_coords(affine=affine, height=height, width=width),
            data=mean.astype(numpy.float32),
            dims=("y", "x"),
            name=darray.name,
        ).chunk()
    )
