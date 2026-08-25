import collections.abc
import logging
import math
import tempfile
import typing

import numpy
import rasterio
import rasterio.enums
import rasterio.shutil
import rasterio.transform
import rasterio.vrt
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
EQUAL_AREA_CRS = "EPSG:6933"


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
        filename=path_to_vector, columns=id_column_names + name_column_names
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
        .sort_values(by="square_meters", ascending=False)
        .dissolve(by="id")
        .drop(columns="square_meters")
        .reset_index(drop=False)
        .to_file(path_to_flatgeobuf, driver="FlatGeobuf", SPATIAL_INDEX=True)
    )


def validate_geotiff(path_to_geotiff: str) -> None:
    logger.info(f"Validating integrity of {path_to_geotiff=:s}")
    with rasterio.open(path_to_geotiff) as dataset:
        assert isinstance(dataset, rasterio.DatasetReader)
        assert dataset.crs is not None
        assert dataset.crs.to_epsg() == 4326
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


def downscale_darray(
    darray: xarray.DataArray,
    epsg: int,
    height: int,
    resampling: rasterio.enums.Resampling,
    transform: tuple[float, float, float, float, float, float],
    width: int,
) -> xarray.DataArray:
    import rioxarray

    logger.info(f"Downscaling {darray.name=:s}")

    import dask.diagnostics

    path_to_unscaled = tempfile.NamedTemporaryFile(  # noqa: SIM115
        delete=False, suffix=".glad.tif"
    ).name
    chunk_size = 1 << 11  # 2k
    logger.debug(f"Serializing to {path_to_unscaled=:s} with {chunk_size=:d}")
    darray = darray.assign_attrs(long_name=darray.name).chunk(chunks=chunk_size)
    with dask.diagnostics.ProgressBar():
        darray.rio.to_raster(
            path_to_unscaled,
            blockxsize=chunk_size,
            blockysize=chunk_size,
            compress="ZSTD",
            driver="GTiff",
            lock=True,
            num_threads="all_cpus",
            tiled=True,
            BIGTIFF="IF_SAFER",
            ZSTD_LEVEL=1,
        )
    with (
        rasterio.open(path_to_unscaled) as unscaled_fp,
        rasterio.vrt.WarpedVRT(
            unscaled_fp,
            crs=f"EPSG:{epsg:d}",
            height=height,
            resampling=resampling,
            transform=rasterio.transform.Affine.from_gdal(*transform),
            width=width,
        ) as scaled_vrt,
        tempfile.NamedTemporaryFile(delete=False, suffix=".warped.vrt") as scaled_fp,
    ):
        assert unscaled_fp.crs.to_epsg() == epsg
        logger.debug(f"Writing warped VRT to {scaled_fp.name=:s}")
        rasterio.shutil.copy(scaled_vrt, scaled_fp.name, driver="VRT")
    ret = rioxarray.open_rasterio(
        scaled_fp.name,
        chunks=get_chunk_size(
            dtypes=[numpy.dtype("float32")],
            # 128 kiB ~ 1 GiB x (12 ÷ 4_000 resolution) ÷ 14 bands
            max_bytes_per_chunk=1 << 17,
        ),
        lock=False,
    )
    assert isinstance(ret, xarray.DataArray)
    ret = ret.drop_vars("spatial_ref").isel(band=0, drop=True)
    return unify_dtype_and_no_data(darray=ret.rename(ret.attrs.pop("long_name")))
