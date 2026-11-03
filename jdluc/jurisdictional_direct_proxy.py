import logging

import numpy
import rasterio.enums
import xarray

from jdluc import emit, geo, harmonize, storage, tiling
from jdluc.datasets import gpw_grassland, tang_cropgrids

logger = logging.getLogger(__name__)


assert tang_cropgrids.YEAR in gpw_grassland.YEARS


def get_name_to_band(dset: xarray.Dataset) -> dict[str, xarray.DataArray]:
    """The 30 m bands this leg divides, each per hectare or a fraction of the pixel.

    Each emissions component is split by the land class its pixel became, and discounted over
    the spans: the shares that divide it come from one year, so no span needs its own band.
    """
    span_to_component_to_emissions = emit.get_span_to_component_to_emissions(dset=dset)
    return (
        {
            f"{land_class:s}-{component!s}-emissions:tco2e-per-ha": emit.get_linear_discounted_total(
                span_to_value={
                    span: component_to_emissions[component].where(
                        dset["conversion"].isin(conversions), other=0
                    )
                    for span, component_to_emissions in span_to_component_to_emissions.items()
                }
            )
            for land_class, conversions in (
                (emit.CROPLAND, emit.TO_CROPLAND),
                (emit.PASTURELAND, emit.TO_PASTURE),
            )
            for component in emit.EmissionComponent
        }
        | {
            name: dset[name]
            for name in (
                f"{emit.CROPLAND:s}-peatland-occupation:tco2e-per-ha",
                "dropped-emissions:tco2e-per-ha",
                f"{emit.PASTURELAND:s}-peatland-occupation:tco2e-per-ha",
            )
        }
        | {
            # NB: a mask here and an averaging downscale after, so this arrives as a cell fraction
            f"pasture:fraction:{tang_cropgrids.YEAR:d}": (
                dset[
                    gpw_grassland.DATASET.fully_qualified_band_names[
                        gpw_grassland.YEARS.index(tang_cropgrids.YEAR)
                    ]
                ]
                == gpw_grassland.Grassland.CULTIVATED
            ).astype(numpy.float32)
        }
    )


@storage.cache_to_zarr(version=0)
def get_downscaled_luc_emissions(tile_id: str) -> xarray.Dataset:
    logger.info("Computing emissions on the GLAD grid")
    luc_and_emissions = geo.exact_merge(
        harmonize.workflow(
            dataset_names=harmonize.Stack.LUC_AND_EMISSIONS.value,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.GLAD,
        ),
        # NB: this should call the harmonize workflow with identical args and hit
        # the cache from the preceding call
        emit.workflow(tile_id=tile_id),
    )

    logger.info("Downsampling emissions from the GLAD grid to the CROPGRIDS grid")
    grid = harmonize.Grid.from_tile_id_resolution(
        resolution=tiling.TileResolution.CROPGRIDS, tile_id=tile_id
    )
    return xarray.Dataset(
        {
            name: geo.downscale_darray(
                darray=band.rename(name),
                epsg=grid.epsg,
                height=grid.resolution.y,
                # NB: without an equal-area CRS, this presents a small error
                resampling=rasterio.enums.Resampling.average,
                transform=grid.transform,
                width=grid.resolution.x,
            )
            for name, band in get_name_to_band(dset=luc_and_emissions).items()
        }
    )
