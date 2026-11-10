import collections
import collections.abc
import logging

import numpy
import pandas
import rasterio.enums
import xarray
from rioxarray.exceptions import NoDataInBounds

from jdluc import emit, geo, harmonize, storage, tiling, utils
from jdluc.datasets import (
    DatasetName,
    faostat,
    gpw_grassland,
    gpw_livestock,
    tang_cropgrids,
    worldbank_jurisdictions,
)

logger = logging.getLogger(__name__)


assert tang_cropgrids.YEAR in gpw_grassland.YEARS
assert tang_cropgrids.YEAR in gpw_livestock.YEARS

type Commodity = tang_cropgrids.Crop | gpw_livestock.Livestock
assert not {e.name for e in tang_cropgrids.Crop} & {
    e.name for e in gpw_livestock.Livestock
}

DATASET_NAMES = (DatasetName.GPW_LIVESTOCK, DatasetName.TANG_CROPGRIDS_CROP_AREA)
HERD_HEADS_COLUMN = gpw_livestock.get_herd_heads_column(year=tang_cropgrids.YEAR)


def get_name_to_band(
    assessment_year: int, dset: xarray.Dataset
) -> dict[str, xarray.DataArray]:
    """The 30 m bands this leg divides, each per hectare or a fraction of the pixel.

    Each emissions component is discounted by its pixel's conversion year and split by the land
    class the pixel became: the shares that divide it come from one year, so no span needs its own
    band. The source carbon no destination claimed is discounted the same way.
    """
    component_to_emissions = {
        component: emit.get_linear_discounted(
            assessment_year=assessment_year,
            conversion_year=dset["conversion-year"],
            darray=emissions,
        )
        for component, emissions in emit.get_component_to_emissions(dset=dset).items()
    }
    return (
        {
            f"{land_class:s}-{component!s}-emissions:tco2e-per-ha": component_to_emissions[
                component
            ].where(dset["conversion"].isin(conversions), other=0)
            for land_class, conversions in (
                (emit.CROPLAND, emit.TO_CROPLAND),
                (emit.PASTURELAND, emit.TO_PASTURE),
            )
            for component in emit.EmissionComponent
        }
        | {
            name: dset[name]
            for name in (
                f"{emit.CROPLAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha",
                f"{emit.PASTURELAND:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha",
            )
        }
        | {
            "dropped-emissions:tco2e-per-ha": emit.get_linear_discounted(
                assessment_year=assessment_year,
                conversion_year=dset["conversion-year"],
                darray=dset["dropped-emissions:tco2e-per-ha"],
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
def get_downscaled_luc_emissions(assessment_year: int, tile_id: str) -> xarray.Dataset:
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
        emit.workflow(assessment_year=assessment_year, tile_id=tile_id),
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
            for name, band in get_name_to_band(
                assessment_year=assessment_year, dset=luc_and_emissions
            ).items()
        }
    )


def get_crop_hectares(
    crop: tang_cropgrids.Crop, dset: xarray.Dataset
) -> xarray.DataArray:
    """`crop`'s physical area in each cell, in hectares; zero over the ocean, which CROPGRIDS
    leaves missing."""
    return dset[
        tang_cropgrids.DATASET.fully_qualified_band_names[
            list(tang_cropgrids.Crop).index(crop)
        ]
    ].fillna(0)


def get_crop_to_area_share(
    crops: tuple[tang_cropgrids.Crop, ...], dset: xarray.Dataset
) -> dict[tang_cropgrids.Crop, xarray.DataArray]:
    """Each crop's share of a cell's crop area, over every crop CROPGRIDS maps."""
    total_area = sum(
        (get_crop_hectares(crop=crop, dset=dset) for crop in tang_cropgrids.Crop),
        start=xarray.DataArray(numpy.float32(0)),
    )
    total_area = total_area.where(total_area > 0)
    return {
        # Share is zero when no crop occupies the cell at all
        crop: (get_crop_hectares(crop=crop, dset=dset) / total_area).fillna(0)
        for crop in crops
    }


def get_commodity_name_to_totals(
    commodities: tuple[Commodity, ...],
    commodity_to_share: dict[Commodity, xarray.DataArray],
    dset: xarray.Dataset,
) -> dict[str, dict[str, float]]:
    """Each commodity's emissions and area, summed over a jurisdiction's cells.

    A crop takes its crop-area share of everything charged to cropland, and a livestock commodity
    its grazing share of everything charged to pasture, in every component and in peat occupation
    alike. Beside them sit `DROPPED` and `UNATTRIBUTED` (see `emit.NonCommodity`); here
    `UNATTRIBUTED` is what the shares leave of the charge: cropland's where CROPGRIDS maps no crop,
    and the shares of commodities not asked for.
    """
    hectares = dset["hectares-per-cell:ha"]
    land_class_to_component_to_emissions = {
        land_class: {
            component: dset[f"{land_class:s}-{component!s}-emissions:tco2e-per-ha"]
            * hectares
            for component in emit.EmissionComponent
        }
        for land_class in (emit.CROPLAND, emit.PASTURELAND)
    }
    land_class_to_occupation = {
        land_class: dset[f"{land_class:s}-{emit.PEATLAND_OCCUPATION:s}:tco2e-per-ha"]
        * hectares
        for land_class in (emit.CROPLAND, emit.PASTURELAND)
    }

    commodity_to_totals: dict[
        Commodity | emit.NonCommodity, dict[str, xarray.DataArray]
    ] = collections.defaultdict(dict)
    for commodity in commodities:
        totals = commodity_to_totals[commodity]
        share = commodity_to_share[commodity]
        if isinstance(commodity, gpw_livestock.Livestock):
            land_class = emit.PASTURELAND
            totals["commodity_hectares"] = (
                dset[f"pasture:fraction:{tang_cropgrids.YEAR:d}"] * hectares * share
            )
        else:
            land_class = emit.CROPLAND
            totals["commodity_hectares"] = get_crop_hectares(crop=commodity, dset=dset)
        component_to_emissions = {
            component: emissions * share
            for component, emissions in land_class_to_component_to_emissions[
                land_class
            ].items()
        }
        occupation = land_class_to_occupation[land_class] * share
        totals |= {
            component.column: component_to_emissions[component]
            for component in emit.EmissionComponent
        }
        # NB: occupation is a flat annual rate over the area drained, so dividing it back out
        # recovers that area on `emit`'s 30 m intersection
        totals["peatland_commodity_hectares"] = (
            occupation / emit.PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA
        )
        totals["peatland_occupation_emissions_mt"] = occupation
        totals["emissions_mt"] = sum(component_to_emissions.values(), start=occupation)

    # As in `statistical`, the whole herd of every cell with any pasture
    livestock_to_heads = {
        livestock: dset[
            gpw_livestock.get_band_name(species=species, year=tang_cropgrids.YEAR)
        ].where(dset[f"pasture:fraction:{tang_cropgrids.YEAR:d}"] > 0, other=0)
        * hectares
        for livestock, (species,) in gpw_livestock.LIVESTOCK_TO_SPECIES.items()
        if livestock in commodities
    }
    for livestock, heads in livestock_to_heads.items():
        commodity_to_totals[livestock][HERD_HEADS_COLUMN] = heads

    commodity_to_totals[emit.NonCommodity.DROPPED]["emissions_mt"] = (
        dset["dropped-emissions:tco2e-per-ha"] * hectares
    )

    # The whole charge each component carries, as a commodity holding every share would take it
    component_to_charged = {
        component: land_class_to_component_to_emissions[emit.CROPLAND][component]
        + land_class_to_component_to_emissions[emit.PASTURELAND][component]
        for component in emit.EmissionComponent
    }
    unattributed = commodity_to_totals[emit.NonCommodity.UNATTRIBUTED]
    unattributed |= {
        component.column: component_to_charged[component]
        - sum(
            commodity_to_totals[commodity][component.column]
            for commodity in commodities
        )
        for component in emit.EmissionComponent
    }
    unattributed["peatland_occupation_emissions_mt"] = (
        land_class_to_occupation[emit.CROPLAND]
        + land_class_to_occupation[emit.PASTURELAND]
        - sum(
            commodity_to_totals[commodity]["peatland_occupation_emissions_mt"]
            for commodity in commodities
        )
    )
    unattributed["emissions_mt"] = sum(
        (unattributed[component.column] for component in emit.EmissionComponent),
        start=unattributed["peatland_occupation_emissions_mt"],
    )

    return {
        name: {HERD_HEADS_COLUMN: numpy.nan} | totals
        for name, totals in utils.get_sum_totals(
            enum_to_name_to_darray=commodity_to_totals
        ).items()
    }


SCHEMA = {
    "admin_id": str,
    "admin_level": str,
    "commodity_hectares": float,
    "commodity_name": str,
    "emissions_mt": float,
    "forest_emissions_mt": float,
    "grassland_emissions_mt": float,
    "jurisdiction_name": str,
    "peatland_commodity_hectares": float,
    "peatland_conversion_emissions_mt": float,
    "peatland_occupation_emissions_mt": float,
    HERD_HEADS_COLUMN: float,
}


@storage.cache_to_parquet(version=0)
def workflow(
    assessment_year: int,
    commodity_names: tuple[str, ...],
    iso_3166: str,
    tile_id: str,
) -> pandas.DataFrame:
    commodities: tuple[Commodity, ...] = tuple(
        tang_cropgrids.Crop[name]
        if name in tang_cropgrids.Crop.__members__
        else gpw_livestock.Livestock[name]
        for name in commodity_names
    )
    crops = tuple(
        commodity
        for commodity in commodities
        if isinstance(commodity, tang_cropgrids.Crop)
    )
    logger.info(
        f"Computing emissions for {commodities=:}, {assessment_year=:d} and {tile_id=:s}"
    )
    merged = geo.exact_merge(
        get_downscaled_luc_emissions(assessment_year=assessment_year, tile_id=tile_id),
        harmonize.workflow(
            dataset_names=DATASET_NAMES,
            ignore_missing_tiles=True,
            skip_ingest=False,
            tile_id=tile_id,
            tile_resolution=tiling.TileResolution.CROPGRIDS,
        ),
    )
    # Each grazer's livestock units per head, as FAOSTAT counts them for this country
    merged = merged.assign(
        gpw_livestock.get_band_name_to_livestock_units(
            dset=merged,
            species_to_units_per_head={
                gpw_livestock.Species[species.name]: units_per_head
                for species, units_per_head in faostat.get_species_to_livestock_units_per_head(
                    iso_3166=iso_3166,
                    livestock_units=faostat.load(
                        dataset=faostat.LIVESTOCK_PATTERNS_DATASET
                    ),
                    stocks=faostat.load(dataset=faostat.LIVESTOCK_DATASET),
                ).items()
            },
        )
    )
    # NB: on the whole tile, since a clip one cell wide has no spacing to measure a cell by
    merged["hectares-per-cell:ha"] = emit.get_hectares_per_pixel(
        darray=merged["dropped-emissions:tco2e-per-ha"]
    )
    # NB: small enough to hold in memory, which speeds up the many reads for each jurisdiction
    merged = merged.load()

    def it() -> collections.abc.Iterator[dict[str, float | str]]:
        for (
            jurisdiction
        ) in worldbank_jurisdictions.iter_jurisdiction_for_iso_3166_tile_id(
            admin_level=worldbank_jurisdictions.AdminLevel.PROVINCIAL,
            iso_3166=iso_3166,
            tile_id=tile_id,
        ):
            logger.info(f"Clipping to provincial geometry from {jurisdiction.id=:s}")
            try:
                clipped = geo.clip_dset(dset=merged, geometry=jurisdiction.geometry)
            except NoDataInBounds as exc:
                logger.warning(repr(exc))
            else:
                logger.info(
                    f"Populating emissions for {jurisdiction.id=!s} over "
                    f"{len(commodities):d} commodities"
                )
                commodity_name_to_totals = get_commodity_name_to_totals(
                    commodities=commodities,
                    commodity_to_share=get_crop_to_area_share(crops=crops, dset=clipped)
                    | gpw_livestock.get_livestock_to_grazing_share(
                        dset=clipped, year=tang_cropgrids.YEAR
                    ),
                    dset=clipped,
                )
                for commodity_name, totals in commodity_name_to_totals.items():
                    yield totals | {
                        "admin_id": jurisdiction.id,
                        "admin_level": jurisdiction.level,
                        "commodity_name": commodity_name,
                        "jurisdiction_name": jurisdiction.name,
                    }

    if data := list(it()):
        first_record, *_ = data
        assert set(first_record) == set(SCHEMA)
    return (
        pandas.DataFrame.from_records(columns=list(SCHEMA), data=data)
        .astype(SCHEMA)
        .set_index(["admin_level", "admin_id", "commodity_name"])
    )
