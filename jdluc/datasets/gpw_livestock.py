"""Global Pasture Watch | Annual livestock headcount, FAOSTAT-adjusted, 2000-2022

license: CC BY 4.0

year: 2000, 2005, 2010, 2020

Parente, L., Ehrmann, S., Hengl, T., Fritz, S., Bonannella, C., Malek, Ž., Gonzalez Fischer, C., Perez, K., Stanimirova, R., Meyer, C., Wisser, D., Cinardi, G., and Sloat, L. (2026). Global distribution of cattle, horses, goats, sheep and buffaloes at 1 km resolution for 2000-2022 based on subnational census data and spatiotemporal machine learning. PeerJ 14, e21494. https://doi.org/10.7717/peerj.21494
Parente, L. et al. (2025). Global Pasture Watch - Annual livestock headcount layers for cattle, goats, sheep, horses, and buffaloes at 1-km 2000-2022 (FAOSTAT-adjusted) (Part-1), v1-rc [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17491242 -- cattle, goats, horses and sheep
Parente, L. et al. (2025). Global Pasture Watch - Annual livestock headcount layers for cattle, goats, sheep, horses, and buffaloes at 1-km 2000-2022 (FAOSTAT-adjusted) (Part-2), v1-rc [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17494177 -- buffaloes

https://github.com/wri/global-pasture-watch

# Methodology

- Random Forest over 128 covariates, GPW's own grassland and cropland extent among them, trained on
  55,336 census polygons across 147 countries, then scaled so each country-year matches FAOSTAT
  stocks
- Test-set CCC 0.60 for cattle (RMSE 105 heads/km2), 0.55-0.69 for the others; it cannot tell
  extensive grazing from feedlots

One 1 km headcount raster per species and year on Interrupted Goode Homolosine (ESRI:54052). Each
equal-area cell is 1 km2, so a headcount / 100 is heads/ha, which is what is stored, warped by
averaging onto `tiling.TileResolution.GPW_LIVESTOCK`.
"""

import enum
import itertools
import logging
import os
import tempfile

import numpy
import rasterio
import rasterio.enums
import rasterio.transform
import rasterio.warp
import xarray

from jdluc import tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


YEARS = (2000, 2005, 2010, 2020)
KM_2_PER_HECTARE = 10_000 / 1_000_000


class Species(enum.StrEnum):
    BUFFALO = enum.auto()
    CATTLE = enum.auto()
    GOAT = enum.auto()
    HORSE = enum.auto()
    SHEEP = enum.auto()


SPECIES_TO_ZENODO_RECORD = {
    Species.BUFFALO: 17494177,
    Species.CATTLE: 17491242,
    Species.GOAT: 17491242,
    Species.HORSE: 17491242,
    Species.SHEEP: 17491242,
}
assert set(SPECIES_TO_ZENODO_RECORD) == set(Species)

SPECIES_YEARS = tuple(sorted(itertools.product(Species, YEARS)))
BAND_NAMES = [f"{species:s}:heads-per-ha:{year:d}" for species, year in SPECIES_YEARS]


@utils.threadsafe_cache
def get_path_to_source(species: Species, year: int) -> str:
    name = f"gpw_{species:s}.headcount.faostat_rf_m_1km_s_{year:d}0101_{year:d}1231_go_esri.54052_v1.tif"
    # NB: deliberately not a TemporaryDirectory, so that the source outlives this call
    path_to_source = os.path.join(
        tempfile.mkdtemp(), f"gpw-livestock-{species:s}-{year:d}"
    )
    utils.save_remote_url_to_local_path(
        local_path=path_to_source,
        params={},
        remote_url=f"https://zenodo.org/api/records/{SPECIES_TO_ZENODO_RECORD[species]:d}/files/{name:s}/content",
    )
    return path_to_source


def _save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
    max_latitude, min_longitude = tiling.get_lat_lon_for_tile_id(tile_id=tile_id)
    resolution = tiling.TileResolution.GPW_LIVESTOCK
    transform = rasterio.transform.from_origin(
        min_longitude, max_latitude, *resolution.degrees_per_pixel
    )
    with rasterio.open(
        NUM_THREADS="ALL_CPUS",
        compress="deflate",
        count=len(SPECIES_YEARS),
        crs=4326,
        driver="GTiff",
        dtype=DATASET.dtype,
        fp=local_path,
        height=resolution.y,
        interleave="band",
        mode="w",
        nodata=DATASET.no_data,
        tiled=True,
        transform=transform,
        width=resolution.x,
    ) as dataset:
        for band_idx, (species, year) in enumerate(SPECIES_YEARS, start=1):
            logger.info(f"Warping {species:s} {year=:d} onto {tile_id=:s}")
            heads_per_km_2 = numpy.full(
                (resolution.y, resolution.x), numpy.nan, dtype=DATASET.dtype
            )
            with rasterio.open(
                fp=get_path_to_source(species=species, year=year)
            ) as source:
                assert source.crs == rasterio.CRS.from_user_input("ESRI:54052")
                assert source.res == (1000, 1000)
                rasterio.warp.reproject(
                    destination=heads_per_km_2,
                    dst_crs=4326,
                    dst_nodata=numpy.nan,
                    dst_transform=transform,
                    resampling=rasterio.enums.Resampling.average,
                    source=rasterio.band(source, 1),
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    src_transform=source.transform,
                )
            dataset.write(
                # NB: convention is missings = no heads, and so are the source's few negative
                # headcounts (three coastal pixels on the Macau coast)
                numpy.clip(numpy.nan_to_num(heads_per_km_2, nan=0), min=0)
                * KM_2_PER_HECTARE,
                band_idx,
            )


DATASET = base.RasterDataset(
    band_names=BAND_NAMES,
    band_type=base.BandType.INTENSIVE,
    dtype="float32",
    no_data=None,
    partitioning=tiling.Partitioning.TEN_DEGREE_TILE,
    product_name="livestock",
    save_tile_id_to_local_path=_save_tile_id_to_local_path,
    source_name="gpw",
    version="v0",
)


def get_band_name(species: Species, year: int) -> str:
    """The fully-qualified band carrying `species`' density, in heads/ha, in `year`."""
    return DATASET.fully_qualified_band_names[SPECIES_YEARS.index((species, year))]


@enum.unique
class Livestock(enum.StrEnum):
    """The commodities dividing pasture's share of a cell; PASTURE keeps what the others don't."""

    BEEF_CATTLE = enum.auto()
    MILK_CATTLE = enum.auto()
    PASTURE = enum.auto()


# Beef and milk both come from the whole cattle herd
LIVESTOCK_TO_SPECIES = {
    Livestock.BEEF_CATTLE: Species.CATTLE,
    Livestock.MILK_CATTLE: Species.CATTLE,
}
assert set(LIVESTOCK_TO_SPECIES) == set(Livestock) - {Livestock.PASTURE}
# Milk's part of the dairy herd's emissions, by biophysical allocation (section 5.4.2) at the worked
# example's 0.851 (appendix 10.5); FAOSTAT does not report the dairy herd's live-weight sales the
# formula needs. IDF. 2022. The IDF global Carbon Footprint standard for the dairy sector. In:
# Bulletin of the IDF No. 520/2022. International Dairy Federation (ed.), Brussels.
MILK_ALLOCATION = 0.85


def get_band_name_to_livestock_units(
    dset: xarray.Dataset, species_to_units_per_head: dict[Species, float]
) -> dict[str, xarray.DataArray]:
    """Each grazer's density in livestock units/ha, for assigning beside its density in heads.

    Livestock units weigh how much one head of a species grazes, so they divide pasture between
    grazers; production counts heads, and reads the heads bands.
    """
    return {
        f"{species:s}:livestock-units-per-ha:{year:d}": dset[
            get_band_name(species=species, year=year)
        ]
        * species_to_units_per_head[species]
        for species, year in SPECIES_YEARS
    }


def get_livestock_to_grazing_share(
    dairy_share: float, dset: xarray.Dataset, year: int
) -> dict[Livestock, xarray.DataArray]:
    """Each commodity's share of a cell's pasture.

    Cattle take their livestock units over all grazers'. Of that, milk takes `MILK_ALLOCATION` of
    the dairy herd's part, `dairy_share`, and beef the rest.
    """
    total = sum(
        (dset[f"{species:s}:livestock-units-per-ha:{year:d}"] for species in Species),
        start=xarray.DataArray(numpy.float32(0)),
    )
    # Share is zero where no grazer is mapped at all
    cattle = (
        dset[f"{Species.CATTLE:s}:livestock-units-per-ha:{year:d}"]
        / total.where(total > 0)
    ).fillna(0)
    milk = MILK_ALLOCATION * dairy_share
    return {
        Livestock.BEEF_CATTLE: (1 - milk) * cattle,
        Livestock.MILK_CATTLE: milk * cattle,
        Livestock.PASTURE: 1 - cattle,
    }


def get_livestock_hectares(
    dset: xarray.Dataset, grazing_share: xarray.DataArray, year: int
) -> xarray.DataArray:
    """The hectares of a cell's pasture a livestock commodity's grazing share takes."""
    return (
        dset[f"pasture:fraction:{year:d}"]
        * dset["hectares-per-cell:ha"]
        * grazing_share
    )


def get_herd_heads(
    dset: xarray.Dataset, species: Species, year: int
) -> xarray.DataArray:
    """A cell's whole herd of `species` where it holds any pasture, and none where it holds none.

    GPW's density is per hectare of the whole cell, so it is not scaled by the pasture fraction.
    """
    return (
        dset[get_band_name(species=species, year=year)].where(
            dset[f"pasture:fraction:{year:d}"] > 0, other=0
        )
        * dset["hectares-per-cell:ha"]
    )


def get_herd_heads_column(year: int) -> str:
    return f"herd_heads_{year:d}"
