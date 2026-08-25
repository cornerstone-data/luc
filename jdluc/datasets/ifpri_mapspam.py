"""International Food Policy Research Institute | Spatial Production Allocation Model (MapSPAM)

license: CC BY 4.0

year: 2000, 2005, 2010, 2020

International Food Policy Research Institute (IFPRI), 2026, "Global Spatially-Disaggregated Crop Production Statistics Data for 2020 Version 2.0 Release 2", https://doi.org/10.7910/DVN/SWPENT, Harvard Dataverse, V5
International Food Policy Research Institute, 2019, "Global Spatially-Disaggregated Crop Production Statistics Data for 2010 Version 2.0", https://doi.org/10.7910/DVN/PRFF8V, Harvard Dataverse, V4
International Food Policy Research Institute (IFPRI); International Institute for Applied Systems Analysis (IIASA), 2016, "Global Spatially-Disaggregated Crop Production Statistics Data for 2005 Version 3.2", https://doi.org/10.7910/DVN/DHXBJX, Harvard Dataverse, V9
International Food Policy Research Institute, 2019, "Global Spatially-Disaggregated Crop Production Statistics Data for 2000 Version 3.0.7", https://doi.org/10.7910/DVN/A50I2T, Harvard Dataverse, V1

https://www.mapspam.info/

# Methodology

- Downscales sub-national crop statistics to a ~10 km (5 arc-minute) grid
- Cross-entropy approach: allocates production across grid cells using priors
  from cropland extent, irrigation, suitability, population, and crop prices
- Contains 46 (2020), 42 (2010), 42 (2005), and 20 (2000) separate crops
- Output variables per crop/system: physical area, harvested area, yield,
  production (also a value-of-production layer)
"""

import enum
import logging
import os
import tempfile

import numpy
import rasterio
import xarray

from jdluc import config, tiling, utils
from jdluc.datasets import base

logger = logging.getLogger(__name__)


@enum.unique
class Crop2000(enum.StrEnum):
    BANP = "Banana And Plantain"  # = BANA + PLNT
    BARL = "Barley"
    BEAN = "Bean"
    CASS = "Cassava"
    COFF = "Coffee"  # = COFF + RCOF
    COTT = "Cotton"
    GROU = "Groundnut"
    MAIZ = "Maize"
    MILL = "Millet"  # = MILL + PMIL
    OFIB = "Other Fibre Crops"
    OOIL = "Other Oil Crops"  # = CNUT + OILP + SUNF + RAPE + SESA + OOIL
    OPUL = "Other Pulses"  # = CHIC + COWP + PIGE + LENT + OPUL
    OTHE = "Rest Of Crops"  # ≠ REST
    POTA = "Potato"
    RICE = "Rice"
    SORG = "Sorghum"
    SOYB = "Soybean"
    SUGB = "Sugarbeet"
    SUGC = "Sugarcane"
    SWPY = "Sweet Potato And Yam"  # = SWPO + YAMS
    WHEA = "Wheat"


@enum.unique
class Crop2005(enum.StrEnum):
    ACOF = "Arabica Coffee"
    BANA = "Banana"
    BARL = "Barley"
    BEAN = "Bean"
    CASS = "Cassava"
    CHIC = "Chickpea"
    CNUT = "Coconut"
    COCO = "Cocoa"
    COTT = "Cotton"
    COWP = "Cowpea"
    GROU = "Groundnut"
    LENT = "Lentil"
    MAIZ = "Maize"
    OCER = "Other Cereals"
    OFIB = "Other Fibre Crops"
    OILP = "Oilpalm"
    OOIL = "Other Oil Crops"
    OPUL = "Other Pulses"
    ORTS = "Other Roots"
    PIGE = "Pigeon Pea"
    PLNT = "Plantain"
    PMIL = "Pearl Millet"
    POTA = "Potato"
    RAPE = "Rapeseed"
    RCOF = "Robusta Coffee"
    REST = "Rest Of Crops"
    RICE = "Rice"
    SESA = "Sesame Seed"
    SMIL = "Small Millet"
    SORG = "Sorghum"
    SOYB = "Soybean"
    SUGB = "Sugarbeet"
    SUGC = "Sugarcane"
    SUNF = "Sunflower"
    SWPO = "Sweet Potato"
    TEAS = "Tea"
    TEMF = "Temperate Fruit"
    TOBA = "Tobacco"
    TROF = "Other Tropical Fruit"
    VEGE = "Other Vegetables"
    WHEA = "Wheat"
    YAMS = "Yams"


@enum.unique
class Crop2010(enum.StrEnum):
    # NB: the 2005/2010 values are the same but we'd like distinct enums
    ACOF = "Arabica Coffee"
    BANA = "Banana"
    BARL = "Barley"
    BEAN = "Bean"
    CASS = "Cassava"
    CHIC = "Chickpea"
    CNUT = "Coconut"
    COCO = "Cocoa"
    COTT = "Cotton"
    COWP = "Cowpea"
    GROU = "Groundnut"
    LENT = "Lentil"
    MAIZ = "Maize"
    OCER = "Other Cereals"
    OFIB = "Other Fibre Crops"
    OILP = "Oilpalm"
    OOIL = "Other Oil Crops"
    OPUL = "Other Pulses"
    ORTS = "Other Roots"
    PIGE = "Pigeon Pea"
    PLNT = "Plantain"
    PMIL = "Pearl Millet"
    POTA = "Potato"
    RAPE = "Rapeseed"
    RCOF = "Robusta Coffee"
    REST = "Rest Of Crops"
    RICE = "Rice"
    SESA = "Sesame Seed"
    SMIL = "Small Millet"
    SORG = "Sorghum"
    SOYB = "Soybean"
    SUGB = "Sugarbeet"
    SUGC = "Sugarcane"
    SUNF = "Sunflower"
    SWPO = "Sweet Potato"
    TEAS = "Tea"
    TEMF = "Temperate Fruit"
    TOBA = "Tobacco"
    TROF = "Other Tropical Fruit"
    VEGE = "Other Vegetables"
    WHEA = "Wheat"
    YAMS = "Yams"


@enum.unique
class Crop2020(enum.StrEnum):
    BANA = "Banana"
    BARL = "Barley"
    BEAN = "Bean"
    CASS = "Cassava"
    CHIC = "Chickpea"
    CITR = "Citrus"
    CNUT = "Coconut"
    COCO = "Cocoa"
    COFF = "Arabica Coffee"
    COTT = "Cotton"
    COWP = "Cowpea"
    GROU = "Groundnut"
    LENT = "Lentil"
    MAIZ = "Maize"
    MILL = "Small Millet"
    OCER = "Other Cereals"
    OFIB = "Other Fibre Crops"
    OILP = "Oilpalm"
    ONIO = "Onion"
    OOIL = "Other Oil Crops"
    OPUL = "Other Pulses"
    ORTS = "Other Roots"
    PIGE = "Pigeon Pea"
    PLNT = "Plantain"
    PMIL = "Pearl Millet"
    POTA = "Potato"
    RAPE = "Rapeseed"
    RCOF = "Robusta Coffee"
    REST = "Rest Of Crops"
    RICE = "Rice"
    RUBB = "Rubber"
    SESA = "Sesame Seed"
    SORG = "Sorghum"
    SOYB = "Soybean"
    SUGB = "Sugarbeet"
    SUGC = "Sugarcane"
    SUNF = "Sunflower"
    SWPO = "Sweet Potato"
    TEAS = "Tea"
    TEMF = "Temperate Fruit"
    TOBA = "Tobacco"
    TOMA = "Tomato"
    TROF = "Other Tropical Fruit"
    VEGE = "Other Vegetables"
    WHEA = "Wheat"
    YAMS = "Yams"


# Crops which are represented consistently across snapshots with the same enum
# and can be compared directly.  Even though the OOIL (other oil crops) enum is
# shared, the addition of oil crops across snapshots shrinks the crops that enum
# represents.
SHARED_CROP_NAMES = {
    Crop2000.BARL.name,
    Crop2000.BEAN.name,
    Crop2000.CASS.name,
    Crop2000.COTT.name,
    Crop2000.GROU.name,
    Crop2000.MAIZ.name,
    Crop2000.POTA.name,
    Crop2000.RICE.name,
    Crop2000.SORG.name,
    Crop2000.SOYB.name,
    Crop2000.SUGB.name,
    Crop2000.SUGC.name,
    Crop2000.WHEA.name,
}
assert all(
    hasattr(crop_cls, name)
    for crop_cls in (Crop2000, Crop2005, Crop2010, Crop2020)
    for name in SHARED_CROP_NAMES
)
assert all(
    len({crop_cls[name].value for crop_cls in (Crop2000, Crop2005, Crop2010, Crop2020)})
    == 1
    for name in SHARED_CROP_NAMES
)

# The 2000 snapshot contains 6 "group" crops which are disaggregated in later snapshots.
# Assuming the mix of crops within a group for a given 10km pixel is unchanged from 2000
# to a later reference year, we can decompose the 2000 crop group into its constituent crops.
GROUP_TO_CONSTITUENT_NAMES: dict[str, set[str]] = {
    Crop2000.BANP.name: {Crop2005.BANA.name, Crop2005.PLNT.name},
    Crop2000.COFF.name: {Crop2005.ACOF.name, Crop2005.RCOF.name},
    Crop2000.MILL.name: {Crop2005.PMIL.name, Crop2005.SMIL.name},
    Crop2000.OOIL.name: {
        Crop2005.CNUT.name,
        Crop2005.OILP.name,
        Crop2005.SUNF.name,
        Crop2005.RAPE.name,
        Crop2005.SESA.name,
        Crop2005.OOIL.name,
    },
    Crop2000.OPUL.name: {
        Crop2005.CHIC.name,
        Crop2005.COWP.name,
        Crop2005.PIGE.name,
        Crop2005.LENT.name,
        Crop2005.OPUL.name,
    },
    Crop2000.SWPY.name: {Crop2005.SWPO.name, Crop2005.YAMS.name},
}
DECOMPOSITION_YEAR = 2000
# Reference years for the decomposition, pooled into a single within-group split rather than
# ranked by nearness to 2000: a pixel the 2000 snapshot uses but 2005 does not is an
# inconsistency between MapSPAM's snapshots, not a crop that arrived later.
DECOMPOSITION_REFERENCE_YEARS = (2005, 2010, 2020)
CONSTITUENT_TO_GROUP_NAME = {
    constituent: group_name
    for group_name, constituents in GROUP_TO_CONSTITUENT_NAMES.items()
    for constituent in constituents
}
assert set(SHARED_CROP_NAMES).isdisjoint(GROUP_TO_CONSTITUENT_NAMES)

RECOVERABLE_CROP_NAMES = SHARED_CROP_NAMES | set(CONSTITUENT_TO_GROUP_NAME)
assert len(RECOVERABLE_CROP_NAMES) == len(SHARED_CROP_NAMES) + len(
    CONSTITUENT_TO_GROUP_NAME
)

# When an "other" group contains no area in ANY reference year, there is no basis for
# dividing the crops within the decomposition year and we divide crops evenly across the
# constituents.  This can result in tropical crops being attributed to non-tropical geos.
# Instead, we'd like to route the unattributable remainder to the residual.
GROUP_TO_RESIDUAL_NAME: dict[str, str] = {
    Crop2000.OOIL.name: Crop2005.OOIL.name,
    Crop2000.OPUL.name: Crop2005.OPUL.name,
}
assert all(
    residual in GROUP_TO_CONSTITUENT_NAMES[group]
    for group, residual in GROUP_TO_RESIDUAL_NAME.items()
)


CropClsType = type[Crop2000] | type[Crop2005] | type[Crop2010] | type[Crop2020]
YEAR_TO_CROP_CLS: dict[int, CropClsType] = {
    2000: Crop2000,
    2005: Crop2005,
    2010: Crop2010,
    2020: Crop2020,
}
CROP_CLS_TO_YEAR: dict[CropClsType, int] = {
    crop_cls: year for year, crop_cls in YEAR_TO_CROP_CLS.items()
}
YEARS = sorted(YEAR_TO_CROP_CLS)

# Every crop name crossing this module's boundary is one of two things.  A *canonical* name is a
# `CANONICAL_CROP_CLS` name -- the spelling `statistical.Crop` and `RECOVERABLE_CROP_NAMES` use,
# stable across years, and what `get_canonical_quantity` takes.  A *reported* name is whatever a
# single year's own taxonomy calls that crop, which is what the bands are keyed by and what
# `get_band_name` and `get_reported_quantity` take.  2005 is the canonical one because it is the
# finest taxonomy every later year can be renamed back into, and the one the decomposition splits
# 2000's groups into.  `get_reported_crop_name` is the only hop between the two.
CANONICAL_CROP_CLS: type[Crop2005] = Crop2005


def get_renamed_crop_name(canonical_crop_name: str, year: int) -> str:
    """`canonical_crop_name` as `year`'s taxonomy spells it, matched on the shared crop label.

    The rename alone -- it cannot see groups, so it is wrong for a constituent in the decompose
    year.  `get_reported_crop_name` is the one to call; this stays separate because `datasets_test`
    checks that function against it.
    """
    return YEAR_TO_CROP_CLS[year](CANONICAL_CROP_CLS[canonical_crop_name].value).name


assert all(
    get_renamed_crop_name(canonical_crop_name=constituent, year=year)
    for constituent in CONSTITUENT_TO_GROUP_NAME
    for year in YEARS
    if year != DECOMPOSITION_YEAR
)


def is_reported_as_group(canonical_crop_name: str, year: int) -> bool:
    """Whether `year` reports `canonical_crop_name` only as part of a coarser group.

    True for a group constituent in the decompose year and nothing else: that snapshot's taxonomy
    carries no band of its own for the crop, which is the reason the decomposition exists.
    """
    return (
        year == DECOMPOSITION_YEAR and canonical_crop_name in CONSTITUENT_TO_GROUP_NAME
    )


def get_reported_crop_name(canonical_crop_name: str, year: int) -> str:
    """The crop `year` reports `canonical_crop_name`'s quantity under.

    Its own name in most years, renamed where that year renames it -- 2020 calls arabica coffee
    COFF and small millet MILL -- and the group it belongs to where the year is too coarse to
    name it separately.  This is the canonical-to-reported hop every band lookup needs: `dset` is
    keyed by reported names, so `get_band_name` and `get_reported_quantity` take the result.
    """
    if is_reported_as_group(canonical_crop_name=canonical_crop_name, year=year):
        return CONSTITUENT_TO_GROUP_NAME[canonical_crop_name]
    else:
        return get_renamed_crop_name(canonical_crop_name=canonical_crop_name, year=year)


# The crops each snapshot reports that no recoverable crop accounts for, in that snapshot's own
# naming: nothing can divide by them, so they can only be carried as a lump.  Spelled out rather
# than derived so that a change to MapSPAM's taxonomy breaks the import, instead of silently
# changing which crops the share denominator can reach.
YEAR_TO_UNRECOVERABLE_CROP_NAMES: dict[int, set[str]] = {
    2000: {"OFIB", "OTHE"},
    2005: {
        "COCO",
        "OCER",
        "OFIB",
        "ORTS",
        "REST",
        "TEAS",
        "TEMF",
        "TOBA",
        "TROF",
        "VEGE",
    },
    2010: {
        "COCO",
        "OCER",
        "OFIB",
        "ORTS",
        "REST",
        "TEAS",
        "TEMF",
        "TOBA",
        "TROF",
        "VEGE",
    },
    2020: {
        "CITR",
        "COCO",
        "OCER",
        "OFIB",
        "ONIO",
        "ORTS",
        "REST",
        "RUBB",
        "TEAS",
        "TEMF",
        "TOBA",
        "TOMA",
        "TROF",
        "VEGE",
    },
}
assert {
    year: {e.name for e in YEAR_TO_CROP_CLS[year]}
    - {
        get_reported_crop_name(canonical_crop_name=name, year=year)
        for name in RECOVERABLE_CROP_NAMES
    }
    for year in YEARS
} == YEAR_TO_UNRECOVERABLE_CROP_NAMES


# Snapshot pairs whose crop names can be matched directly.  Every pair involving the decompose year
# is excluded: it is a different MapSPAM release (v3.0.7 against v3.2) and its unrecoverable
# vocabulary is nearly disjoint from 2005's -- a catch-all OTHE leaves and nine crops appear in its
# place, holding 560 Mha against their 161 and correlating at r = 0.17 per pixel.  A name on only
# one side of that boundary says the release did not use it, not that the crop was absent.
SPANS_WITH_COMPARABLE_CROP_NAMES: set[tuple[int, int]] = {
    (2005, 2010),
    (2005, 2020),
    (2010, 2020),
}
assert {
    (before, after)
    for before in YEARS
    for after in YEARS
    if before < after and DECOMPOSITION_YEAR not in (before, after)
} == SPANS_WITH_COMPARABLE_CROP_NAMES


class Quantity(enum.StrEnum):
    PRODUCTION = "mt"
    PHYSICAL_AREA = "ha"


def get_raster_dataset(
    crop_cls: CropClsType,
    dataset_id: int,
    no_data: float | int | None,
    prefix: str,
    quantity: Quantity,
    suffix: str,
) -> base.RasterDataset:
    def save_tile_id_to_local_path(local_path: str, tile_id: str) -> None:
        logger.info("POST'ing to the guestbook for a signed URL")
        with utils.get_requests_session().request(
            json=config.Config.from_dot_env().get_json_as_object(
                "harvard_dataverse_guestbook_json"
            ),
            method="POST",
            url=f"https://dataverse.harvard.edu/api/access/datafile/{dataset_id:d}",
        ) as response:
            response.raise_for_status()
            signed_url: str = response.json()["data"]["signedUrl"]

        with tempfile.TemporaryDirectory() as local_dir:
            path_to_zip = os.path.join(local_dir, "data.zip")
            utils.save_remote_url_to_local_path(
                local_path=path_to_zip, params={}, remote_url=signed_url
            )
            logger.info(f"Extracting .tif from {path_to_zip=:s}")
            filenames = [f"{prefix:s}{crop.name}{suffix:s}.tif" for crop in crop_cls]
            a_filename = next(iter(filenames))
            with rasterio.open(f"/vsizip/{path_to_zip:s}/{a_filename:s}") as dataset:
                profile = dataset.meta.copy()
                src_transform = dataset.transform
                src_height, src_width = dataset.shape

            res_x, res_y = src_transform.a, -src_transform.e
            full_width, full_height = round(360 / res_x), round(180 / res_y)
            col_off = round((src_transform.c - (-180)) / res_x)
            row_off = round((90 - src_transform.f) / res_y)

            profile.update(
                blockxsize=512,
                blockysize=512,
                compress="deflate",
                count=len(filenames),
                driver="GTiff",
                height=full_height,
                nodata=no_data,
                predictor=3,
                tiled=True,
                transform=rasterio.Affine(res_x, 0, -180, 0, -res_y, 90),
                width=full_width,
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(fp=local_path, mode="w", **profile) as dataset:
                for idx, filename in enumerate(sorted(filenames), start=1):
                    with rasterio.open(
                        f"/vsizip/{path_to_zip:s}/{filename:s}"
                    ) as source:
                        data = source.read(1)
                    data[~numpy.isfinite(data) | (data <= -1e30)] = no_data
                    canvas = numpy.full(
                        (full_height, full_width), no_data, dtype=data.dtype
                    )
                    canvas[
                        row_off : row_off + src_height, col_off : col_off + src_width
                    ] = data
                    dataset.write(canvas, idx)

    return base.RasterDataset(
        band_names=[
            ":".join((crop.value.lower().replace(" ", "-"), quantity.value))
            for crop in crop_cls
        ],
        band_type=base.BandType.EXTENSIVE,
        no_data=no_data,
        partitioning=tiling.Partitioning.WHOLE_WORLD,
        product_name=f"mapspam-{quantity.name.lower().replace('_', '-'):s}-{CROP_CLS_TO_YEAR[crop_cls]:d}",
        save_tile_id_to_local_path=save_tile_id_to_local_path,
        source_name="ifpri",
        version="v0",
    )


PRODUCTION_2000 = get_raster_dataset(
    crop_cls=Crop2000,
    dataset_id=3666795,
    no_data=-1,
    prefix="spam2000V3r107_global_R_",
    quantity=Quantity.PRODUCTION,
    suffix="_A",
)
PHYSICAL_AREA_2000 = get_raster_dataset(
    crop_cls=Crop2000,
    dataset_id=3666793,
    no_data=-1,
    prefix="spam2000V3r107_global_P_",
    quantity=Quantity.PHYSICAL_AREA,
    suffix="_A",
)
PRODUCTION_2005 = get_raster_dataset(
    crop_cls=Crop2005,
    dataset_id=3086560,
    no_data=float("nan"),
    prefix="geotiff_global_prod/SPAM2005V3r2_global_P_TA_",
    quantity=Quantity.PRODUCTION,
    suffix="_A",
)
PHYSICAL_AREA_2005 = get_raster_dataset(
    crop_cls=Crop2005,
    dataset_id=3086559,
    no_data=float("nan"),
    prefix="geotiff_global_phys_area/SPAM2005V3r2_global_A_TA_",
    quantity=Quantity.PHYSICAL_AREA,
    suffix="_A",
)
PRODUCTION_2010 = get_raster_dataset(
    crop_cls=Crop2010,
    dataset_id=3985009,
    prefix="spam2010V2r0_global_P_",
    no_data=-1,
    quantity=Quantity.PRODUCTION,
    suffix="_A",
)
PHYSICAL_AREA_2010 = get_raster_dataset(
    crop_cls=Crop2010,
    dataset_id=3985010,
    prefix="spam2010V2r0_global_A_",
    no_data=-1,
    quantity=Quantity.PHYSICAL_AREA,
    suffix="_A",
)
PRODUCTION_2020 = get_raster_dataset(
    crop_cls=Crop2020,
    dataset_id=13827043,
    no_data=float("nan"),
    prefix="spam2020V2r2_global_production/spam2020_V2r2_global_P_",
    quantity=Quantity.PRODUCTION,
    suffix="_A",
)
PHYSICAL_AREA_2020 = get_raster_dataset(
    crop_cls=Crop2020,
    dataset_id=13827041,
    no_data=float("nan"),
    prefix="spam2020V2r2_global_physical_area/spam2020_V2r2_global_A_",
    quantity=Quantity.PHYSICAL_AREA,
    suffix="_A",
)

YEAR_TO_PRODUCTION_DATASET = {
    2000: PRODUCTION_2000,
    2005: PRODUCTION_2005,
    2010: PRODUCTION_2010,
    2020: PRODUCTION_2020,
}
assert set(YEAR_TO_PRODUCTION_DATASET) == set(YEAR_TO_CROP_CLS)
YEAR_TO_PHYSICAL_AREA_DATASET = {
    2000: PHYSICAL_AREA_2000,
    2005: PHYSICAL_AREA_2005,
    2010: PHYSICAL_AREA_2010,
    2020: PHYSICAL_AREA_2020,
}
assert set(YEAR_TO_PHYSICAL_AREA_DATASET) == set(YEAR_TO_PRODUCTION_DATASET)


def get_band_name(quantity: Quantity, reported_crop_name: str, year: int) -> str:
    """The fully-qualified band carrying `reported_crop_name`'s `quantity` in `year`.

    Takes a reported name -- it indexes `year`'s own taxonomy, so a canonical name that year
    renames or groups is a `KeyError`.  Resolve with `get_reported_crop_name` first.

    Positional: the band list is built from the same enum in the same order, so a crop's index in
    the taxonomy is its index among the bands.  `datasets_test` pins that for every year and
    quantity, since nothing in the band names themselves would reveal a reordering.
    """
    crop_cls = YEAR_TO_CROP_CLS[year]
    name_to_idx = {name: idx for idx, name in enumerate(e.name for e in crop_cls)}
    idx = name_to_idx[reported_crop_name]
    year_to_dataset = (
        YEAR_TO_PHYSICAL_AREA_DATASET
        if quantity == Quantity.PHYSICAL_AREA
        else YEAR_TO_PRODUCTION_DATASET
    )
    return year_to_dataset[year].fully_qualified_band_names[idx]


def get_reported_quantity(
    dset: xarray.Dataset, quantity: Quantity, reported_crop_name: str, year: int
) -> xarray.DataArray:
    """`reported_crop_name`'s `quantity` exactly as `year` stores it.

    The as-reported twin of `get_canonical_quantity`: same quantity, no decomposition and no
    renaming, so it takes a reported name where that one takes a canonical one.
    """
    variable_name = get_band_name(
        quantity=quantity, reported_crop_name=reported_crop_name, year=year
    )
    # NB: a crop absent from a snapshot (or no-data there) counts as zero hectares
    return dset[variable_name].fillna(0)


def get_canonical_quantity(
    canonical_crop_name: str,
    dset: xarray.Dataset,
    quantity: Quantity,
    year: int,
) -> xarray.DataArray:
    """`canonical_crop_name`'s `quantity` in `year`.

    Does whatever `year` requires to honor a canonical name -- following that year's renaming,
    and decomposing a group where the year is too coarse to name the crop -- so that one name
    means one crop across every year.  `get_reported_quantity` is the raw counterpart.
    """
    if is_reported_as_group(canonical_crop_name=canonical_crop_name, year=year):
        # Decompose the grouped crops into their constituents, assuming the within-group
        # proportions match those the reference years pool to.  Each quantity is split by its
        # own distribution -- splitting production by AREA share would assume every constituent
        # of the group yields the same in this pixel, and bananas outyield plantains 2-3x.
        group_name = CONSTITUENT_TO_GROUP_NAME[canonical_crop_name]
        siblings = sorted(GROUP_TO_CONSTITUENT_NAMES[group_name])

        def reference(canonical_name: str) -> xarray.DataArray:
            ret = sum(
                get_reported_quantity(
                    dset=dset,
                    quantity=quantity,
                    reported_crop_name=get_reported_crop_name(
                        canonical_crop_name=canonical_name, year=reference_year
                    ),
                    year=reference_year,
                )
                for reference_year in DECOMPOSITION_REFERENCE_YEARS
            )
            assert isinstance(ret, xarray.DataArray)
            return ret

        reference_group = sum(reference(canonical_name=s) for s in siblings)
        assert isinstance(reference_group, xarray.DataArray)
        constituent_share = xarray.where(
            reference_group > 0,
            reference(canonical_name=canonical_crop_name)
            # NB: this avoids `RuntimeWarning: invalid value encountered in divide`
            / reference_group.where(reference_group > 0, other=1),
            # No reference year places the group here
            (
                # The group has a catch-all constituent, so give it the whole unattributable
                # remainder rather than handing a slice to every named sibling
                float(canonical_crop_name == GROUP_TO_RESIDUAL_NAME[group_name])
                if group_name in GROUP_TO_RESIDUAL_NAME
                # No catch-all to route to, so an even split is the only option among named crops
                else 1 / len(siblings)
            ),
        )
        return (
            get_reported_quantity(
                dset=dset,
                quantity=quantity,
                # NB: the group is already a reported name -- it exists only in the decompose year
                reported_crop_name=group_name,
                year=DECOMPOSITION_YEAR,
            )
            * constituent_share
        )
    else:
        # Simple lookup
        return get_reported_quantity(
            dset=dset,
            quantity=quantity,
            reported_crop_name=get_reported_crop_name(
                canonical_crop_name=canonical_crop_name, year=year
            ),
            year=year,
        )
