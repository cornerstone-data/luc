import collections.abc
import math
import types
import typing
import warnings

import pytest
import xarray

from jdluc import emit
from jdluc.datasets import faostat, gpw_livestock, ifpri_mapspam
from jdluc.datasets.gpw_livestock import (
    Livestock,
    get_band_name_to_livestock_units,
    get_livestock_to_grazing_share,
)
from jdluc.statistical import (
    HERD_HEADS_COLUMN,
    MAPSPAM_SPANS,
    Commodity,
    Crop,
    get_commodity_name_to_totals,
    get_crop_to_area_share,
    get_crop_to_expansion_share,
)

AREA = ifpri_mapspam.Quantity.PHYSICAL_AREA
PRODUCTION = ifpri_mapspam.Quantity.PRODUCTION
# Crops `Crop` does claim, but which are not in every taxonomy under one name -- read off the
# taxonomy rather than hardcoded so a MapSPAM revision cannot quietly turn them into something
# else.  Nothing to do with MapSPAM's "Other ..." crops: OTHER_0 is arabica coffee.
OTHER_0, OTHER_1 = sorted(
    {e.name for e in ifpri_mapspam.YEAR_TO_CROP_CLS[2005]}
    - ifpri_mapspam.SHARED_CROP_NAMES
)[:2]
# The same, narrowed to one the 2020 snapshot also names (ACOF, the first for 2005, is not).
OTHER_2020 = sorted(
    {e.name for e in ifpri_mapspam.YEAR_TO_CROP_CLS[2005]}
    & {e.name for e in ifpri_mapspam.YEAR_TO_CROP_CLS[2020]}
    - ifpri_mapspam.SHARED_CROP_NAMES
)[0]
# Crops no `Crop` claims at all, so nothing can divide by them: they only ever move as a lump.
UNATTRIBUTED_0, UNATTRIBUTED_1 = sorted(
    ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[2005]
)[:2]


# `get_hectares_per_pixel` reads its spacing off these coords, and uniform values keep
# `get_value` below exact
GRID = {"y": [0.1, 0.2], "x": [0.1, 0.2]}
PIXELS = 4
# FAO's livestock units per head for South America, written out so every expected share below
# can be checked by hand
UNITS_PER_HEAD = {
    gpw_livestock.Species.BUFFALO: 0.70,
    gpw_livestock.Species.CATTLE: 0.70,
    gpw_livestock.Species.GOAT: 0.10,
    gpw_livestock.Species.HORSE: 0.65,
    gpw_livestock.Species.SHEEP: 0.10,
}


def get_value(darray: xarray.DataArray) -> float:
    return float(darray.mean())


def get_dset(
    values: dict[tuple[ifpri_mapspam.Quantity, int], dict[str, float]],
    density: dict[tuple[gpw_livestock.Species, int], float] | None = None,
    pasture_fraction: dict[int, float] | None = None,
    units_per_head: dict[gpw_livestock.Species, float] | None = None,
) -> xarray.Dataset:
    # A band for every crop of every (quantity, year) -- the real dset always carries all four
    # snapshots.  Unset crops, and unset years, are 0.  Pasture rides along as a cell fraction, and
    # each grazer as a density in heads/ha, weighed into livestock units as `workflow` weighs them,
    # and each cell's hectares are measured on the whole grid, as `workflow` measures them.
    dset = xarray.Dataset(
        {
            ifpri_mapspam.get_band_name(
                quantity=quantity, reported_crop_name=crop.name, year=year
            ): xarray.DataArray(
                float(values.get((quantity, year), {}).get(crop.name, 0.0))
            )
            for quantity in ifpri_mapspam.Quantity
            for year in ifpri_mapspam.YEARS
            for crop in ifpri_mapspam.YEAR_TO_CROP_CLS[year]
        }
        | {
            f"pasture:fraction:{year:d}": xarray.DataArray(
                float((pasture_fraction or {}).get(year, 0.0))
            )
            for year in ifpri_mapspam.YEARS
        }
        | {
            gpw_livestock.get_band_name(species=species, year=year): xarray.DataArray(
                float(heads_per_ha)
            )
            for (species, year), heads_per_ha in (
                dict.fromkeys(gpw_livestock.SPECIES_YEARS, 0.0) | (density or {})
            ).items()
        }
    ).expand_dims(GRID)
    return dset.assign(
        get_band_name_to_livestock_units(
            dset=dset, species_to_units_per_head=units_per_head or UNITS_PER_HEAD
        )
        | {
            "hectares-per-cell:ha": emit.get_hectares_per_pixel(
                darray=dset["pasture:fraction:2020"]
            )
        }
    )


def get_dset_for_areas(
    areas: dict[int, dict[str, float]],
    density: dict[tuple[gpw_livestock.Species, int], float] | None = None,
    pasture_fraction: dict[int, float] | None = None,
) -> xarray.Dataset:
    return get_dset(
        density=density,
        pasture_fraction=pasture_fraction,
        values={(AREA, year): by_crop for year, by_crop in areas.items()},
    )


def get_density(
    heads_per_ha: dict[gpw_livestock.Species, float],
) -> dict[tuple[gpw_livestock.Species, int], float]:
    """The same density for each grazer in every snapshot."""
    return {
        (species, year): density
        for species, density in heads_per_ha.items()
        for year in gpw_livestock.YEARS
    }


HECTARES_PER_CELL = get_value(darray=get_dset(values={})["hectares-per-cell:ha"])


@pytest.mark.parametrize(
    ("areas", "before", "after", "expected"),
    (
        pytest.param(
            {2005: {}, 2010: {"MAIZ": 100.0, "SOYB": 50.0, OTHER_0: 50.0}},
            2005,
            2010,
            {Crop.MAIZE: 0.5, Crop.SOYBEAN: 0.25},
            id="each-crop-takes-its-own-gross-expansion-over-the-total",
        ),
        pytest.param(
            {2005: {"MAIZ": 100.0}, 2010: {"MAIZ": 0.0, "SOYB": 50.0}},
            2005,
            2010,
            {Crop.MAIZE: 0.0, Crop.SOYBEAN: 1.0},
            id="a-contracting-crop-clips-to-zero-rather-than-going-negative",
        ),
        pytest.param(
            {2005: {}, 2010: {}},
            2005,
            2010,
            {Crop.MAIZE: 0.0},
            id="no-expansion-anywhere-masks-the-total-instead-of-dividing-by-zero-then-fills-it-back-to-zero",
        ),
        pytest.param(
            # 2000 names only the coarse BANP group, so its 100 ha has to be split before any
            # expansion can be measured.  2005 and 2020 pool to BANA 205 / PLNT 35 of 240 (2010
            # is empty), giving BANA 100 * 205/240 == 85.4 ha in 2000 and PLNT 14.6.  Note 2020
            # does double duty: it is both a decomposition reference and the `after` snapshot.
            # BANA then expands 175 - 85.4 == 89.6 and PLNT 25 - 14.6 == 10.4, of 100 together.
            {
                2000: {"BANP": 100.0},
                2005: {"BANA": 30.0, "PLNT": 10.0},
                2020: {"BANA": 175.0, "PLNT": 25.0},
            },
            2000,
            2020,
            {Crop.BANANA: 0.8958333, Crop.PLANTAIN: 0.1041667},
            id="a-2000-group-is-decomposed-by-the-pooled-reference-fractions",
        ),
        pytest.param(
            {2000: {"BANP": 100.0}, 2005: {}, 2020: {"BANA": 80.0, "PLNT": 120.0}},
            2000,
            2020,
            {Crop.BANANA: 0.40, Crop.PLANTAIN: 0.60},
            id="a-later-reference-decomposes-the-group-when-the-nearest-is-absent-rather-than-an-even-split",
        ),
        pytest.param(
            # OTHER_0 expands by exactly the 100 ha OTHER_1 gives up.  Netting the pair before
            # clipping would leave a denominator of 100 rather than 200, and MAIZE and OTHER_0
            # would each read 100/100 -- charging the cell's emissions twice over.
            {2005: {OTHER_1: 100.0}, 2010: {"MAIZ": 100.0, OTHER_0: 100.0}},
            2005,
            2010,
            {Crop.MAIZE: 0.5, Crop(OTHER_0): 0.5},
            id="clipping-each-expansion-before-summing-keeps-a-contracting-sibling-from-pushing-shares-above-one",
        ),
        pytest.param(
            # Netting is all that can honestly be said about a lump nothing divides by:
            # UNATTRIBUTED_1's expansion is cancelled by UNATTRIBUTED_0 giving up as much.
            {2005: {UNATTRIBUTED_0: 50.0}, 2010: {"MAIZ": 100.0, UNATTRIBUTED_1: 50.0}},
            2005,
            2010,
            {Crop.MAIZE: 1.0},
            id="crops-it-cannot-attribute-stay-lumped-and-net-against-each-other-leaving-maize-the-whole-cell",
        ),
        pytest.param(
            # MapSPAM 2010 counted rubber in rest of crops
            {2010: {"REST": 100.0}, 2020: {"MAIZ": 100.0, "RUBB": 100.0}},
            2010,
            2020,
            {Crop.MAIZE: 1.0},
            id="a-crop-2020-first-names-is-not-expansion-where-it-only-left-its-earlier-group",
        ),
        pytest.param(
            {2010: {"REST": 100.0}, 2020: {"MAIZ": 100.0, "RUBB": 200.0}},
            2010,
            2020,
            {Crop.MAIZE: 0.5},
            id="a-crop-2020-first-names-that-outgrew-its-earlier-group-dilutes",
        ),
        pytest.param(
            {2010: {}, 2020: {"MAIZ": 100.0, UNATTRIBUTED_0: 100.0}},
            2010,
            2020,
            {Crop.MAIZE: 0.5},
            id="a-crop-reported-in-both-snapshots-really-did-expand-and-still-dilutes",
        ),
        pytest.param(
            # 2000 is a different MapSPAM release and shares almost no unattributed names with
            # 2005, so the lump is compared as a whole rather than name by name
            {2000: {}, 2005: {"MAIZ": 100.0, UNATTRIBUTED_0: 100.0}},
            2000,
            2005,
            {Crop.MAIZE: 0.5},
            id="the-lump-still-dilutes-across-the-2000-release-boundary",
        ),
    ),
)
def test_get_crop_to_expansion_share(
    areas: dict[int, dict[str, float]],
    before: int,
    after: int,
    expected: dict[Crop, float],
) -> None:
    dset = get_dset_for_areas(areas=areas)
    # Every case asks for all of `Crop`, so each one also checks the invariant that the shares
    # never over-attribute what was converted to cropland
    shares = get_crop_to_expansion_share(
        after=after, before=before, crops=tuple(Crop), dset=dset
    )
    assert sum(get_value(darray=shares[crop]) for crop in Crop) <= 1.0
    assert {crop: get_value(shares[crop]) for crop in expected} == pytest.approx(
        expected
    )


def test_get_crop_to_expansion_share_does_not_renormalise_onto_the_crops_asked_for() -> (
    None
):
    # MAIZE's half of the cell's expansion, whichever other crops are asked for alongside it
    shares = get_crop_to_expansion_share(
        after=2010,
        before=2005,
        crops=(Crop.MAIZE,),
        dset=get_dset_for_areas(
            areas={2005: {}, 2010: {"MAIZ": 100.0, "SOYB": 50.0, OTHER_0: 50.0}}
        ),
    )
    assert get_value(shares[Crop.MAIZE]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("areas", "year", "expected"),
    (
        pytest.param(
            {2020: {"MAIZ": 100.0, "SOYB": 50.0, OTHER_2020: 50.0}},
            2020,
            {Crop.MAIZE: 0.5, Crop.SOYBEAN: 0.25},
            id="each-crop-takes-its-own-area-over-the-total-occupied",
        ),
        pytest.param(
            {2020: {}},
            2020,
            {Crop.MAIZE: 0.0},
            id="a-cell-no-crop-occupies-is-zero-not-nan",
        ),
    ),
)
def test_get_crop_to_area_share(
    areas: dict[int, dict[str, float]], year: int, expected: dict[Crop, float]
) -> None:
    dset = get_dset_for_areas(areas=areas)
    shares = get_crop_to_area_share(crops=tuple(Crop), dset=dset, year=year)
    assert sum(get_value(share) for share in shares.values()) <= 1.0
    assert {crop: get_value(shares[crop]) for crop in expected} == pytest.approx(
        expected
    )


def test_get_crop_to_area_share_does_not_renormalise_onto_the_crops_asked_for() -> None:
    # MAIZE's half of the occupied cropland, whichever other crops are asked for alongside it
    shares = get_crop_to_area_share(
        crops=(Crop.MAIZE,),
        dset=get_dset_for_areas(
            areas={2020: {"MAIZ": 100.0, "SOYB": 50.0, OTHER_2020: 50.0}}
        ),
        year=2020,
    )
    assert get_value(shares[Crop.MAIZE]) == pytest.approx(0.5)


def test_get_crop_to_area_share_survives_zero_expansion() -> None:
    # The regression this function exists for: long-established cropland whose area has not
    # changed over the window.  Expansion share drops it entirely (and with it 100% of the
    # cell's peatland occupation emissions); area share still allocates it.
    areas = {"MAIZ": 100.0, "SOYB": 100.0}
    dset = get_dset_for_areas(areas={2000: areas, 2020: areas})
    expansion_shares = get_crop_to_expansion_share(
        after=2020, before=2000, crops=(Crop.MAIZE,), dset=dset
    )
    assert get_value(expansion_shares[Crop.MAIZE]) == 0.0
    area_shares = get_crop_to_area_share(crops=(Crop.MAIZE,), dset=dset, year=2020)
    assert get_value(area_shares[Crop.MAIZE]) == 0.5


def test_get_crop_to_expansion_share_is_reclassification_invariant() -> None:
    # The same +50 ha of non-MAIZE expansion, spread over one band and then over two.  MAIZE's
    # share depends on how much other cropland expanded, never on how MapSPAM chose to file it.
    # ("Residual" is avoided here: in `ifpri_mapspam` it means a group's catch-all constituent.)
    lumped = get_dset_for_areas(areas={2005: {}, 2010: {"MAIZ": 100.0, OTHER_0: 50.0}})
    split = get_dset_for_areas(
        areas={2005: {}, 2010: {"MAIZ": 100.0, OTHER_0: 20.0, OTHER_1: 30.0}}
    )
    shares_lumped = get_crop_to_expansion_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=lumped
    )
    shares_split = get_crop_to_expansion_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=split
    )
    assert get_value(shares_lumped[Crop.MAIZE]) == get_value(shares_split[Crop.MAIZE])


def test_get_crop_to_expansion_share_tolerates_nodata_absent_crops() -> None:
    # Real rasters carry nodata, not zero, where a crop is absent.  That must not poison the
    # denominator with NaN and take the whole cell down with it.
    dset = get_dset_for_areas(areas={2005: {}, 2010: {"MAIZ": 100.0}})
    dset[
        ifpri_mapspam.get_band_name(
            quantity=AREA, reported_crop_name=OTHER_0, year=2010
        )
    ] = xarray.zeros_like(other=dset["pasture:fraction:2020"]) * float("nan")
    shares = get_crop_to_expansion_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=dset
    )
    # 100 / 100; NaN absent crop counts as 0 ha
    assert get_value(shares[Crop.MAIZE]) == 1.0


def test_get_crop_to_expansion_share_decomposition_does_not_warn_on_zero_reference() -> (
    None
):
    # No reference year places BANP, so the within-group split divides by a zero pool.  The
    # decomposition guards that denominator rather than dividing and discarding the NaN, which
    # would work but bury every run in RuntimeWarnings.
    dset = get_dset_for_areas(areas={2000: {"BANP": 100.0}})
    with warnings.catch_warnings():
        # Any 0/0 divide becomes a failure
        warnings.simplefilter("error", RuntimeWarning)
        get_crop_to_expansion_share(
            after=2020,
            before=2000,
            crops=(Crop.BANANA, Crop.PLANTAIN),
            dset=dset,
        )


@pytest.mark.parametrize(
    ("values", "quantity", "year", "expected"),
    (
        pytest.param(
            {(PRODUCTION, 2020): {"MAIZ": 42.0}},
            PRODUCTION,
            2020,
            {"MAIZ": 42.0},
            id="a-shared-crop-is-a-straight-per-year-lookup",
        ),
        # The next two share a dset whose 2005 area and production distributions disagree:
        # plantain holds a quarter of the area but only a tenth of the production, i.e. it
        # yields less.  Each quantity must therefore be split by its own distribution.
        pytest.param(
            {
                (AREA, 2005): {"BANA": 30.0, "PLNT": 10.0},
                (PRODUCTION, 2005): {"BANA": 180.0, "PLNT": 20.0},
                (AREA, 2000): {"BANP": 400.0},
                (PRODUCTION, 2000): {"BANP": 200.0},
            },
            AREA,
            2000,
            # 400 ha of BANP split 3:1, the 2005 area ratio
            {"BANA": 300.0, "PLNT": 100.0},
            id="area-splits-by-the-area-distribution",
        ),
        pytest.param(
            {
                (AREA, 2005): {"BANA": 30.0, "PLNT": 10.0},
                (PRODUCTION, 2005): {"BANA": 180.0, "PLNT": 20.0},
                (AREA, 2000): {"BANP": 400.0},
                (PRODUCTION, 2000): {"BANP": 200.0},
            },
            PRODUCTION,
            2000,
            # 200 t of BANP split 9:1, the 2005 production ratio.  Inheriting the area split
            # above would have handed plantain 50 t instead of 20 t.
            {"BANA": 180.0, "PLNT": 20.0},
            id="production-splits-by-its-own-distribution-not-the-area-one",
        ),
        pytest.param(
            # The reference years disagree: 2005 alone would have given BANA 150 t.
            {
                (PRODUCTION, 2005): {"BANA": 30.0, "PLNT": 10.0},
                (PRODUCTION, 2010): {"BANA": 5.0, "PLNT": 75.0},
                (PRODUCTION, 2020): {"BANA": 5.0, "PLNT": 75.0},
                (PRODUCTION, 2000): {"BANP": 200.0},
            },
            PRODUCTION,
            2000,
            {"BANA": 40.0, "PLNT": 160.0},
            id="every-reference-year-pools-into-one-split-rather-than-the-nearest-deciding-alone",
        ),
        pytest.param(
            # An even split would have fabricated 100 t of plantain in a pixel MapSPAM says
            # grows nothing but bananas.
            {(PRODUCTION, 2020): {"BANA": 40.0}, (PRODUCTION, 2000): {"BANP": 200.0}},
            PRODUCTION,
            2000,
            {"BANA": 200.0, "PLNT": 0.0},
            id="a-later-reference-carries-the-whole-split-when-2005-and-2010-are-empty",
        ),
        pytest.param(
            # The fallback the two OOIL cases below exist to avoid wherever a catch-all does
            # exist.
            {(PRODUCTION, 2000): {"BANP": 200.0}},
            PRODUCTION,
            2000,
            {"BANA": 100.0, "PLNT": 100.0},
            id="an-even-split-is-the-last-resort-when-no-reference-places-a-group-with-no-catch-all",
        ),
        pytest.param(
            # The reference years pool the same way for every group, catch-all or not.
            {
                (PRODUCTION, 2005): {},
                (PRODUCTION, 2020): {"RAPE": 300.0, "SUNF": 100.0},
                (PRODUCTION, 2000): {"OOIL": 400.0},
            },
            PRODUCTION,
            2000,
            {
                "RAPE": 300.0,
                "SUNF": 100.0,
                "OOIL": 0.0,
                "CNUT": 0.0,
                "OILP": 0.0,
                "SESA": 0.0,
            },
            id="naming-a-crop-in-any-reference-year-beats-the-catch-all",
        ),
        pytest.param(
            # Splitting evenly instead put a sixth of Canada's "Other Oil Crops" under oil palm
            # and another sixth under coconut, neither of which Canada grows.
            {(AREA, 2005): {}, (PRODUCTION, 2000): {"OOIL": 600.0}},
            PRODUCTION,
            2000,
            {
                "OOIL": 600.0,
                "CNUT": 0.0,
                "OILP": 0.0,
                "RAPE": 0.0,
                "SESA": 0.0,
                "SUNF": 0.0,
            },
            id="the-catch-all-takes-the-whole-group-when-no-reference-year-places-any-constituent",
        ),
        pytest.param(
            {
                (PRODUCTION, 2005): {"OILP": 300.0, "CNUT": 100.0},
                (PRODUCTION, 2000): {"OOIL": 400.0},
            },
            PRODUCTION,
            2000,
            {"OILP": 300.0, "CNUT": 100.0, "OOIL": 0.0},
            id="reference-shares-beat-the-catch-all-constituent-whenever-a-reference-year-places-the-group",
        ),
    ),
)
def test_get_canonical_quantity(
    values: dict[tuple[ifpri_mapspam.Quantity, int], dict[str, float]],
    quantity: ifpri_mapspam.Quantity,
    year: int,
    expected: dict[str, float],
) -> None:
    dset = get_dset(values=values)
    assert {
        canonical_crop_name: get_value(
            ifpri_mapspam.get_canonical_quantity(
                canonical_crop_name=canonical_crop_name,
                dset=dset,
                quantity=quantity,
                year=year,
            )
        )
        for canonical_crop_name in expected
    } == pytest.approx(expected)


@pytest.mark.parametrize("group_name", sorted(ifpri_mapspam.GROUP_TO_CONSTITUENT_NAMES))
def test_get_canonical_quantity_conserves_the_group_total(group_name: str) -> None:
    # Whichever fallback applies, the constituents must account for the whole group: an absent
    # reference year is a question of WHICH crop receives the production, never how much.
    total = 900.0
    dset = get_dset(values={(AREA, 2005): {}, (PRODUCTION, 2000): {group_name: total}})
    got = sum(
        get_value(
            ifpri_mapspam.get_canonical_quantity(
                canonical_crop_name=constituent,
                dset=dset,
                quantity=PRODUCTION,
                year=2000,
            )
        )
        for constituent in sorted(ifpri_mapspam.GROUP_TO_CONSTITUENT_NAMES[group_name])
    )
    assert got == pytest.approx(total)


# Each snapshot's share of the reduction is 3.125% for 2000, 12.5% for 2005, 46.875% for 2010
# and 37.5% for 2020.  The expected totals below are written out rather than recomputed from
# those: a test that redoes the weighting can agree with a wrong implementation.


# The default for `get_name_to_totals`'s shares, read-only so every call can share it
NO_SHARES: collections.abc.Mapping[Commodity, float] = types.MappingProxyType({})
NO_EMISSIONS: collections.abc.Mapping[emit.EmissionComponent, float] = (
    types.MappingProxyType({})
)


def get_name_to_totals(
    dset: xarray.Dataset,
    commodity_to_share: collections.abc.Mapping[Commodity, float] = NO_SHARES,
    crop_to_expansion_share: collections.abc.Mapping[
        Crop, float
    ] = types.MappingProxyType({}),
    cropland_emissions: collections.abc.Mapping[
        emit.EmissionComponent, float
    ] = NO_EMISSIONS,
    cropland_occupation: float = 0.0,
    pasture_emissions: collections.abc.Mapping[
        emit.EmissionComponent, float
    ] = NO_EMISSIONS,
    pasture_occupation: float = 0.0,
) -> dict[str, dict[str, float]]:
    """`get_commodity_name_to_totals` over WHEAT and the livestock rows on `get_dset`'s 2x2 grid.

    Every band and share a caller does not set is zero, so each test moves one thing. Each
    emissions band is discounted and set in the 2010-2020 span alone, and the shares are the same
    in every span.
    """
    commodities = (Crop.WHEAT, *Livestock)

    def get_shares(
        shares: collections.abc.Mapping[Commodity, float],
    ) -> dict[Commodity, xarray.DataArray]:
        return {
            commodity: xarray.DataArray(share)
            for commodity, share in (
                dict.fromkeys(commodities, 0.0) | dict(shares)
            ).items()
        }

    grid = xarray.zeros_like(other=dset["pasture:fraction:2020"])
    return get_commodity_name_to_totals(
        commodities=commodities,
        commodity_to_share=get_shares(shares=commodity_to_share),
        dset=dset.assign(
            {
                f"{emit.CROPLAND:s}-peatland-occupation:tco2e-per-ha": grid
                + cropland_occupation,
                "dropped-emissions:tco2e-per-ha": grid,
                f"{emit.PASTURELAND:s}-peatland-occupation:tco2e-per-ha": grid
                + pasture_occupation,
            }
            | {
                f"{land_class:s}-{component!s}-emissions:tco2e-per-ha:{before:d}-{after:d}": grid
                + (
                    (dict.fromkeys(emit.EmissionComponent, 0.0) | dict(emissions))[
                        component
                    ]
                    if (before, after) == (2010, 2020)
                    else 0.0
                )
                for land_class, emissions in (
                    (emit.CROPLAND, cropland_emissions),
                    (emit.PASTURELAND, pasture_emissions),
                )
                for component in emit.EmissionComponent
                for before, after in MAPSPAM_SPANS
            }
        ),
        span_to_crop_to_share={
            span: {
                crop: xarray.DataArray(share)
                for crop, share in (
                    {Crop.WHEAT: 0.0} | dict(crop_to_expansion_share)
                ).items()
            }
            for span in MAPSPAM_SPANS
        },
    )


def test_commodity_hectares_is_the_last_snapshots_area() -> None:
    crop = Crop.WHEAT
    totals = get_name_to_totals(
        dset=get_dset(
            values={
                (AREA, year): {crop.value: area}
                for year, area in {
                    2000: 1.0,
                    2005: 10.0,
                    2010: 100.0,
                    2020: 1000.0,
                }.items()
            }
        ),
    )[crop.name]
    assert totals["commodity_hectares"] == pytest.approx(PIXELS * 1000.0)


CATTLE = gpw_livestock.Species.CATTLE
SHEEP = gpw_livestock.Species.SHEEP


WHEAT = Crop.WHEAT.name
BEEF = Livestock.BEEF_CATTLE.name
PASTURE = Livestock.PASTURE.name
UNATTRIBUTED = emit.NonCommodity.UNATTRIBUTED.name
FOREST = emit.EmissionComponent.FOREST
GRASSLAND = emit.EmissionComponent.GRASSLAND
PEAT = emit.EmissionComponent.PEATLAND_CONVERSION


# Expected values are per hectare of the cell. Each emissions band is already discounted and the
# shares are the same in every span, so a conversion component is band x share.
@pytest.mark.parametrize(
    ("pasture_fraction", "kwargs", "expected"),
    (
        pytest.param(
            0.0,
            {
                "commodity_to_share": {
                    Crop.WHEAT: 0.25,
                    Livestock.BEEF_CATTLE: 0.6,
                    Livestock.PASTURE: 0.4,
                },
                "cropland_occupation": 2.0,
                "pasture_occupation": 5.0,
            },
            {
                (WHEAT, "peatland_occupation_emissions_mt"): 0.5,
                (BEEF, "peatland_occupation_emissions_mt"): 3.0,
                (PASTURE, "peatland_occupation_emissions_mt"): 2.0,
            },
            id="crops-divide-the-cropland-occupation-band-and-livestock-the-pasture-one",
        ),
        pytest.param(
            0.0,
            {
                "commodity_to_share": {
                    Livestock.BEEF_CATTLE: 0.6,
                    Livestock.PASTURE: 0.4,
                },
                "crop_to_expansion_share": {Crop.WHEAT: 0.25},
                "cropland_emissions": {FOREST: 2.0, PEAT: 4.0},
                "pasture_emissions": {GRASSLAND: 5.0},
            },
            {
                (WHEAT, "forest_emissions_mt"): 0.5,
                (WHEAT, "grassland_emissions_mt"): 0.0,
                (WHEAT, "peatland_conversion_emissions_mt"): 1.0,
                (BEEF, "forest_emissions_mt"): 0.0,
                (BEEF, "grassland_emissions_mt"): 3.0,
                (PASTURE, "grassland_emissions_mt"): 2.0,
            },
            id="crops-divide-what-was-converted-to-cropland-and-livestock-what-was-converted-to-pasture",
        ),
        pytest.param(
            0.0,
            {
                "commodity_to_share": {Crop.WHEAT: 0.1, Livestock.PASTURE: 1.0},
                "crop_to_expansion_share": {Crop.WHEAT: 0.5},
                "cropland_emissions": {FOREST: 1.0, GRASSLAND: 10.0, PEAT: 100.0},
                "cropland_occupation": 1000.0,
            },
            # Charged: forest 1, grassland 10, peat conversion 100 and occupation 1000, so 1111 in
            # all, every unit of it on cropland; wheat's row and the unattributed row split each
            # between them, and the pasture row takes none of it
            {
                (WHEAT, "forest_emissions_mt"): 0.5,
                (WHEAT, "grassland_emissions_mt"): 5.0,
                (WHEAT, "peatland_conversion_emissions_mt"): 50.0,
                (WHEAT, "peatland_occupation_emissions_mt"): 100.0,
                (WHEAT, "emissions_mt"): 155.5,
                (PASTURE, "emissions_mt"): 0.0,
                (UNATTRIBUTED, "forest_emissions_mt"): 0.5,
                (UNATTRIBUTED, "grassland_emissions_mt"): 5.0,
                (UNATTRIBUTED, "peatland_conversion_emissions_mt"): 50.0,
                (UNATTRIBUTED, "peatland_occupation_emissions_mt"): 900.0,
                (UNATTRIBUTED, "emissions_mt"): 955.5,
            },
            id="the-unattributed-row-is-what-the-shares-leave-of-the-charge-so-the-rows-sum-to-it",
        ),
        pytest.param(
            0.0,
            {
                "commodity_to_share": {Crop.WHEAT: 1.0 + 1e-6},
                "crop_to_expansion_share": {Crop.WHEAT: 1.0 + 1e-6},
                "cropland_emissions": {FOREST: 1.0},
                "cropland_occupation": 1.0,
            },
            {(UNATTRIBUTED, "emissions_mt"): 0.0},
            id="shares-rounding-above-one-leave-the-unattributed-row-at-zero-not-below",
        ),
        pytest.param(
            0.0,
            {
                "commodity_to_share": {Livestock.PASTURE: 1.0},
                "pasture_occupation": emit.PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA,
            },
            {(PASTURE, "peatland_commodity_hectares"): 1.0},
            id="peatland-hectares-are-the-occupation-divided-back-by-its-annual-rate",
        ),
        pytest.param(
            # Without area every per-hectare figure for the row is undefined; with heads it would
            # publish an emission factor for grazers it does not name
            0.5,
            {"commodity_to_share": {Livestock.PASTURE: 1.0}},
            {
                (PASTURE, "commodity_hectares"): 0.5,
                (PASTURE, HERD_HEADS_COLUMN): math.nan,
            },
            id="the-pasture-row-carries-hectares-but-no-heads",
        ),
    ),
)
def test_get_commodity_name_to_totals(
    pasture_fraction: float,
    kwargs: dict[str, typing.Any],
    expected: dict[tuple[str, str], float],
) -> None:
    name_to_totals = get_name_to_totals(
        dset=get_dset(
            pasture_fraction=dict.fromkeys(ifpri_mapspam.YEARS, pasture_fraction),
            values={},
        ),
        **kwargs,
    )
    assert {
        (name, column): name_to_totals[name][column] for name, column in expected
    } == pytest.approx(
        {key: value * PIXELS * HECTARES_PER_CELL for key, value in expected.items()},
        nan_ok=True,
    )


def test_a_clip_one_cell_wide_keeps_the_hectares_of_its_cells() -> None:
    # One column of the 2x2 grid holds half of every total the whole grid holds
    dset = get_dset(
        pasture_fraction=dict.fromkeys(ifpri_mapspam.YEARS, 0.5),
        values={(AREA, 2020): {Crop.WHEAT.value: 100.0}},
    )
    whole, column = (
        get_name_to_totals(
            commodity_to_share={Crop.WHEAT: 0.5, Livestock.PASTURE: 0.25},
            crop_to_expansion_share={Crop.WHEAT: 0.5},
            cropland_emissions={FOREST: 10.0},
            cropland_occupation=2.0,
            dset=clipped,
            pasture_emissions={GRASSLAND: 10.0},
        )
        for clipped in (dset, dset.isel(x=[0]))
    )
    assert {
        (name, key): value
        for name, totals in column.items()
        for key, value in totals.items()
    } == pytest.approx(
        {
            (name, key): value / 2
            for name, totals in whole.items()
            for key, value in totals.items()
        },
        nan_ok=True,
    )


@pytest.mark.parametrize(
    ("dairy_share", "heads_per_ha", "expected"),
    (
        pytest.param(
            0.0,
            {CATTLE: 0.1, SHEEP: 0.7},
            {Livestock.BEEF_CATTLE: 0.5, Livestock.PASTURE: 0.5},
            id="a-head-weighs-its-livestock-units-so-seven-sheep-graze-what-one-head-of-cattle-does",
        ),
        pytest.param(
            0.0,
            {CATTLE: 0.1},
            {Livestock.BEEF_CATTLE: 1.0, Livestock.PASTURE: 0.0},
            id="cattle-alone-take-the-whole-pasture",
        ),
        pytest.param(
            0.0,
            {},
            {Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 1.0},
            id="pasture-no-grazer-is-mapped-on-stays-with-the-residual-row",
        ),
        pytest.param(
            0.4,
            {CATTLE: 0.1, SHEEP: 0.7},
            {Livestock.BEEF_CATTLE: 0.33, Livestock.MILK_CATTLE: 0.17},
            id="milk-takes-its-allocation-of-the-dairy-herds-part-of-cattles-share",
        ),
    ),
)
def test_get_livestock_to_grazing_share(
    dairy_share: float,
    heads_per_ha: dict[gpw_livestock.Species, float],
    expected: dict[Livestock, float],
) -> None:
    shares = get_livestock_to_grazing_share(
        dairy_share=dairy_share,
        dset=get_dset(density=get_density(heads_per_ha=heads_per_ha), values={}),
        year=2020,
    )
    assert {
        livestock: get_value(darray=shares[livestock]) for livestock in expected
    } == (pytest.approx(expected))


@pytest.mark.parametrize(
    "units_per_head",
    (
        pytest.param(UNITS_PER_HEAD, id="south-america-coefficients"),
        pytest.param(UNITS_PER_HEAD | {CATTLE: 1.4}, id="cattle-weighed-double"),
    ),
)
@pytest.mark.parametrize(
    ("pasture_fraction", "expected_heads_per_ha"),
    (
        pytest.param(
            0.5,
            0.1,
            id="the-whole-herd-counts-where-the-cell-holds-any-pasture-not-a-fraction-weighted-part",
        ),
        pytest.param(
            0.0,
            0.0,
            id="a-herd-in-cells-without-pasture-counts-no-heads",
        ),
    ),
)
@pytest.mark.parametrize(
    ("livestock", "pasture_share"),
    (
        pytest.param(Livestock.BEEF_CATTLE, 0.66, id="beef"),
        pytest.param(Livestock.MILK_CATTLE, 0.34, id="milk"),
    ),
)
def test_each_cattle_row_reports_the_whole_herd(
    livestock: Livestock,
    pasture_fraction: float,
    pasture_share: float,
    expected_heads_per_ha: float,
    units_per_head: dict[gpw_livestock.Species, float],
) -> None:
    # Livestock units weigh grazing, not the herd
    totals = get_name_to_totals(
        commodity_to_share={livestock: pasture_share},
        dset=get_dset(
            density=get_density(heads_per_ha={CATTLE: 0.1}),
            pasture_fraction=dict.fromkeys(ifpri_mapspam.YEARS, pasture_fraction),
            units_per_head=units_per_head,
            values={},
        ),
    )[livestock.name]
    assert totals[HERD_HEADS_COLUMN] == pytest.approx(
        PIXELS * HECTARES_PER_CELL * expected_heads_per_ha
    )
    # A grazer's area is its grazing share of the cell's pasture
    assert totals["commodity_hectares"] == pytest.approx(
        PIXELS * HECTARES_PER_CELL * pasture_fraction * pasture_share
    )


def test_faostat_species_match_gpw_livestock() -> None:
    """Every grazer GPW maps has FAOSTAT stocks and carcass weights to price it by."""
    assert {e.name for e in faostat.Species} == {e.name for e in gpw_livestock.Species}
