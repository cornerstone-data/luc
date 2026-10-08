import collections.abc
import math
import types
import typing
import warnings

import pytest
import xarray

from jdluc import emit
from jdluc.datasets import gpw_livestock, ifpri_mapspam
from jdluc.statistical import (
    MAPSPAM_SNAPSHOT_YEARS,
    SPAN_TO_MAPSPAM_SPAN,
    Commodity,
    Crop,
    Livestock,
    get_band_name_to_livestock_units,
    get_commodity_name_to_totals,
    get_commodity_to_peatland_conversion_share,
    get_commodity_to_share,
    get_crop_to_area_share,
    get_livestock_to_grazing_share,
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
# A crop no `Crop` claims that MapSPAM only starts reporting in 2020.  Compared in 2020's own
# naming, since two attributed crops are renamed there (ACOF -> COFF, SMIL -> MILL) and would
# otherwise look 2020-only themselves.
UNATTRIBUTED_2020_ONLY = sorted(
    ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[2020]
    - ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[2010]
)[0]


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
    # each grazer as a density in heads/ha, weighed into livestock units as `workflow` weighs them.
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


HECTARES_PER_CELL = get_value(
    emit.get_hectares_per_pixel(darray=get_dset(values={})["pasture:fraction:2020"])
)


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
            # Counting UNATTRIBUTED_2020_ONLY's whole area as expansion from zero would halve
            # MAIZE's share, although nothing was necessarily planted.
            {2010: {}, 2020: {"MAIZ": 100.0, UNATTRIBUTED_2020_ONLY: 100.0}},
            2010,
            2020,
            {Crop.MAIZE: 1.0},
            id="a-crop-the-earlier-snapshot-never-reported-is-newly-named-not-newly-grown-so-it-is-dropped",
        ),
        pytest.param(
            # The converse of the case above -- same shape, one crop different -- and the reason
            # it cannot simply drop every unattributed crop.
            {2010: {}, 2020: {"MAIZ": 100.0, UNATTRIBUTED_0: 100.0}},
            2010,
            2020,
            {Crop.MAIZE: 0.5},
            id="a-crop-reported-in-both-snapshots-really-did-expand-and-still-dilutes",
        ),
        pytest.param(
            # 2000 is a different MapSPAM release and shares almost no unattributed names with
            # 2005, so the "reported in both" test above would reject nearly the whole lump here
            # rather than stabilise it -- hence the span's absence from
            # `SPANS_WITH_COMPARABLE_CROP_NAMES`.
            {2000: {}, 2005: {"MAIZ": 100.0, UNATTRIBUTED_0: 100.0}},
            2000,
            2005,
            {Crop.MAIZE: 0.5},
            id="the-lump-still-dilutes-across-the-2000-release-boundary",
        ),
    ),
)
def test_get_commodity_to_share(
    areas: dict[int, dict[str, float]],
    before: int,
    after: int,
    expected: dict[Crop, float],
) -> None:
    dset = get_dset_for_areas(areas=areas)
    # Every case asks for all of `Crop`, so each one also checks the invariant that the shares
    # never over-attribute the cell
    shares = get_commodity_to_share(
        after=after, before=before, crops=tuple(Crop), dset=dset
    )
    assert sum(get_value(share) for share in shares.values()) <= 1.0
    assert {crop: get_value(shares[crop]) for crop in expected} == pytest.approx(
        expected
    )
    # No pasture moves in any of these cases, so it takes nothing off the crops
    assert all(get_value(shares[livestock]) == 0.0 for livestock in Livestock)


def test_get_commodity_to_share_does_not_renormalise_onto_the_crops_asked_for() -> None:
    # MAIZE's half of the cell's expansion, whichever other crops are asked for alongside it
    shares = get_commodity_to_share(
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
    expansion_shares = get_commodity_to_share(
        after=2020, before=2000, crops=(Crop.MAIZE,), dset=dset
    )
    assert get_value(expansion_shares[Crop.MAIZE]) == 0.0
    area_shares = get_crop_to_area_share(crops=(Crop.MAIZE,), dset=dset, year=2020)
    assert get_value(area_shares[Crop.MAIZE]) == 0.5


def test_get_commodity_to_share_is_reclassification_invariant() -> None:
    # The same +50 ha of non-MAIZE expansion, spread over one band and then over two.  MAIZE's
    # share depends on how much other cropland expanded, never on how MapSPAM chose to file it.
    # ("Residual" is avoided here: in `ifpri_mapspam` it means a group's catch-all constituent.)
    lumped = get_dset_for_areas(areas={2005: {}, 2010: {"MAIZ": 100.0, OTHER_0: 50.0}})
    split = get_dset_for_areas(
        areas={2005: {}, 2010: {"MAIZ": 100.0, OTHER_0: 20.0, OTHER_1: 30.0}}
    )
    shares_lumped = get_commodity_to_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=lumped
    )
    shares_split = get_commodity_to_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=split
    )
    assert get_value(shares_lumped[Crop.MAIZE]) == get_value(shares_split[Crop.MAIZE])


def test_get_commodity_to_share_tolerates_nodata_absent_crops() -> None:
    # Real rasters carry nodata, not zero, where a crop is absent.  That must not poison the
    # denominator with NaN and take the whole cell down with it.
    dset = get_dset_for_areas(areas={2005: {}, 2010: {"MAIZ": 100.0}})
    dset[
        ifpri_mapspam.get_band_name(
            quantity=AREA, reported_crop_name=OTHER_0, year=2010
        )
    ] = xarray.zeros_like(other=dset["pasture:fraction:2020"]) * float("nan")
    shares = get_commodity_to_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=dset
    )
    # 100 / 100; NaN absent crop counts as 0 ha
    assert get_value(shares[Crop.MAIZE]) == 1.0


def test_get_commodity_to_share_decomposition_does_not_warn_on_zero_reference() -> None:
    # No reference year places BANP, so the within-group split divides by a zero pool.  The
    # decomposition guards that denominator rather than dividing and discarding the NaN, which
    # would work but bury every run in RuntimeWarnings.
    dset = get_dset_for_areas(areas={2000: {"BANP": 100.0}})
    with warnings.catch_warnings():
        # Any 0/0 divide becomes a failure
        warnings.simplefilter("error", RuntimeWarning)
        get_commodity_to_share(
            after=2020, before=2000, crops=(Crop.BANANA, Crop.PLANTAIN), dset=dset
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


# Defaults for `get_name_to_totals`, read-only so every call can safely share them. A test passes
# only the shares and yields it cares about, and everything else is zero.
NO_SHARES: collections.abc.Mapping[Commodity, float] = types.MappingProxyType({})
NO_YIELD: collections.abc.Mapping[int, float] = types.MappingProxyType(
    dict.fromkeys(MAPSPAM_SNAPSHOT_YEARS, 0.0)
)


def get_name_to_totals(
    dset: xarray.Dataset,
    commodity_to_share: collections.abc.Mapping[Commodity, float] = NO_SHARES,
    cropland_occupation: float = 0.0,
    cropland_pulse: float = 0.0,
    forest_emissions: float = 0.0,
    grassland_emissions: float = 0.0,
    occupation_shares: collections.abc.Mapping[Commodity, float] = NO_SHARES,
    pasture_occupation: float = 0.0,
    pasture_pulse: float = 0.0,
    peatland_conversion_shares: collections.abc.Mapping[Commodity, float] = NO_SHARES,
    year_to_kg_per_head: collections.abc.Mapping[int, float] = NO_YIELD,
) -> dict[str, dict[str, float]]:
    """`get_commodity_name_to_totals` over WHEAT and the livestock rows on `get_dset`'s 2x2 grid.

    Every band, share and rate a caller does not set is zero, so each test moves one thing. The
    shares are the same in every span.
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
        dset=dset.assign(
            {
                f"{emit.CROPLAND:s}-peatland-occupation:tco2e-per-ha": grid
                + cropland_occupation,
                "dropped-emissions:tco2e-per-ha": grid,
                f"{emit.PASTURELAND:s}-peatland-occupation:tco2e-per-ha": grid
                + pasture_occupation,
            }
            | {
                f"{name:s}:tco2e-per-ha:{before:d}-{after:d}": grid + emissions
                for name, emissions in {
                    "forest": forest_emissions,
                    "grassland": grassland_emissions,
                    f"{emit.CROPLAND:s}-peatland-conversion": cropland_pulse,
                    f"{emit.PASTURELAND:s}-peatland-conversion": pasture_pulse,
                }.items()
                for before, after in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT
            }
        ),
        occupation_shares=get_shares(shares=occupation_shares),
        span_to_commodity_to_peatland_conversion_share=dict.fromkeys(
            SPAN_TO_MAPSPAM_SPAN.values(), get_shares(shares=peatland_conversion_shares)
        ),
        span_to_commodity_to_share=dict.fromkeys(
            SPAN_TO_MAPSPAM_SPAN.values(), get_shares(shares=commodity_to_share)
        ),
        species_to_year_to_kg_per_head={
            species: dict(year_to_kg_per_head) for species in gpw_livestock.Species
        },
    )


def get_totals(
    area_by_year: dict[int, float], production_by_year: dict[int, float]
) -> dict[str, float]:
    crop = Crop.WHEAT
    return get_name_to_totals(
        dset=get_dset(
            values={
                (AREA, year): {crop.value: area} for year, area in area_by_year.items()
            }
            | {
                (PRODUCTION, year): {crop.value: production}
                for year, production in production_by_year.items()
            }
        ),
    )[crop.name]


@pytest.mark.parametrize(
    ("area_by_year", "expected_per_pixel"),
    (
        pytest.param(
            {2000: 1.0, 2005: 10.0, 2010: 100.0, 2020: 1000.0},
            423.15625,
            id="every-snapshot-enters-at-the-weight-its-own-span-carries",
        ),
        pytest.param(
            # The defect this replaced: zero area against twenty years of production made every
            # per-hectare figure for the crop meaningless.
            {2000: 100.0, 2005: 100.0},
            15.625,
            id="a-crop-abandoned-before-2020-still-reports-the-area-it-held-rather-than-zero",
        ),
    ),
)
def test_commodity_hectares_is_the_discounted_mean_of_the_snapshots(
    area_by_year: dict[int, float], expected_per_pixel: float
) -> None:
    totals = get_totals(area_by_year=area_by_year, production_by_year={})
    assert totals["commodity_hectares"] == pytest.approx(PIXELS * expected_per_pixel)


def test_a_constant_yield_survives_the_window() -> None:
    # What the shared window buys: area and production reduce over the same spans with the same
    # weights, so a crop whose yield never changes reports exactly that yield -- whatever its
    # area did in between, and whatever weights the spans carry.
    areas = {2000: 1.0, 2005: 50.0, 2010: 7.0, 2020: 0.0}
    totals = get_totals(
        area_by_year=areas,
        production_by_year={year: area * 3.0 for year, area in areas.items()},
    )
    assert totals["production_mt"] / totals["commodity_hectares"] == pytest.approx(3.0)


CATTLE = gpw_livestock.Species.CATTLE
SHEEP = gpw_livestock.Species.SHEEP


@pytest.mark.parametrize(
    ("areas", "pasture_fraction", "density", "expected"),
    (
        pytest.param(
            {2005: {}, 2010: {"MAIZ": 100.0}},
            {2010: 100.0 / HECTARES_PER_CELL},
            {},
            {Crop.MAIZE: 0.5, Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 0.5},
            id="pasture-expanding-by-the-crops-own-hectares-takes-half-the-cell-from-it",
        ),
        pytest.param(
            {2005: {}, 2010: {}},
            {2010: 0.25},
            {},
            {Crop.MAIZE: 0.0, Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 1.0},
            id="the-frontier-cell-where-only-pasture-expands-charges-pasture-not-nobody",
        ),
        pytest.param(
            {2005: {}, 2010: {"MAIZ": 100.0}},
            {2005: 0.5, 2010: 0.0},
            {},
            {Crop.MAIZE: 1.0, Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 0.0},
            id="contracting-pasture-clips-to-zero-rather-than-shrinking-the-denominator",
        ),
        pytest.param(
            {2005: {}, 2010: {"MAIZ": 100.0}},
            {2010: 100.0 / HECTARES_PER_CELL},
            get_density(heads_per_ha={CATTLE: 0.01}),
            {Crop.MAIZE: 0.5, Livestock.BEEF_CATTLE: 0.5, Livestock.PASTURE: 0.0},
            id="cattle-alone-take-the-pasture-share-and-the-crop-keeps-its-own",
        ),
        pytest.param(
            {2005: {}, 2010: {"MAIZ": 100.0}},
            {2010: 100.0 / HECTARES_PER_CELL},
            get_density(heads_per_ha={CATTLE: 0.1, SHEEP: 0.7}),
            {Crop.MAIZE: 0.5, Livestock.BEEF_CATTLE: 0.25, Livestock.PASTURE: 0.25},
            id="grazers-divide-the-pasture-share-by-livestock-units-and-the-crop-keeps-its-own",
        ),
        pytest.param(
            {2005: {}, 2010: {"MAIZ": 100.0}},
            {2005: 0.5, 2010: 0.5},
            {(SHEEP, 2005): 0.7, (CATTLE, 2010): 0.1},
            {Crop.MAIZE: 1.0, Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 0.0},
            id="cattle-replacing-sheep-on-unchanged-pasture-is-not-expansion",
        ),
        pytest.param(
            {2005: {}, 2010: {}},
            {2010: 0.25},
            {(SHEEP, 2005): 0.7, (CATTLE, 2010): 0.1},
            {Crop.MAIZE: 0.0, Livestock.BEEF_CATTLE: 1.0, Livestock.PASTURE: 0.0},
            id="new-pasture-goes-to-whoever-grazes-it-at-the-end-of-the-span",
        ),
    ),
)
def test_get_commodity_to_share_charges_pasture_and_its_grazers(
    areas: dict[int, dict[str, float]],
    pasture_fraction: dict[int, float],
    density: dict[tuple[gpw_livestock.Species, int], float],
    expected: dict[Commodity, float],
) -> None:
    dset = get_dset_for_areas(
        areas=areas, density=density, pasture_fraction=pasture_fraction
    )
    shares = get_commodity_to_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=dset
    )
    assert {
        commodity: get_value(shares[commodity]) for commodity in expected
    } == pytest.approx(expected)


@pytest.mark.parametrize(
    ("areas", "pasture_fraction", "density", "expected"),
    (
        pytest.param(
            {2010: {}, 2020: {"MAIZ": 100.0}},
            {2020: 100.0 / HECTARES_PER_CELL},
            {},
            {Crop.MAIZE: 1.0, Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 1.0},
            id="pasture-expanding-beside-the-crop-takes-none-of-the-cropland-pulse",
        ),
        pytest.param(
            {2010: {}, 2020: {}},
            {2020: 0.25},
            {},
            {Crop.MAIZE: 0.0, Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 1.0},
            id="where-no-crop-expanded-the-cropland-pulse-goes-to-nobody",
        ),
        pytest.param(
            {2010: {}, 2020: {"MAIZ": 100.0, UNATTRIBUTED_0: 100.0}},
            {},
            {},
            {Crop.MAIZE: 0.5},
            id="crops-it-cannot-attribute-keep-their-share-of-the-cropland-pulse",
        ),
        pytest.param(
            {2010: {}, 2020: {"MAIZ": 100.0}},
            {2010: 0.5, 2020: 0.5},
            {(CATTLE, 2020): 0.1, (SHEEP, 2020): 0.7},
            {Crop.MAIZE: 1.0, Livestock.BEEF_CATTLE: 0.5, Livestock.PASTURE: 0.5},
            id="the-pasture-pulse-goes-to-the-grazers-where-only-crops-expanded",
        ),
        pytest.param(
            {2010: {}, 2020: {}},
            {2010: 0.5, 2020: 0.5},
            {(SHEEP, 2010): 0.7, (CATTLE, 2020): 0.1},
            {Crop.MAIZE: 0.0, Livestock.BEEF_CATTLE: 1.0, Livestock.PASTURE: 0.0},
            id="the-pasture-pulse-goes-to-whoever-grazes-at-the-end-of-the-span",
        ),
    ),
)
def test_get_commodity_to_peatland_conversion_share(
    areas: dict[int, dict[str, float]],
    pasture_fraction: dict[int, float],
    density: dict[tuple[gpw_livestock.Species, int], float],
    expected: dict[Commodity, float],
) -> None:
    dset = get_dset_for_areas(
        areas=areas, density=density, pasture_fraction=pasture_fraction
    )
    shares = get_commodity_to_peatland_conversion_share(
        after=2020, before=2010, crops=(Crop.MAIZE,), dset=dset
    )
    assert {
        commodity: get_value(shares[commodity]) for commodity in expected
    } == pytest.approx(expected)
    # Unlike the crops', the grazers' shares always divide the whole of their half
    assert sum(
        get_value(shares[livestock]) for livestock in Livestock
    ) == pytest.approx(1.0)


WHEAT = Crop.WHEAT.name
BEEF = Livestock.BEEF_CATTLE.name
PASTURE = Livestock.PASTURE.name
UNATTRIBUTED = emit.NonCommodity.UNATTRIBUTED.name


# Expected values are per hectare of the cell. Every band and share is the same in each span, so a
# conversion component is band x share x 0.2, the four spans' discount weights summed.
@pytest.mark.parametrize(
    ("pasture_fraction", "kwargs", "expected"),
    (
        pytest.param(
            0.0,
            {
                "cropland_occupation": 2.0,
                "occupation_shares": {
                    Crop.WHEAT: 0.25,
                    Livestock.BEEF_CATTLE: 0.6,
                    Livestock.PASTURE: 0.4,
                },
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
                    Crop.WHEAT: 0.5,
                    Livestock.BEEF_CATTLE: 0.3,
                    Livestock.PASTURE: 0.2,
                },
                "cropland_pulse": 2.0,
                "pasture_pulse": 5.0,
                "peatland_conversion_shares": {
                    Crop.WHEAT: 0.25,
                    Livestock.BEEF_CATTLE: 0.6,
                    Livestock.PASTURE: 0.4,
                },
            },
            {
                (WHEAT, "peatland_conversion_emissions_mt"): 0.1,
                (BEEF, "peatland_conversion_emissions_mt"): 0.6,
                (PASTURE, "peatland_conversion_emissions_mt"): 0.4,
            },
            id="crops-divide-the-cropland-pulse-and-livestock-the-pasture-pulse-by-their-own-shares",
        ),
        pytest.param(
            0.0,
            {
                "commodity_to_share": {Livestock.PASTURE: 0.25},
                "forest_emissions": 10.0,
            },
            {(PASTURE, "emissions_mt"): 0.5},
            id="the-pasture-row-takes-its-pooled-share-of-every-conversion-not-only-pasture-ones",
        ),
        pytest.param(
            0.0,
            {
                "commodity_to_share": {Crop.WHEAT: 0.5},
                "cropland_occupation": 1000.0,
                "cropland_pulse": 100.0,
                "forest_emissions": 1.0,
                "grassland_emissions": 10.0,
                "occupation_shares": {Crop.WHEAT: 0.1},
                "peatland_conversion_shares": {Crop.WHEAT: 0.25},
            },
            {
                (WHEAT, "forest_emissions_mt"): 0.1,
                (WHEAT, "grassland_emissions_mt"): 1.0,
                (WHEAT, "peatland_conversion_emissions_mt"): 5.0,
                (WHEAT, "peatland_occupation_emissions_mt"): 100.0,
                (WHEAT, "emissions_mt"): 106.1,
            },
            id="emissions-mt-is-the-components-plus-occupation-each-by-its-own-share",
        ),
        pytest.param(
            0.0,
            {
                "commodity_to_share": {Crop.WHEAT: 0.5},
                "cropland_occupation": 1000.0,
                "cropland_pulse": 100.0,
                "forest_emissions": 1.0,
                "grassland_emissions": 10.0,
                "occupation_shares": {Crop.WHEAT: 0.1},
                "peatland_conversion_shares": {Crop.WHEAT: 0.25},
            },
            # Charged: forest 0.2, grassland 2.0, peat conversion 20 and occupation 1000, so
            # 1022.2 in all; wheat's row and the unattributed row split each between them
            {
                (WHEAT, "forest_emissions_mt"): 0.1,
                (WHEAT, "grassland_emissions_mt"): 1.0,
                (WHEAT, "peatland_conversion_emissions_mt"): 5.0,
                (WHEAT, "peatland_occupation_emissions_mt"): 100.0,
                (WHEAT, "emissions_mt"): 106.1,
                (UNATTRIBUTED, "forest_emissions_mt"): 0.1,
                (UNATTRIBUTED, "grassland_emissions_mt"): 1.0,
                (UNATTRIBUTED, "peatland_conversion_emissions_mt"): 15.0,
                (UNATTRIBUTED, "peatland_occupation_emissions_mt"): 900.0,
                (UNATTRIBUTED, "emissions_mt"): 916.1,
                (UNATTRIBUTED, "production_mt"): math.nan,
            },
            id="the-unattributed-row-is-what-the-shares-leave-of-the-charge-so-the-rows-sum-to-it",
        ),
        pytest.param(
            0.0,
            {
                "occupation_shares": {Livestock.PASTURE: 1.0},
                "pasture_occupation": emit.PEATLAND_EMISSIONS_ANNUAL_TCO2E_PER_HA,
            },
            {(PASTURE, "peatland_commodity_hectares"): 1.0},
            id="peatland-hectares-are-the-occupation-divided-back-by-its-annual-rate",
        ),
        pytest.param(
            # Without area every per-hectare figure for the row is undefined; with production it
            # would publish an emission factor for grazers it does not name
            0.5,
            {},
            {
                (PASTURE, "commodity_hectares"): 0.5,
                (PASTURE, "production_mt"): math.nan,
            },
            id="the-pasture-row-carries-hectares-but-no-production",
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


@pytest.mark.parametrize(
    ("heads_per_ha", "expected"),
    (
        pytest.param(
            {CATTLE: 0.1, SHEEP: 0.7},
            {Livestock.BEEF_CATTLE: 0.5, Livestock.PASTURE: 0.5},
            id="a-head-weighs-its-livestock-units-so-seven-sheep-graze-what-one-head-of-cattle-does",
        ),
        pytest.param(
            {CATTLE: 0.1},
            {Livestock.BEEF_CATTLE: 1.0, Livestock.PASTURE: 0.0},
            id="cattle-alone-take-the-whole-pasture",
        ),
        pytest.param(
            {},
            {Livestock.BEEF_CATTLE: 0.0, Livestock.PASTURE: 1.0},
            id="pasture-no-grazer-is-mapped-on-stays-with-the-residual-row",
        ),
    ),
)
def test_get_livestock_to_grazing_share(
    heads_per_ha: dict[gpw_livestock.Species, float],
    expected: dict[Livestock, float],
) -> None:
    shares = get_livestock_to_grazing_share(
        dset=get_dset(density=get_density(heads_per_ha=heads_per_ha), values={}),
        year=2020,
    )
    assert {livestock: get_value(shares[livestock]) for livestock in Livestock} == (
        pytest.approx(expected)
    )


@pytest.mark.parametrize(
    "units_per_head",
    (
        pytest.param(UNITS_PER_HEAD, id="south-america-coefficients"),
        pytest.param(UNITS_PER_HEAD | {CATTLE: 1.4}, id="cattle-weighed-double"),
    ),
)
@pytest.mark.parametrize(
    ("pasture_fraction", "year_to_kg_per_head", "expected_kg_per_ha"),
    (
        pytest.param(
            0.5,
            dict.fromkeys(MAPSPAM_SNAPSHOT_YEARS, 40.0),
            4.0,
            id="the-whole-herd-counts-where-the-cell-holds-any-pasture-not-a-fraction-weighted-part",
        ),
        pytest.param(
            0.0,
            dict.fromkeys(MAPSPAM_SNAPSHOT_YEARS, 40.0),
            0.0,
            id="a-herd-in-cells-without-pasture-earns-no-production",
        ),
        pytest.param(
            # 0.1 heads/ha x 40 kg in 2020 alone, at 2020's 37.5% of the window
            0.5,
            {2000: 0.0, 2005: 0.0, 2010: 0.0, 2020: 40.0},
            1.5,
            id="each-snapshot-pairs-its-own-heads-with-its-own-rate",
        ),
    ),
)
def test_the_beef_row_reports_its_herds_carcass_weight(
    pasture_fraction: float,
    year_to_kg_per_head: dict[int, float],
    expected_kg_per_ha: float,
    units_per_head: dict[gpw_livestock.Species, float],
) -> None:
    # Production counts heads, so however a head of cattle is weighed in livestock units, the herd's
    # carcass weight is the same
    totals = get_name_to_totals(
        dset=get_dset(
            density=get_density(heads_per_ha={CATTLE: 0.1}),
            pasture_fraction=dict.fromkeys(ifpri_mapspam.YEARS, pasture_fraction),
            units_per_head=units_per_head,
            values={},
        ),
        year_to_kg_per_head=year_to_kg_per_head,
    )[Livestock.BEEF_CATTLE.name]
    assert totals["production_mt"] == pytest.approx(
        PIXELS * HECTARES_PER_CELL * expected_kg_per_ha / 1_000
    )
    # Cattle alone graze the cell, so the beef row holds all its pasture
    assert totals["commodity_hectares"] == pytest.approx(
        PIXELS * HECTARES_PER_CELL * pasture_fraction
    )
