import warnings

import pytest
import xarray

from jdluc import emit
from jdluc.datasets import gfw_global_peatlands, ifpri_mapspam
from jdluc.statistical import (
    GLAD_TO_MAPSPAM_SPAN,
    Crop,
    get_crop_name_to_totals,
    get_crop_to_area_share,
    get_crop_to_share,
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


def get_dset(
    values: dict[tuple[ifpri_mapspam.Quantity, int], dict[str, float]],
) -> xarray.Dataset:
    # A (scalar) band for every crop of every (quantity, year) -- the real dset always carries
    # all four snapshots.  Unset crops, and unset years, are 0.
    return xarray.Dataset(
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
    )


def get_dset_for_areas(
    areas: dict[int, dict[str, float]],
) -> xarray.Dataset:
    return get_dset(values={(AREA, year): by_crop for year, by_crop in areas.items()})


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
            # The total is masked to NaN rather than dividing 0 by 0; `.fillna(0)` is what turns
            # that back into the 0.0 asserted here.
            {2005: {}, 2010: {}},
            2005,
            2010,
            {Crop.MAIZE: 0.0},
            id="no-expansion-anywhere-masks-the-total-instead-of-dividing-by-zero",
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
            # 2005 names neither constituent, so 2020 carries the pool alone and splits the 2000
            # group 40/60 by area -- not the 50/50 an even split would fabricate.
            {2000: {"BANP": 100.0}, 2005: {}, 2020: {"BANA": 80.0, "PLNT": 120.0}},
            2000,
            2020,
            {Crop.BANANA: 0.40, Crop.PLANTAIN: 0.60},
            id="a-later-reference-carries-the-decomposition-when-the-nearest-is-absent",
        ),
        pytest.param(
            # OTHER_0 expands by exactly the 100 ha OTHER_1 gives up.  Each crop's expansion is
            # clipped at zero before being summed, so the denominator is 200 and MAIZE and
            # OTHER_0 take half the cell each.  A denominator that instead netted the two
            # before clipping would see the pair contribute nothing, leaving 100 -- and MAIZE
            # and OTHER_0 would each read 100/100, charging the cell's emissions twice over.
            {2005: {OTHER_1: 100.0}, 2010: {"MAIZ": 100.0, OTHER_0: 100.0}},
            2005,
            2010,
            {Crop.MAIZE: 0.5, Crop(OTHER_0): 0.5},
            id="a-contracting-sibling-cannot-push-the-shares-above-one",
        ),
        pytest.param(
            # No `Crop` can claim an unattributed crop's expansion, so the lump is netted whole
            # instead: MAIZE keeps the entire cell even though UNATTRIBUTED_1 expanded, because
            # UNATTRIBUTED_0 gave up as much.  Netting is all we can honestly say about a lump
            # nothing divides by.
            {2005: {UNATTRIBUTED_0: 50.0}, 2010: {"MAIZ": 100.0, UNATTRIBUTED_1: 50.0}},
            2005,
            2010,
            {Crop.MAIZE: 1.0},
            id="crops-it-cannot-attribute-stay-lumped-and-net-against-each-other",
        ),
        pytest.param(
            # MapSPAM first reports UNATTRIBUTED_2020_ONLY in 2020, so its whole area would read
            # as expansion from zero and halve MAIZE's share -- although nothing was necessarily
            # planted.  It is newly named, not newly grown, so it is dropped from the denominator.
            {2010: {}, 2020: {"MAIZ": 100.0, UNATTRIBUTED_2020_ONLY: 100.0}},
            2010,
            2020,
            {Crop.MAIZE: 1.0},
            id="a-crop-the-earlier-snapshot-never-reported-is-ignored",
        ),
        pytest.param(
            # The converse, and the reason the case above cannot just drop every unattributed
            # crop: both snapshots report UNATTRIBUTED_0, so its 100 ha really is expansion and
            # must dilute MAIZE.  Same shape as the case above, one crop different.
            {2010: {}, 2020: {"MAIZ": 100.0, UNATTRIBUTED_0: 100.0}},
            2010,
            2020,
            {Crop.MAIZE: 0.5},
            id="a-crop-reported-in-both-snapshots-still-dilutes",
        ),
        pytest.param(
            # 2000 is a different MapSPAM release and shares almost no unattributed names with
            # 2005, so the "reported in both" test above would reject nearly the whole lump here
            # rather than stabilise it.  That is why the span is left out of
            # `SPANS_WITH_COMPARABLE_CROP_NAMES` and the 2005 crops still dilute.
            {2000: {}, 2005: {"MAIZ": 100.0, UNATTRIBUTED_0: 100.0}},
            2000,
            2005,
            {Crop.MAIZE: 0.5},
            id="the-lump-survives-the-2000-release-boundary",
        ),
    ),
)
def test_get_crop_to_share(
    areas: dict[int, dict[str, float]],
    before: int,
    after: int,
    expected: dict[Crop, float],
) -> None:
    dset = get_dset_for_areas(areas=areas)
    # Every case asks for all of `Crop`, so each one also checks the invariant that the shares
    # never over-attribute the cell
    shares = get_crop_to_share(after=after, before=before, crops=tuple(Crop), dset=dset)
    assert sum(float(share) for share in shares.values()) <= 1.0
    assert {crop: float(shares[crop]) for crop in expected} == pytest.approx(expected)
    # Shares are absolute: asking for fewer crops must not renormalise onto the ones asked for
    subset = get_crop_to_share(
        after=after, before=before, crops=tuple(expected), dset=dset
    )
    assert {crop: float(subset[crop]) for crop in expected} == pytest.approx(expected)


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
    assert sum(float(share) for share in shares.values()) <= 1.0
    assert {crop: float(shares[crop]) for crop in expected} == pytest.approx(expected)
    # Shares are absolute: asking for fewer crops must not renormalise onto the ones asked for
    subset = get_crop_to_area_share(crops=tuple(expected), dset=dset, year=year)
    assert {crop: float(subset[crop]) for crop in expected} == pytest.approx(expected)


def test_get_crop_to_area_share_survives_zero_expansion() -> None:
    # The regression this function exists for: long-established cropland whose area has not
    # changed over the window.  Expansion share drops it entirely (and with it 100% of the
    # cell's peatland occupation emissions); area share still allocates it.
    areas = {"MAIZ": 100.0, "SOYB": 100.0}
    dset = get_dset_for_areas(areas={2000: areas, 2020: areas})
    expansion_shares = get_crop_to_share(
        after=2020, before=2000, crops=(Crop.MAIZE,), dset=dset
    )
    assert float(expansion_shares[Crop.MAIZE]) == 0.0
    area_shares = get_crop_to_area_share(crops=(Crop.MAIZE,), dset=dset, year=2020)
    assert float(area_shares[Crop.MAIZE]) == 0.5


def test_get_crop_to_share_is_reclassification_invariant() -> None:
    # The same +50 ha of non-MAIZE expansion, spread over one band and then over two.  MAIZE's
    # share depends on how much other cropland expanded, never on how MapSPAM chose to file it.
    # ("Residual" is avoided here: in `ifpri_mapspam` it means a group's catch-all constituent.)
    lumped = get_dset_for_areas(areas={2005: {}, 2010: {"MAIZ": 100.0, OTHER_0: 50.0}})
    split = get_dset_for_areas(
        areas={2005: {}, 2010: {"MAIZ": 100.0, OTHER_0: 20.0, OTHER_1: 30.0}}
    )
    shares_lumped = get_crop_to_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=lumped
    )
    shares_split = get_crop_to_share(
        after=2010, before=2005, crops=(Crop.MAIZE,), dset=split
    )
    assert float(shares_lumped[Crop.MAIZE]) == float(shares_split[Crop.MAIZE])


def test_get_crop_to_share_tolerates_nodata_absent_crops() -> None:
    # Real rasters carry nodata, not zero, where a crop is absent.  That must not poison the
    # denominator with NaN and take the whole cell down with it.
    dset = get_dset_for_areas(areas={2005: {}, 2010: {"MAIZ": 100.0}})
    dset[
        ifpri_mapspam.get_band_name(
            quantity=AREA, reported_crop_name=OTHER_0, year=2010
        )
    ] = xarray.DataArray(float("nan"))
    shares = get_crop_to_share(after=2010, before=2005, crops=(Crop.MAIZE,), dset=dset)
    assert float(shares[Crop.MAIZE]) == 1.0  # 100 / 100; NaN absent crop counts as 0 ha


def test_get_crop_to_share_decomposition_does_not_warn_on_zero_reference() -> None:
    # No reference year places BANP, so the within-group split divides by a zero pool.  The
    # decomposition guards that denominator rather than dividing and discarding the NaN, which
    # would work but bury every run in RuntimeWarnings.
    dset = get_dset_for_areas(areas={2000: {"BANP": 100.0}})
    with warnings.catch_warnings():
        # Any 0/0 divide becomes a failure
        warnings.simplefilter("error", RuntimeWarning)
        get_crop_to_share(
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
            # All three reference years contribute and they disagree; pooling gives BANA a fifth
            # where 2005 alone would have given it 150 t.
            {
                (PRODUCTION, 2005): {"BANA": 30.0, "PLNT": 10.0},
                (PRODUCTION, 2010): {"BANA": 5.0, "PLNT": 75.0},
                (PRODUCTION, 2020): {"BANA": 5.0, "PLNT": 75.0},
                (PRODUCTION, 2000): {"BANP": 200.0},
            },
            PRODUCTION,
            2000,
            {"BANA": 40.0, "PLNT": 160.0},
            id="every-reference-year-pools-into-one-split",
        ),
        pytest.param(
            # 2005 and 2010 are empty, so 2020 alone decides, and it names only bananas: BANP's
            # whole 200 t goes to BANA.  An even split would have fabricated 100 t of plantain
            # in a pixel MapSPAM says grows nothing but bananas.
            {(PRODUCTION, 2020): {"BANA": 40.0}, (PRODUCTION, 2000): {"BANP": 200.0}},
            PRODUCTION,
            2000,
            {"BANA": 200.0, "PLNT": 0.0},
            id="a-later-reference-carries-the-split-when-2005-is-empty",
        ),
        pytest.param(
            # No reference year places BANP here and it has no catch-all constituent to route
            # the remainder to, so an even split is the only option left.  This is the fallback
            # the two OOIL cases below exist to avoid wherever a catch-all does exist.
            {(PRODUCTION, 2000): {"BANP": 200.0}},
            PRODUCTION,
            2000,
            {"BANA": 100.0, "PLNT": 100.0},
            id="an-even-split-is-the-last-resort-for-groups-with-no-catch-all",
        ),
        pytest.param(
            # A catch-all changes what happens with NO evidence, not how evidence is used: the
            # reference years pool the same way for every group.
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
            # With no area for any constituent in ANY reference year there is no basis for
            # dividing the group.  Splitting evenly instead put a sixth of Canada's "Other Oil
            # Crops" under oil palm and another sixth under coconut, neither of which Canada grows.
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
            id="the-catch-all-takes-the-whole-group-when-no-reference-places-it",
        ),
        pytest.param(
            # The catch-all is only a fallback for having no evidence at all.  Here 2005 does
            # place the group, so the proportional split wins outright and OOIL-the-constituent
            # gets nothing -- despite OOIL also being the group's catch-all.
            {
                (PRODUCTION, 2005): {"OILP": 300.0, "CNUT": 100.0},
                (PRODUCTION, 2000): {"OOIL": 400.0},
            },
            PRODUCTION,
            2000,
            {"OILP": 300.0, "CNUT": 100.0, "OOIL": 0.0},
            id="reference-shares-win-whenever-a-reference-year-places-the-group",
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
        canonical_crop_name: float(
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
        float(
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
PIXELS = 4


def get_totals(
    area_by_year: dict[int, float], production_by_year: dict[int, float]
) -> dict[str, float]:
    """`get_crop_name_to_totals` over one crop on a 2x2 grid.

    The emissions and peatland bands it also reads are zero throughout, and so are the shares:
    only the area and production columns are under test here.
    """
    crop = Crop.WHEAT
    zero = xarray.DataArray(0.0)
    dset = get_dset(
        values={(AREA, year): {crop.value: area} for year, area in area_by_year.items()}
        | {
            (PRODUCTION, year): {crop.value: production}
            for year, production in production_by_year.items()
        }
    ).expand_dims({"y": [0.1, 0.2], "x": [0.1, 0.2]})
    # `get_hectares_per_pixel` reads the peatland band's own coords, so these carry the grid
    grid = xarray.zeros_like(other=dset[next(iter(dset.data_vars))])
    return get_crop_name_to_totals(
        crop_to_span_to_share={
            crop: dict.fromkeys(GLAD_TO_MAPSPAM_SPAN.values(), zero)
        },
        dset=dset.assign(
            {
                gfw_global_peatlands.DATASET.fully_qualified_band_name: grid,
                "peatland-occupation:tco2e-per-ha": grid,
            }
            | {
                f"{source:s}:tco2e-per-ha:{before:d}-{after:d}": grid
                for source in ("emissions", "forest", "peatland_conversion")
                for before, after in emit.SPAN_TO_LINEAR_DISCOUNT_WEIGHT
            }
        ),
        occupation_shares={crop: zero},
    )[crop.name]


def test_crop_hectares_is_the_discounted_mean_of_the_snapshots() -> None:
    totals = get_totals(
        area_by_year={2000: 1.0, 2005: 10.0, 2010: 100.0, 2020: 1000.0},
        production_by_year={},
    )
    assert totals["crop_hectares"] == pytest.approx(PIXELS * 423.15625)


def test_crop_hectares_survives_a_crop_abandoned_before_2020() -> None:
    # The defect this replaced: no area in the final snapshot against twenty years of
    # production, which made every per-hectare figure for the crop meaningless.
    totals = get_totals(
        area_by_year={2000: 100.0, 2005: 100.0}, production_by_year={2000: 1.0}
    )
    assert totals["crop_hectares"] == pytest.approx(PIXELS * 15.625)


def test_a_constant_yield_survives_the_window() -> None:
    # What the shared window buys: area and production reduce over the same spans with the same
    # weights, so a crop whose yield never changes reports exactly that yield -- whatever its
    # area did in between, and whatever weights the spans carry.
    areas = {2000: 1.0, 2005: 50.0, 2010: 7.0, 2020: 0.0}
    totals = get_totals(
        area_by_year=areas,
        production_by_year={year: area * 3.0 for year, area in areas.items()},
    )
    assert totals["production_mt"] / totals["crop_hectares"] == pytest.approx(3.0)
