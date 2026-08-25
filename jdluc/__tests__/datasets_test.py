import itertools

import pytest

from jdluc.datasets import ifpri_mapspam
from jdluc.datasets.glad_glcluc import flatten_ranges
from jdluc.datasets.worldbank_jurisdictions import (
    AdminLevel,
    get_jurisdiction_for_admin_level,
    get_ten_degree_tile_ids_for_admin_id,
    iter_jurisdiction_for_iso_3166_tile_id,
)
from jdluc.tiling import get_box_for_tile_id


def test_flatten_ranges() -> None:
    assert flatten_ranges(range(5), range(5, 10), range(10, 15)) == list(range(15))


@pytest.mark.integration
def test_get_ten_degree_tile_ids_for_country() -> None:
    iso_3166 = "BRA"  # Brazil
    expected = [
        "00N_040W",
        "00N_050W",
        "00N_060W",
        "00N_070W",
        "00N_080W",
        "10N_050W",
        "10N_060W",
        "10N_070W",
        "10N_080W",
        "10S_040W",
        "10S_050W",
        "10S_060W",
        "10S_070W",
        "10S_080W",
        # This tile is not in the GFW tileset
        # "20S_030W",
        "20S_050W",
        "20S_060W",
        "30S_060W",
    ]
    assert (
        sorted(
            get_ten_degree_tile_ids_for_admin_id(
                admin_id=iso_3166, admin_level=AdminLevel.NATIONAL
            )
        )
        == expected
    )


@pytest.mark.integration
def test_iter_jurisdiction_for_iso_3166_tile_id_loses_no_province() -> None:
    # The tile prefilter is only safe because every province overlaps at least one of the tiles
    # its country covers: that is what keeps attribute.merge_dfs summing over an unchanged
    # index, and it is why the prefilter needed no cache version bump. A province is allowed to
    # be absent only when it covers no published tile at all.
    iso_3166 = "BRA"
    seen = {
        jurisdiction.id
        for tile_id in get_ten_degree_tile_ids_for_admin_id(
            admin_id=iso_3166, admin_level=AdminLevel.NATIONAL
        )
        for jurisdiction in iter_jurisdiction_for_iso_3166_tile_id(
            admin_level=AdminLevel.PROVINCIAL, iso_3166=iso_3166, tile_id=tile_id
        )
    }
    expected = {
        str(admin_id)
        for admin_id in get_jurisdiction_for_admin_level(
            admin_level=AdminLevel.PROVINCIAL
        ).index
        if str(admin_id).startswith(iso_3166)
        and get_ten_degree_tile_ids_for_admin_id(
            admin_id=str(admin_id), admin_level=AdminLevel.PROVINCIAL
        )
    }
    assert seen == expected


@pytest.mark.integration
def test_iter_jurisdiction_for_iso_3166_tile_id_only_yields_overlapping() -> None:
    # And the filter must actually filter: a province yielded for a tile has to intersect it,
    # otherwise the prefilter is a no-op and the speedup is imaginary.
    tile_id = "10S_050W"
    box = get_box_for_tile_id(tile_id=tile_id)
    yielded = list(
        iter_jurisdiction_for_iso_3166_tile_id(
            admin_level=AdminLevel.PROVINCIAL, iso_3166="BRA", tile_id=tile_id
        )
    )
    assert yielded, f"{tile_id} is a Brazilian tile, so something should overlap it"
    assert all(box.intersects(j.geometry) for j in yielded)


@pytest.mark.parametrize("year", sorted(ifpri_mapspam.YEARS))
def test_get_reported_crop_name_always_names_a_band_that_year_has(year: int) -> None:
    # The load-bearing property: whatever it returns must be readable from that snapshot.  A
    # rename or a new group that broke this would otherwise surface as a KeyError deep in a
    # pipeline run.
    names = {e.name for e in ifpri_mapspam.YEAR_TO_CROP_CLS[year]}
    for crop_name in sorted(ifpri_mapspam.RECOVERABLE_CROP_NAMES):
        assert (
            ifpri_mapspam.get_reported_crop_name(
                canonical_crop_name=crop_name, year=year
            )
            in names
        )


@pytest.mark.parametrize("year", sorted(ifpri_mapspam.YEARS))
def test_reported_names_collapse_only_in_the_decompose_year(year: int) -> None:
    # 19 constituents share 6 groups in 2000, so 13 names disappear; every other year names all
    # 32 separately.  This is the whole reason the decomposition exists, asserted directly.
    reported = [
        ifpri_mapspam.get_reported_crop_name(canonical_crop_name=name, year=year)
        for name in sorted(ifpri_mapspam.RECOVERABLE_CROP_NAMES)
    ]
    collapsed = len(reported) - len(set(reported))
    assert collapsed == (13 if year == ifpri_mapspam.DECOMPOSITION_YEAR else 0)


@pytest.mark.parametrize(
    "year", sorted(set(ifpri_mapspam.YEARS) - {ifpri_mapspam.DECOMPOSITION_YEAR})
)
def test_get_reported_crop_name_defers_to_the_rename_outside_2000(
    year: int,
) -> None:
    for crop_name in sorted(ifpri_mapspam.RECOVERABLE_CROP_NAMES):
        assert ifpri_mapspam.get_reported_crop_name(
            canonical_crop_name=crop_name, year=year
        ) == ifpri_mapspam.get_renamed_crop_name(
            canonical_crop_name=crop_name, year=year
        )


def test_is_reported_as_group_is_exactly_the_constituents_in_2000() -> None:
    grouped = {
        (crop_name, year)
        for year in ifpri_mapspam.YEARS
        for crop_name in ifpri_mapspam.RECOVERABLE_CROP_NAMES
        if ifpri_mapspam.is_reported_as_group(canonical_crop_name=crop_name, year=year)
    }
    assert grouped == {
        (crop_name, ifpri_mapspam.DECOMPOSITION_YEAR)
        for crop_name in ifpri_mapspam.CONSTITUENT_TO_GROUP_NAME
    }


def test_comparable_spans_agree_with_the_vocabularies() -> None:
    # Not a restatement of the constant: it checks the constant against how much of the
    # unrecoverable vocabulary the two snapshots actually share.  Listing a pair as comparable
    # when their names barely overlap is what would make name-matching measure the taxonomy
    # instead of the land.
    for before, after in itertools.combinations(sorted(ifpri_mapspam.YEARS), 2):
        one, two = (
            ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[before],
            ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[after],
        )
        shared = len(one & two) / len(one | two)
        if (before, after) in ifpri_mapspam.SPANS_WITH_COMPARABLE_CROP_NAMES:
            assert shared > 0.5, (
                f"{before}->{after} called comparable but shares {shared:.0%}"
            )
        else:
            assert shared < 0.2, (
                f"{before}->{after} called incomparable but shares {shared:.0%}"
            )


@pytest.mark.parametrize("quantity", sorted(ifpri_mapspam.Quantity))
@pytest.mark.parametrize("year", sorted(ifpri_mapspam.YEARS))
def test_get_band_name_is_positionally_consistent(
    year: int, quantity: ifpri_mapspam.Quantity
) -> None:
    # `get_band_name` indexes the band list by the crop's position in the taxonomy
    # enum.  Nothing about a band name reveals a reordering, so a crop silently reading another
    # crop's raster is the failure this exists to catch.  Checked by matching each band against
    # the slugified enum *value*, which is what the band name is built from.
    for crop in ifpri_mapspam.YEAR_TO_CROP_CLS[year]:
        band = ifpri_mapspam.get_band_name(
            quantity=quantity, reported_crop_name=crop.name, year=year
        )
        assert crop.value.lower().replace(" ", "-") in band, (
            f"{year} {quantity} {crop.name} resolved to {band}"
        )


@pytest.mark.parametrize("quantity", sorted(ifpri_mapspam.Quantity))
@pytest.mark.parametrize("year", sorted(ifpri_mapspam.YEARS))
def test_get_band_name_is_injective(
    year: int, quantity: ifpri_mapspam.Quantity
) -> None:
    # Two crops resolving to one band would silently double-count it
    bands = [
        ifpri_mapspam.get_band_name(
            quantity=quantity, reported_crop_name=crop.name, year=year
        )
        for crop in ifpri_mapspam.YEAR_TO_CROP_CLS[year]
    ]
    assert len(set(bands)) == len(bands)


@pytest.mark.parametrize("year", sorted(ifpri_mapspam.YEARS))
def test_unrecoverable_crop_names_partition_the_taxonomy(year: int) -> None:
    # Every crop a snapshot reports is either reachable from a recoverable crop or unrecoverable,
    # never both and never neither -- otherwise the share denominator would drop or double-count
    # a band.
    reported = {e.name for e in ifpri_mapspam.YEAR_TO_CROP_CLS[year]}
    unrecoverable = ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[year]
    reachable = {
        ifpri_mapspam.get_reported_crop_name(canonical_crop_name=name, year=year)
        for name in ifpri_mapspam.RECOVERABLE_CROP_NAMES
    }
    assert unrecoverable | reachable == reported
    assert not (unrecoverable & reachable)


def test_unrecoverable_crop_names_grow_with_the_taxonomy() -> None:
    # 2000 leaves only its two catch-alls unreachable; later releases name crops the 2000
    # taxonomy has no route to, and 2020 adds four more again.
    counts = {
        year: len(ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES[year])
        for year in ifpri_mapspam.YEARS
    }
    assert counts == {2000: 2, 2005: 10, 2010: 10, 2020: 14}
