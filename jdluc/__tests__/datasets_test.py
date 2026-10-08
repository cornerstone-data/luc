import itertools

import geopandas
import pandas
import pytest
import shapely

from jdluc.datasets import NAME_TO_CLS, base, faostat, ifpri_mapspam
from jdluc.datasets.glad_glcluc import flatten_ranges
from jdluc.datasets.worldbank_jurisdictions import (
    AdminLevel,
    get_jurisdiction_for_admin_level,
    get_ten_degree_tile_ids_for_admin_id,
    iter_jurisdiction_for_iso_3166_tile_id,
    with_taiwan_carved_out_of_china,
    with_taiwan_province_separated,
)
from jdluc.tiling import get_box_for_tile_id


def test_flatten_ranges() -> None:
    assert flatten_ranges(range(5), range(5, 10), range(10, 15)) == list(range(15))


@pytest.mark.integration
def test_get_ten_degree_tile_ids_for_admin_id() -> None:
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
        # This tile is not in the GNW tileset
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


@pytest.mark.integration
def test_every_admin_1_prefix_has_an_admin_0_row() -> None:
    # Provinces are selected by their country's `admin_id` prefix, so a province whose prefix
    # names no national row is missing from every output. Upstream, Taiwan was one.
    national = set(
        get_jurisdiction_for_admin_level(admin_level=AdminLevel.NATIONAL).index
    )
    prefixes = {
        str(admin_id)[:3]
        for admin_id in get_jurisdiction_for_admin_level(
            admin_level=AdminLevel.PROVINCIAL
        ).index
    }
    assert prefixes - national == set()


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


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(dataset, id=name)
        for name, dataset in NAME_TO_CLS.items()
        if isinstance(dataset, base.TabularDataset)
    ],
)
def test_a_tabular_datasets_are_hashable(dataset: base.TabularDataset) -> None:
    assert hash(dataset)


def get_faostat_frame(
    column_name: str, key_to_value: dict[tuple[str, str, int], float]
) -> pandas.DataFrame:
    """A table shaped like `faostat.load`'s, one column over (country, species, year)."""
    return pandas.DataFrame.from_records(
        data=[
            {
                "admin_level": AdminLevel.NATIONAL.name,
                "admin_id": iso_3166,
                "jurisdiction_name": iso_3166,
                "commodity_name": commodity_name,
                "year": year,
                column_name: value,
            }
            for (iso_3166, commodity_name, year), value in key_to_value.items()
        ]
    ).set_index(faostat.IDX_COLUMN_NAMES)


NO_LIVESTOCK_UNITS = dict.fromkeys(faostat.Species, 0.0)
CATTLE = faostat.Species.CATTLE


@pytest.mark.parametrize(
    ("livestock_units", "stocks", "expected"),
    (
        pytest.param(
            {
                ("BRA", "CATTLE", 2000): 70.0,
                ("BRA", "CATTLE", 2005): 70.0,
                ("BRA", "CATTLE", 2010): 95.0,
            },
            {
                ("BRA", "CATTLE", 2000): 100.0,
                ("BRA", "CATTLE", 2005): 100.0,
                ("BRA", "CATTLE", 2010): 100.0,
            },
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="a-year-whose-heads-were-since-revised-does-not-move-the-median",
        ),
        pytest.param(
            # Buffalo as FAOSTAT gives South America's, at zero, and sheep with no figure at all:
            # neither falls back to another coefficient
            {("BRA", "BUFFALO", 2020): 0.0, ("BRA", "CATTLE", 2020): 70.0},
            {
                ("BRA", "BUFFALO", 2020): 100.0,
                ("BRA", "CATTLE", 2020): 100.0,
                ("BRA", "SHEEP", 2020): 100.0,
            },
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="a-grazer-given-zero-or-no-livestock-units-counts-none",
        ),
        pytest.param(
            {("BRA", "CATTLE", 2000): 10.0, ("BRA", "CATTLE", 2020): 70.0},
            {("BRA", "CATTLE", 2000): 0.0, ("BRA", "CATTLE", 2020): 100.0},
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="a-year-without-heads-is-left-out-rather-than-divided-by",
        ),
        pytest.param(
            {("ARG", "CATTLE", 2020): 50.0, ("BRA", "CATTLE", 2020): 70.0},
            {("ARG", "CATTLE", 2020): 100.0, ("BRA", "CATTLE", 2020): 100.0},
            NO_LIVESTOCK_UNITS | {CATTLE: 0.70},
            id="only-the-countrys-own-rows-count",
        ),
    ),
)
def test_get_species_to_livestock_units_per_head(
    livestock_units: dict[tuple[str, str, int], float],
    stocks: dict[tuple[str, str, int], float],
    expected: dict[faostat.Species, float],
) -> None:
    assert faostat.get_species_to_livestock_units_per_head(
        iso_3166="BRA",
        livestock_units=get_faostat_frame(
            column_name="stocks_livestock_units", key_to_value=livestock_units
        ),
        stocks=get_faostat_frame(column_name="stocks_head", key_to_value=stocks),
    ) == pytest.approx(expected)


MAINLAND_CHINA = shapely.Polygon(
    shell=[
        (122.7, 37.4),
        (121.6, 28.3),
        (109.9, 20.2),
        (100.1, 21.5),
        (89.0, 27.3),
        (78.4, 32.5),
        (73.5, 39.4),
        (86.9, 49.1),
        (105.0, 41.6),
        (120.9, 53.3),
        (134.8, 48.4),
        (130.6, 42.4),
    ]
)
TAIWAN = shapely.Polygon(
    shell=[
        (120.9, 21.9),
        (120.0, 23.1),
        (120.2, 23.8),
        (121.0, 25.0),
        (121.5, 25.3),
        (121.9, 25.1),
        (121.9, 24.5),
        (121.4, 23.1),
    ]
)
JAPAN = shapely.Polygon(
    shell=[
        (130.7, 31.0),
        (129.7, 32.8),
        (130.0, 33.8),
        (132.6, 35.5),
        (136.8, 37.5),
        (139.0, 37.9),
        (140.0, 40.6),
        (140.0, 41.6),
        (140.4, 43.3),
        (141.7, 45.5),
        (145.3, 44.3),
        (145.8, 43.3),
        (143.2, 41.9),
        (141.5, 41.4),
        (142.1, 39.5),
        (141.0, 38.2),
        (140.9, 35.7),
        (138.8, 34.6),
        (135.8, 33.4),
        (133.0, 32.7),
        (131.9, 31.6),
    ]
)


def test_with_taiwan_carved_out_of_china() -> None:
    from_world_bank = geopandas.GeoDataFrame(
        crs=4326,
        data={"ISO_A3": ["CHN", "JPN"], "NAM_0": ["China", "Japan"]},
        geometry=[shapely.union(MAINLAND_CHINA, TAIWAN), JAPAN],
    )
    result = with_taiwan_carved_out_of_china(admin_0=from_world_bank).set_index(
        "ISO_A3"
    )
    assert result.crs == from_world_bank.crs
    assert len(result) == 3
    assert list(result.columns) == ["NAM_0", "geometry"]
    assert set(result.index) == {"CHN", "JPN", "TWN"}
    assert result.loc["TWN"]["NAM_0"] == "Taiwan"
    assert shapely.equals(result.loc["CHN"].geometry, MAINLAND_CHINA)
    assert shapely.equals(result.loc["JPN"].geometry, JAPAN)
    assert shapely.equals(result.loc["TWN"].geometry, TAIWAN)


def test_with_taiwan_province_separated() -> None:
    from_world_bank = geopandas.GeoDataFrame(
        crs=4326,
        data={
            "ADM1CD_c": ["CHNXXX", "JPNXXX", "TWN001"],
            "ISO_A3": ["CHN", "JPN", "CHN"],
            "NAM_0": ["China", "Japan", "China"],
            "NAM_1": ["merged", "merged", "Taiwan Sheng"],
        },
        geometry=[MAINLAND_CHINA, JAPAN, TAIWAN],
    )
    result = with_taiwan_province_separated(admin_1=from_world_bank).set_index(
        "ADM1CD_c"
    )
    assert result.crs == from_world_bank.crs
    assert len(result) == 3
    assert list(result.columns) == ["ISO_A3", "NAM_0", "NAM_1", "geometry"]
    assert set(result.index) == {"CHNXXX", "JPNXXX", "TWN001"}
    assert result.loc["TWN001"]["ISO_A3"] == "TWN"
    assert result.loc["TWN001"]["NAM_0"] == "Taiwan"
    assert shapely.equals(result.loc["CHNXXX"].geometry, MAINLAND_CHINA)
    assert shapely.equals(result.loc["JPNXXX"].geometry, JAPAN)
    assert shapely.equals(result.loc["TWN001"].geometry, TAIWAN)
