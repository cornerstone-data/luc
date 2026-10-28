import geopandas
import pytest
import shapely

from jdluc.datasets.worldbank_jurisdictions import (
    AdminLevel,
    get_jurisdiction_for_admin_level,
    get_ten_degree_tile_ids_for_admin_id,
    iter_jurisdiction_for_iso_3166_tile_id,
    with_taiwan_carved_out_of_china,
    with_taiwan_province_separated,
)
from jdluc.tiling import get_box_for_tile_id


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
