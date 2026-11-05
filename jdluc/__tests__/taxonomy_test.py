import pytest

from jdluc import jurisdictional_direct, statistical
from jdluc.attribute import Methodology
from jdluc.datasets import faostat, ifpri_mapspam
from jdluc.taxonomy import METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID

JURISDICTIONAL_DIRECT_NAME_TO_COMMODITY_ID = (
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[Methodology.JURISDICTIONAL_DIRECT]
)
STATISTICAL_NAME_TO_COMMODITY_ID = METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
    Methodology.STATISTICAL
]


# The statistical leg's crop of the same name already has its FAOSTAT item, chosen independently
@pytest.mark.parametrize(
    "crop", list(jurisdictional_direct.Crop), ids=lambda crop: crop.name.lower()
)
def test_jurisdictional_direct_codes_match_statistical_items(
    crop: jurisdictional_direct.Crop,
) -> None:
    statistical_crop = statistical.Crop[crop.name]
    item_code = faostat.CROP_TO_ITEM_CODE[faostat.Crop[statistical_crop.value]]
    assert JURISDICTIONAL_DIRECT_NAME_TO_COMMODITY_ID[crop.name] == item_code


@pytest.mark.parametrize(
    "crop",
    [crop for crop in statistical.Crop if crop.value in faostat.Crop.__members__],
    ids=lambda crop: crop.name.lower(),
)
def test_statistical_codes_match_faostat_items(crop: statistical.Crop) -> None:
    item_code = faostat.CROP_TO_ITEM_CODE[faostat.Crop[crop.value]]
    assert STATISTICAL_NAME_TO_COMMODITY_ID[crop.name] == item_code


def test_both_coffees_are_faostat_coffee() -> None:
    item_code = faostat.COMBINED_CROP_NAME_TO_ITEM_CODE[
        ifpri_mapspam.Crop2000.COFF.name
    ]
    assert (
        STATISTICAL_NAME_TO_COMMODITY_ID[statistical.Crop.ARABICA_COFFEE.name]
        == item_code
    )
    assert (
        STATISTICAL_NAME_TO_COMMODITY_ID[statistical.Crop.ROBUSTA_COFFEE.name]
        == item_code
    )
