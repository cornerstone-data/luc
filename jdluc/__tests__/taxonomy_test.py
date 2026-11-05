import pytest

from jdluc import jurisdictional_direct, statistical
from jdluc.attribute import Methodology
from jdluc.datasets import faostat
from jdluc.taxonomy import METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID


# The statistical leg's crop of the same name already has its FAOSTAT item, chosen independently
@pytest.mark.parametrize(
    "crop", list(jurisdictional_direct.Crop), ids=lambda crop: crop.name.lower()
)
def test_jurisdictional_direct_codes_match_statistical_items(
    crop: jurisdictional_direct.Crop,
) -> None:
    statistical_crop = statistical.Crop[crop.name]
    item_code = faostat.CROP_TO_ITEM_CODE[faostat.Crop[statistical_crop.value]]
    assert (
        METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
            Methodology.JURISDICTIONAL_DIRECT
        ][crop.name]
        == item_code
    )
