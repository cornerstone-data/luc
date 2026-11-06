import pytest

from jdluc import jurisdictional_direct, statistical
from jdluc.attribute import Methodology
from jdluc.taxonomy import METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID

JURISDICTIONAL_DIRECT_NAME_TO_COMMODITY_ID = (
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[Methodology.JURISDICTIONAL_DIRECT]
)
STATISTICAL_NAME_TO_COMMODITY_ID = METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
    Methodology.STATISTICAL
]


# The two legs' entries are written separately, so each checks the other
@pytest.mark.parametrize(
    "crop", list(jurisdictional_direct.Crop), ids=lambda crop: crop.name.lower()
)
def test_jurisdictional_direct_codes_match_statistical_codes(
    crop: jurisdictional_direct.Crop,
) -> None:
    assert (
        JURISDICTIONAL_DIRECT_NAME_TO_COMMODITY_ID[crop.name]
        == STATISTICAL_NAME_TO_COMMODITY_ID[statistical.Crop[crop.name].name]
    )
