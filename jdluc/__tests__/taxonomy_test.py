import pytest

from jdluc import jurisdictional_direct, statistical
from jdluc.attribute import Methodology
from jdluc.datasets import tang_cropgrids
from jdluc.taxonomy import METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID

JURISDICTIONAL_DIRECT_NAME_TO_COMMODITY_ID = (
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[Methodology.JURISDICTIONAL_DIRECT]
)
STATISTICAL_NAME_TO_COMMODITY_ID = METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
    Methodology.STATISTICAL
]
PROXY_NAME_TO_COMMODITY_ID = METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
    Methodology.JURISDICTIONAL_DIRECT_PROXY
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


@pytest.mark.parametrize(
    "crop",
    [crop for crop in tang_cropgrids.Crop if crop.name in statistical.Crop.__members__],
    ids=lambda crop: crop.name.lower(),
)
def test_proxy_codes_match_statistical_codes(crop: tang_cropgrids.Crop) -> None:
    assert (
        PROXY_NAME_TO_COMMODITY_ID[crop.name]
        == STATISTICAL_NAME_TO_COMMODITY_ID[crop.name]
    )
