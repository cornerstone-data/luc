"""Checks on how WRI's crops are put onto the commodity ids `trace` keys its rows on."""

import pandas
import pytest

from jdluc import statistical
from jdluc.datasets import faostat
from jdluc.taxonomy import InventedCommodityId
from validation import prepare


@pytest.mark.parametrize(
    ("crop", "commodity_id"),
    [
        pytest.param(statistical.Crop.ARABICA_COFFEE, 656, id="arabica-coffee"),
        pytest.param(statistical.Crop.ROBUSTA_COFFEE, 656, id="robusta-coffee"),
        pytest.param(statistical.Crop.PEARL_MILLET, 79, id="pearl-millet"),
        pytest.param(statistical.Crop.SMALL_MILLET, 79, id="small-millet"),
    ],
)
def test_a_split_wri_crop_reaches_faostat_s_one_item(
    crop: statistical.Crop, commodity_id: int
) -> None:
    """WRI publishes the two coffees and the two millets apart, as MapSPAM 2005 onward does, where
    FAOSTAT has one item for each, so both halves are compared as that item."""
    assert prepare.WRI_CROP_CODE_TO_COMMODITY_ID[crop.value] == commodity_id


def test_get_compared_rows_keeps_only_anchored_rows() -> None:
    frame = pandas.DataFrame(
        {
            "commodity_id": [
                56,  # maize
                656,  # coffee
                faostat.SPECIES_TO_MEAT_ITEM_CODE[faostat.Species.CATTLE],
                InventedCommodityId.PASTURE,
                InventedCommodityId.DROPPED,
                InventedCommodityId.UNATTRIBUTED,
            ]
        }
    ).astype({"commodity_id": int})
    assert prepare.get_compared_rows(frame=frame)["commodity_id"].tolist() == [
        56,
        656,
        faostat.SPECIES_TO_MEAT_ITEM_CODE[faostat.Species.CATTLE],
    ]
