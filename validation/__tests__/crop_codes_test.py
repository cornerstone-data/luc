"""Checks on how captured rows, keyed by FAOSTAT item, are put back into MapSPAM codes."""

import pandas
import pytest

from jdluc import statistical
from jdluc.attribute import Methodology
from jdluc.datasets import ifpri_mapspam
from jdluc.taxonomy import METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID, CommodityId
from validation import prepare

STATISTICAL_NAME_TO_COMMODITY_ID = METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
    Methodology.STATISTICAL
]


@pytest.mark.parametrize(
    "crop",
    [
        crop
        for crop in statistical.Crop
        if crop.value not in prepare.SPLIT_CROP_CODE_TO_COMBINED_CROP_CODE
    ],
    ids=lambda crop: crop.name.lower(),
)
def test_a_crop_with_its_own_item_keeps_its_own_code(crop: statistical.Crop) -> None:
    commodity_id = STATISTICAL_NAME_TO_COMMODITY_ID[crop.name]
    assert prepare.COMMODITY_ID_TO_CROP_CODE[commodity_id] == crop.value


@pytest.mark.parametrize(
    ("crop", "combined_crop_code"),
    [
        pytest.param(
            statistical.Crop.ARABICA_COFFEE,
            ifpri_mapspam.Crop2000.COFF.name,
            id="arabica-coffee",
        ),
        pytest.param(
            statistical.Crop.ROBUSTA_COFFEE,
            ifpri_mapspam.Crop2000.COFF.name,
            id="robusta-coffee",
        ),
        pytest.param(
            statistical.Crop.PEARL_MILLET,
            ifpri_mapspam.Crop2000.MILL.name,
            id="pearl-millet",
        ),
        pytest.param(
            statistical.Crop.SMALL_MILLET,
            ifpri_mapspam.Crop2000.MILL.name,
            id="small-millet",
        ),
    ],
)
def test_a_split_crop_is_compared_under_its_combined_code(
    crop: statistical.Crop, combined_crop_code: str
) -> None:
    commodity_id = STATISTICAL_NAME_TO_COMMODITY_ID[crop.name]
    assert prepare.COMMODITY_ID_TO_CROP_CODE[commodity_id] == combined_crop_code
    assert (
        prepare.SPLIT_CROP_CODE_TO_COMBINED_CROP_CODE[crop.value] == combined_crop_code
    )


# Written separately from `taxonomy`, so each checks the other
@pytest.mark.parametrize(
    "crop",
    [
        crop
        for crop in statistical.Crop
        if crop.value in prepare.FAOSTAT_ITEM_CODE_TO_CROP_CODE.values()
    ],
    ids=lambda crop: crop.name.lower(),
)
def test_faostat_items_match_taxonomy_codes(crop: statistical.Crop) -> None:
    commodity_id = STATISTICAL_NAME_TO_COMMODITY_ID[crop.name]
    assert prepare.FAOSTAT_ITEM_CODE_TO_CROP_CODE[commodity_id] == crop.value


def test_with_crop_codes_keeps_only_crop_rows() -> None:
    frame = pandas.DataFrame(
        {
            "commodity_id": [
                CommodityId.MAIZE,
                CommodityId.COFFEE_GREEN,
                CommodityId.CATTLE_MEAT,
                CommodityId.PASTURE,
                CommodityId.DROPPED,
                CommodityId.UNATTRIBUTED,
            ]
        }
    ).astype({"commodity_id": int})
    assert prepare.with_crop_codes(frame=frame)["crop_code"].tolist() == [
        statistical.Crop.MAIZE.value,
        ifpri_mapspam.Crop2000.COFF.name,
    ]
