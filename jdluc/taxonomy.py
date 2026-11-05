"""The commodity codes that key `trace`'s output, and each leg's commodities mapped onto them.

`attribute` names each leg's rows by its own map: CDL crops for the jurisdictional direct leg,
MapSPAM crops for the statistical leg and CROPGRIDS crops for the proxy leg.
`METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID` maps those names onto codes, one entry per leg. A
code is a FAOSTAT item code where FAOSTAT has an item, and a negative code of our own where it does
not:

- -1 and -2: DROPPED and UNATTRIBUTED, which every leg has
- -3: PASTURE, the pasture no grazer takes
"""

import enum

from jdluc import attribute, emit, jurisdictional_direct


@enum.unique
class CommodityId(enum.IntEnum):
    WHEAT = 15
    RICE = 27
    BARLEY = 44
    MAIZE = 56
    SORGHUM = 83
    POTATOES = 116
    SUGAR_CANE = 156
    SUGAR_BEET = 157
    BEANS_DRY = 176
    SOYA_BEANS = 236
    SEED_COTTON = 328
    CATTLE_MEAT = 867
    DROPPED = -1
    UNATTRIBUTED = -2
    PASTURE = -3


# FAOSTAT's item names, as its bulk downloads spell them
COMMODITY_ID_TO_NAME = {
    CommodityId.WHEAT: "Wheat",
    CommodityId.RICE: "Rice",
    CommodityId.BARLEY: "Barley",
    CommodityId.MAIZE: "Maize (corn)",
    CommodityId.SORGHUM: "Sorghum",
    CommodityId.POTATOES: "Potatoes",
    CommodityId.SUGAR_CANE: "Sugar cane",
    CommodityId.SUGAR_BEET: "Sugar beet",
    CommodityId.BEANS_DRY: "Beans, dry",
    CommodityId.SOYA_BEANS: "Soya beans",
    CommodityId.SEED_COTTON: "Seed cotton, unginned",
    CommodityId.CATTLE_MEAT: "Meat of cattle with the bone, fresh or chilled",
    CommodityId.DROPPED: "DROPPED",
    CommodityId.UNATTRIBUTED: "UNATTRIBUTED",
    CommodityId.PASTURE: "PASTURE",
}
assert set(COMMODITY_ID_TO_NAME) == set(CommodityId)
assert len(set(COMMODITY_ID_TO_NAME.values())) == len(COMMODITY_ID_TO_NAME)


# COTTON is seed cotton: `usda_nass_quickstats` converts NASS's ginned lint to it
METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID = {
    attribute.Methodology.JURISDICTIONAL_DIRECT: {
        emit.NonCommodity.DROPPED.name: CommodityId.DROPPED,
        emit.NonCommodity.UNATTRIBUTED.name: CommodityId.UNATTRIBUTED,
        jurisdictional_direct.Crop.BARLEY.name: CommodityId.BARLEY,
        jurisdictional_direct.Crop.BEAN.name: CommodityId.BEANS_DRY,
        jurisdictional_direct.Crop.COTTON.name: CommodityId.SEED_COTTON,
        jurisdictional_direct.Crop.MAIZE.name: CommodityId.MAIZE,
        jurisdictional_direct.Crop.POTATO.name: CommodityId.POTATOES,
        jurisdictional_direct.Crop.RICE.name: CommodityId.RICE,
        jurisdictional_direct.Crop.SORGHUM.name: CommodityId.SORGHUM,
        jurisdictional_direct.Crop.SOYBEAN.name: CommodityId.SOYA_BEANS,
        jurisdictional_direct.Crop.SUGARBEET.name: CommodityId.SUGAR_BEET,
        jurisdictional_direct.Crop.SUGARCANE.name: CommodityId.SUGAR_CANE,
        jurisdictional_direct.Crop.WHEAT.name: CommodityId.WHEAT,
    },
}
assert all(
    set(commodity_name_to_commodity_id)
    == {
        *attribute.get_commodity_names(methodology=methodology),
        *(non_commodity.name for non_commodity in emit.NonCommodity),
    }
    for methodology, commodity_name_to_commodity_id in (
        METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID.items()
    )
)
assert len(
    set(
        METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
            attribute.Methodology.JURISDICTIONAL_DIRECT
        ].values()
    )
) == len(
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
        attribute.Methodology.JURISDICTIONAL_DIRECT
    ]
)
