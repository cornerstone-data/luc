"""The commodity codes that key `trace`'s output, and each leg's commodities mapped onto them.

`attribute` names each leg's rows by its own map: CDL crops for the jurisdictional direct leg,
MapSPAM crops for the statistical leg and CROPGRIDS crops for the proxy leg.
`METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID` maps those names onto codes, one entry per leg, and
`trace` adds together the rows that share a code. A code is a FAOSTAT item code where FAOSTAT has an
item, and a negative code of our own where it does not:

- -1 and -2: DROPPED and UNATTRIBUTED, which every leg has
- -3: PASTURE, the pasture no grazer takes
- -201 and -202: OTHER_OILCROPS and OTHER_PULSES, MapSPAM groups of several FAOSTAT items
"""

import enum

from jdluc import attribute, emit, jurisdictional_direct, statistical


@enum.unique
class CommodityId(enum.IntEnum):
    WHEAT = 15
    RICE = 27
    BARLEY = 44
    MAIZE = 56
    MILLET = 79
    SORGHUM = 83
    POTATOES = 116
    SWEET_POTATOES = 122
    CASSAVA = 125
    YAMS = 137
    SUGAR_CANE = 156
    SUGAR_BEET = 157
    BEANS_DRY = 176
    CHICK_PEAS = 191
    COW_PEAS = 195
    PIGEON_PEAS = 197
    LENTILS = 201
    SOYA_BEANS = 236
    GROUNDNUTS = 242
    COCONUTS = 249
    OIL_PALM_FRUIT = 254
    SUNFLOWER_SEED = 267
    RAPE_OR_COLZA_SEED = 270
    SESAME_SEED = 289
    SEED_COTTON = 328
    BANANAS = 486
    PLANTAINS = 489
    COFFEE_GREEN = 656
    CATTLE_MEAT = 867
    DROPPED = -1
    UNATTRIBUTED = -2
    PASTURE = -3
    OTHER_OILCROPS = -201
    OTHER_PULSES = -202


# FAOSTAT's item names, as its bulk downloads spell them
COMMODITY_ID_TO_NAME = {
    CommodityId.WHEAT: "Wheat",
    CommodityId.RICE: "Rice",
    CommodityId.BARLEY: "Barley",
    CommodityId.MAIZE: "Maize (corn)",
    CommodityId.MILLET: "Millet",
    CommodityId.SORGHUM: "Sorghum",
    CommodityId.POTATOES: "Potatoes",
    CommodityId.SWEET_POTATOES: "Sweet potatoes",
    CommodityId.CASSAVA: "Cassava, fresh",
    CommodityId.YAMS: "Yams",
    CommodityId.SUGAR_CANE: "Sugar cane",
    CommodityId.SUGAR_BEET: "Sugar beet",
    CommodityId.BEANS_DRY: "Beans, dry",
    CommodityId.CHICK_PEAS: "Chick peas, dry",
    CommodityId.COW_PEAS: "Cow peas, dry",
    CommodityId.PIGEON_PEAS: "Pigeon peas, dry",
    CommodityId.LENTILS: "Lentils, dry",
    CommodityId.SOYA_BEANS: "Soya beans",
    CommodityId.GROUNDNUTS: "Groundnuts, excluding shelled",
    CommodityId.COCONUTS: "Coconuts, in shell",
    CommodityId.OIL_PALM_FRUIT: "Oil palm fruit",
    CommodityId.SUNFLOWER_SEED: "Sunflower seed",
    CommodityId.RAPE_OR_COLZA_SEED: "Rape or colza seed",
    CommodityId.SESAME_SEED: "Sesame seed",
    CommodityId.SEED_COTTON: "Seed cotton, unginned",
    CommodityId.BANANAS: "Bananas",
    CommodityId.PLANTAINS: "Plantains and cooking bananas",
    CommodityId.COFFEE_GREEN: "Coffee, green",
    CommodityId.CATTLE_MEAT: "Meat of cattle with the bone, fresh or chilled",
    CommodityId.DROPPED: "DROPPED",
    CommodityId.UNATTRIBUTED: "UNATTRIBUTED",
    CommodityId.PASTURE: "PASTURE",
    CommodityId.OTHER_OILCROPS: "OTHER_OILCROPS",
    CommodityId.OTHER_PULSES: "OTHER_PULSES",
}
assert set(COMMODITY_ID_TO_NAME) == set(CommodityId)
assert len(set(COMMODITY_ID_TO_NAME.values())) == len(COMMODITY_ID_TO_NAME)


# COTTON is seed cotton: `usda_nass_quickstats` converts NASS's ginned lint to it, as MapSPAM
# carries it. MapSPAM splits coffee into arabica and robusta and millet into pearl and small, where
# FAOSTAT has one item for each
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
    attribute.Methodology.STATISTICAL: {
        emit.NonCommodity.DROPPED.name: CommodityId.DROPPED,
        emit.NonCommodity.UNATTRIBUTED.name: CommodityId.UNATTRIBUTED,
        statistical.Crop.ARABICA_COFFEE.name: CommodityId.COFFEE_GREEN,
        statistical.Crop.BANANA.name: CommodityId.BANANAS,
        statistical.Crop.BARLEY.name: CommodityId.BARLEY,
        statistical.Crop.BEAN.name: CommodityId.BEANS_DRY,
        statistical.Crop.CASSAVA.name: CommodityId.CASSAVA,
        statistical.Crop.CHICKPEA.name: CommodityId.CHICK_PEAS,
        statistical.Crop.COCONUT.name: CommodityId.COCONUTS,
        statistical.Crop.COTTON.name: CommodityId.SEED_COTTON,
        statistical.Crop.COWPEA.name: CommodityId.COW_PEAS,
        statistical.Crop.GROUNDNUT.name: CommodityId.GROUNDNUTS,
        statistical.Crop.LENTIL.name: CommodityId.LENTILS,
        statistical.Crop.MAIZE.name: CommodityId.MAIZE,
        statistical.Crop.OILPALM.name: CommodityId.OIL_PALM_FRUIT,
        statistical.Crop.OTHER_OILCROPS.name: CommodityId.OTHER_OILCROPS,
        statistical.Crop.OTHER_PULSES.name: CommodityId.OTHER_PULSES,
        statistical.Crop.PEARL_MILLET.name: CommodityId.MILLET,
        statistical.Crop.PIGEONPEA.name: CommodityId.PIGEON_PEAS,
        statistical.Crop.PLANTAIN.name: CommodityId.PLANTAINS,
        statistical.Crop.POTATO.name: CommodityId.POTATOES,
        statistical.Crop.RAPESEED.name: CommodityId.RAPE_OR_COLZA_SEED,
        statistical.Crop.RICE.name: CommodityId.RICE,
        statistical.Crop.ROBUSTA_COFFEE.name: CommodityId.COFFEE_GREEN,
        statistical.Crop.SESAME_SEED.name: CommodityId.SESAME_SEED,
        statistical.Crop.SMALL_MILLET.name: CommodityId.MILLET,
        statistical.Crop.SORGHUM.name: CommodityId.SORGHUM,
        statistical.Crop.SOYBEAN.name: CommodityId.SOYA_BEANS,
        statistical.Crop.SUGARBEET.name: CommodityId.SUGAR_BEET,
        statistical.Crop.SUGARCANE.name: CommodityId.SUGAR_CANE,
        statistical.Crop.SUNFLOWER.name: CommodityId.SUNFLOWER_SEED,
        statistical.Crop.SWEET_POTATO.name: CommodityId.SWEET_POTATOES,
        statistical.Crop.WHEAT.name: CommodityId.WHEAT,
        statistical.Crop.YAM.name: CommodityId.YAMS,
        statistical.Livestock.BEEF_CATTLE.name: CommodityId.CATTLE_MEAT,
        statistical.Livestock.PASTURE.name: CommodityId.PASTURE,
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
assert set(
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
        attribute.Methodology.JURISDICTIONAL_DIRECT
    ]
) == {
    *(crop.name for crop in jurisdictional_direct.Crop),
    *(non_commodity.name for non_commodity in emit.NonCommodity),
}
assert set(
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[attribute.Methodology.STATISTICAL]
) == {
    *(crop.name for crop in statistical.Crop),
    *(livestock.name for livestock in statistical.Livestock),
    *(non_commodity.name for non_commodity in emit.NonCommodity),
}
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
