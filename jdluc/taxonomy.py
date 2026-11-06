"""The commodity codes that key `trace`'s output, and each leg's commodities mapped onto them.

`attribute` names each leg's rows by its own map: CDL crops for the jurisdictional direct leg,
MapSPAM crops for the statistical leg and CROPGRIDS crops for the proxy leg.
`METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID` maps those names onto codes, one entry per leg, and
`trace` adds together the rows that share a code. A code is a FAOSTAT item code where FAOSTAT has an
item, and a negative code of our own where it does not:

- -1 and -2: DROPPED and UNATTRIBUTED, which every leg has
- -3: PASTURE, the pasture no grazer takes
- -101 to -116: the proxy leg's fodder crops, which have no production
- -201 and -202: OTHER_OILCROPS and OTHER_PULSES, MapSPAM groups of several FAOSTAT items

`CommodityId` holds every code of our own and each FAOSTAT item the repo refers to by name. The
proxy leg maps most of its crops onto bare item codes.
"""

import enum

from jdluc import attribute, emit, jurisdictional_direct, statistical
from jdluc.datasets import tang_cropgrids


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
    ALFALFA = -101
    BEETS_FOR_FODDER = -102
    CABBAGE_FOR_FODDER = -103
    CARROTS_FOR_FODDER = -104
    CLOVER = -105
    FORAGE_PRODUCTS_OTHER = -106
    GRASSES_OTHER = -107
    LEGUMES_OTHER = -108
    MAIZE_FOR_FORAGE_AND_SILAGE = -109
    MIXED_GRASSES_AND_LEGUMES = -110
    GREEN_OILSEEDS_FOR_FODDER = -111
    RYE_GRASS_FOR_FORAGE_AND_SILAGE = -112
    SORGHUM_FOR_FORAGE_AND_SILAGE = -113
    SWEDES_FOR_FODDER = -114
    TURNIPS_FOR_FODDER = -115
    VEGETABLES_AND_ROOTS_FOR_FODDER = -116
    OTHER_OILCROPS = -201
    OTHER_PULSES = -202


# FAOSTAT's item names, as its bulk downloads spell them, and the fodder crops' names in the
# CROPGRIDS publisher's crop table
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
    CommodityId.ALFALFA: "Alfalfa",
    CommodityId.BEETS_FOR_FODDER: "Beets for fodder",
    CommodityId.CABBAGE_FOR_FODDER: "Cabbage for fodder",
    CommodityId.CARROTS_FOR_FODDER: "Carrots for fodder",
    CommodityId.CLOVER: "Clover",
    CommodityId.FORAGE_PRODUCTS_OTHER: "Forage products, other",
    CommodityId.GRASSES_OTHER: "Grasses, other",
    CommodityId.LEGUMES_OTHER: "Legumes, other",
    CommodityId.MAIZE_FOR_FORAGE_AND_SILAGE: "Maize for forage and silage",
    CommodityId.MIXED_GRASSES_AND_LEGUMES: "Mixed grasses and legumes",
    CommodityId.GREEN_OILSEEDS_FOR_FODDER: "Green oilseeds for fodder",
    CommodityId.RYE_GRASS_FOR_FORAGE_AND_SILAGE: "Rye grass for forage and silage",
    CommodityId.SORGHUM_FOR_FORAGE_AND_SILAGE: "Sorghum for forage and silage",
    CommodityId.SWEDES_FOR_FODDER: "Swedes for fodder",
    CommodityId.TURNIPS_FOR_FODDER: "Turnips for fodder",
    CommodityId.VEGETABLES_AND_ROOTS_FOR_FODDER: "Vegetables and roots for fodder",
    CommodityId.OTHER_OILCROPS: "OTHER_OILCROPS",
    CommodityId.OTHER_PULSES: "OTHER_PULSES",
}
assert set(COMMODITY_ID_TO_NAME) == set(CommodityId)
assert len(set(COMMODITY_ID_TO_NAME.values())) == len(COMMODITY_ID_TO_NAME)


# COTTON is seed cotton: `usda_nass_quickstats` converts NASS's ginned lint to it, as MapSPAM
# carries it. MapSPAM splits coffee into arabica and robusta and millet into pearl and small, where
# FAOSTAT has one item for each
METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID: dict[
    attribute.Methodology, dict[str, int]
] = {
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
    attribute.Methodology.JURISDICTIONAL_DIRECT_PROXY: {
        emit.NonCommodity.DROPPED.name: CommodityId.DROPPED,
        emit.NonCommodity.UNATTRIBUTED.name: CommodityId.UNATTRIBUTED,
        statistical.Livestock.BEEF_CATTLE.name: CommodityId.CATTLE_MEAT,
        statistical.Livestock.PASTURE.name: CommodityId.PASTURE,
        # The publisher's crop table (`CROPS_matching_v17.xlsx` in figshare's `CODES.zip`), through
        # the authors' 2020 FAOSTAT extract. Both kapok crops are 310, kapok fruit
        tang_cropgrids.Crop.ABACA.name: 809,
        tang_cropgrids.Crop.AGAVE.name: 800,
        tang_cropgrids.Crop.ALMOND.name: 221,
        tang_cropgrids.Crop.ANISEETC.name: 711,
        tang_cropgrids.Crop.APPLE.name: 515,
        tang_cropgrids.Crop.APRICOT.name: 526,
        tang_cropgrids.Crop.ARECA.name: 226,
        tang_cropgrids.Crop.ARTICHOKE.name: 366,
        tang_cropgrids.Crop.ASPARAGUS.name: 367,
        tang_cropgrids.Crop.AVOCADO.name: 572,
        tang_cropgrids.Crop.BAMBARA.name: 203,
        tang_cropgrids.Crop.BANANA.name: 486,
        tang_cropgrids.Crop.BARLEY.name: 44,
        tang_cropgrids.Crop.BEAN.name: 176,
        tang_cropgrids.Crop.BERRYNES.name: 558,
        tang_cropgrids.Crop.BLUEBERRY.name: 552,
        tang_cropgrids.Crop.BRAZIL.name: 216,
        tang_cropgrids.Crop.BROADBEAN.name: 181,
        tang_cropgrids.Crop.BUCKWHEAT.name: 89,
        tang_cropgrids.Crop.CABBAGE.name: 358,
        tang_cropgrids.Crop.CANARYSEED.name: 101,
        tang_cropgrids.Crop.CAROB.name: 461,
        tang_cropgrids.Crop.CARROT.name: 426,
        tang_cropgrids.Crop.CASHEW.name: 217,
        tang_cropgrids.Crop.CASHEWAPPLE.name: 591,
        tang_cropgrids.Crop.CASSAVA.name: 125,
        tang_cropgrids.Crop.CASTOR.name: 265,
        tang_cropgrids.Crop.CAULIFLOWER.name: 393,
        tang_cropgrids.Crop.CEREALNES.name: 108,
        tang_cropgrids.Crop.CHERRY.name: 531,
        tang_cropgrids.Crop.CHESTNUT.name: 220,
        tang_cropgrids.Crop.CHICKPEA.name: 191,
        tang_cropgrids.Crop.CHICORY.name: 459,
        tang_cropgrids.Crop.CHILLEETC.name: 401,
        tang_cropgrids.Crop.CINNAMON.name: 693,
        tang_cropgrids.Crop.CITRUSNES.name: 512,
        tang_cropgrids.Crop.CLOVE.name: 698,
        tang_cropgrids.Crop.COCOA.name: 661,
        tang_cropgrids.Crop.COCONUT.name: 249,
        tang_cropgrids.Crop.COFFEE.name: 656,
        tang_cropgrids.Crop.COTTON.name: 328,
        tang_cropgrids.Crop.COWPEA.name: 195,
        tang_cropgrids.Crop.CRANBERRY.name: 554,
        tang_cropgrids.Crop.CUCUMBERETC.name: 397,
        tang_cropgrids.Crop.CURRANT.name: 550,
        tang_cropgrids.Crop.DATE.name: 577,
        tang_cropgrids.Crop.EGGPLANT.name: 399,
        tang_cropgrids.Crop.FIBRENES.name: 821,
        tang_cropgrids.Crop.FIG.name: 569,
        tang_cropgrids.Crop.FONIO.name: 94,
        tang_cropgrids.Crop.FRUITNES.name: 619,
        tang_cropgrids.Crop.GARLIC.name: 406,
        tang_cropgrids.Crop.GINGER.name: 720,
        tang_cropgrids.Crop.GOOSEBERRY.name: 549,
        tang_cropgrids.Crop.GRAPE.name: 560,
        tang_cropgrids.Crop.GRAPEFRUITETC.name: 507,
        tang_cropgrids.Crop.GREENBEAN.name: 414,
        tang_cropgrids.Crop.GREENCORN.name: 446,
        tang_cropgrids.Crop.GREENONION.name: 402,
        tang_cropgrids.Crop.GREENPEA.name: 417,
        tang_cropgrids.Crop.GROUNDNUT.name: 242,
        tang_cropgrids.Crop.HAZELNUT.name: 225,
        tang_cropgrids.Crop.HEMP.name: 777,
        tang_cropgrids.Crop.HEMPSEED.name: 336,
        tang_cropgrids.Crop.HOP.name: 677,
        tang_cropgrids.Crop.JUTE.name: 780,
        tang_cropgrids.Crop.KAPOKFIBER.name: 310,
        tang_cropgrids.Crop.KAPOKSEED.name: 310,
        tang_cropgrids.Crop.KARITE.name: 263,
        tang_cropgrids.Crop.KIWI.name: 592,
        tang_cropgrids.Crop.KOLANUT.name: 224,
        tang_cropgrids.Crop.LEMONLIME.name: 497,
        tang_cropgrids.Crop.LENTIL.name: 201,
        tang_cropgrids.Crop.LETTUCE.name: 372,
        tang_cropgrids.Crop.LINSEED.name: 333,
        tang_cropgrids.Crop.LUPIN.name: 210,
        tang_cropgrids.Crop.MAIZE.name: 56,
        tang_cropgrids.Crop.MANGO.name: 571,
        tang_cropgrids.Crop.MATE.name: 671,
        tang_cropgrids.Crop.MELONETC.name: 568,
        tang_cropgrids.Crop.MELONSEED.name: 299,
        tang_cropgrids.Crop.MILLET.name: 79,
        tang_cropgrids.Crop.MIXEDGRAIN.name: 103,
        tang_cropgrids.Crop.MUSHROOM.name: 449,
        tang_cropgrids.Crop.MUSTARD.name: 292,
        tang_cropgrids.Crop.NUTMEG.name: 702,
        tang_cropgrids.Crop.NUTNES.name: 234,
        tang_cropgrids.Crop.OATS.name: 75,
        tang_cropgrids.Crop.OILPALM.name: 254,
        tang_cropgrids.Crop.OILSEEDNES.name: 339,
        tang_cropgrids.Crop.OKRA.name: 430,
        tang_cropgrids.Crop.OLIVE.name: 260,
        tang_cropgrids.Crop.ONION.name: 403,
        tang_cropgrids.Crop.ORANGE.name: 490,
        tang_cropgrids.Crop.PAPAYA.name: 600,
        tang_cropgrids.Crop.PEA.name: 187,
        tang_cropgrids.Crop.PEACHETC.name: 534,
        tang_cropgrids.Crop.PEAR.name: 521,
        tang_cropgrids.Crop.PEPPER.name: 687,
        tang_cropgrids.Crop.PEPPERMINT.name: 748,
        tang_cropgrids.Crop.PERSIMMON.name: 587,
        tang_cropgrids.Crop.PIGEONPEA.name: 197,
        tang_cropgrids.Crop.PINEAPPLE.name: 574,
        tang_cropgrids.Crop.PISTACHIO.name: 223,
        tang_cropgrids.Crop.PLANTAIN.name: 489,
        tang_cropgrids.Crop.PLUM.name: 536,
        tang_cropgrids.Crop.POPPY.name: 296,
        tang_cropgrids.Crop.POTATO.name: 116,
        tang_cropgrids.Crop.PULSENES.name: 211,
        tang_cropgrids.Crop.PUMPKINETC.name: 394,
        tang_cropgrids.Crop.PYRETHRUM.name: 754,
        tang_cropgrids.Crop.QUINCE.name: 523,
        tang_cropgrids.Crop.QUINOA.name: 92,
        tang_cropgrids.Crop.RAMIE.name: 788,
        tang_cropgrids.Crop.RAPESEED.name: 270,
        tang_cropgrids.Crop.RASBERRY.name: 547,
        tang_cropgrids.Crop.RICE.name: 27,
        tang_cropgrids.Crop.ROOTNES.name: 149,
        tang_cropgrids.Crop.RUBBER.name: 836,
        tang_cropgrids.Crop.RYE.name: 71,
        tang_cropgrids.Crop.SAFFLOWER.name: 280,
        tang_cropgrids.Crop.SESAME.name: 289,
        tang_cropgrids.Crop.SISAL.name: 789,
        tang_cropgrids.Crop.SORGHUM.name: 83,
        tang_cropgrids.Crop.SOURCHERRY.name: 530,
        tang_cropgrids.Crop.SOYBEAN.name: 236,
        tang_cropgrids.Crop.SPICENES.name: 723,
        tang_cropgrids.Crop.SPINACH.name: 373,
        tang_cropgrids.Crop.STONEFRUITNES.name: 541,
        tang_cropgrids.Crop.STRAWBERRY.name: 544,
        tang_cropgrids.Crop.STRINGBEAN.name: 423,
        tang_cropgrids.Crop.SUGARBEET.name: 157,
        tang_cropgrids.Crop.SUGARCANE.name: 156,
        tang_cropgrids.Crop.SUGARNES.name: 161,
        tang_cropgrids.Crop.SUNFLOWER.name: 267,
        tang_cropgrids.Crop.SWEETPOTATO.name: 122,
        tang_cropgrids.Crop.TANGETC.name: 495,
        tang_cropgrids.Crop.TARO.name: 136,
        tang_cropgrids.Crop.TEA.name: 667,
        tang_cropgrids.Crop.TOBACCO.name: 826,
        tang_cropgrids.Crop.TOMATO.name: 388,
        tang_cropgrids.Crop.TRITICALE.name: 97,
        tang_cropgrids.Crop.TROPICALNES.name: 603,
        tang_cropgrids.Crop.TUNG.name: 275,
        tang_cropgrids.Crop.VANILLA.name: 692,
        tang_cropgrids.Crop.VEGFOR.name: 463,
        tang_cropgrids.Crop.VETCH.name: 205,
        tang_cropgrids.Crop.WALNUT.name: 222,
        tang_cropgrids.Crop.WATERMELON.name: 567,
        tang_cropgrids.Crop.WHEAT.name: 15,
        tang_cropgrids.Crop.YAM.name: 137,
        tang_cropgrids.Crop.YAUTIA.name: 135,
        # FAOSTAT's own items, where the publisher's table names none or a retired one
        tang_cropgrids.Crop.FLAX.name: 771,  # replaced 773, flax fibre and tow
        tang_cropgrids.Crop.GREENBROADBEAN.name: 420,
        tang_cropgrids.Crop.JUTELIKEFIBER.name: 782,  # formerly jute-like fibres
        # Judgement: FAOSTAT's dry chillies and peppers names Pimenta, and popcorn is maize grain
        tang_cropgrids.Crop.PIMENTO.name: 689,
        tang_cropgrids.Crop.POPCORN.name: 56,
        # Fodder, which FAOSTAT has no item for
        tang_cropgrids.Crop.ALFALFA.name: CommodityId.ALFALFA,
        tang_cropgrids.Crop.BEETFOR.name: CommodityId.BEETS_FOR_FODDER,
        tang_cropgrids.Crop.CABBAGEFOR.name: CommodityId.CABBAGE_FOR_FODDER,
        tang_cropgrids.Crop.CARROTFOR.name: CommodityId.CARROTS_FOR_FODDER,
        tang_cropgrids.Crop.CLOVER.name: CommodityId.CLOVER,
        tang_cropgrids.Crop.FORNES.name: CommodityId.FORAGE_PRODUCTS_OTHER,
        tang_cropgrids.Crop.GRASSNES.name: CommodityId.GRASSES_OTHER,
        tang_cropgrids.Crop.LEGUMENES.name: CommodityId.LEGUMES_OTHER,
        tang_cropgrids.Crop.MAIZEFOR.name: CommodityId.MAIZE_FOR_FORAGE_AND_SILAGE,
        tang_cropgrids.Crop.MIXEDGRASS.name: CommodityId.MIXED_GRASSES_AND_LEGUMES,
        tang_cropgrids.Crop.OILSEEDFOR.name: CommodityId.GREEN_OILSEEDS_FOR_FODDER,
        tang_cropgrids.Crop.RYEFOR.name: CommodityId.RYE_GRASS_FOR_FORAGE_AND_SILAGE,
        tang_cropgrids.Crop.SORGHUMFOR.name: CommodityId.SORGHUM_FOR_FORAGE_AND_SILAGE,
        tang_cropgrids.Crop.SWEDEFOR.name: CommodityId.SWEDES_FOR_FODDER,
        tang_cropgrids.Crop.TURNIPFOR.name: CommodityId.TURNIPS_FOR_FODDER,
        tang_cropgrids.Crop.VEGETABLENES.name: CommodityId.VEGETABLES_AND_ROOTS_FOR_FODDER,
    },
}
assert set(METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID) == set(attribute.Methodology)
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
assert set(
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
        attribute.Methodology.JURISDICTIONAL_DIRECT_PROXY
    ]
) == {
    *(crop.name for crop in tang_cropgrids.Crop),
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
