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

`InventedCommodityId` holds the codes of our own. FAOSTAT's codes are bare integers, and their names
come from the FAOSTAT crop table in `trace`.
"""

import enum

from jdluc import attribute, emit, jurisdictional_direct, statistical
from jdluc.datasets import faostat, tang_cropgrids


@enum.unique
class InventedCommodityId(enum.IntEnum):
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


# The names the FAOSTAT crop table does not carry: those of our own codes, from the CROPGRIDS
# publisher's crop table for the fodder crops, and cattle meat's, which is livestock
COMMODITY_ID_TO_NAME: dict[int, str] = {
    InventedCommodityId.DROPPED: "DROPPED",
    InventedCommodityId.UNATTRIBUTED: "UNATTRIBUTED",
    InventedCommodityId.PASTURE: "PASTURE",
    InventedCommodityId.ALFALFA: "Alfalfa",
    InventedCommodityId.BEETS_FOR_FODDER: "Beets for fodder",
    InventedCommodityId.CABBAGE_FOR_FODDER: "Cabbage for fodder",
    InventedCommodityId.CARROTS_FOR_FODDER: "Carrots for fodder",
    InventedCommodityId.CLOVER: "Clover",
    InventedCommodityId.FORAGE_PRODUCTS_OTHER: "Forage products, other",
    InventedCommodityId.GRASSES_OTHER: "Grasses, other",
    InventedCommodityId.LEGUMES_OTHER: "Legumes, other",
    InventedCommodityId.MAIZE_FOR_FORAGE_AND_SILAGE: "Maize for forage and silage",
    InventedCommodityId.MIXED_GRASSES_AND_LEGUMES: "Mixed grasses and legumes",
    InventedCommodityId.GREEN_OILSEEDS_FOR_FODDER: "Green oilseeds for fodder",
    InventedCommodityId.RYE_GRASS_FOR_FORAGE_AND_SILAGE: "Rye grass for forage and silage",
    InventedCommodityId.SORGHUM_FOR_FORAGE_AND_SILAGE: "Sorghum for forage and silage",
    InventedCommodityId.SWEDES_FOR_FODDER: "Swedes for fodder",
    InventedCommodityId.TURNIPS_FOR_FODDER: "Turnips for fodder",
    InventedCommodityId.VEGETABLES_AND_ROOTS_FOR_FODDER: "Vegetables and roots for fodder",
    InventedCommodityId.OTHER_OILCROPS: "OTHER_OILCROPS",
    InventedCommodityId.OTHER_PULSES: "OTHER_PULSES",
    faostat.SPECIES_TO_MEAT_ITEM_CODE[faostat.Species.CATTLE]: (
        "Meat of cattle with the bone, fresh or chilled"
    ),
}
assert set(COMMODITY_ID_TO_NAME) == {
    *InventedCommodityId,
    faostat.SPECIES_TO_MEAT_ITEM_CODE[faostat.Species.CATTLE],
}
assert len(set(COMMODITY_ID_TO_NAME.values())) == len(COMMODITY_ID_TO_NAME)


# COTTON is seed cotton: `usda_nass_quickstats` converts NASS's ginned lint to it, as MapSPAM
# carries it. MapSPAM splits coffee into arabica and robusta and millet into pearl and small, where
# FAOSTAT has one item for each
METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID: dict[
    attribute.Methodology, dict[str, int]
] = {
    attribute.Methodology.JURISDICTIONAL_DIRECT: {
        emit.NonCommodity.DROPPED.name: InventedCommodityId.DROPPED,
        emit.NonCommodity.UNATTRIBUTED.name: InventedCommodityId.UNATTRIBUTED,
        jurisdictional_direct.Crop.BARLEY.name: 44,
        jurisdictional_direct.Crop.BEAN.name: 176,
        jurisdictional_direct.Crop.COTTON.name: 328,
        jurisdictional_direct.Crop.MAIZE.name: 56,
        jurisdictional_direct.Crop.POTATO.name: 116,
        jurisdictional_direct.Crop.RICE.name: 27,
        jurisdictional_direct.Crop.SORGHUM.name: 83,
        jurisdictional_direct.Crop.SOYBEAN.name: 236,
        jurisdictional_direct.Crop.SUGARBEET.name: 157,
        jurisdictional_direct.Crop.SUGARCANE.name: 156,
        jurisdictional_direct.Crop.WHEAT.name: 15,
    },
    attribute.Methodology.STATISTICAL: {
        emit.NonCommodity.DROPPED.name: InventedCommodityId.DROPPED,
        emit.NonCommodity.UNATTRIBUTED.name: InventedCommodityId.UNATTRIBUTED,
        statistical.Crop.ARABICA_COFFEE.name: 656,
        statistical.Crop.BANANA.name: 486,
        statistical.Crop.BARLEY.name: 44,
        statistical.Crop.BEAN.name: 176,
        statistical.Crop.CASSAVA.name: 125,
        statistical.Crop.CHICKPEA.name: 191,
        statistical.Crop.COCONUT.name: 249,
        statistical.Crop.COTTON.name: 328,
        statistical.Crop.COWPEA.name: 195,
        statistical.Crop.GROUNDNUT.name: 242,
        statistical.Crop.LENTIL.name: 201,
        statistical.Crop.MAIZE.name: 56,
        statistical.Crop.OILPALM.name: 254,
        statistical.Crop.OTHER_OILCROPS.name: InventedCommodityId.OTHER_OILCROPS,
        statistical.Crop.OTHER_PULSES.name: InventedCommodityId.OTHER_PULSES,
        statistical.Crop.PEARL_MILLET.name: 79,
        statistical.Crop.PIGEONPEA.name: 197,
        statistical.Crop.PLANTAIN.name: 489,
        statistical.Crop.POTATO.name: 116,
        statistical.Crop.RAPESEED.name: 270,
        statistical.Crop.RICE.name: 27,
        statistical.Crop.ROBUSTA_COFFEE.name: 656,
        statistical.Crop.SESAME_SEED.name: 289,
        statistical.Crop.SMALL_MILLET.name: 79,
        statistical.Crop.SORGHUM.name: 83,
        statistical.Crop.SOYBEAN.name: 236,
        statistical.Crop.SUGARBEET.name: 157,
        statistical.Crop.SUGARCANE.name: 156,
        statistical.Crop.SUNFLOWER.name: 267,
        statistical.Crop.SWEET_POTATO.name: 122,
        statistical.Crop.WHEAT.name: 15,
        statistical.Crop.YAM.name: 137,
        statistical.Livestock.BEEF_CATTLE.name: faostat.SPECIES_TO_MEAT_ITEM_CODE[
            faostat.Species.CATTLE
        ],
        statistical.Livestock.PASTURE.name: InventedCommodityId.PASTURE,
    },
    attribute.Methodology.JURISDICTIONAL_DIRECT_PROXY: {
        emit.NonCommodity.DROPPED.name: InventedCommodityId.DROPPED,
        emit.NonCommodity.UNATTRIBUTED.name: InventedCommodityId.UNATTRIBUTED,
        statistical.Livestock.BEEF_CATTLE.name: faostat.SPECIES_TO_MEAT_ITEM_CODE[
            faostat.Species.CATTLE
        ],
        statistical.Livestock.PASTURE.name: InventedCommodityId.PASTURE,
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
        tang_cropgrids.Crop.ALFALFA.name: InventedCommodityId.ALFALFA,
        tang_cropgrids.Crop.BEETFOR.name: InventedCommodityId.BEETS_FOR_FODDER,
        tang_cropgrids.Crop.CABBAGEFOR.name: InventedCommodityId.CABBAGE_FOR_FODDER,
        tang_cropgrids.Crop.CARROTFOR.name: InventedCommodityId.CARROTS_FOR_FODDER,
        tang_cropgrids.Crop.CLOVER.name: InventedCommodityId.CLOVER,
        tang_cropgrids.Crop.FORNES.name: InventedCommodityId.FORAGE_PRODUCTS_OTHER,
        tang_cropgrids.Crop.GRASSNES.name: InventedCommodityId.GRASSES_OTHER,
        tang_cropgrids.Crop.LEGUMENES.name: InventedCommodityId.LEGUMES_OTHER,
        tang_cropgrids.Crop.MAIZEFOR.name: InventedCommodityId.MAIZE_FOR_FORAGE_AND_SILAGE,
        tang_cropgrids.Crop.MIXEDGRASS.name: InventedCommodityId.MIXED_GRASSES_AND_LEGUMES,
        tang_cropgrids.Crop.OILSEEDFOR.name: InventedCommodityId.GREEN_OILSEEDS_FOR_FODDER,
        tang_cropgrids.Crop.RYEFOR.name: InventedCommodityId.RYE_GRASS_FOR_FORAGE_AND_SILAGE,
        tang_cropgrids.Crop.SORGHUMFOR.name: InventedCommodityId.SORGHUM_FOR_FORAGE_AND_SILAGE,
        tang_cropgrids.Crop.SWEDEFOR.name: InventedCommodityId.SWEDES_FOR_FODDER,
        tang_cropgrids.Crop.TURNIPFOR.name: InventedCommodityId.TURNIPS_FOR_FODDER,
        tang_cropgrids.Crop.VEGETABLENES.name: InventedCommodityId.VEGETABLES_AND_ROOTS_FOR_FODDER,
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

# FAOSTAT crop items no CROPGRIDS crop maps to: co-products, counted in the crop they come from
CO_PRODUCT_ITEM_CODE_TO_ITEM_CODE = {
    256: 254,  # palm kernels
    311: 310,  # kapokseed
    329: 328,  # cotton seed
    378: 125,  # cassava leaves
    767: 328,  # cotton lint
    778: 310,  # kapok fibre
    813: 249,  # coir
}
assert set(CO_PRODUCT_ITEM_CODE_TO_ITEM_CODE.values()) <= set(
    METHODOLOGY_TO_COMMODITY_NAME_TO_COMMODITY_ID[
        attribute.Methodology.JURISDICTIONAL_DIRECT_PROXY
    ].values()
)
# The rest: crops CROPGRIDS does not map (jojoba, tallowtree, leeks and other pome fruits)
UNMAPPED_CROP_ITEM_CODES = {277, 305, 407, 542}
