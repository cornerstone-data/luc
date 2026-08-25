"""Materialize the GADM-to-World-Bank admin-1 key map, once, for review and committing.

WRI publishes provincial factors against GADM's `GID_1` (`USA.10_1`); the capture keys provinces on
the World Bank's `ADM1CD_c` (`USA001`). Nothing published joins the two, so they are matched here on
country and province name -- which is fuzzy, and therefore something to do once, review, and commit,
rather than at the top of every run.

Runtime then does a strict lookup and fails on a missing key. That is deliberate: silently dropping
a
province would quietly shrink a rollup and make its coverage look better than it is.

A tool rather than part of `validation/` because its output is the artifact: it runs rarely, reads a
93 MiB GeoPackage, needs geopandas, and nothing on the reporting path imports it. `validation/` is
left as the reporting path, with no geospatial dependency and no large inputs.

The map is checked at the end of the run rather than by a test suite, and a failed check exits
nonzero. That is deliberate for a generator -- the report's exit code never depends on its results,
but a build tool that emits a broken artifact should fail loudly.

  uv run --with geopandas python tools/build-national-mappings.py
  uv run --with geopandas python tools/build-national-mappings.py --overwrite
"""

import argparse
import collections.abc
import dataclasses
import json
import logging
import pathlib
import unicodedata

import geopandas
import pandas

from jdluc import tiling, utils
from jdluc.datasets import worldbank_jurisdictions
from validation import prepare, pull, schema

logger = logging.getLogger(__name__)

OUTPUT = pull.DATA / "gadm_to_world_bank_admin_1.json"
# Names GADM and the World Bank spell differently enough that no normalization will join them. Kept
# here rather than in the output so a re-run cannot silently lose a hand-made decision.
OVERRIDES = pull.DATA / "admin_1_overrides.json"
# Orbae's provincial ids are opaque (`USA-20230119-1`), so its provinces join on name like GADM's do.
# Every country in the export is attempted; the countries are read off it rather than listed, so the
# scope cannot drift from the data.
ORBAE_OUTPUT = pull.DATA / "orbae_to_world_bank_admin_1.json"

# An equal-area projection, so a province's area does not depend on its latitude.
EQUAL_AREA_CRS = "EPSG:6933"
SQUARE_METRES_PER_SQUARE_KILOMETRE = 1e6

WRI_REVISION = "559fe23eb752e9df270a1bf93e7f290044026bab"
GADM_KEY_URL = (
    f"https://raw.githubusercontent.com/wri/GCSC/{WRI_REVISION:s}/data/gadm_admin_keys/"
    "key_gadm_adm1.csv"
)
# Administrative-type words one side appends and the other does not: the World Bank writes "Anhui
# Sheng" where GADM writes "Anhui". Stripped from both sides, so neither spelling is privileged.
ADMINISTRATIVE_WORDS = frozenset(
    {
        "administrativeregion",
        "autonomousregion",
        "canton",
        "capital",
        "city",
        "county",
        "department",
        "departamento",
        "district",
        "division",
        "emirate",
        "governorate",
        "krai",
        "krong",
        "kray",
        "municipality",
        "oblast",
        "oblasti",
        "okrug",
        "parish",
        "prefecture",
        "province",
        "provincia",
        "regiao",
        "region",
        "republic",
        "sheng",
        "state",
        "territory",
        "voivodeship",
        "zhou",
    }
)
# A floor rather than an expected value: the share moves whenever either source is revised, and
# pinning it exactly would fail on every legitimate change. It is here to catch a collapse -- a
# renamed column matching nothing -- not to police normal drift. 62.8% at the current revisions.
MINIMUM_MATCHED_SHARE = 0.55
# The Orbae map's own floor. Lower than GADM's because a third of its misses are structural rather
# than spelling -- Cote d'Ivoire's regions against districts, France's post-2016 regions against
# pre-reform ones -- and no map can close those. 77.1% at the current export.
MINIMUM_ORBAE_MATCHED_SHARE = 0.70


def get_keys(name: str) -> set[str]:
    """Every comparable form of a name, because one side sometimes gives two.

    The World Bank writes Canadian provinces bilingually -- "British Columbia / Colombie-
    Britannique"
    -- where GADM gives English only, so each alternative is offered separately as well as the
    whole.
    """
    parts = [part for part in str(name).split("/") if part.strip()]
    return {normalize(name=part) for part in [name, *parts]} - {""}


def normalize(name: str) -> str:
    """A comparable form of a province name: unaccented, lowercase, alphanumeric only.

    Accents are the largest single source of mismatch -- "Bie" against "Bié" -- and neither side is
    consistently accented, so both are folded rather than one being corrected to the other.
    """
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(
        character
        for character in decomposed
        if (not unicodedata.combining(character) and character.isalnum())
        or character.isspace()
    )
    words = [word for word in stripped.casefold().split() if word]
    kept = [word for word in words if word not in ADMINISTRATIVE_WORDS] or words
    return "".join(kept)


def download_world_bank(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    worldbank_jurisdictions.ADMIN_1_DATASET.save_tile_id_to_local_path(
        str(path), tiling.WHOLE_WORLD_TILE_ID
    )


def download_gadm_keys(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_remote_url_to_local_path(
        local_path=str(path), params={}, remote_url=GADM_KEY_URL
    )


def load_gadm(path: pathlib.Path) -> pandas.DataFrame:
    """GADM's admin-1 units, minus the 26 of 3,691 rows that cannot identify one.

    Twenty-three are territories with no admin-1 subdivision -- Antarctica, Aruba, the Caspian Sea --
    and three name a real province but record GID_1 as the literal string "NA". Dropping both keeps
    them out of the unmatched tally, where they would read as coverage we failed to get.

    Four GID_1s are each assigned to two provinces (GBR.1_1 to England *and* Wales), which is why
    `match` requires a name to identify exactly one unit per side rather than taking the first.
    """
    frame = pandas.read_csv(path).dropna(subset=["GID_1", "NAME_1"])
    frame["keys"] = frame["NAME_1"].map(get_keys)
    return frame.rename(columns={"GID_0": "iso_3166", "GID_1": "gadm_id"})


def load_world_bank(path: pathlib.Path) -> pandas.DataFrame:
    """Provinces with their land area, which is what makes coverage meaningful.

    Counting units treats a 12 km2 island as equal to a province the size of France, and it misleads
    in both directions: Brazil matches 87% of its units but 100% of its area, while Algeria matches
    96% of its units and only 74% of its area, missing two Saharan wilayas of 607,000 km2 between
    them. Area is read here because the GeoPackage already has it, where a cropland weighting would
    need a capture.
    """
    geometries = geopandas.read_file(path).to_crs(EQUAL_AREA_CRS)
    frame = pandas.DataFrame(geometries.drop(columns="geometry"))
    frame["square_kilometres"] = (
        geometries.geometry.area / SQUARE_METRES_PER_SQUARE_KILOMETRE
    )
    frame["keys"] = frame["NAM_1"].map(get_keys)
    return frame.rename(columns={"ISO_A3": "iso_3166", "ADM1CD_c": "world_bank_id"})


def match_orbae(
    orbae_names: dict[str, set[str]], world_bank: pandas.DataFrame
) -> dict[str, str]:
    """Orbae province name to World Bank admin id, per country.

    Exact after the same normalization GADM gets, and 502 of 651 provinces resolve that way. The 149
    that do not split into two kinds, and only one is fixable:

    Naming, worth overrides. Russia is the bulk at 68 -- "Adygea" against "Adygeya Rep.", "Altai Krai"
    against "Altayskiy Kray" -- transliteration stems that no normalization reaches. Czechia, Croatia,
    China, Thailand and Poland add a further 36 of the same kind.

    Structural, and no key map can express it. Cote d'Ivoire gives Orbae's 26 regions against the World
    Bank's 14 districts, France gives Orbae's 13 post-2016 regions against the World Bank's 22
    pre-reform ones, and the Solomon Islands has no World Bank subdivision at all. These are
    many-to-one or absent, the same shape as CAF's 17 prefectures against 7 regions.
    """
    matched: dict[str, str] = {}
    for iso_3166 in sorted(orbae_names):
        provinces = world_bank[world_bank["iso_3166"] == iso_3166]
        by_key: dict[str, set[str]] = {}
        for world_bank_id, keys in zip(
            provinces["world_bank_id"], provinces["keys"], strict=True
        ):
            for key in keys:
                by_key.setdefault(key, set()).add(world_bank_id)
        for name in sorted(orbae_names.get(iso_3166, set())):
            candidates = {
                world_bank_id
                for key in get_keys(name=name)
                for world_bank_id in by_key.get(key, set())
            }
            if len(candidates) == 1:
                matched[f"{iso_3166:s}:{name:s}"] = next(iter(candidates))
    return matched


def match(
    gadm: pandas.DataFrame, overrides: dict[str, str], world_bank: pandas.DataFrame
) -> dict[str, str]:
    """GADM id to World Bank id, wherever a name identifies exactly one province on each side.

    A match is a join on (country, normalized name), kept only where it is unambiguous in both
    directions. Ambiguity is refused rather than guessed: two provinces normalizing alike would
    otherwise file one's emissions under the other's name, permanently and silently.

    Overrides win, and any automatic match colliding with one is dropped, so a hand-made decision
    cannot be half-applied.
    """
    pairs = gadm.explode("keys")[["iso_3166", "gadm_id", "keys"]].merge(
        world_bank.explode("keys")[["iso_3166", "world_bank_id", "keys"]],
        on=["iso_3166", "keys"],
    )
    unambiguous = pairs[
        pairs["gadm_id"].map(pairs.groupby("gadm_id")["world_bank_id"].nunique()).eq(1)
        & pairs["world_bank_id"]
        .map(pairs.groupby("world_bank_id")["gadm_id"].nunique())
        .eq(1)
    ].drop_duplicates(subset="gadm_id")
    overridden = set(overrides.values())
    return {
        gadm_id: world_bank_id
        for gadm_id, world_bank_id in zip(
            unambiguous["gadm_id"], unambiguous["world_bank_id"], strict=True
        )
        if gadm_id not in overrides and world_bank_id not in overridden
    } | overrides


def get_coverage(claimed: set[str], world_bank: pandas.DataFrame) -> dict[str, float]:
    """The share of each country's land area whose province the key map can name.

    This is what eligibility filter E3 reads. It is a proxy -- the real weight is cropland,
    and Algeria's missing area is mostly desert -- but it is available without a capture, where
    cropland is not, and it beats counting units as equal.
    """
    world_bank = world_bank.assign(matched=world_bank["world_bank_id"].isin(claimed))
    by_country = world_bank.groupby("iso_3166")
    return {
        str(iso_3166): float(
            group.loc[group["matched"], "square_kilometres"].sum()
            / group["square_kilometres"].sum()
        )
        for iso_3166, group in by_country
        if group["square_kilometres"].sum()
    }


def get_unmatched(
    claimed: set[str], frame: pandas.DataFrame, id_column: str
) -> dict[str, list[str]]:
    """What each side has left over, by country, so the gap is a number and not an impression."""
    leftover = frame[~frame[id_column].isin(claimed)]
    return {
        str(iso_3166): sorted(group[id_column])
        for iso_3166, group in leftover.groupby("iso_3166")
    }


def read_overrides() -> dict[str, str]:
    """Hand-made matches, each carrying the two names it reconciles.

    The names are in the file rather than only here so an override can be reviewed without opening
    two other sources: "Kracheh / Kratie" is checkable at a glance, `KHM.11_1 -> KHM011` is not.
    """
    if not OVERRIDES.exists():
        return {}
    return {
        gadm_id: entry["world_bank_id"]
        for gadm_id, entry in json.loads(OVERRIDES.read_text()).items()
    }


@dataclasses.dataclass(frozen=True)
class Check:
    """One property the finished map must have, and what was found instead where it does not."""

    name: str
    passed: bool
    detail: str


def iter_checks(
    gadm: pandas.DataFrame,
    matched: dict[str, str],
    overrides: dict[str, str],
    world_bank: pandas.DataFrame,
) -> collections.abc.Iterator[Check]:
    """Verify the finished map rather than trusting the code that built it.

    These check the artifact that gets committed, not the functions that produced it, so a bad map
    fails whichever route produced it -- including a hand-edited overrides file. A failure means the
    map would file one jurisdiction's emissions under another's, silently and permanently.
    """
    world_bank_ids = list(matched.values())
    duplicated = sorted(
        {
            world_bank_id
            for world_bank_id in world_bank_ids
            if world_bank_ids.count(world_bank_id) > 1
        }
    )
    yield Check(
        name="injective",
        passed=not duplicated,
        detail=(
            f"{len(duplicated):d} World Bank id(s) claimed by more than one GADM unit: "
            f"{', '.join(duplicated[:5])}"
            if duplicated
            else f"{len(world_bank_ids):d} World Bank ids, each claimed exactly once"
        ),
    )

    unknown_gadm = sorted(set(matched) - set(gadm["gadm_id"]))
    unknown_world_bank = sorted(set(world_bank_ids) - set(world_bank["world_bank_id"]))
    yield Check(
        name="ids-exist",
        passed=not unknown_gadm and not unknown_world_bank,
        detail=(
            f"{len(unknown_gadm):d} GADM and {len(unknown_world_bank):d} World Bank id(s) in the "
            f"map are absent from their source: "
            f"{', '.join(unknown_gadm[:3] + unknown_world_bank[:3])}"
            if unknown_gadm or unknown_world_bank
            else "every id in the map exists in the source it came from"
        ),
    )

    dropped = sorted(
        gadm_id
        for gadm_id, world_bank_id in overrides.items()
        if matched.get(gadm_id) != world_bank_id
    )
    yield Check(
        name="overrides-applied",
        passed=not dropped,
        detail=(
            f"{len(dropped):d} hand-written override(s) missing from the map or overwritten: "
            f"{', '.join(dropped)}"
            if dropped
            else f"all {len(overrides):d} overrides present with their intended value"
        ),
    )

    share = len(matched) / len(gadm) if len(gadm) else 0.0
    yield Check(
        name="coverage-floor",
        passed=share >= MINIMUM_MATCHED_SHARE,
        detail=(
            f"{share:.1%} of GADM units matched against a {MINIMUM_MATCHED_SHARE:.0%} floor"
            + (
                "; a collapse this size usually means a source renamed a column"
                if share < MINIMUM_MATCHED_SHARE
                else ""
            )
        ),
    )


def get_orbae_province_names() -> dict[str, set[str]]:
    """Every province name Orbae publishes a factor for, by country.

    Read through `prepare` rather than reparsed here, so the names this matches are exactly the names
    the runtime looks up. A second parser would drift.
    """
    provincial = prepare.read_orbae()
    provincial = provincial[provincial["admin_level"] == schema.PROVINCIAL]
    names: dict[str, set[str]] = {}
    for row in provincial.to_dict("records"):
        names.setdefault(str(row["iso_3166"]), set()).add(str(row["jurisdiction_name"]))
    return names


def iter_orbae_checks(
    matched: dict[str, str], names: dict[str, set[str]], world_bank: pandas.DataFrame
) -> collections.abc.Iterator[Check]:
    """The same properties the GADM map is held to, over the Orbae map."""
    claimed = list(matched.values())
    duplicated = sorted({one for one in claimed if claimed.count(one) > 1})
    yield Check(
        name="orbae-injective",
        passed=not duplicated,
        detail=(
            f"{len(duplicated):d} World Bank id(s) claimed by more than one Orbae province: "
            f"{', '.join(duplicated[:5])}"
            if duplicated
            else f"{len(claimed):d} World Bank ids, each claimed exactly once"
        ),
    )
    unknown = sorted(set(claimed) - set(world_bank["world_bank_id"]))
    yield Check(
        name="orbae-ids-exist",
        passed=not unknown,
        detail=(
            f"{len(unknown):d} id(s) absent from the World Bank frame: {', '.join(unknown[:3])}"
            if unknown
            else "every id in the map exists in the World Bank frame"
        ),
    )
    total = sum(map(len, names.values()))
    share = len(matched) / total if total else 0.0
    yield Check(
        name="orbae-coverage-floor",
        passed=share >= MINIMUM_ORBAE_MATCHED_SHARE,
        detail=(
            f"{share:.1%} of {total:d} Orbae provinces matched against a "
            f"{MINIMUM_ORBAE_MATCHED_SHARE:.0%} floor"
        ),
    )


def write_orbae_map(
    names: dict[str, set[str]], world_bank: pandas.DataFrame
) -> dict[str, str]:
    """Materialize the Orbae map, recording coverage and every unmatched province.

    The unmatched list is in the output rather than only in a log: a third of it is structural and
    will never close, so it is a standing fact about the anchor rather than a to-do.
    """
    matched = match_orbae(orbae_names=names, world_bank=world_bank)
    ORBAE_OUTPUT.write_text(
        json.dumps(
            {
                "orbae_export": prepare.ORBAE_EXPORT.name,
                "world_bank_source": {
                    "dataset": worldbank_jurisdictions.ADMIN_1_DATASET.product_name,
                    "version": worldbank_jurisdictions.ADMIN_1_DATASET.version,
                },
                "matched": dict(sorted(matched.items())),
                "coverage": {
                    iso_3166: round(
                        sum(1 for name in these if f"{iso_3166:s}:{name:s}" in matched)
                        / len(these),
                        3,
                    )
                    for iso_3166, these in sorted(names.items())
                },
                "unmatched": {
                    iso_3166: sorted(
                        name
                        for name in these
                        if f"{iso_3166:s}:{name:s}" not in matched
                    )
                    for iso_3166, these in sorted(names.items())
                    if any(f"{iso_3166:s}:{name:s}" not in matched for name in these)
                },
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"Wrote {ORBAE_OUTPUT}")
    return matched


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pull.CACHE,
        help="where the downloads land; not committed, and reused unless --overwrite",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-download both inputs"
    )
    args = parser.parse_args()

    gadm_path = args.cache / "key_gadm_adm1.csv"
    world_bank_path = args.cache / "world_bank_admin_1.gpkg"
    if args.overwrite or not gadm_path.exists():
        download_gadm_keys(path=gadm_path)
    if args.overwrite or not world_bank_path.exists():
        download_world_bank(path=world_bank_path)

    overrides = read_overrides()
    gadm = load_gadm(path=gadm_path)
    world_bank = load_world_bank(path=world_bank_path)
    matched = match(gadm=gadm, overrides=overrides, world_bank=world_bank)
    unmatched_gadm = get_unmatched(
        claimed=set(matched), frame=gadm, id_column="gadm_id"
    )
    unmatched_world_bank = get_unmatched(
        claimed=set(matched.values()), frame=world_bank, id_column="world_bank_id"
    )

    pull.DATA.mkdir(parents=True, exist_ok=True)
    # No timestamp: the inputs' revisions are the identity, and a clock would make two branches
    # disagree about an identical map.
    OUTPUT.write_text(
        json.dumps(
            {
                "gadm_source": {"url": GADM_KEY_URL, "revision": WRI_REVISION},
                "world_bank_source": {
                    "dataset": worldbank_jurisdictions.ADMIN_1_DATASET.product_name,
                    "version": worldbank_jurisdictions.ADMIN_1_DATASET.version,
                },
                "matched": dict(sorted(matched.items())),
                "unmatched_gadm": {
                    iso_3166: sorted(ids)
                    for iso_3166, ids in sorted(unmatched_gadm.items())
                },
                "unmatched_world_bank": dict(sorted(unmatched_world_bank.items())),
                "area_coverage": {
                    iso_3166: round(share, 4)
                    for iso_3166, share in sorted(
                        get_coverage(
                            claimed=set(matched.values()), world_bank=world_bank
                        ).items()
                    )
                },
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    logger.info(f"Wrote {OUTPUT}")

    total = len(gadm)
    print(
        f"\n{len(matched):d} of {total:d} GADM units matched ({len(matched) / total:.1%}); "
        f"{sum(map(len, unmatched_gadm.values())):d} unmatched over "
        f"{len(unmatched_gadm):d} countries"
    )
    print(f"  overrides applied: {len(overrides):d}  ->  {OVERRIDES}")

    checks = list(
        iter_checks(
            gadm=gadm, matched=matched, overrides=overrides, world_bank=world_bank
        )
    )
    # Orbae is optional: its export is a supplied file, and the GADM map must not depend on it.
    if prepare.ORBAE_EXPORT.exists():
        names = get_orbae_province_names()
        orbae_matched = write_orbae_map(names=names, world_bank=world_bank)
        checks += list(
            iter_orbae_checks(matched=orbae_matched, names=names, world_bank=world_bank)
        )
        print(
            f"  orbae: {len(orbae_matched):d} of {sum(map(len, names.values())):d} provinces "
            f"matched  ->  {ORBAE_OUTPUT}"
        )
    else:
        logger.warning(f"{prepare.ORBAE_EXPORT} absent; skipping the Orbae map")
    print()
    for check in checks:
        print(
            f"  {'ok  ' if check.passed else 'FAIL'} {check.name:20s} {check.detail:s}"
        )
    failed = [check.name for check in checks if not check.passed]
    if failed:
        # The maps are still on disk, deliberately: a failed check is easier to diagnose against the
        # written file than against a run that refused to produce one.
        print(
            f"\n{len(failed):d} check(s) failed: {', '.join(failed)}. "
            f"{OUTPUT} is not safe to commit."
        )
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
