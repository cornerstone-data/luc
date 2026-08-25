"""Retrieve the external anchors, exactly as published, and record what was retrieved.

Not `jdluc.ingest`, which mirrors a registered dataset into managed storage, normalized and
tile-partitioned, so the pipeline can compute over it. This keeps an anchor's published bytes as they
are, under `data/.cache/raw/<source>/<revision>/`, and records each in `data/sources.lock.json`.
Reading an anchor is then a local file read, which is what lets the report run offline.

Every lock entry has one shape, whichever writer produced it: `path` is where the bytes are, relative
to the cache root, `sha256` and `bytes` describe them, `origin` says where they came from and is
absent for a supplied file that was never retrieved, and `revision` appears where one is pinned. Both
are strings a reader cannot tell apart by inspection, so keeping them in named fields is what stops a
consumer reading one as the other.

FAOSTAT is absent from the retrieval loop: it arrives as an ingested `jdluc` dataset, and
`record_digest` pins the ingested parquet instead of an upstream URL.

**The lock is committed; the bytes are not.** The anchors run to 11 MiB across 128 files, and none of
it belongs in a repository when the lock reproduces all of it: a committed record keyed on
content makes a stale or swapped file visible in a diff, where inherited CSVs with no provenance
could not be traced to a source at all. Every source pins a revision, so a re-run reproduces rather
than fetching whatever is published today -- WRI to a commit, FAOSTAT to its digest, since it
publishes no revision. A fresh checkout runs `pull` once and the digests confirm it got what the lock
describes.

  Retrieval only; run it with `python -m validation --stage pull`.
"""

import collections.abc
import dataclasses
import hashlib
import json
import logging
import pathlib
import shutil
import urllib.request

logger = logging.getLogger(__name__)

# The data directory's layout, held here because `prepare` already imports this module and the
# tools/ generators need the same paths. Everything in DATA is committed except CACHE.
DATA = pathlib.Path(__file__).resolve().parent / "data"
LOCK = DATA / "sources.lock.json"
CACHE = DATA / ".cache"
RAW = CACHE / "raw"
# What `validation.capture` writes. Not committed: these are re-keyed extracts of the pipeline's own
# parquet outputs, so a copy here would be a second one. Unlike everything else under CACHE they are
# not re-fetchable either -- producing them takes hours of pipeline -- so a report carrying
# capture-dependent numbers reproduces only alongside the run that made them. `prepare` reads both
# back, so the report can render a capture's results without importing the module that produced them.
CAPTURE = CACHE / "capture"
EFS = CAPTURE / "efs.parquet"
FOREST_POOLS = CAPTURE / "forest_pools.parquet"

# The commit these files were read at. Bumping it is a deliberate act: every anchor number moves.
WRI_REVISION = "559fe23eb752e9df270a1bf93e7f290044026bab"
WRI_ROOT = f"https://raw.githubusercontent.com/wri/GCSC/{WRI_REVISION:s}/data"
# All 42, because the national rollup needs every commodity, not just the ones we model. The 42 are
# not a choice of ours: they are the crops SPAM maps in 2005, 2010 *and* 2020, which is what the
# allocation needs, since a crop with only one snapshot has no expansion to allocate on (Fitts et al.
# 2025a; docs/validation.md).
WRI_CROP_CODES = (
    "ACOF",
    "BANA",
    "BARL",
    "BEAN",
    "CASS",
    "CHIC",
    "CNUT",
    "COCO",
    "COTT",
    "COWP",
    "GROU",
    "LENT",
    "MAIZ",
    "OCER",
    "OFIB",
    "OILP",
    "OOIL",
    "OPUL",
    "ORTS",
    "PIGE",
    "PLNT",
    "PMIL",
    "POTA",
    "RAPE",
    "RCOF",
    "REST",
    "RICE",
    "SESA",
    "SMIL",
    "SORG",
    "SOYB",
    "SUGB",
    "SUGC",
    "SUNF",
    "SWPO",
    "TEAS",
    "TEMF",
    "TOBA",
    "TROF",
    "VEGE",
    "WHEA",
    "YAMS",
)
# The four crops SPAM 2020 v2 adds, which therefore have a yield factor and no emission factor --
# `EF_ADM0_RUBB_CO2.csv` and its three siblings 404 at WRI_REVISION. Spelled out rather than derived
# from the taxonomy, for the same reason `ifpri_mapspam.YEAR_TO_UNRECOVERABLE_CROP_NAMES` is: a
# derived list would absorb a new crop silently, where the point is to notice. Rubber is the one that
# matters -- the largest crop our own leg cannot model, so a factor appearing here would be the first
# external anchor it has ever had.
WRI_CROPS_WITHOUT_FACTORS = ("CITR", "ONIO", "RUBB", "TOMA")
CHUNK_BYTES = 1 << 22


@dataclasses.dataclass(frozen=True)
class Grain:
    """A grain WRI publishes factors at, and what differs between the two."""

    directory: str
    gas_scopes: tuple[str, ...]
    name: str
    token: str


# Both gas scopes nationally: one product differing by CH4 and N2O, and having both means a
# comparison
# never has to guess which an anchor figure came from. Provincially the 0.5% cannot matter to a
# shape.
WRI_GRAINS = (
    Grain(
        directory="sLUC_emission_factors/deforestation_emission_factors_admin0",
        gas_scopes=("CO2", "CO2e"),
        name="national",
        token="ADM0",
    ),
    Grain(
        directory="sLUC_emission_factors/deforestation_emission_factors_adm1",
        gas_scopes=("CO2",),
        name="provincial",
        token="ADM1",
    ),
)
# The yield factors are what make a provincial comparison possible at all: WRI does not publish
# deforestation-linked production provincially, but factor over yield factor is an emissions
# intensity
# per hectare, with no production term left in it.
WRI_YIELD_FACTORS = (("national", "gadm0"), ("provincial", "gadm1"))


@dataclasses.dataclass(frozen=True)
class Remote:
    """One file to retrieve, and where it lands.

    `relative_path` leads with the source and carries the revision, so the source is read back off
    it
    rather than repeated, and a pin bump is additive: the previous pin stays readable.
    """

    revision: str
    url: str
    relative_path: pathlib.PurePosixPath

    @property
    def path(self) -> pathlib.Path:
        return RAW / self.relative_path

    @property
    def source(self) -> str:
        return self.relative_path.parts[0]


def get_wri_factor_remote(crop_code: str, gas_scope: str, grain: Grain) -> Remote:
    name = f"EF_{grain.token:s}_{crop_code:s}_{gas_scope:s}.csv"
    return Remote(
        revision=WRI_REVISION,
        url=f"{WRI_ROOT:s}/{grain.directory:s}/individual_commodities_{gas_scope:s}/{name:s}",
        relative_path=pathlib.PurePosixPath(
            f"wri/{WRI_REVISION[:12]:s}/{grain.name:s}/{gas_scope:s}/{name:s}"
        ),
    )


def get_wri_yield_remote(grain_name: str) -> Remote:
    suffix = dict(WRI_YIELD_FACTORS)[grain_name]
    return Remote(
        revision=WRI_REVISION,
        url=f"{WRI_ROOT:s}/yield_factors/yield_factor_{suffix:s}.csv",
        relative_path=pathlib.PurePosixPath(
            f"wri/{WRI_REVISION[:12]:s}/yield_factors/{grain_name:s}.csv"
        ),
    )


def get_grain(grain_name: str) -> Grain:
    """The grain WRI publishes at, by the name the rest of the tool uses for it."""
    by_name = {grain.name: grain for grain in WRI_GRAINS}
    assert grain_name in by_name, (
        f"unknown grain {grain_name!r}; WRI publishes at {', '.join(sorted(by_name))}"
    )
    return by_name[grain_name]


def get_pulled_path(remote: Remote) -> pathlib.Path:
    """Where `pull` put a file, refusing to hand back one that is not there.

    Readers resolve a path through the same `Remote` that fetched it, rather than rebuilding the
    pinned layout at each call site, so the writer and its readers agree by construction instead of
    by convention.

    Absence raises, and that is the point. The lock names a file for every crop at every grain, so
    a missing one is a half-run `pull` rather than a fact about what WRI publishes -- and a reader
    that shrugged at it would report thinner eligibility in the voice of a result.
    """
    assert remote.path.exists(), (
        f"{remote.path} is absent, though {LOCK.name} pins it. That is a broken or half-run pull "
        f"rather than a gap in the anchor: run `python -m validation --stage pull`"
    )
    return remote.path


def iter_remotes() -> collections.abc.Iterator[Remote]:
    """Every anchor file the validation reads."""
    for grain in WRI_GRAINS:
        for gas_scope in grain.gas_scopes:
            for crop_code in WRI_CROP_CODES:
                yield get_wri_factor_remote(
                    crop_code=crop_code, gas_scope=gas_scope, grain=grain
                )
    for grain_name, _ in WRI_YIELD_FACTORS:
        yield get_wri_yield_remote(grain_name=grain_name)


def get_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(remote: Remote) -> None:
    remote.path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"GET {remote.url:s}")
    # Streamed, not read whole: the FAOSTAT archive is 32 MiB and there is no reason to hold it.
    with (
        urllib.request.urlopen(remote.url, timeout=300) as response,
        remote.path.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, CHUNK_BYTES)


def read_lock() -> dict[str, dict[str, dict[str, object]]]:
    return json.loads(LOCK.read_text()) if LOCK.exists() else {}


def write_lock(lock: dict[str, dict[str, dict[str, object]]]) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    # Sorted and indented so a bump is a readable diff, and with no timestamp: the content is the
    # identity, and a clock would make two branches disagree about identical files.
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote {LOCK}")


def get_record(
    path: pathlib.Path, origin: str | None = None, revision: str | None = None
) -> dict[str, object]:
    """One lock entry, in the single shape every writer produces and every reader can rely on.

    `path` is always where the bytes are, relative to the cache root; `origin` is always where they
    came from, and is absent for a supplied file that was never retrieved. Keeping the two apart is
    the point: their values are indistinguishable by inspection, so a reader that conflated them
    would produce a confident, wrong answer.
    """
    record: dict[str, object] = {
        "path": str(path.relative_to(CACHE)),
        "sha256": get_sha256(path=path),
        "bytes": path.stat().st_size,
    }
    if origin is not None:
        record["origin"] = origin
    if revision is not None:
        record["revision"] = revision
    return record


def record_digest(
    key: str, source: str, path: pathlib.Path, origin: str | None = None
) -> None:
    """Pin an artifact this tool depends on but does not retrieve.

    FAOSTAT arrives through `jdluc.ingest` rather than through `pull`, so there is no upstream URL to
    lock. What can still be pinned is the ingested parquet we actually compute from, which is better
    provenance than the bytes it came from: it identifies the artifact the numbers were derived off.
    `TabularDataset` records only a hand-maintained `version`, so without this the version would be an
    assertion with nothing checking it.
    """
    lock = read_lock()
    lock.setdefault(source, {})[key] = get_record(path=path, origin=origin)
    write_lock(lock=lock)


def workflow(overwrite: tuple[str, ...], remotes: tuple[Remote, ...]) -> dict[str, int]:
    """Retrieve what is missing or stale, and record every file's digest.

    A file whose digest matches the lock is left alone, so a re-run is cheap and offline. One that
    no
    longer matches is reported rather than accepted: either it was edited here, or the upstream
    moved
    under a pin saying it could not have.
    """
    lock = read_lock()
    counts = {"downloaded": 0, "reused": 0, "changed": 0}
    for remote in remotes:
        key = str(remote.relative_path)
        recorded = lock.get(remote.source, {}).get(key, {})
        wanted = remote.source in overwrite
        if not wanted and remote.path.exists() and recorded.get("sha256"):
            if get_sha256(path=remote.path) == recorded["sha256"]:
                counts["reused"] += 1
                continue
            logger.warning(
                f"{key:s} no longer matches its recorded digest; re-retrieving"
            )
            counts["changed"] += 1
        download(remote=remote)
        counts["downloaded"] += 1
        lock.setdefault(remote.source, {})[key] = get_record(
            path=remote.path, origin=remote.url, revision=remote.revision
        )
    write_lock(lock=lock)
    return counts
