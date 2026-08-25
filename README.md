# Cornerstone LUC

This repo contains an experimental methodology and data pipeline for estimating the land use change (LUC) related emissions associated with agricultural commodities. This methodology allocates LUC emissions to crops in proportion to their displacement of natural ecosystems, based on high-resolution satellite imagery. It's primarily intended for use in corporate GHG inventories, and follows the new GHGP Land Sector and Removals Standard. It supports two attribution methodologies over a shared per-pixel emissions core: the high-resolution "jurisdictional direct land use change" (jdLUC) calculation where detailed crop maps exist, and a coarser-resolution statistical approach for regions where they don't.

As a proof of concept, the jurisdictional-direct leg covers eleven crops grown in the United States (barley, dry beans, cotton, maize, potato, rice, sorghum, soybean, sugarbeet, sugarcane and wheat). Although U.S. land use change emissions are relatively modest contributors to global totals, the U.S. agricultural sector is well studied and has strong data infrastructure, which makes it a good place to start testing methods. The statistical leg extends the same per-pixel emissions core to global crop coverage, using coarser sub-national statistics (IFPRI MapSPAM) where high-resolution crop maps aren't available.

> **Assessing LSRS conformance?** The [executive summary](docs/executive_summary.md) gives a concise account of how this methodology maps to the GHGP Land Sector and Removals Standard — which requirements it meets, the key modeling choices behind them, and where it deviates or remains a work in progress.

![Land conversion and soy expansion drive LUC emissions in Matopiba, Brazil](docs/figures/soy-brazil-matopiba-methodology.png)

*A worked example of the methodology on real data: soy-driven land conversion and the resulting land-use-change emissions in Matopiba, Brazil. For clarity the maps show only the 2000 and 2020 endpoints, but the pipeline uses all five GLAD epochs (2000, 2005, 2010, 2015, 2020). See [`docs/methodology.md`](docs/methodology.md) for the full walkthrough.*

## Why are we publishing this?

Land use change emissions are one of the key targets for scaled global emissions reductions over the next five years. But measurement of LUC emissions is much harder and more uncertain than energy sector emissions, requiring sophisticated analysis of remote sensing imagery and complex modeling of carbon stocks and flows.

Although there are a number of LUC datasets already available for corporate GHG inventories, the emissions factors they publish differ by factors of 5x or more, thanks to differing choices at various points in the complex LUC modeling chain. Some of the LUC methodologies are very well documented, others less so. But even with the best methodology papers, it's extremely challenging to trace the origin of emissions factor differences to specific methodology decisions, or to see which choices are causing the biggest swings.

Therefore, we've come to believe the best way for the corporate measurement ecosystem to get to credible and stable LUC numbers is to shift to open-source LUC models, at least for the base data cleaning, harmonization, and math. Runnable code is the clearest documentation and the strongest platform for collaboration.

The methodology and technical decisions in this repo are intended as a starting point for discussion and collaboration. The LUC space is early. But we thought the best way to get a good conversation going was to actually publish a working open-source implementation.

## Data access

The pipeline's published artifact is the emissions-factor table: one `emissions-factors.parquet` per data version deposited in a public data archive that mints a DOI per version. The archive is still being set up. Each version's git tag names the commit that produced the table.

```python
>>> import pandas
>>> emission_factors = pandas.read_parquet("emissions-factors.parquet")
```

See `docs/data.md` for the full column reference and `docs/coverage.md` for the countries and crops a version covers.

The harmonized inputs and per-pixel emissions behind that table are pipeline outputs rather than published artifacts — zarr stores far too large to deposit alongside it. Reproduce them with the steps under [Running the pipeline](#running-the-pipeline); sharing the underlying maps is open work.

The data is licensed [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please follow the latest attribution guidance in ATTRIBUTION.md.

## Methodology and architecture

Once you're ready to look under the hood:

- `docs/executive_summary.md` — the short, non-technical overview: what jdLUC is, why LUC emissions matter, and the headline takeaways. Start here if you're new.
- `docs/methodology.md` — the high-level overview of the methodology: the datasets behind it, how we quantify per-pixel emissions, and how the jurisdictional-direct and statistical attribution legs produce emissions factors. The *what* we compute and *why*, from a scientific standpoint.
- `docs/architecture.md` — the system architecture and the rationale behind it: why we build on the [Pangeo](https://pangeo.io/) stack, how the five-stage pipeline is structured, and the tooling, storage, and caching choices that make it reproducible on a single host. The *how* it's built and *why those choices*.
- `docs/data.md` — how to get the data products, grids, and full schemas for the harmonized inputs, per-pixel emissions, and emissions-factor table.
- `docs/coverage.md` — the full list of countries and crops the pipeline produces emissions factors for, by ISO 3166-1 alpha-3 code.
- `docs/validation.md` — how the pipeline is measured against external datasets: how the (country, crop) targets are chosen, which relationships are held as regression controls, and what such a comparison can and cannot conclude. The tooling lives in `validation/` and reports against the gaps in `docs/further_research.md`.

## Running and contributing

### Getting set up

You'll need [uv](https://docs.astral.sh/uv/getting-started/installation/) installed as the Python env manager. Storage is either local directories or a GCS bucket: `INGEST_ROOT` and `SCRATCH_ROOT` in `.env` take any fsspec-supported URI, and only a `gs://` root needs a GCP project. Every source dataset is fetched over https, so ingest needs no cloud account either way.

```bash
# Copy the example env file and fill in your values.
cp .env.example .env
# Fill in the values in .env, including for USDA QuickStats and Harvard Dataverse

# Sync Python dependencies into the project venv.
uv sync
```

**Local roots.** Point `INGEST_ROOT` and `SCRATCH_ROOT` at absolute paths and create them up front:

```bash
mkdir -p "${HOME}/luc/ingest" "${HOME}/luc/scratch"
```

Budget the disk: the two zarr stores dominate, at 12 TiB for the continent-scale benchmark in [`docs/architecture.md`](docs/architecture.md#single-host-by-design).

**A `gs://` root** additionally needs application-default credentials, which gcsfs finds on its own and GDAL's `/vsigs` reader does not:

```bash
# Authenticate gcloud application-default credentials (for GCS access).
gcloud auth application-default login --project "${GCP_PROJECT}"

# Point GDAL at those credentials.
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.config/gcloud/application_default_credentials.json"
```

### Running the pipeline

Every stage is scoped by one or more **ISO 3166 alpha-3 country codes**, or by `--backfill` for a global run. Stages 4–5 additionally take a `--methodology-name` (`STATISTICAL` or `JURISDICTIONAL_DIRECT`). The example below reproduces Honduras via the statistical leg. Honduras is the cheapest end-to-end run in the corpus: the unit of work is a 10° tile, and it falls inside a single one (`20N_090W`). Swap in `USA` for the jurisdictional-direct leg, which needs the US-only CDL and NASS inputs:

```bash
# 1. Ingest each source dataset for the countries (positional: dataset, then ISO codes).
#    Repeat per DATASET in the inventory (see docs/methodology.md).
uv run python -m jdluc.ingest GLAD_GLCLUC HND

# 2. Harmonize the countries' tiles onto the common grid (--grid-name defaults to GLAD ~30 m).
uv run python jdluc/harmonize.py HND

# 3. Compute per-pixel land-conversion emissions (writes zarr).
uv run python jdluc/emit.py HND

# 4. Attribute emissions to crops for one or more countries (writes the rollup parquet).
uv run python jdluc/attribute.py HND --methodology-name STATISTICAL

# 5. Reduce the rollup to the emissions-factor table (writes parquet).
uv run python jdluc/trace.py HND --methodology-name STATISTICAL
```

Stages 1–3 derive their tiles from the country boundaries exactly as stage 4 does, so pre-warming them covers precisely what attribution will request. Each stage pulls its cached upstreams, so re-running a later stage recomputes only what's missing. The artifacts these stages write are documented in `docs/data.md`.

### Running tests

```bash
uv run pytest
```

### Linting the codebase

```bash
uv run pre-commit run
```

### Contributing

The repo is maintained by the Cornerstone Sustainability Data Initiative team. We welcome contributions via pull requests. For more open-ended discussion, feel free to open an issue.
