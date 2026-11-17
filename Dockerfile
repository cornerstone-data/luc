# syntax=docker/dockerfile:1.19
# Build from a full clone, not a git worktree: a worktree's .git is a file pointing outside the
# build context, and the image needs the repository for git provenance (utils.get_git_version).

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# The rasterio/GDAL wheels vendor GDAL but still link libexpat, which the slim image omits.
# safe.directory lets git read /app when the container runs as a user other than its owner.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY --exclude=.git . .
RUN uv sync --frozen --no-dev

# .git in its own last layer, so a commit that changes no source rebuilds only this layer.
COPY .git .git

ENV PATH="/app/.venv/bin:${PATH}"
