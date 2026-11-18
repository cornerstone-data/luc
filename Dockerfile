# syntax=docker/dockerfile:1.19
FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2

COPY --from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /uvx /bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python

WORKDIR /app

RUN : \
    && apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        git \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app \
    && :

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

COPY --exclude=.git . .
RUN uv sync --frozen --no-dev

COPY .git .git
RUN git rev-parse --verify HEAD

ENV PATH="/app/.venv/bin:${PATH}"
