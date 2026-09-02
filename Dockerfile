# syntax=docker/dockerfile:1


FROM python:3.12-slim-bookworm AS base


COPY --from=ghcr.io/astral-sh/uv:latest \
    /uv \
    /uvx \
    /bin/


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy


WORKDIR /app


# ---------------------------------------------------------
# Stage 1: Download the production embedding model
# ---------------------------------------------------------

FROM base AS model-downloader


COPY pyproject.toml uv.lock ./


RUN uv sync \
    --frozen \
    --no-install-project


COPY scripts/download_embedding_model.py \
    ./scripts/download_embedding_model.py


RUN uv run \
    --no-sync \
    python \
    -c "from scripts.download_embedding_model import download; download('Xenova/all-MiniLM-L6-v2')"


# ---------------------------------------------------------
# Stage 2: Production application
# ---------------------------------------------------------

FROM base AS runtime


COPY pyproject.toml uv.lock ./


RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project


ENV PATH="/app/.venv/bin:$PATH"


COPY src ./src

COPY frontend ./frontend

COPY data/processed ./data/processed


COPY --from=model-downloader \
    /app/models \
    ./models


EXPOSE 8000


HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"


CMD ["sh", "-c", "uvicorn src.pet_first_aid_assistant.api:app --host 0.0.0.0 --port ${PORT:-8000}"]