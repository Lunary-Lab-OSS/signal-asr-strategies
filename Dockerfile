FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/signal-asr-strategies/.venv/bin:$PATH" \
    SIGNAL_ASR_HOST=0.0.0.0 \
    SIGNAL_ASR_PORT=18500 \
    SIGNAL_ASR_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 asr

WORKDIR /opt/signal-asr-strategies

# Dependency layer: cached unless packaging metadata changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra server --extra cpu --no-install-project --no-editable

# Project layer: rebuilt when the package source changes.
COPY README.md LICENSE ./
COPY signal_asr ./signal_asr
RUN uv sync --locked --no-dev --extra server --extra cpu --no-editable

# Only the model directory is writable by the runtime user; the virtualenv
# and package source stay root-owned.
RUN mkdir -p /models && chown asr:asr /models
USER asr
WORKDIR /models

EXPOSE 18500
HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=60s \
  CMD curl -fsS "http://127.0.0.1:${SIGNAL_ASR_PORT}/health" || exit 1

# The entry point reads SIGNAL_ASR_HOST/SIGNAL_ASR_PORT, so
# `docker run -e SIGNAL_ASR_PORT=...` and the health check stay in sync.
CMD ["signal-asr-server"]
