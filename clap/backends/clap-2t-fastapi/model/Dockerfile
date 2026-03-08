FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 체크포인트 다운로드 (빌드 타임, ~2.35GB)
RUN mkdir -p /app/checkpoints && \
    wget -q -O /app/checkpoints/music_audioset_epoch_15_esc_90.14.pt \
    "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

ENV CLAP_CHECKPOINT_PATH=/app/checkpoints/music_audioset_epoch_15_esc_90.14.pt

COPY app/ app/
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
