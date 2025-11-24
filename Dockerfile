ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=online

# System dependencies for PyTorch / PennyLane and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Torch install is parameterized to support CPU (default) or CUDA builds via build args
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC="torch==2.3.1 torchvision==0.18.1"

RUN pip install --no-cache-dir --upgrade pip && \
    # Optional: strip torch/torchvision from requirements to avoid overriding the chosen spec
    python - <<'PY'
from pathlib import Path
import re

req = Path('requirements.txt').read_text().splitlines()
filtered = [line for line in req if not re.match(r'^(torch|torchvision)\b', line.strip())]
Path('requirements.filtered.txt').write_text('\n'.join(filtered) + '\n')
PY
RUN pip install --no-cache-dir --extra-index-url ${TORCH_INDEX_URL} ${TORCH_SPEC} && \
    pip install --no-cache-dir -r requirements.filtered.txt && \
    pip cache purge

COPY . .

ENV PYTHONPATH=/app

# Default to an interactive shell so training commands can be launched manually
CMD ["bash"]
