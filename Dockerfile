# Light CUDA image (compatible GPU25, Solaris, laptop)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git libgomp1 libgl1 libglib2.0-0 libjpeg-dev zlib1g-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --upgrade pip

COPY requirements.txt .

# ----------------------------------------------------
#  PyTorch light-nightly WITHOUT CUDA libraries
#  -> avoids downloading 3–6 GB of dependencies
# ----------------------------------------------------
RUN pip install --no-cache-dir --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# CuPy (CUDA 12.x) – ultra compatible, ultra léger
RUN pip install --no-cache-dir cupy-cuda12x

# PennyLane GPU backend
RUN pip install --no-cache-dir pennylane pennylane-lightning[gpu]

# Project deps
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .
ENV PYTHONPATH=/app

CMD ["bash"]
