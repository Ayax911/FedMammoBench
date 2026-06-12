# Multi-stage Dockerfile for fedmammobench.
# Default build is CPU-only. For CUDA, override BASE_IMAGE at build time, e.g.:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 -t fedmammobench:gpu .

ARG BASE_IMAGE=python:3.11-slim-bookworm

FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# System deps. libgl1 for opencv, libgomp1 for sklearn/torch threading,
# git for editable installs from VCS if you ever need it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# If BASE_IMAGE is the CUDA image it won't ship Python; install it.
RUN command -v python3.11 >/dev/null 2>&1 || ( \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip && \
        ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
        rm -rf /var/lib/apt/lists/* \
    )

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy the project sources and install the package itself.
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install -e .

# Default to an interactive shell; CI / experiments override this.
CMD ["bash"]
