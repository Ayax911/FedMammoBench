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

# Ensure python3.11 is the active interpreter.
# CUDA base images ship Python 3.10; Debian/Ubuntu disable ensurepip for system Python,
# so pip is bootstrapped via get-pip.py (downloaded with Python's own urllib).
RUN if ! python3.11 --version >/dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.11 python3.11-venv && \
        rm -rf /var/lib/apt/lists/*; \
    fi && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    python3.11 -c "\
import urllib.request; \
open('/tmp/get-pip.py','wb').write(urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py').read())" && \
    python3.11 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

# Now copy the project sources and install the package itself.
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN python -m pip install -e .

# Default to an interactive shell; CI / experiments override this.
CMD ["bash"]
