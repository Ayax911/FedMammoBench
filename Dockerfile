# Imagen de SOLO ENTORNO para FedMammoBench -- no copia código.
#
# Reemplaza el Dockerfile anterior (Python 3.11, PYTHONPATH=/app/src,
# `pip install -e .`), que documentaba un paquete que ya no existe: no hay
# `pyproject.toml` en este árbol (decisión deliberada, ver CLAUDE.md — todo
# corre como `.venv/bin/python -m src.<módulo>`, sin instalación ni
# PYTHONPATH) y CLAUDE.md ya marcaba ese Dockerfile como "no puede construir
# contra este árbol".
#
# Esta imagen instala EXACTAMENTE lo que requirements.txt declara (incluido
# el pin de flwr para src/federated/) y nada más. El repo se monta en
# /workspace vía bind mount (ver docker-compose.federated.yaml) y los
# comandos corren `python -m src.federated.{server,client}` desde ahí --
# igual que en la workstation sin Docker, sin reconstruir la imagen por
# cada cambio de código.
#
# CPU por defecto. Para CUDA:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 -t fedmammobench:gpu .
# (BASE_IMAGE debe traer una libc/glibc compatible con las wheels de
# torch+CUDA de requirements.txt -- verificar en la workstation antes del
# primer build real; ver docs/FEDERATED_DESIGN.md, riesgo #6.)
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl1/libglib2.0-0: Pillow con backends de imagen completos (mismo motivo
# que el Dockerfile anterior). libgomp1: threading de torch. build-essential:
# por si alguna wheel no trae binario para esta base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA base images no siempre traen Python 3.12 -- si no está, se instala
# junto al Python del sistema de la imagen base (igual que el Dockerfile
# anterior hacía para 3.11).
RUN if ! python3.12 --version >/dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev && \
        rm -rf /var/lib/apt/lists/* && \
        python3.12 -m ensurepip --upgrade; \
    fi && \
    ln -sf "$(command -v python3.12)" /usr/local/bin/python

WORKDIR /workspace

# Solo requirements.txt en build time -- el resto del repo llega por bind
# mount (docker-compose.federated.yaml), así que esta capa se cachea hasta
# que requirements.txt cambie, sin importar cuánto cambie src/.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

# Sin COPY de src/ ni configs/, sin `pip install -e .`: no hay paquete que
# instalar (ver CLAUDE.md). El entrypoint real lo da docker-compose.federated.yaml.
CMD ["bash"]
