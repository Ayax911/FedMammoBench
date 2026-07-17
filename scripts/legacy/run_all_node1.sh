#!/usr/bin/env bash
# run_all_node1.sh — Lanza todos los experimentos en el Nodo 1 (CMMD / China).
#
# Ejecutar en la MÁQUINA DEL NODO 1.
#
# Variables OBLIGATORIAS a definir antes de ejecutar:
#   SERVER_ADDRESS  — IP y puerto del servidor (ej. export SERVER_ADDRESS=192.168.1.10:8080)
#   IMAGE_ROOT_HOST — ruta al directorio de imágenes CMMD en este host
#
# Uso:
#   export SERVER_ADDRESS=192.168.1.10:8080
#   export IMAGE_ROOT_HOST=/ruta/local/a/las/imágenes
#   bash scripts/run_all_node1.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

NODE_ID=1 \
MANIFEST_HOST="$REPO_DIR/manifests/node1_manifest.csv" \
    bash "$REPO_DIR/scripts/run_all_experiments_node.sh"
