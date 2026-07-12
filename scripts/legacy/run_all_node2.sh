#!/usr/bin/env bash
# run_all_node2.sh — Lanza todos los experimentos en el Nodo 2 (InBreast / Portugal).
#
# Ejecutar en la MÁQUINA DEL NODO 2.
#
# Variables OBLIGATORIAS a definir antes de ejecutar:
#   SERVER_ADDRESS  — IP y puerto del servidor (ej. export SERVER_ADDRESS=192.168.1.10:8080)
#   IMAGE_ROOT_HOST — ruta al directorio de imágenes InBreast en este host
#
# Uso:
#   export SERVER_ADDRESS=192.168.1.10:8080
#   export IMAGE_ROOT_HOST=/ruta/local/a/las/imágenes
#   bash scripts/run_all_node2.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

NODE_ID=2 \
MANIFEST_HOST="$REPO_DIR/manifests/node2_manifest.csv" \
    bash "$REPO_DIR/scripts/run_all_experiments_node.sh"
