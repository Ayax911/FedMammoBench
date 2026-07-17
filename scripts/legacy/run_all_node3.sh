#!/usr/bin/env bash
# run_all_node3.sh — Lanza todos los experimentos en el Nodo 3 (CDD-CESM / Egipto).
#
# Ejecutar en la MÁQUINA DEL NODO 3.
#
# Variables OBLIGATORIAS a definir antes de ejecutar:
#   SERVER_ADDRESS  — IP y puerto del servidor (ej. export SERVER_ADDRESS=192.168.1.10:8080)
#   IMAGE_ROOT_HOST — ruta al directorio de imágenes CDD-CESM en este host
#
# Uso:
#   export SERVER_ADDRESS=192.168.1.10:8080
#   export IMAGE_ROOT_HOST=/ruta/local/a/las/imágenes
#   bash scripts/run_all_node3.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

NODE_ID=3 \
MANIFEST_HOST="$REPO_DIR/manifests/node3_manifest.csv" \
    bash "$REPO_DIR/scripts/run_all_experiments_node.sh"
