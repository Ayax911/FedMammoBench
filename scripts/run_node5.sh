#!/usr/bin/env bash
# run_node5.sh — Lanza el nodo 5 (dmid) usando el manifest generado en manifests/
# Las imágenes están en el mismo Mammo-Bench que el nodo 0 (mismo servidor).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

NODE_ID=5 \
MANIFEST_HOST="$REPO_DIR/manifests/node5_manifest.csv" \
bash "$REPO_DIR/scripts/run_node.sh"
