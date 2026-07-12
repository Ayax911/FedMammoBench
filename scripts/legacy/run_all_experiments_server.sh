#!/usr/bin/env bash
# run_all_experiments_server.sh — Lanza todos los experimentos en secuencia en el servidor.
#
# Ejecutar en la MÁQUINA SERVIDOR. Cada experimento bloquea hasta que se completan
# todas las rondas (los nodos deben estar corriendo su script equivalente en paralelo).
#
# Uso:
#   bash scripts/run_all_experiments_server.sh
#
# Variables de entorno opcionales:
#   IMAGE_TAG    — imagen Docker (por defecto fedmammobench:latest)
#   EXPERIMENTS  — lista de experimentos a ejecutar (por defecto: los 4 de Fase 1)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTOS — define la lista con la variable EXPERIMENTS si quieres varios.
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS="${EXPERIMENTS:-exp01_fedavg}"

echo "=========================================="
echo "  fedmammobench — Lote Servidor"
echo "  Experimentos: $EXPERIMENTS"
echo "  Repo: $REPO_DIR"
echo "=========================================="
echo ""

EXP_NUM=0
for EXP in $EXPERIMENTS; do
    EXP_NUM=$((EXP_NUM + 1))
    CONFIG="configs/${EXP}_resnet50_server.yaml"

    if [[ ! -f "$REPO_DIR/$CONFIG" ]]; then
        echo "ERROR: config no encontrada: $REPO_DIR/$CONFIG" >&2
        exit 1
    fi

    echo ""
    echo "══════════════════════════════════════════"
    echo "  Experimento $EXP_NUM: $EXP"
    echo "  Config: $CONFIG"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════"

    SERVER_CONFIG="$CONFIG" \
    IMAGE_TAG="${IMAGE_TAG:-fedmammobench:latest}" \
        bash "$REPO_DIR/scripts/run_server.sh"

    echo ""
    echo "✓ $EXP completado — $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "=========================================="
echo "  Lote completo: $EXP_NUM experimentos"
echo "  Artefactos en: $REPO_DIR/runs/"
echo "=========================================="
