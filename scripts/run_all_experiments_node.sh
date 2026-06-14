#!/usr/bin/env bash
# run_all_experiments_node.sh — Lanza todos los experimentos en secuencia en un nodo cliente.
#
# Ejecutar en la MÁQUINA NODO. Cada experimento bloquea hasta que el servidor
# complete todas las rondas; luego el nodo avanza al siguiente experimento.
# El servidor debe arrancar cada experimento antes (o casi simultáneamente) que los nodos.
#
# Uso:
#   NODE_ID=1 \
#   SERVER_ADDRESS=192.168.1.10:8080 \
#   MANIFEST_HOST=/ruta/local/node1_manifest.csv \
#   IMAGE_ROOT_HOST=/ruta/local/imágenes \
#   bash scripts/run_all_experiments_node.sh
#
# Variables OBLIGATORIAS (sin valor por defecto):
#   NODE_ID         — ID numérico de este nodo (1–5)
#   SERVER_ADDRESS  — IP y puerto del servidor (ej. 192.168.1.10:8080)
#   MANIFEST_HOST   — ruta absoluta al CSV manifest de este nodo en el host
#   IMAGE_ROOT_HOST — ruta absoluta al directorio de imágenes en el host
#
# Variables opcionales:
#   IMAGE_TAG    — imagen Docker (por defecto fedmammobench:latest)
#   EXPERIMENTS  — lista de experimentos a ejecutar

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

# Validar variables obligatorias
: "${NODE_ID:?Debes definir NODE_ID (ej. NODE_ID=1)}"
: "${SERVER_ADDRESS:?Debes definir SERVER_ADDRESS (ej. SERVER_ADDRESS=192.168.1.10:8080)}"
: "${MANIFEST_HOST:?Debes definir MANIFEST_HOST (ruta al CSV manifest del nodo)}"
: "${IMAGE_ROOT_HOST:?Debes definir IMAGE_ROOT_HOST (ruta al directorio de imágenes)}"

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTOS — debe coincidir exactamente con el orden del servidor
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS="${EXPERIMENTS:-exp01_fedavg exp04_fedyogi exp05_fedadam exp06_fedprox}"

echo "=========================================="
echo "  fedmammobench — Lote Nodo ${NODE_ID}"
echo "  Servidor  : $SERVER_ADDRESS"
echo "  Manifest  : $MANIFEST_HOST"
echo "  Imágenes  : $IMAGE_ROOT_HOST"
echo "  Experimentos: $EXPERIMENTS"
echo "=========================================="
echo ""

EXP_NUM=0
for EXP in $EXPERIMENTS; do
    EXP_NUM=$((EXP_NUM + 1))
    CONFIG="configs/${EXP}_resnet50_client.yaml"

    if [[ ! -f "$REPO_DIR/$CONFIG" ]]; then
        echo "ERROR: config no encontrada: $REPO_DIR/$CONFIG" >&2
        exit 1
    fi

    echo ""
    echo "══════════════════════════════════════════"
    echo "  Nodo ${NODE_ID} — Experimento $EXP_NUM: $EXP"
    echo "  Config: $CONFIG"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════"

    NODE_ID="$NODE_ID" \
    SERVER_ADDRESS="$SERVER_ADDRESS" \
    MANIFEST_HOST="$MANIFEST_HOST" \
    IMAGE_ROOT_HOST="$IMAGE_ROOT_HOST" \
    CLIENT_CONFIG="$CONFIG" \
    IMAGE_TAG="${IMAGE_TAG:-fedmammobench:latest}" \
        bash "$REPO_DIR/scripts/run_node.sh"

    echo ""
    echo "✓ Nodo ${NODE_ID} — $EXP completado — $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "=========================================="
echo "  Lote completo: $EXP_NUM experimentos (Nodo ${NODE_ID})"
echo "=========================================="
