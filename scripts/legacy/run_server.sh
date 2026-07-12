#!/usr/bin/env bash
# run_server.sh — Lanza el servidor central FL (fedmammobench) en Docker.
#
# USO:
#   bash scripts/run_server.sh
#
# LOGS:
#   En el host: runs/exp01_fedavg_resnet50_6nodes/server.log
#   En vivo:    docker logs -f fedmammo-server
#
# NOTA: El servidor bloquea hasta que MIN_AVAILABLE_CLIENTS nodos se conecten.
#       Los nodos de otras máquinas deben apuntar a la IP LAN de este servidor.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta si es necesario
# ─────────────────────────────────────────────────────────────────────────────

# Config YAML del servidor.
SERVER_CONFIG="${SERVER_CONFIG:-configs/exp01_fedavg_resnet50_server.yaml}"

# Tag de la imagen Docker.
IMAGE_TAG="${IMAGE_TAG:-fedmammobench:latest}"

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — no editar
# ─────────────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
CONTAINER_NAME="fedmammo-server"

# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIONES
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -f "$REPO_DIR/$SERVER_CONFIG" ]]; then
    echo "ERROR: config no encontrada: $REPO_DIR/$SERVER_CONFIG" >&2
    exit 1
fi
if [[ ! -f "$REPO_DIR/weights/RadImageNet-resnet50.pth" ]]; then
    echo "ERROR: pesos no encontrados: $REPO_DIR/weights/RadImageNet-resnet50.pth" >&2
    exit 1
fi

# Detener servidor anterior si existiera.
if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    echo "AVISO: deteniendo servidor anterior '$CONTAINER_NAME'..."
    docker rm -f "$CONTAINER_NAME"
fi

mkdir -p "$REPO_DIR/runs"

# Obtener IP LAN para que los nodos remotos sepan a dónde conectarse.
LAN_IP=$(hostname -I | awk '{print $1}')

echo "=========================================="
echo "  fedmammobench — Servidor Central"
echo "=========================================="
echo "  Escuchando en : 0.0.0.0:8080"
echo "  IP LAN        : $LAN_IP"
echo "  Nodos remotos deben usar: $LAN_IP:8080"
echo "  Config        : $SERVER_CONFIG"
echo "  Logs host     : $REPO_DIR/runs/exp01_fedavg_resnet50_6nodes/server.log"
echo "=========================================="
echo ""

docker run --rm \
    --name "$CONTAINER_NAME" \
    --network host \
    -e PYTHONUNBUFFERED=1 \
    -v "$REPO_DIR/configs:/app/configs:ro" \
    -v "$REPO_DIR/scripts:/app/scripts:ro" \
    -v "$REPO_DIR/src:/app/src:ro" \
    -v "$REPO_DIR/weights:/app/weights:ro" \
    -v "$REPO_DIR/runs:/app/runs" \
    "$IMAGE_TAG" \
    python scripts/run_server.py \
        --config "$SERVER_CONFIG"
