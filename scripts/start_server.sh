#!/usr/bin/env bash
# start_server.sh — arranca el servidor central Flower gRPC.
#
# Uso:
#   bash scripts/start_server.sh [CONFIG] [IP_BIND] [PUERTO]
#
# Ejemplos:
#   bash scripts/start_server.sh                              # config por defecto, 0.0.0.0:8080
#   bash scripts/start_server.sh configs/exp01_5nodes_server.yaml
#   bash scripts/start_server.sh configs/exp01_5nodes_server.yaml 192.168.14.184 8080
#
# El servidor espera a que MIN_AVAILABLE_CLIENTS nodos se conecten antes de
# iniciar la primera ronda. Arranca PRIMERO el servidor, luego los clientes.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ── Activar entorno virtual ──────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    : # ya hay un venv activo
else
    echo "[server] AVISO: no se encontró .venv — usando Python del sistema"
fi

# ── Parámetros ───────────────────────────────────────────────────────────────
CONFIG="${1:-configs/exp01_fedavg_resnet50_server.yaml}"
BIND_IP="${2:-0.0.0.0}"
PORT="${3:-8080}"

# ── Variable de entorno para pesos RadImageNet ───────────────────────────────
export FEDMAMMOBENCH_RADIMAGENET_DIR="${FEDMAMMOBENCH_RADIMAGENET_DIR:-$REPO_ROOT/weights}"

if [[ ! -d "$FEDMAMMOBENCH_RADIMAGENET_DIR" ]]; then
    echo "[server] AVISO: FEDMAMMOBENCH_RADIMAGENET_DIR no existe: $FEDMAMMOBENCH_RADIMAGENET_DIR"
fi

LAN_IP=$(hostname -I | awk '{print $1}')

echo "════════════════════════════════════════════════════════════════════"
echo "  Servidor federated learning"
echo "  Config  : $CONFIG"
echo "  Escucha : ${BIND_IP}:${PORT}"
echo "  IP LAN  : ${LAN_IP}:${PORT}  ← usa esta IP en los nodos cliente"
echo "  Pesos   : $FEDMAMMOBENCH_RADIMAGENET_DIR"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python scripts/run_server.py \
    --config "$CONFIG" \
    --address "${BIND_IP}:${PORT}"
