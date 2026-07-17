#!/usr/bin/env bash
# start_client.sh — arranca un nodo cliente Flower gRPC.
#
# Uso:
#   bash scripts/start_client.sh <NODE_ID> <SERVER_IP> [DATA_DIR] [MANIFEST] [CONFIG]
#
# Argumentos:
#   NODE_ID    Entero único del nodo (0, 1, 2, 3, 4 …)
#   SERVER_IP  IP LAN del servidor central (sin puerto), ej. 192.168.14.184
#   DATA_DIR   Directorio raíz de imágenes  (default: data/mammobench/images)
#   MANIFEST   CSV manifest de este nodo    (default: manifests/node<ID>_manifest.csv)
#   CONFIG     YAML de configuración        (default: configs/exp01_fedavg_resnet50_client.yaml)
#
# Ejemplos:
#   bash scripts/start_client.sh 0 192.168.14.184
#   bash scripts/start_client.sh 2 192.168.14.184 /mnt/datos/imagenes manifests/node2_manifest.csv
#   NODE_ID=1 SERVER_IP=192.168.14.184 bash scripts/start_client.sh
#
# Variables de entorno:
#   FEDMAMMOBENCH_RADIMAGENET_DIR  — ruta al directorio con RadImageNet-resnet50.pth
#   SERVER_PORT                    — puerto del servidor (default 8080)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ── Parámetros ────────────────────────────────────────────────────────────────
NODE_ID="${1:-${NODE_ID:?'ERROR: Falta NODE_ID. Uso: start_client.sh <NODE_ID> <SERVER_IP>'}}"
SERVER_IP="${2:-${SERVER_IP:?'ERROR: Falta SERVER_IP. Uso: start_client.sh <NODE_ID> <SERVER_IP>'}}"
DATA_DIR="${3:-data/mammobench/images}"
MANIFEST="${4:-manifests/node${NODE_ID}_manifest.csv}"
CONFIG="${5:-configs/exp01_fedavg_resnet50_client.yaml}"
PORT="${SERVER_PORT:-8080}"

# ── Activar entorno virtual ───────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    : # venv ya activo
else
    echo "[nodo${NODE_ID}] AVISO: no se encontró .venv — usando Python del sistema"
fi

# ── Pesos RadImageNet ─────────────────────────────────────────────────────────
export FEDMAMMOBENCH_RADIMAGENET_DIR="${FEDMAMMOBENCH_RADIMAGENET_DIR:-$REPO_ROOT/weights}"

# ── Verificaciones previas ────────────────────────────────────────────────────
ERRORS=0
if [[ ! -f "$MANIFEST" ]]; then
    echo "[nodo${NODE_ID}] ERROR: manifest no encontrado: $MANIFEST"
    ERRORS=1
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "[nodo${NODE_ID}] ERROR: directorio de imágenes no encontrado: $DATA_DIR"
    ERRORS=1
fi
if ! ls "${FEDMAMMOBENCH_RADIMAGENET_DIR}"/*.pth &>/dev/null; then
    echo "[nodo${NODE_ID}] ERROR: no hay archivos .pth en FEDMAMMOBENCH_RADIMAGENET_DIR=${FEDMAMMOBENCH_RADIMAGENET_DIR}"
    ERRORS=1
fi
if [[ $ERRORS -ne 0 ]]; then
    exit 1
fi

MANIFEST_ROWS=$(tail -n +2 "$MANIFEST" | wc -l)

echo "════════════════════════════════════════════════════════════════════"
echo "  Nodo cliente  : $NODE_ID"
echo "  Servidor      : ${SERVER_IP}:${PORT}"
echo "  Config        : $CONFIG"
echo "  Manifest      : $MANIFEST  ($MANIFEST_ROWS filas)"
echo "  Imágenes      : $DATA_DIR"
echo "  Pesos         : $FEDMAMMOBENCH_RADIMAGENET_DIR"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python scripts/run_client.py \
    --config "$CONFIG" \
    --server "${SERVER_IP}:${PORT}" \
    --client-id "$NODE_ID" \
    --manifest "$MANIFEST" \
    --data-dir "$DATA_DIR"
