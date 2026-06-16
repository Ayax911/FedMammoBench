#!/usr/bin/env bash
# run_node.sh — Lanza un nodo cliente FL (fedmammobench) en Docker con GPU.
#
# EDITA LAS VARIABLES DEL BLOQUE "CONFIGURACIÓN" PARA CADA NODO.
#
# USO:
#   bash scripts/run_node.sh           # lanza node 0 con los valores por defecto
#   NODE_ID=1 bash scripts/run_node.sh # lanza node 1 (sobreescribiendo NODE_ID)
#
# LOGS y RESULTADOS:
#   En el host:  runs/exp01_fedavg_resnet50/client_<NODE_ID>/client.log
#   En vivo:     docker logs -f fedmammo-node<NODE_ID>
#
# NOTA: El cliente Flower reintenta la conexión con backoff exponencial.
#       Si el servidor aún no está arriba, el nodo espera y reintenta.
#       Los datos y pesos se validan ANTES de intentar conectar al servidor.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas variables para cada nodo
# ─────────────────────────────────────────────────────────────────────────────

# ID del nodo (0-5). Debe ser único entre todos los nodos del experimento.
NODE_ID="${NODE_ID:-0}"

# Dirección del servidor FL (HOST:PUERTO).
# Si el servidor corre en esta misma máquina: 127.0.0.1:8080
# Si corre en otra máquina: <IP_LAN>:8080
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"

# Ruta absoluta en el HOST al manifest CSV de este nodo.
MANIFEST_HOST="${MANIFEST_HOST:-/home/imagenesmedicas/Descargas/mini-ddsm.csv}"

# Directorio raíz de imágenes en el HOST.
# Las rutas en el manifest son relativas a esta carpeta.
IMAGE_ROOT_HOST="${IMAGE_ROOT_HOST:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"

# Config YAML del cliente (relativo a la raíz del repo, se monta en /app).
CLIENT_CONFIG="${CLIENT_CONFIG:-configs/exp01_fedavg_resnet50_client.yaml}"

# Tag de la imagen Docker.
IMAGE_TAG="${IMAGE_TAG:-fedmammobench:latest}"

# Usar GPU (1) o CPU (0).
USE_GPU="${USE_GPU:-1}"

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — no editar
# ─────────────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
CONTAINER_NAME="fedmammo-node${NODE_ID}"
MANIFEST_CONTAINER="/app/manifests/node${NODE_ID}.csv"

# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIONES
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -f "$MANIFEST_HOST" ]]; then
    echo "ERROR: manifest no encontrado en: $MANIFEST_HOST" >&2
    exit 1
fi
if [[ ! -d "$IMAGE_ROOT_HOST" ]]; then
    echo "ERROR: directorio de imágenes no encontrado: $IMAGE_ROOT_HOST" >&2
    exit 1
fi
if [[ ! -f "$REPO_DIR/weights/RadImageNet-resnet50.pth" ]]; then
    echo "ERROR: pesos no encontrados en: $REPO_DIR/weights/RadImageNet-resnet50.pth" >&2
    exit 1
fi

# Detener contenedor anterior si existiera.
if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    echo "AVISO: deteniendo contenedor anterior '$CONTAINER_NAME'..."
    docker rm -f "$CONTAINER_NAME"
fi

# Crear carpeta de runs (el contenedor la monta rw).
mkdir -p "$REPO_DIR/runs"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL COMANDO docker run
# ─────────────────────────────────────────────────────────────────────────────

GPU_FLAG=""
if [[ "$USE_GPU" == "1" ]]; then
    GPU_FLAG="--gpus all"
fi

echo "=========================================="
echo "  fedmammobench — Nodo ${NODE_ID}"
echo "=========================================="
echo "  Servidor  : $SERVER_ADDRESS"
echo "  Manifest  : $MANIFEST_HOST"
echo "  Imágenes  : $IMAGE_ROOT_HOST"
echo "  Config    : $CLIENT_CONFIG"
echo "  GPU       : $([ "$USE_GPU" = "1" ] && echo "sí" || echo "no")"
echo "  Logs host : $REPO_DIR/runs/exp01_fedavg_resnet50/client_${NODE_ID}/client.log"
echo "=========================================="
echo ""

# shellcheck disable=SC2086
docker run --rm \
    --name "$CONTAINER_NAME" \
    --network host \
    $GPU_FLAG \
    -e PYTHONUNBUFFERED=1 \
    -v "$REPO_DIR/configs:/app/configs:ro" \
    -v "$REPO_DIR/scripts:/app/scripts:ro" \
    -v "$REPO_DIR/src:/app/src:ro" \
    -v "$REPO_DIR/weights:/app/weights:ro" \
    -v "$REPO_DIR/runs:/app/runs" \
    -v "$IMAGE_ROOT_HOST:/app/images:ro" \
    -v "$MANIFEST_HOST:$MANIFEST_CONTAINER:ro" \
    "$IMAGE_TAG" \
    python scripts/run_client.py \
        --config "$CLIENT_CONFIG" \
        --server "$SERVER_ADDRESS" \
        --client-id "$NODE_ID" \
        --manifest "$MANIFEST_CONTAINER" \
        --data-dir /app/images
