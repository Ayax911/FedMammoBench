#!/usr/bin/env bash
# setup_node.sh — instala fedmammobench en un nodo físico (cliente o servidor).
#
# Uso:
#   bash setup_node.sh [DIRECTORIO_INSTALACION]
#
# Ejemplo:
#   bash setup_node.sh ~/fedmammobench
#
# Requiere:
#   - Python 3.11+  (sudo apt install python3.11 python3.11-venv python3-pip)
#   - git
#   - Conexión a internet (o clonar manualmente y pasar el directorio)
#
# Variables de entorno opcionales:
#   CUDA_VERSION  — fuerza versión CUDA, p.ej. "121" para cu121 (torch CUDA).
#                   Por defecto detecta automáticamente con nvcc.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_URL="https://github.com/Ayax911/Federal-Learning.git"
BRANCH="feature/radimagenet"
INSTALL_DIR="${1:-$HOME/fedmammobench}"

# ── 1. Clonar el repositorio ────────────────────────────────────────────────
if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "[setup] El repositorio ya existe en '$INSTALL_DIR'. Actualizando…"
    git -C "$INSTALL_DIR" fetch origin "$BRANCH"
    git -C "$INSTALL_DIR" checkout "$BRANCH"
    git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
else
    echo "[setup] Clonando rama $BRANCH en '$INSTALL_DIR'…"
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ── 2. Crear entorno virtual ────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo "[setup] Creando entorno virtual…"
    python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip --quiet

# ── 3. Instalar PyTorch (CUDA si está disponible, CPU si no) ────────────────
if [[ -n "${CUDA_VERSION:-}" ]]; then
    echo "[setup] Instalando PyTorch con CUDA${CUDA_VERSION}…"
    pip install --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}" \
        torch torchvision --quiet
elif command -v nvcc &>/dev/null; then
    CUDA_RAW=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    CUDA_SHORT="${CUDA_RAW//.}"     # "12.1" → "121"
    echo "[setup] nvcc detectó CUDA ${CUDA_RAW} → instalando PyTorch cu${CUDA_SHORT}…"
    pip install --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}" \
        torch torchvision --quiet
else
    echo "[setup] No se detectó CUDA → instalando PyTorch CPU…"
    pip install torch torchvision --quiet
fi

# ── 4. Instalar dependencias del proyecto ───────────────────────────────────
echo "[setup] Instalando dependencias…"
pip install -r requirements.txt --quiet
pip install -e . --quiet

# ── 5. Crear directorios necesarios ─────────────────────────────────────────
mkdir -p weights manifests data/mammobench/images runs

# ── 6. Verificar instalación ─────────────────────────────────────────────────
echo ""
echo "[setup] Verificando importaciones…"
python -c "
import torch, flwr, fedmammobench
print(f'  torch     {torch.__version__}')
print(f'  flwr      {flwr.__version__}')
print(f'  CUDA?     {torch.cuda.is_available()}')
print(f'  fedmammobench OK')
"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  Instalación completada en: $INSTALL_DIR"
echo ""
echo "  PASOS SIGUIENTES:"
echo "  1. Copiar pesos RadImageNet al directorio 'weights/':"
echo "       scp usuario@servidor:ruta/RadImageNet-resnet50.pth weights/"
echo "     O descargar de: https://github.com/BMEII-AI/RadImageNet"
echo ""
echo "  2. Copiar manifest de este nodo a 'manifests/'."
echo "     (generado por el servidor con scripts/partition_mammobench.py)"
echo ""
echo "  3. Copiar las imágenes de este nodo a 'data/mammobench/images/'."
echo ""
echo "  4. Exportar la ruta de pesos:"
echo "       export FEDMAMMOBENCH_RADIMAGENET_DIR=\$PWD/weights"
echo ""
echo "  5. Iniciar el cliente (cuando el servidor esté listo):"
echo "       bash scripts/start_client.sh <NODE_ID> <SERVER_IP>"
echo "══════════════════════════════════════════════════════════════════════"
