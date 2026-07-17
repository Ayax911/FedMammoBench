#!/usr/bin/env bash
# verify_setup.sh — verifica que todos los archivos necesarios existan
# antes de ejecutar el experimento con 6 nodos.
#
# Uso:
#   bash scripts/verify_setup.sh <NODE_ID> <SERVER_IP>
#
# Ejemplos:
#   bash scripts/verify_setup.sh 0 192.168.14.184      # Verifica Node0
#   bash scripts/verify_setup.sh server                  # Verifica servidor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ── Parámetros ────────────────────────────────────────────────────────────
ENTITY="${1:-server}"
SERVER_IP="${2:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

check_file() {
  local file="$1"
  local name="$2"
  if [[ -f "$file" ]]; then
    echo -e "${GREEN}✓${NC} $name"
    return 0
  else
    echo -e "${RED}✗${NC} $name: NO ENCONTRADO ($file)"
    return 1
  fi
}

check_dir() {
  local dir="$1"
  local name="$2"
  if [[ -d "$dir" ]]; then
    echo -e "${GREEN}✓${NC} $name"
    return 0
  else
    echo -e "${RED}✗${NC} $name: NO ENCONTRADO ($dir)"
    return 1
  fi
}

check_env() {
  local var="$1"
  if [[ -n "${!var:-}" ]]; then
    echo -e "${GREEN}✓${NC} Variable \$$var = ${!var}"
    return 0
  else
    echo -e "${YELLOW}⚠${NC} Variable \$$var no está seteada"
    return 1
  fi
}

# ── Verificaciones Comunes ─────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "  Verificación de Setup — 6 NODOS + SERVIDOR"
echo "════════════════════════════════════════════════════════════════════"
echo ""

ERRORS=0

# Configuración
echo "[CONFIG]"
check_file configs/exp01_fedavg_resnet50_6nodes_server.yaml "Config servidor (6 nodos)" || ((ERRORS++))
check_file configs/exp01_fedavg_resnet50_client.yaml "Config cliente (todos)" || ((ERRORS++))
echo ""

# Pesos RadImageNet
echo "[PESOS]"
if ls weights/*.pth &>/dev/null; then
  echo -e "${GREEN}✓${NC} Pesos RadImageNet encontrados"
  ls -lh weights/*.pth | awk '{print "    " $9 " (" $5 ")"}'
else
  echo -e "${RED}✗${NC} Pesos RadImageNet: NO ENCONTRADOS"
  echo "   Descargar de: https://github.com/BMEII-AI/RadImageNet"
  ((ERRORS++))
fi
echo ""

# Scripts
echo "[SCRIPTS]"
check_file scripts/run_server.py "Script servidor" || ((ERRORS++))
check_file scripts/run_client.py "Script cliente" || ((ERRORS++))
check_file scripts/partition_mammobench.py "Script partición" || ((ERRORS++))
echo ""

# ── Verificaciones Específicas ─────────────────────────────────────────────

if [[ "$ENTITY" == "server" ]]; then
  echo "[SERVIDOR CENTRAL]"
  check_dir data/mammobench/images "Directorio de imágenes" || ((ERRORS++))
  check_file manifests/node0_manifest.csv "Manifest Node0" || ((ERRORS++))
  check_file manifests/node1_manifest.csv "Manifest Node1" || ((ERRORS++))
  check_file manifests/node2_manifest.csv "Manifest Node2" || ((ERRORS++))
  check_file manifests/node3_manifest.csv "Manifest Node3" || ((ERRORS++))
  check_file manifests/node4_manifest.csv "Manifest Node4" || ((ERRORS++))
  check_file manifests/node5_manifest.csv "Manifest Node5" || ((ERRORS++))
  check_file manifests/server_train_manifest.csv "Manifest Servidor (pre-training)" || echo -e "${YELLOW}⚠${NC} Manifest servidor (opcional para pre-training)"
  echo ""
  check_env FEDMAMMOBENCH_RADIMAGENET_DIR || ((ERRORS++))
  echo ""
  echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
  echo "Servidor listo. Ejecutar con:"
  echo "  bash scripts/start_server.sh configs/exp01_fedavg_resnet50_6nodes_server.yaml"
  echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

elif [[ "$ENTITY" =~ ^[0-5]$ ]]; then
  NODE_ID="$ENTITY"
  echo "[NODO CLIENTE $NODE_ID]"

  if [[ -z "$SERVER_IP" ]]; then
    echo -e "${RED}✗${NC} SERVER_IP no proporcionado"
    echo "  Uso: bash scripts/verify_setup.sh $NODE_ID <SERVER_IP>"
    echo "  Ej:  bash scripts/verify_setup.sh $NODE_ID 192.168.14.184"
    ((ERRORS++))
  else
    echo -e "${GREEN}✓${NC} SERVER_IP = $SERVER_IP"
  fi
  echo ""

  check_dir data/mammobench/images "Directorio de imágenes" || ((ERRORS++))
  check_file "manifests/node${NODE_ID}_manifest.csv" "Manifest Node$NODE_ID" || ((ERRORS++))
  echo ""

  check_env FEDMAMMOBENCH_RADIMAGENET_DIR || ((ERRORS++))
  echo ""

  echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
  echo "Node$NODE_ID listo. Ejecutar con:"
  echo "  bash scripts/start_client.sh $NODE_ID $SERVER_IP"
  echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

else
  echo -e "${RED}ERROR: ENTITY debe ser 'server' o NODE_ID (0-5)${NC}"
  echo "Uso:"
  echo "  bash scripts/verify_setup.sh server"
  echo "  bash scripts/verify_setup.sh 0 192.168.14.184"
  exit 1
fi

echo ""

if [[ $ERRORS -eq 0 ]]; then
  echo -e "${GREEN}✓ TODAS LAS VERIFICACIONES PASARON${NC}"
  exit 0
else
  echo -e "${RED}✗ $ERRORS VERIFICACIONES FALLARON${NC}"
  echo ""
  echo "Checklist de setup:"
  echo "  1. Generar particiones: python scripts/partition_mammobench.py --csv data/mammobench/mammo-bench.csv --out manifests/ --nodes 6"
  echo "  2. Copiar pesos: wget https://github.com/BMEII-AI/RadImageNet/releases/.../RadImageNet-resnet50.pth -O weights/RadImageNet-resnet50.pth"
  echo "  3. Copiar imágenes a data/mammobench/images/"
  echo "  4. Copiar manifests a cada nodo remoto"
  echo "  5. Exportar FEDMAMMOBENCH_RADIMAGENET_DIR"
  exit 1
fi
