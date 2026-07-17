#!/bin/bash
# scripts/run-exp24-26.sh — Ejecuta los 3 experimentos de re-validación en secuencia
#
# Réplica de exp20-22 tras corregir la colisión de IDs entre datasets (manifest
# mamo-bench-split-no-ddsm-rsna.csv regenerado: 0 IDs duplicados, 0 overlap train/val/test).
#
# exp24: Centralizado + RadImageNet (baseline)
# exp25: Federado FedAvg + RadImageNet (20 rounds × 5 epochs)
# exp26: Federado FedProx + RadImageNet (20 rounds × 5 epochs)
#
# Diseñados para aislar el impacto de la estrategia (centralizado vs FedAvg vs FedProx)
# cuando el pretrain es de alta calidad (RadImageNet, no exp15 DDSM-only).
#
# Uso:
#   scripts/run-exp24-26.sh
#   scripts/run-exp24-26.sh --no-clean          # no limpiar contenedores previos
#
# Ejecutar en background (sobrevive a cierre de terminal):
#   nohup scripts/run-exp24-26.sh > runs/_logs/queue/exp24_26_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Environment ───────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO

MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"
export MAMMO_DATA

WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
export WEIGHTS_DIR

IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"
export IMAGE_TAG

NO_CLEAN=false
if [[ "${1:-}" == "--no-clean" ]]; then
  NO_CLEAN=true
  shift || true
fi

# ── Log setup ─────────────────────────────────────────────────────────────
mkdir -p runs/_logs/queue
QUEUE_LOG="runs/_logs/queue/exp24_26_$(date +%Y%m%d_%H%M%S).log"

echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"
echo "  EXPERIMENTS 24–26: RadImageNet Validation Queue" | tee -a "$QUEUE_LOG"
echo "  Start time: $(date)" | tee -a "$QUEUE_LOG"
echo "  Log: $QUEUE_LOG" | tee -a "$QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"
echo "" | tee -a "$QUEUE_LOG"

# ── Helper functions ──────────────────────────────────────────────────────
run_exp() {
  local exp_full_name=$1
  local exp_short_name=$2  # e.g., "exp24", "exp25", "exp26"
  local exp_type=$3        # "centralized" or "federated"

  echo "" | tee -a "$QUEUE_LOG"
  echo -e "${BLUE}┌─ Starting $exp_full_name ($exp_type) ─────────────────────────────────────────────${NC}" | tee -a "$QUEUE_LOG"
  echo "Time: $(date)" | tee -a "$QUEUE_LOG"
  echo "" | tee -a "$QUEUE_LOG"

  if [ "$exp_type" == "centralized" ]; then
    # Centralized: run directly via python (use short name for config path)
    (
      cd "$REPO"
      docker run --rm \
        --gpus all \
        -v "$MAMMO_DATA:/app/data" \
        -v "$WEIGHTS_DIR:/app/weights:ro" \
        -v "$REPO/configs:/app/configs:ro" \
        -v "$REPO/manifests:/app/manifests:ro" \
        -v "$REPO/runs:/app/runs" \
        -e PYTHONUNBUFFERED=1 \
        ayax911/federal-learning:latest \
        python scripts/run_centralized.py \
        --config "configs/$exp_short_name/centralized.yaml" \
        2>&1 | tee -a "$QUEUE_LOG"
    )
  else
    # Federated: use docker-deploy-federated.sh (expects short name like exp21)
    # YAML already defines rounds: 20, so just launch and wait for completion
    (
      cd "$REPO"
      scripts/docker-deploy-federated.sh "$exp_short_name" 2>&1 | tee -a "$QUEUE_LOG"
    )
  fi

  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ $exp_full_name completed successfully${NC}" | tee -a "$QUEUE_LOG"
  else
    echo -e "${RED}✗ $exp_full_name failed with exit code $exit_code${NC}" | tee -a "$QUEUE_LOG"
  fi
  echo -e "${BLUE}└─ End $exp_full_name ─────────────────────────────────────────────────────────────────────${NC}" | tee -a "$QUEUE_LOG"
  echo "Time: $(date)" | tee -a "$QUEUE_LOG"
  echo "" | tee -a "$QUEUE_LOG"

  return $exit_code
}

# ── Run experiments ───────────────────────────────────────────────────────
FAILED=()
SUCCESS=()

# exp20: Centralized baseline
echo -e "${YELLOW}[1/3] Centralized Baseline (exp24_centralized_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_exp "exp24_centralized_radimagenet" "exp24" "centralized" && SUCCESS+=("exp24") || FAILED+=("exp24")

# exp21: Federated FedAvg
echo -e "${YELLOW}[2/3] Federated FedAvg (exp25_fedavg_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_exp "exp25_fedavg_radimagenet" "exp25" "federated" && SUCCESS+=("exp25") || FAILED+=("exp25")

# exp22: Federated FedProx
echo -e "${YELLOW}[3/3] Federated FedProx (exp26_fedprox_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_exp "exp26_fedprox_radimagenet" "exp26" "federated" && SUCCESS+=("exp26") || FAILED+=("exp26")

# ── Summary ───────────────────────────────────────────────────────────────
echo "" | tee -a "$QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"
echo "  QUEUE SUMMARY" | tee -a "$QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"

if [ ${#SUCCESS[@]} -gt 0 ]; then
  echo -e "${GREEN}✓ Completed: ${SUCCESS[*]}${NC}" | tee -a "$QUEUE_LOG"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo -e "${RED}✗ Failed: ${FAILED[*]}${NC}" | tee -a "$QUEUE_LOG"
fi

echo "" | tee -a "$QUEUE_LOG"
echo "End time: $(date)" | tee -a "$QUEUE_LOG"
echo "Log saved to: $QUEUE_LOG" | tee -a "$QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"

# ── Exit with appropriate code ────────────────────────────────────────────
if [ ${#FAILED[@]} -gt 0 ]; then
  exit 1
else
  exit 0
fi
