#!/bin/bash
# scripts/run-exp28-31.sh — Ejecuta el baseline centralizado + trío de estrategias federadas
#
# Mismo manifest corregido que exp24-26 (mamo-bench-split-no-ddsm-rsna.csv: 0 IDs
# duplicados, 0 overlap train/val/test) y mismo warm start (RadImageNet directo,
# igual que exp20/24/25/26 — no exp15/exp27 DDSM pretrain, esa es otra línea).
#
# exp31: Centralizado         + RadImageNet (100 epochs)          — baseline
# exp28: Federado FedAvg      + RadImageNet (20 rounds × 5 epochs)
# exp29: Federado FedYogi     + RadImageNet (20 rounds × 5 epochs)
# exp30: Federado FedProx     + RadImageNet (20 rounds × 5 epochs)
#
# Diseñados para aislar el impacto de centralizado vs. FedAvg vs. FedYogi vs. FedProx
# manteniendo warm start, manifest, LR y régimen de épocas idénticos (100 equivalentes).
#
# Uso:
#   scripts/run-exp28-31.sh
#   scripts/run-exp28-31.sh --no-clean          # no limpiar contenedores previos
#
# Ejecutar en background (sobrevive a cierre de terminal):
#   nohup scripts/run-exp28-31.sh > runs/_logs/queue/exp28_31_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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
QUEUE_LOG="runs/_logs/queue/exp28_31_$(date +%Y%m%d_%H%M%S).log"

echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"
echo "  EXPERIMENTS 28–31: Centralized + Strategy Comparison Queue" | tee -a "$QUEUE_LOG"
echo "  (Centralized / FedAvg / FedYogi / FedProx — all RadImageNet direct warm start)" | tee -a "$QUEUE_LOG"
echo "  Start time: $(date)" | tee -a "$QUEUE_LOG"
echo "  Log: $QUEUE_LOG" | tee -a "$QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"
echo "" | tee -a "$QUEUE_LOG"

# ── Helper function ───────────────────────────────────────────────────────
run_exp() {
  local exp_full_name=$1
  local exp_short_name=$2  # e.g., "exp31", "exp28", "exp29", "exp30"
  local exp_type=$3        # "centralized" or "federated"

  echo "" | tee -a "$QUEUE_LOG"
  echo -e "${BLUE}┌─ Starting $exp_full_name ($exp_type) ─────────────────────────────────────────────${NC}" | tee -a "$QUEUE_LOG"
  echo "Time: $(date)" | tee -a "$QUEUE_LOG"
  echo "" | tee -a "$QUEUE_LOG"

  if [ "$exp_type" == "centralized" ]; then
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
        "$IMAGE_TAG" \
        python scripts/run_centralized.py \
        --config "configs/$exp_short_name/centralized.yaml" \
        2>&1 | tee -a "$QUEUE_LOG"
    )
  else
    local clean_flag=()
    if [ "$NO_CLEAN" = true ]; then
      clean_flag=(--no-clean)
    fi
    (
      cd "$REPO"
      scripts/docker-deploy-federated.sh "$exp_short_name" "${clean_flag[@]}" 2>&1 | tee -a "$QUEUE_LOG"
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

echo -e "${YELLOW}[1/4] Centralized Baseline (exp31_centralized_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_exp "exp31_centralized_radimagenet" "exp31" "centralized" && SUCCESS+=("exp31") || FAILED+=("exp31")

echo -e "${YELLOW}[2/4] Federated FedAvg (exp28_fedavg_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_exp "exp28_fedavg_radimagenet" "exp28" "federated" && SUCCESS+=("exp28") || FAILED+=("exp28")

echo -e "${YELLOW}[3/4] Federated FedYogi (exp29_fedyogi_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_exp "exp29_fedyogi_radimagenet" "exp29" "federated" && SUCCESS+=("exp29") || FAILED+=("exp29")

echo -e "${YELLOW}[4/4] Federated FedProx (exp30_fedprox_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_exp "exp30_fedprox_radimagenet" "exp30" "federated" && SUCCESS+=("exp30") || FAILED+=("exp30")

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
