#!/bin/bash
# Skip exp20, run exp21-22 only

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"
export MAMMO_DATA
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
export WEIGHTS_DIR
IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"
export IMAGE_TAG

mkdir -p runs/_logs/queue
QUEUE_LOG="runs/_logs/queue/exp21_22_$(date +%Y%m%d_%H%M%S).log"

echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"
echo "  EXPERIMENTS 21–22: RadImageNet Federated Validation (exp20 completed)" | tee -a "$QUEUE_LOG"
echo "  Start time: $(date)" | tee -a "$QUEUE_LOG"
echo "  Log: $QUEUE_LOG" | tee -a "$QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"
echo "" | tee -a "$QUEUE_LOG"

run_federated() {
  local exp_full_name=$1
  local exp_short_name=$2
  
  echo "" | tee -a "$QUEUE_LOG"
  echo -e "${BLUE}┌─ Starting $exp_full_name (federated) ─────────────────────────────────────────────${NC}" | tee -a "$QUEUE_LOG"
  echo "Time: $(date)" | tee -a "$QUEUE_LOG"
  echo "" | tee -a "$QUEUE_LOG"

  (
    cd "$REPO"
    scripts/docker-deploy-federated.sh "$exp_short_name" 2>&1 | tee -a "$QUEUE_LOG"
  )

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

FAILED=()
SUCCESS=()

# exp21: Federated FedAvg
echo -e "${YELLOW}[1/2] Federated FedAvg (exp21_fedavg_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_federated "exp21_fedavg_radimagenet" "exp21" && SUCCESS+=("exp21") || FAILED+=("exp21")

# exp22: Federated FedProx
echo -e "${YELLOW}[2/2] Federated FedProx (exp22_fedprox_radimagenet)${NC}" | tee -a "$QUEUE_LOG"
run_federated "exp22_fedprox_radimagenet" "exp22" && SUCCESS+=("exp22") || FAILED+=("exp22")

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
echo "════════════════════════════════════════════════════════════════════════════════" | tee -a "$QUEUE_LOG"

if [ ${#FAILED[@]} -gt 0 ]; then
  exit 1
else
  exit 0
fi
