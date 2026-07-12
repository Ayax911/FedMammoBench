#!/bin/bash
# Post-hoc evaluation for experiments 15-19 on mamo-bench-split.csv

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO="${REPO:-.}"
LOG_DIR="${REPO}/runs"

echo -e "${YELLOW}=== Post-hoc Test Evaluation (exp15-19) ===${NC}"
echo "Test split: mamo-bench-split.csv (all datasets)"
echo ""

# Define experiments
experiments=(
  "exp15:runs/exp15_pretrain_ddsm/exp15_pretrain_ddsm/final.pt"
  "exp16:runs/exp16_centralized_resnet50/exp16_centralized_resnet50/final.pt"
  "exp17:runs/exp17_fedavg_resnet50/exp17_fedavg_resnet50/global_model.pt"
  "exp18:runs/exp18_fedavg_resnet50/exp18_fedavg_resnet50/global_model.pt"
  "exp19:runs/exp19_fedprox_resnet50/exp19_fedprox_resnet50/global_model.pt"
)

# Summary tracking
declare -a results
i=0

for exp_spec in "${experiments[@]}"; do
  exp="${exp_spec%%:*}"
  checkpoint="${exp_spec##*:}"

  echo -e "${YELLOW}[${i}] Evaluating $exp...${NC}"

  # Check checkpoint exists
  if [ ! -f "$REPO/$checkpoint" ]; then
    echo -e "${RED}✗ Checkpoint not found: $checkpoint${NC}"
    results+=("  [✗] $exp: missing checkpoint")
    i=$((i+1))
    continue
  fi

  # Run evaluation
  if fedmammobench-evaluate \
    --config "$REPO/configs/$exp/eval/mammo_bench.yaml" \
    --checkpoint "$REPO/$checkpoint" > "$LOG_DIR/${exp}_test.log" 2>&1; then

    echo -e "${GREEN}✓ $exp test complete${NC}"
    results+=("  [✓] $exp: OK")
  else
    echo -e "${RED}✗ $exp test failed${NC}"
    results+=("  [✗] $exp: FAILED")
  fi

  echo ""
  i=$((i+1))
done

# Summary
echo -e "${YELLOW}=== Test Summary ===${NC}"
for result in "${results[@]}"; do
  echo -e "$result"
done
echo ""
echo "Logs: $LOG_DIR/<exp>_test.log"
