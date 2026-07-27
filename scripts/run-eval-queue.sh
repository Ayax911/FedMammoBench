#!/bin/bash
# Post-hoc evaluation for experiments 15-19 on mamo-bench-split.csv
# Executes sequentially and logs to a single master log file in runs/_logs/eval/.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO="${REPO:-.}"
mkdir -p "$REPO/runs/_logs/eval"
EVAL_LOG="$REPO/runs/_logs/eval/eval_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo -e "$1" | tee -a "$EVAL_LOG"
}

log "${YELLOW}=== Post-hoc Evaluation (exp15-19) ===${NC}"
log "Test split: mamo-bench-split.csv (all datasets)"
log ""

# Define experiments: exp_id:checkpoint_path:output_dir_parent
experiments=(
  "exp15:runs/exp15_pretrain_ddsm/exp15_pretrain_ddsm/final.pt:runs/exp15_pretrain_ddsm"
  "exp16:runs/exp16_centralized_resnet50/weights/final.pt:runs/exp16_centralized_resnet50"
  "exp17:runs/exp17_fedavg_resnet50/weights/global_model.pt:runs/exp17_fedavg_resnet50"
  "exp18:runs/exp18_fedavg_resnet50/weights/global_model.pt:runs/exp18_fedavg_resnet50"
  "exp19:runs/exp19_fedprox_resnet50/weights/global_model.pt:runs/exp19_fedprox_resnet50"
)

# Summary tracking
declare -a results
i=0

for exp_spec in "${experiments[@]}"; do
  exp="${exp_spec%%:*}"
  rest="${exp_spec#*:}"
  checkpoint="${rest%%:*}"
  output_dir_parent="${rest##*:}"

  eval_out_dir="$output_dir_parent/eval/mammo_bench"

  log ""
  log "${YELLOW}[${i}] Evaluating $exp...${NC}"

  # Check checkpoint exists
  if [ ! -f "$REPO/$checkpoint" ]; then
    log "${RED}✗ Checkpoint not found: $checkpoint${NC}"
    results+=("  [✗] $exp: missing checkpoint")
    i=$((i+1))
    continue
  fi

  # Run evaluation
  if fedmammobench-evaluate \
    --config "$REPO/configs/$exp/eval/mammo_bench.yaml" \
    --checkpoint "$REPO/$checkpoint" \
    --output-dir "$REPO/$eval_out_dir" \
    --predictions-out "$REPO/$eval_out_dir/predictions.csv" \
    >> "$EVAL_LOG" 2>&1; then

    log "${GREEN}✓ $exp eval complete${NC}"
    results+=("  [✓] $exp: OK")
  else
    log "${RED}✗ $exp eval failed${NC}"
    results+=("  [✗] $exp: FAILED")
  fi

  i=$((i+1))
done

# Summary
log ""
log "${YELLOW}=== Evaluation Summary ===${NC}"
for result in "${results[@]}"; do
  log "$result"
done
log ""
log "Master log: $EVAL_LOG"
