#!/usr/bin/env bash
# Barrido de dropout sobre exp11 (ganador del barrido de profundidad).
set -euo pipefail
cd "$(dirname "$0")"

configs=(
  configs/exp14_layer4_bce_h1024_dr03.yaml
  configs/exp15_layer4_bce_h1024_dr04.yaml
  configs/exp16_layer4_bce_h1024_dr05.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done
