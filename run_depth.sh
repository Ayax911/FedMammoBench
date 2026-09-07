#!/usr/bin/env bash
# Barrido de profundidad de cabeza sobre el ganador exp08 (layer4 + pos_weight).
set -euo pipefail
cd "$(dirname "$0")"

configs=(
  configs/exp11_layer4_bce_h1024.yaml
  configs/exp12_layer4_bce_h1024_512.yaml
  configs/exp13_layer4_bce_h1024_512_256.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done
