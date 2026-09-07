set -euo pipefail
cd "$(dirname "$0")"

configs=(
  configs/exp07_fullfreeze_bce_2048_2_posweight.yaml
  configs/exp08_layer4_bce_2048_2_posweight.yaml
  configs/exp09_fullfreeze_bce_2048_2.yaml
  configs/exp10_layer4_bce_2048_2.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done