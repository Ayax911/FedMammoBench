set -euo pipefail
cd "$(dirname "$0")"

configs=(
  configs/expcamilo.yaml

)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done