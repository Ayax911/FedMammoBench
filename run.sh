set -euo pipefail
cd "$(dirname "$0")"

configs=(
  configs/exp24_bydatabase_cdd-cesm_standard_mlp_bce.yaml
  configs/exp25_bydatabase_cmmd_standard_mlp_bce.yaml
  configs/exp26_bydatabase_inbreast_standard_mlp_bce.yaml
  configs/exp27_bydatabase_kau-bcmd_standard_mlp_bce.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done