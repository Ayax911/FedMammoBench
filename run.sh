set -euo pipefail
cd "$(dirname "$0")"

configs=(
  configs/exp28_antioverfit_base.yaml
  configs/exp29_antioverfit_wd1e2.yaml
  configs/exp30_antioverfit_labelsmooth.yaml
  configs/exp31_antioverfit_no_inputdrop.yaml
  configs/exp32_antioverfit_radimagenet.yaml
  configs/exp33_pretrain_ablation_imagenet_mismatched_norm.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done