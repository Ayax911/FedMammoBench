set -euo pipefail
cd "$(dirname "$0")"

# Cuenta de W&B de la carpeta FL-JULIAN (../wandb.env), NO la de ~/.netrc: WANDB_API_KEY en el
# entorno tiene prioridad sobre ~/.netrc. Sin esto, las corridas suben a otra cuenta que la de
# exp57 y la de INC. set -a exporta las variables para que las vea el proceso de Python.
set -a
source ../wandb.env
set +a

configs=(
  configs/exp58_pretrain_ablation_imagenet_2048_1024_1024_512_512.yaml
  configs/exp59_pretrain_ablation_imagenet_512.yaml
  configs/exp60_pretrain_ablation_imagenet_256.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done
