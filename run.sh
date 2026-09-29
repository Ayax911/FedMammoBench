set -euo pipefail
cd "$(dirname "$0")"

configs=(
  exp65_resnet18_imagenet_frozen.yaml
  exp66_resnet18_imagenet_unfreeze_layer4.yaml
  exp67_resnet18_imagenet_unfreeze_layer3.yaml
  exp68_resnet18_imagenet_unfreeze_layer2.yaml
  exp69_resnet18_imagenet_unfreeze_layer1.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "configs/$cfg" 2>&1 | tee "runs/${exp_id}.log"
done