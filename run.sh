set -euo pipefail
cd "$(dirname "$0")"

configs=(
  exp65_resnet18_imagenet_frozen.yaml
  exp66_resnet18_imagenet_unfreeze_layer4.yaml
  exp67_resnet18_imagenet_unfreeze_layer3.yaml
  exp68_resnet18_imagenet_unfreeze_layer2.yaml
  exp69_resnet18_imagenet_unfreeze_layer1.yaml
)

# Los 5 experimentos corren en paralelo sobre la misma GPU (un proceso cada uno).
# Cada log va a su propio archivo; si alguno falla, se reporta al final.
pids=()
for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== lanzando $exp_id ==="
  .venv/bin/python -m src.cli --config "configs/$cfg" > "runs/${exp_id}.log" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FALLO: ${configs[$i]} (ver runs/$(basename "${configs[$i]}" .yaml).log)"
    fail=1
  fi
done
exit $fail
