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
# Cada log va a train.log dentro del run_dir de su propio config
# (runs/centralizado/<exp>/train.log); si alguno falla, se reporta al final.
pids=()
logs=()
for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  run_dir=$(.venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['train']['run_dir'])" "configs/$cfg")
  mkdir -p "$run_dir"
  log="$run_dir/train.log"
  echo "=== lanzando $exp_id (log: $log) ==="
  .venv/bin/python -m src.cli --config "configs/$cfg" > "$log" 2>&1 &
  pids+=($!)
  logs+=("$log")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FALLO: ${configs[$i]} (ver ${logs[$i]})"
    fail=1
  fi
done
exit $fail
