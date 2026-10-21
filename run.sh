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

# Experimento baseline que propone el propio repo de FLamby para fed_isic2019
# (pooled training + evaluación, ver flamby/datasets/fed_isic2019/README.md).
# Corre después de los 5 anteriores, no en paralelo con ellos, para no competir
# por la misma GPU. Usa el intérprete del entorno conda "flamby", no el .venv
# de este repo.
flamby_python=/home/lenovo-mirp/miniconda3/envs/flamby/bin/python
flamby_dataset_dir=/media/lenovo-mirp/TB2/DATA_TB2/04_PregradoJulian/FLamby/flamby/datasets/fed_isic2019
flamby_run_dir=runs/flamby/fed_isic2019
mkdir -p "$flamby_run_dir"
flamby_log="$flamby_run_dir/train.log"
echo "=== lanzando flamby_fed_isic2019_benchmark (log: $flamby_log) ==="
if ( cd "$flamby_dataset_dir" && "$flamby_python" benchmark.py --GPU 0 --workers 4 ) \
    > "$flamby_log" 2>&1; then
  :
else
  echo "FALLO: flamby_fed_isic2019_benchmark (ver $flamby_log)"
  fail=1
fi

# Grid de Fed-ISIC2019 ya construido en este repo (src/federated/), replicando las 7
# estrategias x hiperparámetros de FLamby (docs/FEDISIC2019_FLAMBY_REPRO.md §4.3). Una
# celda a la vez: cada celda ya lanza servidor + 6 nodos en paralelo por su cuenta
# (scripts/fed_isic2019/run_cell.py), así que correr varias celdas a la vez encima
# saturaría los 8GB de VRAM de esta GPU.
fedisic_cells=(
  fedisic2019_fedavg_r10
  fedisic2019_fedprox_r10
  fedisic2019_fedadagrad_r10
  fedisic2019_fedadam_r10
  fedisic2019_fedyogi_r10
  fedisic2019_cyclic_r60
  fedisic2019_scaffold_r10
)
for cell in "${fedisic_cells[@]}"; do
  cell_log="runs/federado/$cell/run_cell.log"
  mkdir -p "$(dirname "$cell_log")"
  echo "=== lanzando $cell (log: $cell_log) ==="
  if .venv/bin/python scripts/fed_isic2019/run_cell.py --exp "$cell" > "$cell_log" 2>&1; then
    :
  else
    echo "FALLO: $cell (ver $cell_log)"
    fail=1
  fi
done

exit $fail
