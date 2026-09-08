#!/usr/bin/env bash
# Ablación de pretraining del backbone: exp17 (ImageNet, normalización correcta) vs
# exp18 (RadImageNet, su normalización nativa) vs exp19 (ImageNet, normalización de
# RadImageNet sin corregir -- control "mal alimentado"). Mismo head/optimizer/scheduler/
# loss/train en los tres -- ver cabeceras de cada config para el detalle de qué varía.
set -euo pipefail
cd "$(dirname "$0")"

configs=(
  configs/exp17_pretrain_ablation_imagenet.yaml
  configs/exp18_pretrain_ablation_radimagenet.yaml
  configs/exp19_pretrain_ablation_imagenet_mismatched_norm.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  # El log vive DENTRO de runs/<exp_id>/ -- nunca suelto en runs/ -- para que
  # quede junto al resto de artefactos de esa corrida (train.log, plots/, etc.)
  # en vez de ensuciar runs/ con un archivo por experimento. mkdir -p porque
  # `tee` abre el archivo de salida antes de que src.cli llegue a crear
  # run_dir (Trainer.fit() lo hace recién al entrar a fit()) -- sin esto,
  # `tee` falla con "No such file or directory" en la primera corrida.
  mkdir -p "runs/${exp_id}"
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}/run.log"
done
