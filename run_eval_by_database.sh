#!/usr/bin/env bash
# Backfill del desglose de test por base de datos (test/metrics_by_database.json,
# confusion_matrix_by_database.png, metrics_by_database.png) para exp05..exp16,
# entrenados antes de que DataConfig.by_database_manifests existiera. No reentrena
# nada -- usa src.evaluate sobre el best checkpoint real de cada corrida (el
# best_epoch<N>.pt de mayor N, que es el único que Trainer.fit() llegó a guardar
# como "mejora real" según EarlyStopping/min_delta -- ver src/train/trainer.py).
set -euo pipefail
cd "$(dirname "$0")"

declare -A checkpoints=(
  [exp05_fedmammobench_full_weighted]=best_epoch79.pt
  [exp06_fedmammobench_full_unweighted]=best_epoch162.pt
  [exp07_fullfreeze_bce_2048_2_posweight]=best_epoch65.pt
  [exp08_layer4_bce_2048_2_posweight]=best_epoch71.pt
  [exp09_fullfreeze_bce_2048_2]=best_epoch83.pt
  [exp10_layer4_bce_2048_2]=best_epoch79.pt
  [exp11_layer4_bce_h1024]=best_epoch69.pt
  [exp12_layer4_bce_h1024_512]=best_epoch63.pt
  [exp13_layer4_bce_h1024_512_256]=best_epoch22.pt
  [exp14_layer4_bce_h1024_dr03]=best_epoch20.pt
  [exp15_layer4_bce_h1024_dr04]=best_epoch20.pt
  [exp16_layer4_bce_h1024_dr05]=best_epoch20.pt
)

exps=(
  exp05_fedmammobench_full_weighted
  exp06_fedmammobench_full_unweighted
  exp07_fullfreeze_bce_2048_2_posweight
  exp08_layer4_bce_2048_2_posweight
  exp09_fullfreeze_bce_2048_2
  exp10_layer4_bce_2048_2
  exp11_layer4_bce_h1024
  exp12_layer4_bce_h1024_512
  exp13_layer4_bce_h1024_512_256
  exp14_layer4_bce_h1024_dr03
  exp15_layer4_bce_h1024_dr04
  exp16_layer4_bce_h1024_dr05
)

for exp_id in "${exps[@]}"; do
  cfg="configs/${exp_id}.yaml"
  ckpt="runs/${exp_id}/weights/${checkpoints[$exp_id]}"
  echo "=== evaluating $exp_id ($ckpt) ==="
  # El log vive DENTRO de runs/<exp_id>/ -- nunca suelto en runs/ -- para que
  # quede junto al resto de artefactos de esa corrida (train.log, test/, etc.)
  # en vez de ensuciar runs/ con un archivo por experimento.
  .venv/bin/python -m src.evaluate --config "$cfg" --checkpoint "$ckpt" \
    2>&1 | tee "runs/${exp_id}/eval_by_database.log"
done
