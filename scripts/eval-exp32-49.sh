#!/bin/bash
# scripts/eval-exp32-49.sh — Post-hoc test eval del grid federado exp32–49
#
# Por experimento: global (mammo_bench) + por-nodo (node1..5), con predicciones.
# Checkpoint = runs/<full>/weights/global_model.pt (última ronda). El nombre completo
# se deriva del campo `name:` de configs/<exp>/server.yaml.
#
# Uso:
#   scripts/eval-exp32-49.sh --stage 1            # bloques A + D
#   scripts/eval-exp32-49.sh --block A            # un bloque
#   scripts/eval-exp32-49.sh --all                # los 6 bloques

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"

declare -A BLOCK_EXPS=(
  [A]="exp32 exp33 exp34" [B]="exp35 exp36 exp37" [C]="exp38 exp39 exp40"
  [D]="exp41 exp42 exp43" [E]="exp44 exp45 exp46" [F]="exp47 exp48 exp49"
)
declare -A STAGE_BLOCKS=( [1]="A D" [2]="C" [3]="B E" [4]="F" )
STAGED_ORDER="A D C B E F"

usage() { echo "Uso: scripts/eval-exp32-49.sh [--stage N | --block X ... | --all]"; exit 1; }

BLOCKS_TO_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)   BLOCKS_TO_RUN="$STAGED_ORDER"; shift ;;
    --stage) shift; BLOCKS_TO_RUN="$BLOCKS_TO_RUN ${STAGE_BLOCKS[${1:-}]:-}"; shift ;;
    --block) shift; BLOCKS_TO_RUN="$BLOCKS_TO_RUN ${1:-}"; shift ;;
    -h|--help) usage ;;
    *) echo "Arg desconocido: $1"; usage ;;
  esac
done
BLOCKS_TO_RUN=$(echo "$BLOCKS_TO_RUN" | xargs)
[[ -z "$BLOCKS_TO_RUN" ]] && usage

# Build experiment list (short names) from selected blocks
EXPS=()
for b in $BLOCKS_TO_RUN; do
  [[ -z "${BLOCK_EXPS[$b]:-}" ]] && { echo "Bloque inválido: $b"; usage; }
  for e in ${BLOCK_EXPS[$b]}; do EXPS+=("$e"); done
done

mkdir -p "$REPO/runs/_logs/eval"
EVAL_LOG="$REPO/runs/_logs/eval/eval_$(date +%Y%m%d_%H%M%S).log"

{
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo "  POST-HOC EVAL: grid federado exp32–49 — bloques $BLOCKS_TO_RUN"
  echo "  Start time: $(date)"
  echo "════════════════════════════════════════════════════════════════════════════════"

  run_evaluation() {
    local exp_short=$1 exp_full=$2 checkpoint=$3 config_name=$4 eval_dir=$5
    mkdir -p "$eval_dir"
    local eval_log="$eval_dir/run.log" eval_metrics="$eval_dir/metrics.json"
    local eval_predictions="$eval_dir/predictions.csv" tmp_output="/tmp/eval_${exp_short}_${config_name}.txt"
    echo "  Config: $config_name"

    if docker run --rm --gpus all --network host \
      -v "$REPO/configs:/app/configs:ro" \
      -v "$REPO/manifests:/app/manifests:ro" \
      -v "$WEIGHTS_DIR:/app/weights:ro" \
      -v "$MAMMO_DATA:/app/data:ro" \
      -v "$REPO/runs:/app/runs" \
      "$IMAGE_TAG" \
      fedmammobench-evaluate \
        --config "configs/$exp_short/eval/$config_name.yaml" \
        --checkpoint "/app/$checkpoint" \
        --predictions-out "/app/runs/$exp_full/eval/$config_name/predictions.csv" \
        2>&1 | tee "$eval_log" > "$tmp_output"; then

      python3 - "$tmp_output" "$eval_metrics" << 'EXTRACT_EOF'
import json, sys
tmp, out = sys.argv[1], sys.argv[2]
content = open(tmp).read()
last = content.rfind('}')
if last == -1: sys.exit(1)
depth = 0
for i in range(last, -1, -1):
    if content[i] == '}': depth += 1
    elif content[i] == '{':
        depth -= 1
        if depth == 0:
            json.dump(json.loads(content[i:last+1]), open(out, "w"), indent=2)
            sys.exit(0)
sys.exit(1)
EXTRACT_EOF

      if [ -f "$eval_metrics" ]; then
        python3 - "$eval_metrics" "$eval_predictions" << 'SHOW_EOF'
import json, os, sys
m = json.load(open(sys.argv[1]))
p = os.path.getsize(sys.argv[2])/1024 if os.path.exists(sys.argv[2]) else 0
print(f'    ✓ AUC-ROC: {m.get("roc_auc",0):.4f} | F1: {m.get("f1",0):.4f} | Predictions: {p:.1f}KB')
SHOW_EOF
      else
        echo "    ⚠ JSON extraction failed"
      fi
    else
      echo "    ✗ Evaluation failed"
    fi
    rm -f "$tmp_output"
  }

  for exp_short in "${EXPS[@]}"; do
    exp_full=$(grep -m1 '^name:' "$REPO/configs/$exp_short/server.yaml" | awk '{print $2}')
    checkpoint="runs/$exp_full/weights/global_model.pt"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $exp_full"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$REPO/$checkpoint" ]; then
      echo "  ✗ Checkpoint no encontrado: $checkpoint (¿corrió el entrenamiento?)"; echo ""; continue
    fi
    echo "  Checkpoint: $checkpoint"; echo ""

    echo "  🌍 Global (mammo_bench):"
    run_evaluation "$exp_short" "$exp_full" "$checkpoint" "mammo_bench" "$REPO/runs/$exp_full/eval/mammo_bench"
    echo ""

    echo "  🔗 Per-Node:"
    for node in 1 2 3 4 5; do
      cfg="node${node}_partition"
      [ -f "$REPO/configs/$exp_short/eval/${cfg}.yaml" ] || { echo "    ⊘ node$node sin config"; continue; }
      echo "    📍 Node $node:"
      run_evaluation "$exp_short" "$exp_full" "$checkpoint" "$cfg" "$REPO/runs/$exp_full/eval/${cfg}"
    done
    echo ""
  done

  echo "════════════════════════════════════════════════════════════════════════════════"
  echo "  RESUMEN — última ronda (global_model.pt) en test"
  echo "════════════════════════════════════════════════════════════════════════════════"
  for exp_short in "${EXPS[@]}"; do
    exp_full=$(grep -m1 '^name:' "$REPO/configs/$exp_short/server.yaml" | awk '{print $2}')
    mj="$REPO/runs/$exp_full/eval/mammo_bench/metrics.json"
    if [ -f "$mj" ]; then
      python3 - "$exp_full" "$mj" << 'SUM_EOF'
import json, sys
m = json.load(open(sys.argv[2]))
print(f'📊 {sys.argv[1]}')
print(f'   Global: AUC-ROC={m.get("roc_auc",0):.4f} F1={m.get("f1",0):.4f} Acc={m.get("accuracy",0):.4f}')
SUM_EOF
    else
      echo "📊 $exp_full — sin métricas (eval no completado)"
    fi
  done
  echo ""
  echo "End time: $(date)"
} 2>&1 | tee "$EVAL_LOG"

echo ""
echo "✓ Master log: $EVAL_LOG"
echo "ℹ Recuerda cruzar con la MEJOR ronda-val de runs/<exp>/*/server_federated_metrics.csv"
