#!/bin/bash
# Post-hoc evaluation: exp20, exp21, exp22 (RadImageNet validation)
# Con predicciones y evaluaciones por nodos (replicando exp14)

set -uo pipefail

REPO="${REPO:-.}"
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"

mkdir -p "$REPO/runs/_logs/eval"
EVAL_LOG="$REPO/runs/_logs/eval/eval_$(date +%Y%m%d_%H%M%S).log"

{
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo "  POST-HOC EVALUATION: exp20, exp21, exp22 (Con predicciones)"
  echo "  Start time: $(date)"
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo ""

  # Experiments to evaluate
  experiments=(
    "exp20:exp20_centralized_radimagenet:runs/exp20_centralized_radimagenet/exp20_centralized_radimagenet/final.pt"
    "exp21:exp21_fedavg_radimagenet:runs/exp21_fedavg_radimagenet/exp21_fedavg_radimagenet/global_model.pt"
    "exp22:exp22_fedprox_radimagenet:runs/exp22_fedprox_radimagenet/exp22_fedprox_radimagenet/global_model.pt"
  )

  run_evaluation() {
    local exp_short=$1
    local exp_full=$2
    local checkpoint=$3
    local config_name=$4
    local eval_dir=$5

    mkdir -p "$eval_dir"
    eval_log="$eval_dir/run.log"
    eval_metrics="$eval_dir/metrics.json"
    eval_predictions="$eval_dir/predictions.csv"
    tmp_output="/tmp/eval_${exp_short}_${config_name}.txt"

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

      # Extract JSON from output
      python3 << EXTRACT_EOF
import json
import sys

try:
    with open("$tmp_output", "r") as f:
        content = f.read()

    last_brace = content.rfind('}')
    if last_brace == -1:
        sys.exit(1)

    brace_count = 0
    for i in range(last_brace, -1, -1):
        if content[i] == '}':
            brace_count += 1
        elif content[i] == '{':
            brace_count -= 1
            if brace_count == 0:
                json_str = content[i:last_brace+1]
                data = json.loads(json_str)
                with open("$eval_metrics", "w") as f:
                    json.dump(data, f, indent=2)
                sys.exit(0)

    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"Error: {e}\n")
    sys.exit(1)
EXTRACT_EOF

      if [ -f "$eval_metrics" ]; then
        python3 << SHOW_METRICS
import json
import os
m = json.load(open("$eval_metrics"))
pred_size = os.path.getsize("$eval_predictions") / 1024 if os.path.exists("$eval_predictions") else 0
print(f'    ✓ AUC-ROC: {m.get("roc_auc", 0):.4f} | F1: {m.get("f1", 0):.4f} | Predictions: {pred_size:.1f}KB')
SHOW_METRICS
      else
        echo "    ⚠ JSON extraction failed"
      fi
    else
      echo "    ✗ Evaluation failed"
    fi

    rm -f "$tmp_output"
  }

  # Evaluate each experiment
  for exp_spec in "${experiments[@]}"; do
    IFS=':' read -r exp_short exp_full checkpoint <<< "$exp_spec"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $exp_full"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Verify checkpoint exists
    if [ ! -f "$REPO/$checkpoint" ]; then
      echo "✗ Checkpoint not found: $checkpoint"
      continue
    fi

    echo "  Checkpoint: $checkpoint"
    echo ""

    # Global evaluation (mammo_bench)
    echo "  🌍 Global (mammo_bench):"
    eval_dir="$REPO/runs/$exp_full/eval/mammo_bench"
    run_evaluation "$exp_short" "$exp_full" "$checkpoint" "mammo_bench" "$eval_dir"
    echo ""

    # Per-node evaluations (only for federated: exp21, exp22)
    if [ "$exp_short" != "exp20" ]; then
      echo "  🔗 Per-Node evaluations:"
      for node in 1 2 3 4 5; do
        config_name="node${node}_partition"

        # Check if config exists
        if [ ! -f "$REPO/configs/$exp_short/eval/${config_name}.yaml" ]; then
          echo "    ⊘ node$node config not found, skipping"
          continue
        fi

        echo "    📍 Node $node:"
        eval_dir="$REPO/runs/$exp_full/eval/${config_name}"
        run_evaluation "$exp_short" "$exp_full" "$checkpoint" "$config_name" "$eval_dir"
      done
      echo ""
    fi
  done

  echo "════════════════════════════════════════════════════════════════════════════════"
  echo "  FINAL RESULTS SUMMARY"
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo ""

  for exp_full in exp20_centralized_radimagenet exp21_fedavg_radimagenet exp22_fedprox_radimagenet; do
    echo "📊 $exp_full"

    if [ -f "$REPO/runs/$exp_full/eval/mammo_bench/metrics.json" ]; then
      python3 << SUMMARY
import json
m = json.load(open("$REPO/runs/$exp_full/eval/mammo_bench/metrics.json"))
print(f'   Global: AUC-ROC={m.get("roc_auc", 0):.4f} F1={m.get("f1", 0):.4f} Acc={m.get("accuracy", 0):.4f}')
SUMMARY
    fi

    # Per-node summary for federated
    if [[ "$exp_full" == *"fedavg"* ]] || [[ "$exp_full" == *"fedprox"* ]]; then
      for node in 1 2 3 4 5; do
        metrics_file="$REPO/runs/$exp_full/eval/node${node}_partition/metrics.json"
        if [ -f "$metrics_file" ]; then
          python3 << NODE_SUMMARY
import json
m = json.load(open("$metrics_file"))
print(f'   Node $node: AUC-ROC={m.get("roc_auc", 0):.4f}')
NODE_SUMMARY
        fi
      done
    fi
    echo ""
  done

  echo "End time: $(date)"
  echo "════════════════════════════════════════════════════════════════════════════════"

} 2>&1 | tee "$EVAL_LOG"

echo ""
echo "✓ Master log: $EVAL_LOG"
echo "✓ Per-experiment metrics: runs/exp*/eval/*/metrics.json"
echo "✓ Per-experiment predictions: runs/exp*/eval/*/predictions.csv"
