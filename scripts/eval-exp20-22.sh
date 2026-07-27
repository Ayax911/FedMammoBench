#!/bin/bash
set -uo pipefail

REPO="${REPO:-.}"
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"

mkdir -p "$REPO/runs/_logs/eval"
EVAL_LOG="$REPO/runs/_logs/eval/eval_$(date +%Y%m%d_%H%M%S).log"

{
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo "  EVALUATION: exp20, exp21, exp22 (RadImageNet Validation)"
  echo "  Start time: $(date)"
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo ""
  
  # Experiments to evaluate
  experiments=(
    "exp20:exp20_centralized_radimagenet:runs/exp20_centralized_radimagenet/weights/final.pt"
    "exp21:exp21_fedavg_radimagenet:runs/exp21_fedavg_radimagenet/weights/global_model.pt"
    "exp22:exp22_fedprox_radimagenet:runs/exp22_fedprox_radimagenet/weights/global_model.pt"
  )
  
  for exp_spec in "${experiments[@]}"; do
    IFS=':' read -r exp_short exp_full checkpoint <<< "$exp_spec"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $exp_full"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    eval_dir="$REPO/runs/$exp_full/eval/mammo_bench"
    mkdir -p "$eval_dir"
    
    # Run evaluation and capture output
    eval_log="$eval_dir/run.log"
    eval_metrics="$eval_dir/metrics.json"
    tmp_output="/tmp/eval_output_${exp_short}.txt"
    
    if docker run --rm --gpus all --network host \
      -v "$REPO/configs:/app/configs:ro" \
      -v "$REPO/manifests:/app/manifests:ro" \
      -v "$WEIGHTS_DIR:/app/weights:ro" \
      -v "$MAMMO_DATA:/app/data:ro" \
      -v "$REPO/runs:/app/runs" \
      "$IMAGE_TAG" \
      fedmammobench-evaluate \
        --config "configs/$exp_short/eval/mammo_bench.yaml" \
        --checkpoint "$checkpoint" \
        2>&1 | tee "$eval_log" > "$tmp_output"; then
      
      # Extract JSON from output using Python script in /tmp
      python3 << EXTRACT_EOF
import json
import sys

try:
    with open("$tmp_output", "r") as f:
        content = f.read()
    
    # Find the last complete JSON object
    last_brace = content.rfind('}')
    if last_brace == -1:
        sys.exit(1)
    
    # Find the opening brace for this object
    brace_count = 0
    for i in range(last_brace, -1, -1):
        if content[i] == '}':
            brace_count += 1
        elif content[i] == '{':
            brace_count -= 1
            if brace_count == 0:
                json_str = content[i:last_brace+1]
                # Verify it's valid JSON
                data = json.loads(json_str)
                # Write to output file
                with open("$eval_metrics", "w") as f:
                    json.dump(data, f, indent=2)
                sys.exit(0)
    
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"Error: {e}\n")
    sys.exit(1)
EXTRACT_EOF
      
      if [ -f "$eval_metrics" ]; then
        echo "✓ Evaluation complete"
        echo "  Metrics: $eval_metrics"
        python3 << SHOW_METRICS
import json
m = json.load(open("$eval_metrics"))
print(f'  AUC-ROC: {m.get("roc_auc", 0):.4f}')
print(f'  F1:      {m.get("f1", 0):.4f}')
print(f'  Acc:     {m.get("accuracy", 0):.4f}')
SHOW_METRICS
      else
        echo "⚠ Could not extract JSON from output"
        tail -20 "$tmp_output"
      fi
    else
      echo "✗ Evaluation failed"
    fi
    
    rm -f "$tmp_output"
    echo ""
  done
  
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo "  RESULTS SUMMARY"
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo ""
  
  for exp_full in exp20_centralized_radimagenet exp21_fedavg_radimagenet exp22_fedprox_radimagenet; do
    if [ -f "$REPO/runs/$exp_full/eval/mammo_bench/metrics.json" ]; then
      echo "📊 $exp_full"
      python3 << SUMMARY
import json
m = json.load(open("$REPO/runs/$exp_full/eval/mammo_bench/metrics.json"))
print(f'   AUC-ROC: {m.get("roc_auc", 0):.4f}, F1: {m.get("f1", 0):.4f}, Accuracy: {m.get("accuracy", 0):.4f}')
SUMMARY
      echo ""
    fi
  done
  
  echo "End time: $(date)"
  echo "════════════════════════════════════════════════════════════════════════════════"
  
} 2>&1 | tee "$EVAL_LOG"

echo ""
echo "✓ Evaluation log: $EVAL_LOG"
