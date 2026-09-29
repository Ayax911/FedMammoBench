set -euo pipefail
cd "$(dirname "$0")"

configs=(
  # Las réplicas multi-semilla de exp37 que vivían acá (exp64/65/66) nunca se
  # llegaron a correr (sin runs/exp6{4,5,6}_*/) y se borraron para no dejar
  # configs huérfanos sin resultado -- ver git log de este archivo para
  # recuperarlas si todavía hacen falta. exp64_custom_cnn.yaml de abajo NO es
  # una de ellas -- es un experimento nuevo y distinto que reutiliza el número
  # 64 porque exp61/62/63 (ResNet50/18 scratch/ImageNet) ya tienen resultados
  # reales publicados del lado INC (aunque nunca se corrieron acá en FMB), así
  # que no se reutilizan.
  configs/exp64_custom_cnn.yaml
)

for cfg in "${configs[@]}"; do
  exp_id=$(basename "$cfg" .yaml)
  echo "=== running $exp_id ==="
  .venv/bin/python -m src.cli --config "$cfg" 2>&1 | tee "runs/${exp_id}.log"
done