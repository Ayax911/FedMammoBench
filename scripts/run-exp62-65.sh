#!/bin/bash
# scripts/run-exp62-65.sh — Ablación de profundidad de descongelamiento (cabeza lineal simple)
#
# 4 experimentos centralizados, backbone ResNet50 RadImageNet, cabeza
# Dropout(0.5)+Linear, variando unfreeze_layers (ninguno/layer4/layer3+4/
# layer2+3+4). Ver configs/exp62/centralized.yaml para la tabla completa de
# params entrenables por variante.
#
# Cola con N slots en paralelo (work-stealing, mismo patrón que
# run-exp32-49-parallel.sh): cuando un slot libera, toma el siguiente de la
# cola. Orden por defecto: más liviano primero (exp62 solo cabeza, exp65
# casi fine-tuning completo al final).
#
# Uso:
#   scripts/run-exp62-65.sh                  # los 4, SLOTS=2 por defecto
#   scripts/run-exp62-65.sh --slots 3
#   scripts/run-exp62-65.sh exp63 exp65       # lista propia
#
# Background:
#   nohup scripts/run-exp62-65.sh --slots 2 > runs/_logs/queue/exp62_65_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export REPO
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"; export MAMMO_DATA
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"; export WEIGHTS_DIR
IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"; export IMAGE_TAG
cd "$REPO"

SLOTS=2
EXTRA_EXPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slots) SLOTS="${2:-2}"; shift 2 ;;
    exp*) EXTRA_EXPS+=("$1"); shift ;;
    *) echo -e "${RED}Argumento no reconocido: $1${NC}" >&2; exit 1 ;;
  esac
done

DEFAULT_QUEUE=(exp62 exp63 exp64 exp65)
if [[ ${#EXTRA_EXPS[@]} -gt 0 ]]; then QUEUE=("${EXTRA_EXPS[@]}"); else QUEUE=("${DEFAULT_QUEUE[@]}"); fi

MISSING=()
for exp in "${QUEUE[@]}"; do
  [[ -f "configs/$exp/centralized.yaml" ]] || MISSING+=("$exp")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo -e "${RED}✗ Faltan configs para: ${MISSING[*]}${NC}" >&2
  exit 1
fi

mkdir -p runs/_logs/queue
MASTER_LOG="runs/_logs/queue/exp62_65_$(date +%Y%m%d_%H%M%S).log"
DONEDIR="$(mktemp -d)"
trap 'rm -rf "$DONEDIR"' EXIT

log() { echo -e "$1" | tee -a "$MASTER_LOG"; }

log "════════════════════════════════════════════════════════════════════════════════"
log "  EXPERIMENTOS 62–65 — Ablación profundidad de descongelamiento (cabeza lineal)"
log "  Cola (${#QUEUE[@]}): ${QUEUE[*]}   SLOTS=$SLOTS"
log "  Start: $(date)   MASTER_LOG=$MASTER_LOG"
log "════════════════════════════════════════════════════════════════════════════════"

declare -a SLOT_EXP
for ((i=0; i<SLOTS; i++)); do SLOT_EXP[$i]=""; done
SUCCESS=(); FAILED=()
idx=0

launch_one() {
  local slot="$1" exp="$2"
  local cfg="configs/$exp/centralized.yaml"
  local elog="runs/_logs/queue/${exp}_$(date +%Y%m%d_%H%M%S).log"
  log "${BLUE}[$(date +%H:%M:%S)] ▶ $exp → slot $slot  (log: $elog)${NC}"
  (
    docker run --rm --gpus all \
      -v "$MAMMO_DATA:/app/data" \
      -v "$WEIGHTS_DIR:/app/weights:ro" \
      -v "$REPO/configs:/app/configs:ro" \
      -v "$REPO/manifests:/app/manifests:ro" \
      -v "$REPO/runs:/app/runs" \
      -e PYTHONUNBUFFERED=1 \
      -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
      "$IMAGE_TAG" \
      python scripts/run_centralized.py --config "$cfg" > "$elog" 2>&1
    echo $? > "$DONEDIR/$exp"
  ) &
  SLOT_EXP[$slot]=$exp
}

for ((s=0; s<SLOTS; s++)); do
  [[ $idx -lt ${#QUEUE[@]} ]] || break
  launch_one "$s" "${QUEUE[$idx]}"; idx=$((idx+1))
done

while :; do
  active=0
  for ((s=0; s<SLOTS; s++)); do [[ -n "${SLOT_EXP[$s]}" ]] && active=1; done
  [[ $active -eq 0 ]] && break
  sleep 20
  for ((s=0; s<SLOTS; s++)); do
    exp="${SLOT_EXP[$s]}"
    [[ -z "$exp" ]] && continue
    if [[ -f "$DONEDIR/$exp" ]]; then
      rc=$(cat "$DONEDIR/$exp"); rm -f "$DONEDIR/$exp"
      if [[ "$rc" -eq 0 ]]; then
        log "${GREEN}[$(date +%H:%M:%S)] ✓ $exp OK (slot $s)${NC}"; SUCCESS+=("$exp")
      else
        log "${RED}[$(date +%H:%M:%S)] ✗ $exp FALLÓ rc=$rc (slot $s)${NC}"; FAILED+=("$exp")
      fi
      SLOT_EXP[$s]=""
      if [[ $idx -lt ${#QUEUE[@]} ]]; then
        launch_one "$s" "${QUEUE[$idx]}"; idx=$((idx+1))
      fi
    fi
  done
done

log ""
log "════════════════════════════════════════════════════════════════════════════════"
log "  RESUMEN — $(date)"
[[ ${#SUCCESS[@]} -gt 0 ]] && log "${GREEN}✓ Completos: ${SUCCESS[*]}${NC}"
[[ ${#FAILED[@]}  -gt 0 ]] && log "${RED}✗ Fallidos: ${FAILED[*]}${NC}"
log "  Log: $MASTER_LOG"
log "════════════════════════════════════════════════════════════════════════════════"

[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
