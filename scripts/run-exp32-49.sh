#!/bin/bash
# scripts/run-exp32-49.sh — Grid federado estrategia × régimen ronda/época (exp32–49)
#
# 3 estrategias (fedavg/fedprox/fedadam) × 6 regímenes (bloques A–F).
# Todos: RadImageNet directo, mismo manifest/LR/regularización; solo varían rounds y local_epochs.
#
#   Bloque A: exp32/33/34   5 rondas × 20 épocas   (~3.0 h)
#   Bloque B: exp35/36/37  10 rondas × 20 épocas   (~5.9 h)
#   Bloque C: exp38/39/40  30 rondas ×  5 épocas   (~4.7 h)
#   Bloque D: exp41/42/43   1 ronda  × 100 épocas  (~2.9 h)
#   Bloque E: exp44/45/46   3 rondas × 100 épocas  (~8.7 h)
#   Bloque F: exp47/48/49   5 rondas × 100 épocas  (~14.5 h)
#
# Ejecución ESCALONADA recomendada (revisar entre etapas):
#   Etapa 1: bloques A + D  (~6 h)  ← verificar aquí que FedAdam NO colapse
#   Etapa 2: bloque  C      (~4.7 h)
#   Etapa 3: bloques B + E  (~14.6 h)
#   Etapa 4: bloque  F      (~14.5 h)
#
# Uso:
#   scripts/run-exp32-49.sh --stage 1                 # bloques A + D
#   scripts/run-exp32-49.sh --block A                 # un bloque (3 experimentos)
#   scripts/run-exp32-49.sh --block A --block D       # varios bloques
#   scripts/run-exp32-49.sh --all                     # los 6 bloques en orden escalonado
#   scripts/run-exp32-49.sh --block A --no-clean      # no limpiar contenedores previos
#
# Background (sobrevive al cierre de terminal):
#   nohup scripts/run-exp32-49.sh --stage 1 > runs/_logs/queue/exp32_49_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export REPO
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"; export MAMMO_DATA
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"; export WEIGHTS_DIR
IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"; export IMAGE_TAG

# ── Grid definition ───────────────────────────────────────────────────────
declare -A BLOCK_EXPS=(
  [A]="exp32 exp33 exp34" [B]="exp35 exp36 exp37" [C]="exp38 exp39 exp40"
  [D]="exp41 exp42 exp43" [E]="exp44 exp45 exp46" [F]="exp47 exp48 exp49"
)
declare -A BLOCK_DESC=(
  [A]="5r×20e" [B]="10r×20e" [C]="30r×5e" [D]="1r×100e" [E]="3r×100e" [F]="5r×100e"
)
declare -A STAGE_BLOCKS=( [1]="A D" [2]="C" [3]="B E" [4]="F" )
STAGED_ORDER="A D C B E F"   # orden barato→caro para --all

usage() {
  echo "Uso: scripts/run-exp32-49.sh [--stage N | --block X ... | --all] [--no-clean]"
  echo ""
  echo "  Etapas:  1=A+D   2=C   3=B+E   4=F"
  echo "  Bloques: A=5r×20e B=10r×20e C=30r×5e D=1r×100e E=3r×100e F=5r×100e"
  echo ""
  echo "  Ej: scripts/run-exp32-49.sh --stage 1     # empieza aquí (~6 h)"
  exit 1
}

# ── Parse args ────────────────────────────────────────────────────────────
NO_CLEAN=false
BLOCKS_TO_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-clean) NO_CLEAN=true; shift ;;
    --all)      BLOCKS_TO_RUN="$STAGED_ORDER"; shift ;;
    --stage)    shift; BLOCKS_TO_RUN="$BLOCKS_TO_RUN ${STAGE_BLOCKS[${1:-}]:-}"; shift ;;
    --block)    shift; BLOCKS_TO_RUN="$BLOCKS_TO_RUN ${1:-}"; shift ;;
    -h|--help)  usage ;;
    *)          echo -e "${RED}Arg desconocido: $1${NC}"; usage ;;
  esac
done
# trim + validate
BLOCKS_TO_RUN=$(echo "$BLOCKS_TO_RUN" | xargs)
[[ -z "$BLOCKS_TO_RUN" ]] && usage
for b in $BLOCKS_TO_RUN; do
  [[ -z "${BLOCK_EXPS[$b]:-}" ]] && { echo -e "${RED}Bloque inválido: $b${NC}"; usage; }
done

# ── Log setup ─────────────────────────────────────────────────────────────
mkdir -p "$REPO/runs/_logs/queue"
QUEUE_LOG="$REPO/runs/_logs/queue/exp32_49_$(date +%Y%m%d_%H%M%S).log"

{
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  GRID FEDERADO exp32–49 — Estrategia × Régimen (RadImageNet directo)"
echo "  Bloques a correr: $BLOCKS_TO_RUN"
echo "  Start: $(date)   Log: $QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════"
} | tee -a "$QUEUE_LOG"

# ── Run one experiment (federated) ────────────────────────────────────────
FAILED=(); SUCCESS=()
run_exp() {
  local exp=$1
  echo "" | tee -a "$QUEUE_LOG"
  echo -e "${BLUE}┌─ $exp ─────────────────────────────────────────────${NC}" | tee -a "$QUEUE_LOG"
  echo "Time: $(date)" | tee -a "$QUEUE_LOG"
  local clean_flag=(); [ "$NO_CLEAN" = true ] && clean_flag=(--no-clean)
  ( cd "$REPO" && scripts/docker-deploy-federated.sh "$exp" "${clean_flag[@]}" 2>&1 | tee -a "$QUEUE_LOG" )
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo -e "${GREEN}✓ $exp OK${NC}" | tee -a "$QUEUE_LOG"; SUCCESS+=("$exp")
  else
    echo -e "${RED}✗ $exp FAILED (rc=$rc)${NC}" | tee -a "$QUEUE_LOG"; FAILED+=("$exp")
  fi
  echo -e "${BLUE}└─ fin $exp ($(date)) ─${NC}" | tee -a "$QUEUE_LOG"
}

# ── Main loop ─────────────────────────────────────────────────────────────
for block in $BLOCKS_TO_RUN; do
  echo "" | tee -a "$QUEUE_LOG"
  echo -e "${YELLOW}═══ Bloque $block (${BLOCK_DESC[$block]}): ${BLOCK_EXPS[$block]} ═══${NC}" | tee -a "$QUEUE_LOG"
  for exp in ${BLOCK_EXPS[$block]}; do
    run_exp "$exp"
  done
done

# ── Summary ───────────────────────────────────────────────────────────────
{
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  RESUMEN — $(date)"
[ ${#SUCCESS[@]} -gt 0 ] && echo -e "${GREEN}✓ OK:     ${SUCCESS[*]}${NC}"
[ ${#FAILED[@]}  -gt 0 ] && echo -e "${RED}✗ FALLÓ:  ${FAILED[*]}${NC}"
echo "  Log: $QUEUE_LOG"
echo "════════════════════════════════════════════════════════════════════════════════"
} | tee -a "$QUEUE_LOG"

[ ${#FAILED[@]} -gt 0 ] && exit 1 || exit 0
