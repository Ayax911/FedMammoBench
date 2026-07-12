#!/bin/bash
# scripts/eval-experiments.sh — Evalúa checkpoints (por nodos o en mammobench)
#
# Uso:
#   scripts/eval-experiments.sh <exp> <checkpoint> [--mode nodes|mammobench] [--dataset ddsm|no-ddsm|all] [--predictions]
#
# Ejemplos:
#   scripts/eval-experiments.sh exp17 runs/exp17_fedavg/exp17_fedavg/global_model.pt
#   scripts/eval-experiments.sh exp17 runs/exp17_fedavg/exp17_fedavg/global_model.pt --mode nodes
#   scripts/eval-experiments.sh exp16 runs/exp16_centralized/exp16_centralized/final.pt --dataset no-ddsm --predictions

set -uo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Resolver variables de entorno ────────────────────────────────────────
REPO="${REPO:-.}"
cd "$REPO"

if [ -z "${MAMMO_DATA:-}" ]; then
  MAMMO_DATA="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench"
  export MAMMO_DATA
fi

WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
export WEIGHTS_DIR

IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"
export IMAGE_TAG

mkdir -p runs/_logs/eval
EVAL_LOG="runs/_logs/eval/eval_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo -e "$1" | tee -a "$EVAL_LOG"
}

# ── Parsear argumentos ───────────────────────────────────────────────────
EXP="${1:-}"
CHECKPOINT="${2:-}"
MODE="mammobench"
DATASET="no-ddsm"
PREDICTIONS=false
OUTPUT_BASE="runs"

if [ -z "$EXP" ]; then
  log "${RED}✗ Uso: $0 <exp> <checkpoint> [--mode nodes|mammobench] [--dataset ddsm|no-ddsm|all] [--predictions]${NC}"
  exit 1
fi

shift 2 || true

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --predictions)
      PREDICTIONS=true
      shift
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    *)
      log "${RED}✗ Argumento desconocido: $1${NC}"
      exit 1
      ;;
  esac
done

# ── Validar experimento ──────────────────────────────────────────────────
if [ ! -d "configs/$EXP" ]; then
  log "${RED}✗ configs/$EXP no existe${NC}"
  exit 1
fi

if [ ! -d "configs/$EXP/eval" ]; then
  log "${RED}✗ configs/$EXP/eval/ no existe — no hay evaluaciones pre-definidas${NC}"
  exit 1
fi

# ── Validar checkpoint ───────────────────────────────────────────────────
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  log "${RED}✗ Checkpoint no encontrado: $CHECKPOINT${NC}"
  exit 1
fi

# ── Extraer nombre completo del experimento (del YAML) ──────────────────
# Buscar en server.yaml, client.yaml o cualquier YAML en la carpeta
EXP_NAME=""
for yaml_file in "configs/$EXP"/server.yaml "configs/$EXP"/client.yaml "configs/$EXP"/*.yaml; do
  if [ -f "$yaml_file" ]; then
    EXP_NAME=$(grep "^name:" "$yaml_file" | head -1 | sed 's/^name:\s*//' | tr -d ' ')
    if [ -n "$EXP_NAME" ]; then
      break
    fi
  fi
done

if [ -z "$EXP_NAME" ]; then
  log "${RED}✗ No se encontró 'name:' en configs/$EXP/*.yaml${NC}"
  exit 1
fi

# ── Detectar configs de eval ─────────────────────────────────────────────
log "${YELLOW}=== Evaluación: $EXP (nombre: $EXP_NAME) ===${NC}"
log "Modo: $MODE"
log "Dataset: $DATASET"
log "Checkpoint: $CHECKPOINT"
log "Log: $EVAL_LOG"
log ""

declare -a CONFIGS_TO_RUN

if [ "$MODE" = "nodes" ]; then
  # Buscar todas las configs de nodo
  for cfg in configs/$EXP/eval/node*; do
    if [ -f "$cfg" ]; then
      cfg_name=$(basename "$cfg")
      CONFIGS_TO_RUN+=("$cfg_name")
    fi
  done
  if [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
    log "${YELLOW}⚠ No se encontraron configs node<N>_*.yaml — saltando evaluación por nodos${NC}"
    exit 0
  fi
  log "Configs a ejecutar (por nodos): ${CONFIGS_TO_RUN[*]}"
else
  # Modo mammobench: seleccionar según --dataset
  if [ "$DATASET" = "all" ]; then
    for cfg in configs/$EXP/eval/mammo_bench*.yaml; do
      if [ -f "$cfg" ]; then
        cfg_name=$(basename "$cfg" .yaml)
        CONFIGS_TO_RUN+=("$cfg_name.yaml")
      fi
    done
  else
    if [ "$DATASET" = "ddsm" ]; then
      if [ -f "configs/$EXP/eval/mammo_bench_ddsm.yaml" ]; then
        CONFIGS_TO_RUN+=("mammo_bench_ddsm.yaml")
      else
        CONFIGS_TO_RUN+=("mammo_bench.yaml")
      fi
    else
      # no-ddsm (default)
      if [ -f "configs/$EXP/eval/mammo_bench_no_ddsm.yaml" ]; then
        CONFIGS_TO_RUN+=("mammo_bench_no_ddsm.yaml")
      else
        CONFIGS_TO_RUN+=("mammo_bench.yaml")
      fi
    fi
  fi
  if [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
    log "${RED}✗ No se encontraron configs mammo_bench*.yaml${NC}"
    exit 1
  fi
  log "Configs a ejecutar (mammobench): ${CONFIGS_TO_RUN[*]}"
fi

log ""

# ── Ejecutar evaluaciones ────────────────────────────────────────────────
declare -a RESULTS

for cfg_name in "${CONFIGS_TO_RUN[@]}"; do
  cfg_file="configs/$EXP/eval/$cfg_name"
  eval_out_dir="$OUTPUT_BASE/$EXP_NAME/$EXP_NAME/eval/${cfg_name%.yaml}"

  log "${CYAN}Evaluando $cfg_name...${NC}"
  mkdir -p "$eval_out_dir"

  # Construir comando (usar scripts/run_evaluation.py directamente)
  CMD=(
    docker run --rm --gpus all --network host
    -v "$REPO/configs:/app/configs:ro"
    -v "$REPO/manifests:/app/manifests:ro"
    -v "$WEIGHTS_DIR:/app/weights:ro"
    -v "$MAMMO_DATA:/app/data:ro"
    -v "$REPO/runs:/app/runs"
    "$IMAGE_TAG"
    python scripts/run_evaluation.py
    --config "configs/$EXP/eval/$cfg_name"
    --checkpoint "$CHECKPOINT"
  )

  if [ "$PREDICTIONS" = true ]; then
    CMD+=(--predictions-out "$eval_out_dir/predictions.csv")
  fi

  if "${CMD[@]}" >> "$EVAL_LOG" 2>&1; then
    log "${GREEN}✓ $cfg_name OK${NC}"
    RESULTS+=("$cfg_name: OK")
  else
    log "${RED}✗ $cfg_name FAILED${NC}"
    RESULTS+=("$cfg_name: FAILED")
  fi
  log ""
done

# ── Resumen ──────────────────────────────────────────────────────────────
log "${YELLOW}=== Resumen ===${NC}"
for result in "${RESULTS[@]}"; do
  if [[ "$result" == *": OK" ]]; then
    log "  ${GREEN}✓${NC} $result"
  else
    log "  ${RED}✗${NC} $result"
  fi
done

log ""
log "Outputs: $OUTPUT_BASE/$EXP_NAME/$EXP_NAME/eval/"
log "Master log: $EVAL_LOG"
