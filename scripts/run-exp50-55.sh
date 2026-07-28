#!/bin/bash
# scripts/run-exp50-55.sh — Cabezas MLP sobre RadImageNet congelado (linear probing)
#
# Factorial 3 profundidades x 2 regularizaciones, centralizado, backbone ResNet50
# CONGELADO (freeze_backbone: true → solo entrena la cabeza, 4.95% de los params),
# salida de 2 neuronas + softmax (CrossEntropyLoss), SIN aumento de imágenes,
# 100 épocas, RadImageNet directo (mismo manifest/LR que exp31).
#
#                  dropout 0.0                    dropout 0.2
#   6 ocultas      exp50_mlp6_nodrop              exp53_mlp6_d02     [512,256,128,64,32,16]
#   5 ocultas      exp51_mlp5_nodrop              exp54_mlp5_d02     [512,256,128,64,32]
#   4 ocultas      exp52_mlp4_nodrop              exp55_mlp4_d02     [512,256,128,64]
#
# Por filas varía solo la profundidad; por columnas solo el dropout. Ningún
# experimento depende del resultado de otro → los 6 se lanzan de una sola vez y
# no requieren supervisión (~1.4 h/run estimadas, ~8.4 h en total).
#
# Uso:
#   scripts/run-exp50-55.sh                     # sin argumentos: solo ayuda, no corre nada
#   scripts/run-exp50-55.sh --phase all         # LOS 6, lote completo desatendido (~8.4h)
#   scripts/run-exp50-55.sh --phase nodrop      # exp50 51 52 (~4.2h)
#   scripts/run-exp50-55.sh --phase drop        # exp53 54 55 (~4.2h)
#   scripts/run-exp50-55.sh exp53 exp55         # experimentos sueltos, en el orden dado
#   scripts/run-exp50-55.sh --phase all --dry-run    # imprime los `docker run` sin ejecutar
#
# Ejecutar en background (sobrevive a cierre de terminal):
#   nohup scripts/run-exp50-55.sh --phase all > runs/_logs/queue/exp50_55_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Environment ───────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
cd "$REPO"

MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"
export MAMMO_DATA

WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"
export WEIGHTS_DIR

IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"
export IMAGE_TAG

# ── Fases ─────────────────────────────────────────────────────────────────
declare -A PHASE_EXPS=(
  [nodrop]="exp50 exp51 exp52"
  [drop]="exp53 exp54 exp55"
  [all]="exp50 exp51 exp52 exp53 exp54 exp55"
)
declare -A PHASE_LABEL=(
  [nodrop]="Brazo SIN dropout (3 profundidades)"
  [drop]="Brazo CON dropout 0.2 (3 profundidades)"
  [all]="Factorial completo 3 profundidades x 2 regularizaciones"
)

usage() {
  cat <<'EOF'
Uso: scripts/run-exp50-55.sh [--phase all|nodrop|drop] [exp50 exp53 ...] [--dry-run]

Sin argumentos no corre nada — solo muestra esta ayuda.

  --phase all       exp50 51 52 53 54 55   Factorial completo, desatendido   (~8.4h)
  --phase nodrop    exp50 51 52            Brazo sin dropout                 (~4.2h)
  --phase drop      exp53 54 55            Brazo con dropout 0.2             (~4.2h)
  exp53 exp55 ...   Lista explícita de experimentos, en el orden dado
  --dry-run         Imprime los `docker run` que lanzaría, sin ejecutarlos

Variables de entorno respetadas: MAMMO_DATA, WEIGHTS_DIR, IMAGE_TAG.
EOF
}

DRY_RUN=false
PHASE=""
EXTRA_EXPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    exp*)
      EXTRA_EXPS+=("$1")
      shift
      ;;
    *)
      echo -e "${RED}Argumento no reconocido: $1${NC}" >&2
      usage
      exit 1
      ;;
  esac
done

# ── Resolver la lista de experimentos a correr ───────────────────────────
EXPS=()
if [[ -n "$PHASE" ]]; then
  if [[ -z "${PHASE_EXPS[$PHASE]:-}" ]]; then
    echo -e "${RED}Fase desconocida: '$PHASE'. Usa all | nodrop | drop.${NC}" >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  EXPS+=(${PHASE_EXPS[$PHASE]})
fi
EXPS+=("${EXTRA_EXPS[@]}")

if [[ ${#EXPS[@]} -eq 0 ]]; then
  usage
  exit 0
fi

# ── Validar que los configs existen ANTES de lanzar nada ─────────────────
MISSING=()
for exp in "${EXPS[@]}"; do
  [[ -f "configs/$exp/centralized.yaml" ]] || MISSING+=("$exp")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo -e "${RED}✗ Faltan configs para: ${MISSING[*]}${NC}" >&2
  echo "  Se esperaba configs/<exp>/centralized.yaml para cada uno." >&2
  exit 1
fi

# ── Log setup ─────────────────────────────────────────────────────────────
mkdir -p runs/_logs/queue
MASTER_LOG="runs/_logs/queue/exp50_55_$(date +%Y%m%d_%H%M%S).log"

log(){ echo -e "$1" | tee -a "$MASTER_LOG"; }

log "════════════════════════════════════════════════════════════════════════════════"
log "  EXPERIMENTOS 50–55 — Cabezas MLP sobre RadImageNet CONGELADO (linear probing)"
[[ -n "$PHASE" ]] && log "  ${PHASE_LABEL[$PHASE]:-}"
log "  Cola: ${EXPS[*]}"
[[ "$DRY_RUN" == true ]] && log "  ${YELLOW}[DRY-RUN — no se ejecuta nada]${NC}"
log "  Start: $(date)   MASTER_LOG=$MASTER_LOG"
log "════════════════════════════════════════════════════════════════════════════════"

# ── Runner ────────────────────────────────────────────────────────────────
run_one() {
  local exp="$1"
  local cfg="configs/$exp/centralized.yaml"
  local full_name
  full_name="$(grep -m1 '^name:' "$cfg" | sed 's/^name:[[:space:]]*//')"

  local cmd=(docker run --rm --gpus all
    -v "$MAMMO_DATA:/app/data"
    -v "$WEIGHTS_DIR:/app/weights:ro"
    -v "$REPO/configs:/app/configs:ro"
    -v "$REPO/manifests:/app/manifests:ro"
    -v "$REPO/runs:/app/runs"
    -e PYTHONUNBUFFERED=1
    "$IMAGE_TAG"
    python scripts/run_centralized.py --config "$cfg")

  if [[ "$DRY_RUN" == true ]]; then
    log "${BLUE}[dry-run] $exp ($full_name):${NC}"
    log "  ${cmd[*]}"
    return 0
  fi

  log ""
  log "${BLUE}┌─ Starting $full_name ($exp) ────────────────────────────────────────────${NC}"
  log "Time: $(date)"
  ( "${cmd[@]}" 2>&1 | tee -a "$MASTER_LOG" )
  local exit_code=${PIPESTATUS[0]:-$?}
  if [[ $exit_code -eq 0 ]]; then
    log "${GREEN}✓ $exp ($full_name) completed successfully${NC}"
  else
    log "${RED}✗ $exp ($full_name) failed with exit code $exit_code${NC}"
  fi
  log "${BLUE}└─ End $exp ──────────────────────────────────────────────────────────────${NC}"
  log "Time: $(date)"
  return $exit_code
}

SUCCESS=(); FAILED=()
for exp in "${EXPS[@]}"; do
  if run_one "$exp"; then
    SUCCESS+=("$exp")
  else
    FAILED+=("$exp")
    if [[ "$DRY_RUN" != true ]]; then
      log "${YELLOW}⚠ Deteniendo la cola: $exp falló. Los siguientes NO se lanzan.${NC}"
      break
    fi
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════════════════════"
log "  RESUMEN — $(date)"
[[ ${#SUCCESS[@]} -gt 0 ]] && log "${GREEN}✓ Completos: ${SUCCESS[*]}${NC}"
[[ ${#FAILED[@]}  -gt 0 ]] && log "${RED}✗ Fallidos: ${FAILED[*]}${NC}"
log "  Log: $MASTER_LOG"
log "════════════════════════════════════════════════════════════════════════════════"

[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
