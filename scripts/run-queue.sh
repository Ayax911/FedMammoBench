#!/bin/bash
# scripts/run-queue.sh — Ejecuta varios experimentos Docker EN SECUENCIA.
#
# Por cada experimento de la lista corre sus fases de entrenamiento
# (centralizado/pretrain y/o federado, según qué YAML existan en
# configs/<exp>/) una tras otra, esperando (docker wait) a que cada
# contenedor termine antes de seguir con el siguiente. Si un experimento
# falla, lo registra y CONTINÚA con el siguiente (no detiene la cola).
#
# Si un experimento tiene pretrain.yaml Y server.yaml (ej. exp07, exp12), se
# corren como 2 fases del mismo experimento: primero el pretrain, y solo si
# ese termina OK, el federado (el warm-start depende del checkpoint del
# pretrain). Si el pretrain falla, la fase federada de ESE experimento se
# salta, pero la cola sigue con el siguiente experimento de la lista.
#
# NO incluye evaluación post-hoc (configs/<exp>/eval/*.yaml): requiere elegir
# a mano qué checkpoint usar. Correr esas aparte con /docker-run --only eval
# una vez que la cola termine.
#
# Uso:
#   scripts/run-queue.sh exp15 exp10 exp08
#   scripts/run-queue.sh --file queue.txt      # un experimento por línea, '#' = comentario
#
# Dejarla corriendo en background (sobrevive si cierras la terminal):
#   QUEUE_LOG="runs/_logs/queue/queue_$(date +%Y%m%d_%H%M%S).log"
#   nohup env QUEUE_LOG="$QUEUE_LOG" scripts/run-queue.sh exp15 exp10 exp08 > /dev/null 2>&1 &
#
# Variables de entorno (mismos defaults que scripts/docker-deploy-federated.sh):
#   REPO, MAMMO_DATA, WEIGHTS_DIR, IMAGE_TAG, QUEUE_LOG

set -uo pipefail  # sin -e: un fallo individual no debe matar la cola completa

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Resolver variables de entorno ────────────────────────────────────────
if [ -z "${REPO:-}" ]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  export REPO
fi

if [ -z "${MAMMO_DATA:-}" ]; then
  MAMMO_DATA="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench"
  export MAMMO_DATA
fi

if [ -z "${WEIGHTS_DIR:-}" ]; then
  WEIGHTS_DIR="$REPO/weights"
  export WEIGHTS_DIR
fi

IMAGE_TAG="${IMAGE_TAG:-ayax911/federal-learning:latest}"
export IMAGE_TAG

cd "$REPO"
mkdir -p runs/_logs/queue
QUEUE_LOG="${QUEUE_LOG:-runs/_logs/queue/queue_$(date +%Y%m%d_%H%M%S).log}"
export QUEUE_LOG

log() {
  echo -e "$1" | tee -a "$QUEUE_LOG"
}

# ── Parsear argumentos: lista directa o --file ───────────────────────────
EXPERIMENTS=()
if [ "${1:-}" = "--file" ]; then
  FILE="${2:-}"
  if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    log "${RED}✗ Archivo no encontrado: ${FILE:-<vacío>}${NC}"
    exit 1
  fi
  while IFS= read -r line; do
    line="$(echo "$line" | sed 's/#.*//' | xargs)"
    [ -n "$line" ] && EXPERIMENTS+=("$line")
  done < "$FILE"
else
  EXPERIMENTS=("$@")
fi

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
  log "Uso: $0 <exp1> <exp2> ... | --file <lista.txt>"
  exit 1
fi

log "${YELLOW}=== Cola de experimentos: ${EXPERIMENTS[*]} ===${NC}"
log "REPO: $REPO"
log "MAMMO_DATA: $MAMMO_DATA"
log "WEIGHTS_DIR: $WEIGHTS_DIR"
log "IMAGE_TAG: $IMAGE_TAG"
log "Log: $QUEUE_LOG"
log ""

RESULTS=()

run_centralized_phase() {
  local exp=$1 cfg=$2
  local name="${exp}_queue_centralized"
  log "${YELLOW}[$exp] Fase centralizado ($cfg)...${NC}"
  docker rm -f "$name" >/dev/null 2>&1 || true

  docker run -d --name "$name" --gpus all --network host \
    -v "$REPO/configs:/app/configs:ro" \
    -v "$REPO/manifests:/app/manifests:ro" \
    -v "$WEIGHTS_DIR:/app/weights:ro" \
    -v "$MAMMO_DATA:/app/data:ro" \
    -v "$REPO/runs:/app/runs" \
    "$IMAGE_TAG" \
    python scripts/run_centralized.py --config "configs/$exp/$cfg" \
    >> "$QUEUE_LOG" 2>&1

  local exit_code
  exit_code="$(docker wait "$name" 2>>"$QUEUE_LOG")"
  docker logs "$name" --tail 20 >> "$QUEUE_LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1 || true

  if [ "$exit_code" = "0" ]; then
    log "${GREEN}✓ [$exp] centralizado OK${NC}"
    return 0
  else
    log "${RED}✗ [$exp] centralizado FALLÓ (exit=${exit_code:-?}) — ver $QUEUE_LOG${NC}"
    return 1
  fi
}

run_federated_phase() {
  local exp=$1
  log "${YELLOW}[$exp] Fase federada (servidor + clientes)...${NC}"

  if ! REPO="$REPO" MAMMO_DATA="$MAMMO_DATA" WEIGHTS_DIR="$WEIGHTS_DIR" \
      scripts/docker-deploy-federated.sh "$exp" >> "$QUEUE_LOG" 2>&1; then
    log "${RED}✗ [$exp] federado FALLÓ al arrancar servidor/clientes — ver $QUEUE_LOG${NC}"
    docker rm -f "${exp}_server" "${exp}_client"{1..5} >/dev/null 2>&1 || true
    return 1
  fi

  local exit_code
  exit_code="$(docker wait "${exp}_server" 2>>"$QUEUE_LOG")"
  docker logs "${exp}_server" --tail 30 >> "$QUEUE_LOG" 2>&1
  docker rm -f "${exp}_server" "${exp}_client"{1..5} >/dev/null 2>&1 || true

  if [ "$exit_code" = "0" ]; then
    log "${GREEN}✓ [$exp] federado OK${NC}"
    return 0
  else
    log "${RED}✗ [$exp] federado FALLÓ (exit=${exit_code:-?}) — ver $QUEUE_LOG${NC}"
    return 1
  fi
}

for EXP in "${EXPERIMENTS[@]}"; do
  CFG_DIR="configs/$EXP"
  if [ ! -d "$CFG_DIR" ]; then
    log "${RED}✗ [$EXP] no existe $CFG_DIR — saltando${NC}"
    RESULTS+=("$EXP: FAIL (config no encontrado)")
    log ""
    continue
  fi

  STATUS="OK"

  if [ -f "$CFG_DIR/pretrain.yaml" ]; then
    run_centralized_phase "$EXP" "pretrain.yaml" || STATUS="FAIL (centralizado)"
  elif [ -f "$CFG_DIR/centralized.yaml" ]; then
    run_centralized_phase "$EXP" "centralized.yaml" || STATUS="FAIL (centralizado)"
  fi

  if [ -f "$CFG_DIR/server.yaml" ]; then
    if [ "$STATUS" = "OK" ]; then
      run_federated_phase "$EXP" || STATUS="FAIL (federado)"
    else
      log "${YELLOW}⚠ [$EXP] se salta la fase federada porque el centralizado falló${NC}"
      STATUS="$STATUS + federado saltado"
    fi
  fi

  RESULTS+=("$EXP: $STATUS")
  log ""
done

log "${YELLOW}=== Resumen de la cola ===${NC}"
for r in "${RESULTS[@]}"; do
  if [[ "$r" == *": OK" ]]; then
    log "  ${GREEN}✓${NC} $r"
  else
    log "  ${RED}✗${NC} $r"
  fi
done
log ""
log "Log completo: $QUEUE_LOG"
