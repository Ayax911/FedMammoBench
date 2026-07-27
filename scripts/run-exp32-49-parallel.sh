#!/bin/bash
# scripts/run-exp32-49-parallel.sh — Grid federado exp32–49 con 2 slots en paralelo
#
# La GPU está ~96% ociosa con un solo experimento (cuello de botella = carga de datos
# single-thread, num_workers=0). Este scheduler corre 2 experimentos a la vez en puertos
# separados (8080/8081) con una COLA COMPARTIDA (work-stealing): cuando un slot termina,
# toma el siguiente. VRAM ~31/49 GB con 2 concurrentes (seguro; 3 daría OOM).
#
# Orden de la cola: cheap-first, con un FedAdam BARATO primero (exp34, ~1 h) para detectar
# colapso pronto, y los FedAdam CAROS al final (abortables si exp34 colapsa como FedYogi en
# exp29). Correr en paralelo NO cambia las métricas (cada exp es un proceso aislado, seed=42);
# solo el timing por-ronda queda menos limpio.
#
# Uso:
#   scripts/run-exp32-49-parallel.sh                 # los 18 en el orden diseñado
#   scripts/run-exp32-49-parallel.sh exp34 exp47 ... # una lista propia
#
# Background:
#   nohup scripts/run-exp32-49-parallel.sh > runs/_logs/queue/parallel_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail   # NO -e: un experimento que falle no debe abortar la cola

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export REPO
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"; export MAMMO_DATA
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"; export WEIGHTS_DIR
cd "$REPO"

PORTS=(8080 8081)   # 2 slots (VRAM cap: 2 concurrentes ~31 GB)

# Cola por defecto: exp34 (fedadam A, ~1h) PRIMERO para señal de colapso temprana;
# no-fedadam caros temprano (makespan); fedadam caros (37/40/46/49) AL FINAL (abortables).
DEFAULT_QUEUE=(
  exp34                              # fedadam A 5r×20e  (~1.0h)  ← señal FedAdam temprana
  exp47 exp48                        # fedavg/prox F 5r×100e (~4.8h)
  exp43                              # fedadam D 1r×100e (~1.0h)  ← 2ª señal FedAdam barata
  exp44 exp45                        # fedavg/prox E 3r×100e (~2.9h)
  exp35 exp36                        # fedavg/prox B 10r×20e (~2.0h)
  exp38 exp39                        # fedavg/prox C 30r×5e  (~1.6h)
  exp32 exp33                        # fedavg/prox A 5r×20e  (~1.0h)
  exp41 exp42                        # fedavg/prox D 1r×100e (~1.0h)
  exp40 exp37 exp46 exp49            # fedadam C/B/E/F AL FINAL (abortables si colapsa)
)

if [[ $# -gt 0 ]]; then QUEUE=("$@"); else QUEUE=("${DEFAULT_QUEUE[@]}"); fi

mkdir -p runs/_logs/queue
MASTER_LOG="runs/_logs/queue/exp32_49_parallel_$(date +%Y%m%d_%H%M%S).log"
DONEDIR="$(mktemp -d)"
trap 'rm -rf "$DONEDIR"' EXIT

log() { echo -e "$1" | tee -a "$MASTER_LOG"; }

log "════════════════════════════════════════════════════════════════════════════════"
log "  GRID exp32–49 — 2 SLOTS EN PARALELO (puertos ${PORTS[*]})"
log "  Cola (${#QUEUE[@]}): ${QUEUE[*]}"
log "  Start: $(date)"
log "  MASTER_LOG=$MASTER_LOG"
log "════════════════════════════════════════════════════════════════════════════════"

declare -A PORT_EXP        # port -> exp corriendo (vacío = libre)
SUCCESS=(); FAIL=()
START_TS=$(date +%s)

launch_one() {
  local exp=$1 port=$2
  local elog="runs/_logs/queue/${exp}_$(date +%Y%m%d_%H%M%S).log"
  log "${BLUE}[$(date +%H:%M:%S)] ▶ $exp → puerto $port  (log: $elog)${NC}"
  ( scripts/docker-deploy-federated.sh "$exp" --port="$port" > "$elog" 2>&1; echo $? > "$DONEDIR/$exp" ) &
  PORT_EXP[$port]=$exp
}

# ── Prime los slots ───────────────────────────────────────────────────────
idx=0
for port in "${PORTS[@]}"; do
  [[ $idx -lt ${#QUEUE[@]} ]] || break
  launch_one "${QUEUE[$idx]}" "$port"; idx=$((idx+1))
done

# ── Loop principal (work-stealing) ────────────────────────────────────────
while :; do
  active=0
  for p in "${PORTS[@]}"; do [[ -n "${PORT_EXP[$p]:-}" ]] && active=1; done
  [[ $active -eq 0 ]] && break
  sleep 20
  for port in "${PORTS[@]}"; do
    exp="${PORT_EXP[$port]:-}"
    [[ -z "$exp" ]] && continue
    if [[ -f "$DONEDIR/$exp" ]]; then
      rc=$(cat "$DONEDIR/$exp"); rm -f "$DONEDIR/$exp"
      if [[ "$rc" -eq 0 ]]; then
        log "${GREEN}[$(date +%H:%M:%S)] ✓ $exp OK (puerto $port)${NC}"; SUCCESS+=("$exp")
      else
        log "${RED}[$(date +%H:%M:%S)] ✗ $exp FALLÓ rc=$rc (puerto $port)${NC}"; FAIL+=("$exp")
      fi
      PORT_EXP[$port]=""
      if [[ $idx -lt ${#QUEUE[@]} ]]; then
        launch_one "${QUEUE[$idx]}" "$port"; idx=$((idx+1))
      fi
    fi
  done
done

# ── Resumen ───────────────────────────────────────────────────────────────
ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))
log ""
log "════════════════════════════════════════════════════════════════════════════════"
log "  RESUMEN — $(date)  (tiempo total: ${ELAPSED} min)"
[[ ${#SUCCESS[@]} -gt 0 ]] && log "${GREEN}✓ OK (${#SUCCESS[@]}): ${SUCCESS[*]}${NC}"
[[ ${#FAIL[@]}    -gt 0 ]] && log "${RED}✗ FALLÓ (${#FAIL[@]}): ${FAIL[*]}${NC}"
log "  MASTER_LOG=$MASTER_LOG"
log "════════════════════════════════════════════════════════════════════════════════"

[[ ${#FAIL[@]} -gt 0 ]] && exit 1 || exit 0
