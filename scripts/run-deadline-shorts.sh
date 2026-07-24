#!/bin/bash
# scripts/run-deadline-shorts.sh — Corre los experimentos CORTOS antes de un deadline duro
#
# Contexto: cómputo disponible solo hasta 2026-07-22 06:00. Prioriza los bloques cortos
# (A=5r×20e, D=1r×100e ~1.6h; C=30r×5e ~3.3h) — los pesados (B/E/F) NO caben y se omiten.
# Cortos-primero, fedavg/fedprox antes que fedadam (eta=0.01 aún sin verificar del todo).
#
# - Máx 2 experimentos propios en paralelo + gate de VRAM (convive con exp47 que sigue vivo).
# - No lanza un experimento si no alcanza a terminar antes del deadline.
# - A las 05:50 (10 min de margen) DETIENE todo automáticamente.

set -uo pipefail
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export REPO
MAMMO_DATA="${MAMMO_DATA:-/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench}"; export MAMMO_DATA
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"; export WEIGHTS_DIR
cd "$REPO"

DEADLINE_KILL=$(date -d '2026-07-22 05:50:00' +%s)   # stop TODO 10 min antes de las 6am
PORTS=(8080 8082)                                     # 8081 lo ocupa exp47 (se respeta)
VRAM_GATE=18000                                       # MiB libres mínimos para lanzar
MAX_CONCURRENT=2

# Cola: "exp:segundos_necesarios". Cortos primero; fedavg/fedprox antes que fedadam.
QUEUE=(
  "exp41:6600"  "exp42:6600"                 # D fedavg/fedprox  (~1.6h)
  "exp32:6600"  "exp33:6600"                 # A fedavg/fedprox  (~1.6h)
  "exp43:6600"  "exp34:6600"                 # D/A fedadam (eta=0.01)  (~1.6h)
  "exp38:13500" "exp39:13500"                # C fedavg/fedprox  (~3.3h)
  "exp40:13500"                              # C fedadam (eta=0.01)  (~3.3h)
)

mkdir -p runs/_logs/queue
MASTER_LOG="runs/_logs/queue/deadline_shorts_$(date +%Y%m%d_%H%M%S).log"
DONEDIR="$(mktemp -d)"; trap 'rm -rf "$DONEDIR"' EXIT
log(){ echo -e "$1" | tee -a "$MASTER_LOG"; }

log "════════════════════════════════════════════════════════════════════════════════"
log "  DEADLINE-SHORTS — parar TODO a las $(date -d @$DEADLINE_KILL '+%H:%M')  (10 min antes de 6am)"
log "  Cola: ${QUEUE[*]%%:*}"
log "  Start: $(date)   MASTER_LOG=$MASTER_LOG"
log "════════════════════════════════════════════════════════════════════════════════"

declare -A PORT_EXP; SUCCESS=(); FAIL=(); SKIPPED=()
free_vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
running_count(){ local n=0; for p in "${PORTS[@]}"; do [[ -n "${PORT_EXP[$p]:-}" ]] && n=$((n+1)); done; echo $n; }
free_port(){ for p in "${PORTS[@]}"; do [[ -z "${PORT_EXP[$p]:-}" ]] && { echo "$p"; return; }; done; }

launch_one(){
  local exp=$1 port=$2
  local elog="runs/_logs/queue/${exp}_deadline_$(date +%H%M%S).log"
  log "${BLUE}[$(date +%H:%M:%S)] ▶ $exp → puerto $port${NC}"
  ( scripts/docker-deploy-federated.sh "$exp" --port="$port" > "$elog" 2>&1; echo $? > "$DONEDIR/$exp" ) &
  PORT_EXP[$port]=$exp
}

reap(){
  for port in "${PORTS[@]}"; do
    local exp="${PORT_EXP[$port]:-}"; [[ -z "$exp" ]] && continue
    if [[ -f "$DONEDIR/$exp" ]]; then
      local rc; rc=$(cat "$DONEDIR/$exp"); rm -f "$DONEDIR/$exp"
      if [[ "$rc" -eq 0 ]]; then log "${GREEN}[$(date +%H:%M:%S)] ✓ $exp OK${NC}"; SUCCESS+=("$exp")
      else log "${RED}[$(date +%H:%M:%S)] ✗ $exp rc=$rc${NC}"; FAIL+=("$exp"); fi
      PORT_EXP[$port]=""
    fi
  done
}

qidx=0
while :; do
  now=$(date +%s)
  # deadline duro → parar todo
  if [[ $now -ge $DEADLINE_KILL ]]; then
    log "${YELLOW}[$(date +%H:%M:%S)] ⏰ DEADLINE — deteniendo TODOS los contenedores de entrenamiento${NC}"
    docker ps --format '{{.Names}}' | grep -E '_(server|client[0-9])$' | xargs -r docker stop >/dev/null 2>&1
    break
  fi
  reap
  # ¿lanzar el siguiente?
  if [[ $qidx -lt ${#QUEUE[@]} && $(running_count) -lt $MAX_CONCURRENT ]]; then
    entry="${QUEUE[$qidx]}"; exp="${entry%%:*}"; need="${entry##*:}"
    remaining=$(( DEADLINE_KILL - now ))
    port=$(free_port)
    if [[ $remaining -lt $need ]]; then
      log "${YELLOW}[$(date +%H:%M:%S)] ⤼ omito $exp (faltan ${remaining}s < ${need}s necesarios)${NC}"
      SKIPPED+=("$exp"); qidx=$((qidx+1)); continue
    fi
    if [[ -n "$port" && "$(free_vram)" -ge $VRAM_GATE ]]; then
      launch_one "$exp" "$port"; qidx=$((qidx+1)); sleep 10; continue
    fi
  fi
  # ¿terminamos? (cola agotada y nada corriendo)
  [[ $qidx -ge ${#QUEUE[@]} && $(running_count) -eq 0 ]] && { log "Cola agotada antes del deadline."; break; }
  sleep 20
done

# resumen
log ""
log "════════════════════════════════════════════════════════════════════════════════"
log "  RESUMEN DEADLINE-SHORTS — $(date)"
[[ ${#SUCCESS[@]} -gt 0 ]] && log "${GREEN}✓ Completos (${#SUCCESS[@]}): ${SUCCESS[*]}${NC}"
[[ ${#FAIL[@]}    -gt 0 ]] && log "${RED}✗ Fallaron/detenidos (${#FAIL[@]}): ${FAIL[*]}${NC}"
[[ ${#SKIPPED[@]} -gt 0 ]] && log "${YELLOW}⤼ Omitidos por tiempo (${#SKIPPED[@]}): ${SKIPPED[*]}${NC}"
log "  (exp47 F seguía aparte; revísalo con: ls runs/exp47*/weights/)"
log "════════════════════════════════════════════════════════════════════════════════"
