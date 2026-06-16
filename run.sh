#!/usr/bin/env bash
#
# run.sh — automatización de experimentos de aprendizaje federado (Docker Compose).
#
# Orquesta el servidor central y los nodos cliente, tanto en una sola máquina
# como en máquinas separadas, con o sin GPU.
#
# USO:
#   ./run.sh <comando> [opciones]
#
# COMANDOS:
#   build              Construye la imagen Docker.
#   up                 Levanta TODO en esta máquina (server + node0 + node1).
#   server             Levanta SOLO el servidor central.
#   node <0|1>         Levanta SOLO un nodo cliente.
#   down               Detiene y elimina los contenedores.
#   logs [servicio]    Muestra logs (en vivo). Ej: ./run.sh logs server
#   ps                 Estado de los contenedores.
#   shell [servicio]   Abre una shell dentro de la imagen (debug). Por defecto: server.
#   clean              down + elimina la red. (No borra runs/ ni datos.)
#
# OPCIONES:
#   --gpu              Habilita GPU NVIDIA (añade docker-compose.gpu.yml).
#   --build            Reconstruye la imagen antes de levantar (en up/server/node).
#   -d, --detach       Ejecuta en segundo plano.
#
# EJEMPLOS:
#   ./run.sh build --gpu
#   ./run.sh up --gpu                 # experimento completo local con GPU
#   ./run.sh server --gpu -d          # solo servidor, en background (host servidor)
#   SERVER_ADDRESS=192.168.1.10:8080 ./run.sh node 0 --gpu   # nodo en otro host
#   ./run.sh logs node0
#   ./run.sh down
#
# Los parámetros del experimento (configs, rutas, dirección del servidor) se
# leen de .env (copia .env.example a .env). Ver docs/DOCKER.md.

set -euo pipefail

# Sitúate en la raíz del repo (donde vive este script).
cd "$(dirname "$(readlink -f "$0")")"

# ── Selección del binario de compose (plugin v2 o binario v1) ───────────
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "ERROR: no se encontró 'docker compose' ni 'docker-compose'." >&2
  exit 1
fi

# ── Parseo de flags globales ────────────────────────────────────────────
USE_GPU=0
DO_BUILD=0
DETACH=0
POSITIONAL=()

for arg in "$@"; do
  case "$arg" in
    --gpu)            USE_GPU=1 ;;
    --build)          DO_BUILD=1 ;;
    -d|--detach)      DETACH=1 ;;
    *)                POSITIONAL+=("$arg") ;;
  esac
done
set -- "${POSITIONAL[@]:-}"

CMD="${1:-}"
shift || true

# ── Construye los argumentos -f (base [+ gpu]) ──────────────────────────
FILES=(-f docker-compose.yml)
if [[ "$USE_GPU" -eq 1 ]]; then
  FILES+=(-f docker-compose.gpu.yml)
fi

# Avisa si falta .env (Compose usará los defaults del compose igualmente).
if [[ ! -f .env ]]; then
  echo "AVISO: no existe .env — usando valores por defecto. (cp .env.example .env)" >&2
fi

up_flags() {
  local flags=(up)
  [[ "$DO_BUILD" -eq 1 ]] && flags+=(--build)
  [[ "$DETACH" -eq 1 ]]   && flags+=(-d)
  printf '%s\n' "${flags[@]}"
}

case "$CMD" in
  build)
    "${COMPOSE[@]}" "${FILES[@]}" --profile all build
    ;;

  up)
    mapfile -t UPF < <(up_flags)
    "${COMPOSE[@]}" "${FILES[@]}" --profile all "${UPF[@]}"
    ;;

  server)
    mapfile -t UPF < <(up_flags)
    "${COMPOSE[@]}" "${FILES[@]}" --profile server "${UPF[@]}"
    ;;

  node)
    NODE_ID="${1:-}"
    if [[ "$NODE_ID" != "0" && "$NODE_ID" != "1" ]]; then
      echo "ERROR: especifica el nodo: ./run.sh node 0   (o 1)" >&2
      exit 1
    fi
    mapfile -t UPF < <(up_flags)
    "${COMPOSE[@]}" "${FILES[@]}" --profile "node${NODE_ID}" "${UPF[@]}"
    ;;

  down)
    "${COMPOSE[@]}" "${FILES[@]}" --profile all down
    ;;

  clean)
    "${COMPOSE[@]}" "${FILES[@]}" --profile all down --remove-orphans
    ;;

  logs)
    SVC="${1:-}"
    "${COMPOSE[@]}" "${FILES[@]}" logs -f ${SVC:+"$SVC"}
    ;;

  ps)
    "${COMPOSE[@]}" "${FILES[@]}" --profile all ps
    ;;

  shell)
    SVC="${1:-server}"
    "${COMPOSE[@]}" "${FILES[@]}" run --rm --entrypoint bash "$SVC"
    ;;

  ""|-h|--help|help)
    sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  *)
    echo "ERROR: comando desconocido '$CMD'. Usa './run.sh help'." >&2
    exit 1
    ;;
esac
