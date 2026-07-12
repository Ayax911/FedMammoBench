# /docker-queue — Ejecuta varios experimentos Docker en secuencia

Envuelve `scripts/run-queue.sh`: corre una lista de experimentos uno tras otro (esperando a que cada uno termine antes del siguiente) en background, para poder "dejarlos programados" sin bloquear la sesión.

## Uso
```
/docker-queue <exp1> <exp2> ... [--file <lista.txt>] [--foreground]
```

**Ejemplos:**
- `/docker-queue exp15 exp10 exp08` → los corre en background, devuelve el log al instante
- `/docker-queue --file queue.txt` → lista de experimentos desde un archivo (uno por línea, `#` = comentario)
- `/docker-queue exp15 --foreground` → bloquea la terminal hasta terminar (solo para pruebas cortas)

## Instrucciones

### 1. Resolver variables de entorno

Igual que en `/docker-run`: `REPO`, `MAMMO_DATA`, `WEIGHTS_DIR`, `IMAGE_TAG`. Si `MAMMO_DATA` o `WEIGHTS_DIR` no están seteadas, pregunta al usuario en vez de asumir un valor (ver tabla completa en `docker-run.md`).

### 2. Mostrar qué va a correr, antes de lanzar

Para cada experimento de la lista, revisa `configs/<exp>/` y anticipa las fases que `scripts/run-queue.sh` va a ejecutar:
- `pretrain.yaml` o `centralized.yaml` → fase centralizada
- `server.yaml` (+ `client.yaml`) → fase federada
- ambos → centralizada primero, federada solo si la primera termina OK

Si algún `<exp>` no tiene `configs/<exp>/`, avísalo (la cola lo saltará y lo marcará FAIL).

### 3. Confirmar antes de ejecutar

Son corridas largas (horas) que ocupan la GPU en secuencia. Muestra el resumen de fases por experimento y las variables de entorno resueltas, y pide confirmación antes de lanzar.

### 4. Ejecutar

**Por defecto, siempre en background** (evita bloquear la sesión o el tool call de Claude Code — una corrida federada completa puede tardar horas, muy por encima de cualquier timeout de comando):
```bash
QUEUE_LOG="runs/_logs/queue/queue_$(date +%Y%m%d_%H%M%S).log"
nohup env QUEUE_LOG="$QUEUE_LOG" scripts/run-queue.sh <exp1> <exp2> ... > /dev/null 2>&1 &
disown
echo "PID: $!"
```
Reporta al usuario: el PID y la ruta del `runs/_logs/queue/queue_<timestamp>.log` que escribe `run-queue.sh` con toda la salida (stdout y stderr).

Si el usuario pasó `--foreground` (uso interactivo/pruebas cortas), corre `scripts/run-queue.sh <exp1> <exp2> ...` directamente sin `nohup`, bloqueando hasta terminar.

### 5. Cómo monitorear después

Indica al usuario estos comandos (no hace falta quedarse ejecutándolos por él):
```bash
tail -f runs/_logs/queue/queue_<timestamp>.log   # progreso de la cola completa
docker ps                                         # contenedores activos ahora
docker logs -f <exp>_server                       # logs del experimento en curso
ps -p <PID>                                       # si la cola sigue corriendo
```

### 6. Detener la cola

```bash
kill <PID>                                        # detiene run-queue.sh (no interrumpe el contenedor Docker en curso)
docker rm -f <exp>_server <exp>_client{1..5} <exp>_queue_centralized   # limpia el experimento en curso
```

## Notas

- No relanza `run-queue.sh` con `set -e`: si un experimento falla, la cola sigue con el siguiente (comportamiento fijo del script, no configurable por flag).
- No incluye evaluación post-hoc — usa `/docker-run --only eval` por separado una vez termine la cola.
- Ver `scripts/run-queue.sh` para el detalle de cómo arma cada fase (mismos mounts/env vars que `/docker-run` y `docker-deploy-federated.sh`).
