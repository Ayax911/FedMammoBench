# /docker-run — Ejecuta un experimento en Docker (centralizado, federado o evaluación)

Lanza en contenedores Docker cualquier experimento del proyecto (`configs/<exp>/`), detectando automáticamente qué tipo de corrida es según los YAML presentes, y usando siempre variables de entorno (nunca rutas hardcodeadas) para las carpetas que cambian de una máquina a otra: imágenes y pesos preentrenados.

## Uso
```
/docker-run <exp> [--only pretrain|centralized|federated|eval] [--monitor] [--no-clean]
```

**Ejemplos:**
- `/docker-run exp15` → detecta que solo hay `pretrain.yaml` → un contenedor centralizado
- `/docker-run exp14 --monitor` → servidor + 5 clientes federados, espera ROUND 1
- `/docker-run exp07 --only eval` → corre los 5 YAMLs de `configs/exp07/eval/`

## Variables de entorno

Estas rutas **cambian según la máquina** (dataset de imágenes montado en un disco distinto, pesos preentrenados en otra ubicación) y nunca deben quedar hardcodeadas en un comando `docker run`. Resuélvelas siempre así, en este orden:

| Variable | Qué contiene | Si no está seteada |
|---|---|---|
| `REPO` | Raíz del repo (para montar `configs/`, `manifests/`, `runs/`) | Usa `pwd` del repo actual (ya lo estás ejecutando desde ahí) |
| `MAMMO_DATA` | Carpeta con las imágenes (`Preprocessed_Dataset/`) montada en `/app/data` | **Pregunta al usuario** — no asumas un valor; solo si existe un `.env` con `MAMMOBENCH_DATA_DIR` ofrécelo como sugerencia |
| `WEIGHTS_DIR` | Carpeta con checkpoints preentrenados (RadImageNet, warm-starts) montada en `/app/weights` | Sugiere `$REPO/weights` (default del proyecto) pero permite que el usuario la cambie |
| `IMAGE_TAG` | Imagen Docker a usar | `ayax911/federal-learning:latest` |

Antes de lanzar cualquier contenedor:
1. Comprueba si `REPO`, `MAMMO_DATA`, `WEIGHTS_DIR` ya están exportadas en el shell (`echo $MAMMO_DATA` etc.).
2. Si `MAMMO_DATA` o `WEIGHTS_DIR` no están seteadas, **pregunta al usuario** el valor correcto en vez de inventarlo — son las dos rutas que más varían entre hosts (ver `.env` local si existe, pero no lo copies ciegamente: puede estar desactualizado).
3. Muestra el resumen de variables resueltas antes de construir los comandos `docker run`.

Esto reemplaza el hardcode que tenía `scripts/docker-deploy-federated.sh` (ahora también soporta `WEIGHTS_DIR`, con fallback a `$REPO/weights`).

## Instrucciones

### 1. Detectar el tipo de experimento

Lista `configs/<exp>/` y decide según lo que exista (o según `--only` si se pasó):

| Archivos presentes | Tipo | Entrypoint |
|---|---|---|
| `pretrain.yaml` o `centralized.yaml` | Centralizado | `scripts/run_centralized.py` |
| `server.yaml` + `client.yaml` | Federado | `scripts/run_server.py` + `scripts/run_client.py` |
| `eval/*.yaml` | Evaluación post-hoc | `scripts/run_evaluation.py` |

Un experimento puede tener más de un tipo a la vez (ej. `exp07` tiene `pretrain.yaml` + `server.yaml`/`client.yaml` + `eval/`) — en ese caso pregunta cuál correr, salvo que `--only` ya lo indique.

### 2. Caso centralizado / pretrain

Un solo contenedor:
```bash
docker run -d --name "${EXP}_run" --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$WEIGHTS_DIR:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  "$IMAGE_TAG" \
  python scripts/run_centralized.py --config "configs/${EXP}/pretrain.yaml"
```
(sustituye `pretrain.yaml` por `centralized.yaml` según cuál exista). Verifica arranque con `docker logs "${EXP}_run" | tail -20` a los pocos segundos.

### 3. Caso federado

Si el experimento sigue el patrón estándar de 5 nodos documentado en `configs/README.md` (cmmd, inbreast, cdd-cesm, kau-bcmd, dmid), **reusa el script existente** en vez de reimplementar los `docker run`:
```bash
REPO="$REPO" MAMMO_DATA="$MAMMO_DATA" WEIGHTS_DIR="$WEIGHTS_DIR" \
  scripts/docker-deploy-federated.sh "$EXP" $( [ "$MONITOR" = true ] && echo --monitor ) $( [ "$NO_CLEAN" = true ] && echo --no-clean )
```

Si el experimento tiene una variante de nodos distinta (ej. `exp01/server_6nodes.yaml`) o el usuario pide un mapeo de nodos distinto, construye los `docker run` manualmente siguiendo el patrón de `configs/README.md` (servidor primero, luego un contenedor por nodo con `--client-id` y `--manifest` correspondiente), sustituyendo siempre `$WEIGHTS_DIR` en vez de `$REPO/weights`.

### 4. Caso evaluación

Para cada YAML en `configs/<exp>/eval/*.yaml`, pregunta (si no es obvio por el nombre del experimento) qué checkpoint usar — recuerda la convención de anidado `runs/<name>/<name>/...` (ver nota en `new-exp.md`):
- Checkpoint de pretrain: `runs/<exp>_pretrain_ddsm/<exp>_pretrain_ddsm/final.pt`
- Checkpoint federado agregado: `runs/<exp>_.../<exp>_.../global_model.pt`

```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$WEIGHTS_DIR:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  "$IMAGE_TAG" \
  python scripts/run_evaluation.py \
    --config "configs/${EXP}/eval/<config_name>.yaml" \
    --checkpoint "<ruta_checkpoint>" \
    --split test \
    --output-dir "runs/<exp_name>/eval/<config_name>" \
    --predictions-out "runs/<exp_name>/eval/<config_name>/predictions.csv"
```

### 5. Confirmar antes de ejecutar

Antes de lanzar cualquier `docker run`, muestra el resumen: tipo de corrida detectado, variables de entorno resueltas (`REPO`, `MAMMO_DATA`, `WEIGHTS_DIR`, `IMAGE_TAG`) y los comandos exactos a ejecutar. Pregunta confirmación — son contenedores de larga duración que consumen GPU.

### 6. Verificación y limpieza

- Servidor listo: `docker logs <nombre> | grep "gRPC server running"`
- Cliente listo: `docker logs <nombre> | grep "data loaded"`
- Ver logs en vivo: `docker logs -f <nombre>`
- Detener: `docker stop <nombres>` / limpiar: `docker rm -f <nombres>`

## Cola de varios experimentos en secuencia

Para dejar corriendo varios experimentos uno tras otro (sin confirmación interactiva por cada uno), usa `scripts/run-queue.sh` en vez de invocar este comando repetidamente:
```bash
scripts/run-queue.sh exp15 exp10 exp08
scripts/run-queue.sh --file queue.txt          # un experimento por línea
QUEUE_LOG="runs/_logs/queue/queue_$(date +%Y%m%d_%H%M%S).log"
nohup env QUEUE_LOG="$QUEUE_LOG" scripts/run-queue.sh exp15 exp10 exp08 > /dev/null 2>&1 &   # sobrevive al cierre
```
Detecta el tipo de cada experimento igual que este comando, espera (`docker wait`) a que cada contenedor termine antes de lanzar el siguiente, y si uno falla **continúa con el siguiente** (no detiene la cola) — queda registrado en `runs/queue_<timestamp>.log` y en el resumen final. No incluye evaluación post-hoc (requiere elegir el checkpoint a mano); correr esas con `--only eval` después.

## Notas

- `scripts/docker-deploy-federated.sh` sigue siendo el camino rápido para el patrón federado estándar de 5 nodos — este comando lo reusa en vez de duplicarlo. Ahora acepta `WEIGHTS_DIR` (antes tenía `$REPO/weights` hardcodeado).
- El `.env`/`.env.example`/`run.sh` (docker-compose, exp01 de 2 nodos) son un flujo **legacy** distinto al de `configs/README.md`; no mezclar sus variables (`NODE_DATA_DIR`, `SERVER_ADDRESS=server:8080`, etc.) con `MAMMO_DATA`/`WEIGHTS_DIR` de este comando.
- `weights/` y las imágenes están gitignored (`*.pth`, `*.pt`, `data/`) — por eso sus rutas reales solo existen en el filesystem de cada host y deben pasarse por variable de entorno, nunca commitearse.
