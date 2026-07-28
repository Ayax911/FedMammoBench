# Configs

## Estructura de carpetas

```
configs/
├── base.yaml              # Valores por defecto globales (herencia via `defaults:`)
├── reference.yaml         # Config de referencia anotado con todos los campos
├── exp01/                 # FedAvg baseline sin warm start
│   ├── server.yaml
│   ├── server_6nodes.yaml
│   └── client.yaml
├── exp07/                 # FedAvg con warm start DDSM + cyclic unfreeze
│   ├── pretrain.yaml      # Fase 1: pretrain centralizado en DDSM
│   ├── server.yaml        # Fase 2: servidor FL
│   ├── client.yaml        # Fase 2: clientes FL (todos los nodos comparten este YAML)
│   └── eval/              # Evaluación del modelo global por nodo
│       ├── node1_cmmd.yaml
│       ├── node2_inbreast.yaml
│       ├── node3_cdd-cesm.yaml
│       ├── node4_kau-bcmd.yaml
│       └── node5_dmid.yaml
├── exp08/                 # Centralizado con todos los datos
│   └── centralized.yaml
├── exp09/                 # FedAvg variante
│   ├── server.yaml
│   └── client.yaml
├── exp10/                 # Centralizado variante
│   └── centralized.yaml
├── exp50 .. exp55/        # Factorial linear probing (backbone RadImageNet CONGELADO):
│   ├── centralized.yaml   #   3 profundidades de cabeza MLP x 2 dropouts
│   └── eval/
│       └── mammo_bench.yaml
├── exp56/                 # Igual que exp54 pero epochs=15 + save_best_checkpoint
│   ├── centralized.yaml   #   (ver "Selección de mejor checkpoint" más abajo)
│   └── eval/
│       └── mammo_bench.yaml
└── legacy/                # Configs anteriores al refactor (no usar en experimentos nuevos)
    ├── fedavg_cbis_ddsm.yaml
    ├── radimagenet_*.yaml
    └── ...
```

`configs/exp11` a `exp49` (omitidos arriba por brevedad) siguen el mismo patrón que exp07-10:
carpetas con `server.yaml`/`client.yaml` (federado) o `centralized.yaml` (centralizado), más `eval/`
cuando aplica.

---

## Variables de entorno

Definir en la terminal antes de lanzar cualquier contenedor. `MAMMO_DATA` y `WEIGHTS_DIR` **cambian según la máquina** (dataset y pesos preentrenados suelen vivir en discos externos gitignored) — nunca hardcodear sus rutas en un `docker run`, siempre pasarlas por variable:

```bash
# Raíz del repositorio en el host
REPO="ruta/al/proyecto/Federal-Learning"

# Directorio que contiene Preprocessed_Dataset/ con las imágenes JPG
MAMMO_DATA="/ruta/a/mammo-bench/data"

# Carpeta con checkpoints preentrenados (RadImageNet, warm-starts). Default: $REPO/weights
WEIGHTS_DIR="$REPO/weights"

# Imagen Docker a usar
IMAGE_TAG="ayax911/federal-learning:latest"
```

---

## Mounts comunes

Todos los contenedores usan este conjunto base de mounts:

| Mount | Descripción |
|---|---|
| `-v "$REPO/configs:/app/configs:ro"` | YAMLs de configuración |
| `-v "$REPO/manifests:/app/manifests:ro"` | CSVs de splits por dataset |
| `-v "$WEIGHTS_DIR:/app/weights:ro"` | Checkpoints de pesos preentrenados |
| `-v "$MAMMO_DATA:/app/data:ro"` | Imágenes mamográficas |
| `-v "$REPO/runs:/app/runs"` | Salida de experimentos (escritura) |

El servidor FL no necesita `-v manifests` ni `-v data` (no carga dataset).

---

## Automatización (Claude Code)

Los comandos manuales de este README (`docker run ...`) sirven para entender qué hace cada contenedor, pero para correr experimentos del día a día usa los comandos automatizados — resuelven las variables de entorno, detectan qué tipo de corrida es cada experimento (centralizado, federado, evaluación) leyendo `configs/<exp>/`, y piden confirmación antes de lanzar nada:

| Comando | Qué hace |
|---|---|
| `/docker-run <exp>` | Corre un solo experimento (autodetecta pretrain/centralizado, federado o evaluación) |
| `/docker-queue <exp1> <exp2> ...` | Corre varios experimentos **en secuencia** en background — si uno falla, sigue con el siguiente |
| `scripts/docker-deploy-federated.sh <exp>` | Script bash directo para el patrón federado estándar de 5 nodos (server + cmmd/inbreast/cdd-cesm/kau-bcmd/dmid) |
| `scripts/run-queue.sh <exp1> <exp2> ...` | Script bash directo detrás de `/docker-queue` |

Todos respetan `REPO`/`MAMMO_DATA`/`WEIGHTS_DIR`/`IMAGE_TAG` de la tabla anterior. El `.env`/`.env.example`/`run.sh` (docker-compose) en la raíz del repo son un flujo **legacy** distinto (exp01, 2 nodos, variables `NODE_DATA_DIR`/`SERVER_ADDRESS`) — no mezclar sus variables con las de esta tabla.

---

## exp07 — FedAvg warm start DDSM

### Nodos

| ID | Dataset   | Manifest               | Imágenes |
|----|-----------|------------------------|----------|
| 1  | CMMD      | cmmd-split.csv         | 5 202    |
| 2  | INbreast  | inbreast-split.csv     | 410      |
| 3  | CDD-CESM  | cdd-cesm-split.csv     | 1 003    |
| 4  | KAU-BCMD  | kau-bcmd-split.csv     | 2 206    |
| 5  | DMID      | dmid-split.csv         | 510      |

DDSM (7 808 imágenes) se usa **solo en el pretrain** — no es nodo FL.

---

### Fase 1 — Pretrain centralizado (DDSM)

```bash
docker run -d --name exp07_pretrain --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  ayax911/federal-learning:latest \
  python scripts/run_centralized.py \
    --config configs/exp07/pretrain.yaml
```

Genera: `runs/exp07_pretrain_ddsm/exp07_pretrain_ddsm/final.pt`

---

### Fase 2 — Entrenamiento federado

**Servidor** (lanzar primero):
```bash
docker run -d --name exp07_server --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$REPO/runs:/app/runs" \
  ayax911/federal-learning:latest \
  python scripts/run_server.py \
    --config configs/exp07/server.yaml
```

**Cliente 1 (cmmd):**
```bash
docker run -d --name exp07_client1 --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  ayax911/federal-learning:latest \
  python scripts/run_client.py \
    --config configs/exp07/client.yaml \
    --server 127.0.0.1:8080 \
    --client-id 1 \
    --manifest manifests/cmmd-split.csv
```

**Cliente 2 (inbreast):**
```bash
docker run -d --name exp07_client2 --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  ayax911/federal-learning:latest \
  python scripts/run_client.py \
    --config configs/exp07/client.yaml \
    --server 127.0.0.1:8080 \
    --client-id 2 \
    --manifest manifests/inbreast-split.csv
```

**Cliente 3 (cdd-cesm):**
```bash
docker run -d --name exp07_client3 --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  ayax911/federal-learning:latest \
  python scripts/run_client.py \
    --config configs/exp07/client.yaml \
    --server 127.0.0.1:8080 \
    --client-id 3 \
    --manifest manifests/cdd-cesm-split.csv
```

**Cliente 4 (kau-bcmd):**
```bash
docker run -d --name exp07_client4 --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  ayax911/federal-learning:latest \
  python scripts/run_client.py \
    --config configs/exp07/client.yaml \
    --server 127.0.0.1:8080 \
    --client-id 4 \
    --manifest manifests/kau-bcmd-split.csv
```

**Cliente 5 (dmid):**
```bash
docker run -d --name exp07_client5 --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  ayax911/federal-learning:latest \
  python scripts/run_client.py \
    --config configs/exp07/client.yaml \
    --server 127.0.0.1:8080 \
    --client-id 5 \
    --manifest manifests/dmid-split.csv
```

---

### Evaluación del modelo global por nodo

Checkpoint del modelo federado: `runs/exp07_fedavg_resnet50/exp07_fedavg_resnet50/global_model.pt`

**Pretrain DDSM (baseline fase 1):**
```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs:ro" \
  ayax911/federal-learning:latest \
  python scripts/run_evaluation.py \
    --config configs/exp07/pretrain.yaml \
    --checkpoint runs/exp07_pretrain_ddsm/exp07_pretrain_ddsm/final.pt \
    --split test
```

**Nodo 1 (cmmd):**
```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs:ro" \
  ayax911/federal-learning:latest \
  python scripts/run_evaluation.py \
    --config configs/exp07/eval/node1_cmmd.yaml \
    --checkpoint runs/exp07_fedavg_resnet50/exp07_fedavg_resnet50/global_model.pt \
    --split test
```

**Nodo 2 (inbreast):**
```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs:ro" \
  ayax911/federal-learning:latest \
  python scripts/run_evaluation.py \
    --config configs/exp07/eval/node2_inbreast.yaml \
    --checkpoint runs/exp07_fedavg_resnet50/exp07_fedavg_resnet50/global_model.pt \
    --split test
```

**Nodo 3 (cdd-cesm):**
```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs:ro" \
  ayax911/federal-learning:latest \
  python scripts/run_evaluation.py \
    --config configs/exp07/eval/node3_cdd-cesm.yaml \
    --checkpoint runs/exp07_fedavg_resnet50/exp07_fedavg_resnet50/global_model.pt \
    --split test
```

**Nodo 4 (kau-bcmd):**
```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs:ro" \
  ayax911/federal-learning:latest \
  python scripts/run_evaluation.py \
    --config configs/exp07/eval/node4_kau-bcmd.yaml \
    --checkpoint runs/exp07_fedavg_resnet50/exp07_fedavg_resnet50/global_model.pt \
    --split test
```

**Nodo 5 (dmid):**
```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$REPO/weights:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs:ro" \
  ayax911/federal-learning:latest \
  python scripts/run_evaluation.py \
    --config configs/exp07/eval/node5_dmid.yaml \
    --checkpoint runs/exp07_fedavg_resnet50/exp07_fedavg_resnet50/global_model.pt \
    --split test
```

---

## Selección de mejor checkpoint (centralizado y federado)

Desde la v0.5.0, `training.save_best_checkpoint` permite que `fedmammobench-centralized` guarde y
evalúe el mejor checkpoint visto durante el entrenamiento en vez de siempre usar el de la última
época:

```yaml
training:
  epochs: 15
  save_best_checkpoint: true       # default: false — no cambia el comportamiento si se omite
  best_checkpoint_metric: roc_auc  # default: roc_auc. Debe ser "mayor es mejor":
                                    # roc_auc | f1 | accuracy | auc_pr | precision | recall
```

Con esto activado:

- `weights/final.pt` se sigue guardando igual que siempre (última época).
- `weights/best.pt` se sobrescribe cada vez que `val_<best_checkpoint_metric>` mejora.
- Antes de evaluar el test set, se recarga `best.pt` en vez de usar los pesos de la última época.
- `test_metrics.csv` añade una columna `checkpoint` (`final` o `best_epoch_<N>`) indicando cuál se usó.

**Por qué existe:** en el factorial exp50-55 (linear probing sobre RadImageNet congelado) las 6
corridas alcanzaban su mejor `val_roc_auc` entre la época 3 y 15 de 100, y luego se degradaban por
sobreajuste (`val_loss` sube 3-9x) — el AUC de test reportado terminaba midiendo el checkpoint ya
sobreajustado, no el mejor real. `exp56` (mismos hiperparámetros que `exp54`, `epochs: 15` +
`save_best_checkpoint: true`) es el primer experimento que usa este flag; úsalo como plantilla para
cualquier config nueva con pocas épocas efectivas de convergencia (cabezas pequeñas, backbone
congelado, fine-tuning corto).

Sin selección de mejor checkpoint (el resto de experimentos, exp01-55), no hay early stopping real: si
`training.epochs` es mucho mayor que el punto de convergencia, revisa `metrics.csv` para confirmar que
`val_roc_auc`/`val_loss` no se degradaron antes de confiar en el AUC de test final. Desde la v0.7.0
existe `early_stopping_patience` para detener el entrenamiento en vez de solo diagnosticarlo — ver la
sección "Early stopping (paciencia)" más abajo.

Desde la v0.6.0 existe el equivalente en federado, con los **mismos nombres de campo** pero bajo
`federated:` en vez de `training:`:

```yaml
federated:
  rounds: 20
  save_best_checkpoint: true       # default: false
  best_checkpoint_metric: roc_auc  # default: roc_auc. Igual lista que arriba MENOS auc_pr:
                                    # roc_auc | f1 | accuracy | precision | recall
                                    # (auc_pr no lo reporta FedMammoBenchClient.evaluate en
                                    # federado, así que nunca dispararía un guardado)
```

Diferencias con el centralizado:

- El checkpoint se guarda en `weights/global_best.pt` (junto a `weights/global_model.pt`, que sigue
  siendo siempre la última ronda — nombre distinto de `weights/best.pt` del centralizado a propósito,
  para no confundirlos si algún día un mismo `runs/` mezcla ambos modos).
- Se sobrescribe cada vez que el **promedio ponderado entre nodos** de `val_<best_checkpoint_metric>`
  (la métrica que ya agrega `aggregate_evaluate`) mejora ronda a ronda.
- El servidor **no recarga** `global_best.pt` al terminar — a diferencia del centralizado, el federado
  no tiene un paso de test terminal que recargar (`evaluate_fn` ya corre cada ronda). Evalúa
  `global_best.pt` después con `fedmammobench-evaluate --checkpoint runs/<exp>/weights/global_best.pt`.
- `timing_summary.csv` reporta la ronda y el valor del mejor checkpoint cuando el tracking está activo.

### Early stopping (paciencia)

Desde la v0.7.0, `early_stopping_patience` detiene el entrenamiento/las rondas cuando
`val_<best_checkpoint_metric>` lleva N épocas/rondas consecutivas sin mejorar — cierra la carencia que
el párrafo anterior admitía ("no hay early stopping real"). **Mismo nombre de campo en ambos modos**,
y **requiere `save_best_checkpoint: true`** en la misma sección (el error de validación lo explica si
falta): sin eso no habría forma de saber cuál fue la mejor época/ronda para saber cuándo parar ni qué
pesos conservar.

```yaml
training:                          # o federated: — mismo campo, misma semántica
  save_best_checkpoint: true
  best_checkpoint_metric: roc_auc
  early_stopping_patience: 3       # default: 0 (deshabilitado)
```

Ejemplo concreto: con `best_checkpoint_metric: roc_auc` y `early_stopping_patience: 3`, si la mejor
`val_roc_auc` ocurrió en la época/ronda 4 y las épocas/rondas 5, 6 y 7 no la superan, el entrenamiento
para al terminar la época/ronda 7 (3 sin mejora = patience) — `epochs`/`rounds` configuradas de sobra
nunca se ejecutan. El checkpoint recargado para test (centralizado) o para evaluación post-hoc
(federado, `global_best.pt`) sigue siendo el de la época/ronda 4, nunca el de la 7.

Notas:

- En centralizado, `metrics.csv`/`test_metrics.csv` registran la fila de la época que dispara la
  parada con normalidad; `test_metrics.csv` añade una columna `early_stopped` (bool).
- En federado, no existe ningún hook nativo de Flower para detener el bucle de rondas a mitad de
  camino — se implementa lanzando una excepción interna desde la estrategia envuelta, capturada en
  `server.py` como una parada limpia (log INFO), no un crash. El `finally` del servidor sigue
  guardando `global_model.pt`, escribiendo `final_summary.txt` (que añade una línea "Parada temprana"
  cuando aplica) y generando las gráficas exactamente igual que un run completo.
- Si usas `model.unfreeze_at_epoch`/`local_unfreeze_at_epoch`, ten en cuenta que la parada temprana
  puede ocurrir antes de ese punto — el backbone nunca llegaría a descongelarse, igual que si hubieras
  configurado menos épocas/rondas directamente.
- `0` (default) mantiene el comportamiento anterior sin cambios: corre siempre el presupuesto
  configurado completo.

---

## Resume tras un crash

Desde la v0.8.0, el flag `--resume` (en `run_centralized.py`, `run_federated.py` y `run_server.py`)
permite continuar un entrenamiento interrumpido en vez de perder todo el progreso. No hay ningún campo
nuevo en el YAML — el comportamiento se controla solo con el flag, exactamente igual en la corrida
original y en el relanzamiento:

```bash
python scripts/run_centralized.py --config configs/x.yaml --resume
python scripts/run_federated.py   --config configs/x.yaml --resume
python scripts/run_server.py      --config configs/x.yaml --resume
```

- **Sin checkpoint todavía** (primera vez que se pasa `--resume`): corre normal, pero además empieza a
  escribir un checkpoint de seguridad cada época/ronda (reutiliza `weights/final.pt` /
  `weights/global_model.pt` — no crea archivos nuevos), por si crashea más adelante.
- **Con checkpoint existente**: recarga modelo (+ optimizer + scheduler en centralizado) y continúa
  hasta `training.epochs`/`federated.rounds` — se interpreta como el presupuesto **total** acumulado,
  no "cuánto más". Si el checkpoint ya alcanzó (o superó, por ejemplo si bajaste `epochs` a mano) ese
  total, no hace nada y lo informa por log — es seguro pasar `--resume` de más.
- **Sin `--resume`**: comportamiento idéntico al actual, sin cambios.

**Por qué no basta con el `try/finally` que ya guarda `global_model.pt`/`final.pt` al terminar:** un
crash real en esta máquina compartida (segfault, OOM-killer, reset de driver CUDA) salta por completo la
maquinaria de excepciones de Python — el `finally` nunca corre. Por eso el checkpoint de seguridad se
sobrescribe cada época/ronda, no solo al final.

Notas y limitaciones aceptadas:

- Recuperación por **época/ronda completa** — el trabajo parcial de una época/ronda a medio terminar se
  pierde y se re-corre desde el último punto completo.
- El YAML debe quedar igual entre el crash y el resume, salvo `training.epochs`/`federated.rounds`.
  Cambiar `model.name` falla ruidosamente (carga estricta). Cambiar `federated.num_clients` o
  `partitioning.*` **no** falla ruidosamente — el particionado es determinista solo por `cfg.seed`, así
  que reasignaría en silencio qué datos ve cada cliente.
- En federado, Flower no tiene forma de "empezar a contar en la ronda N" — su contador interno siempre
  reinicia en 1 para un proceso nuevo. El resume compensa aplicando un offset a todo lo que se loguea
  (CSVs, TensorBoard, checkpoints, y la config que reciben los clientes para el descongelamiento
  progresivo) — la estrategia real de Flower nunca ve el offset, solo lo nuestro.
- Estado del `GradScaler` (mixed precision) no se persiste — se readapta solo en pocas iteraciones.
- El tiempo acumulado por nodo (`per_node_timing.csv`) no se combina entre resumes (es por proceso).
- Fuera de alcance por ahora: no está integrado en `scripts/docker-deploy-federated.sh` ni
  `scripts/run-queue.sh` — si crashea, hay que relanzar a mano el mismo comando con `--resume`.

---

## Weights & Biases (monitoreo en tiempo real)

Desde la v0.6.0, tanto `fedmammobench-centralized` como el servidor federado
(`run_server.py`/`fedmammobench-federated`) reportan métricas a [Weights & Biases](https://wandb.ai)
además de TensorBoard/CSV, vía la sección `wandb:` del config:

```yaml
wandb:
  enabled: true              # default: true — activo aunque no pongas esta sección
  project: fedmammobench     # nombre del proyecto en wandb.ai
  entity: null                # null = tu team/usuario por defecto
  run_name: null               # null = usa `name:` del experimento
  group: null                  # null = usa `name:` del experimento (agrupa runs relacionados)
  tags: []
  mode: online                # online | offline | disabled
  log_dir: null                 # null = <output_dir>/<name>/wandb
```

**Es un solo run por experimento, del lado del servidor.** Los clientes/nodos federados (workers Ray en
simulación, o contenedores `run_client.py` en gRPC) **nunca** llaman a `wandb.init` — sus métricas por
nodo llegan igual al run del servidor bajo claves `node_<id>/train_loss`, `node_<id>/val_roc_auc`, etc.,
así que verás paneles por nodo dentro de un único run en la UI de W&B, no un run por nodo.

`enabled: true` es el default (decisión explícita de este proyecto), así que **no hace falta escribir
la sección `wandb:` para que se active** — cualquier config sin ella igual reporta a W&B. Para
desactivarlo en un experimento puntual: `wandb: {enabled: false}`, o `wandb: {mode: disabled}`.

### Cómo usarlo con Docker

El proceso W&B corre **dentro** del contenedor del servidor/centralizado, así que necesita la API key
ahí adentro. `docker-deploy-federated.sh`, `run-queue.sh`, `run-exp{20-22,24-26,28-31,50-55}.sh` y los
workflows de GitHub Actions ya pasan `-e WANDB_API_KEY="${WANDB_API_KEY:-}"` a los contenedores de
servidor/centralizado (nunca a los de cliente/nodo — no la necesitan). Para que llegue algo ahí:

```bash
# Opción 1 — exportar en la shell que lanza el script, antes de correrlo
export WANDB_API_KEY="tu-api-key-de-wandb.ai/authorize"
scripts/docker-deploy-federated.sh exp14

# Opción 2 — guardarla en .env (ya está en .gitignore) y cargarla antes de lanzar
cp .env.example .env   # si no existe aún
# editar .env: WANDB_API_KEY=tu-api-key
set -a; source .env; set +a
scripts/docker-deploy-federated.sh exp14
```

`wandb login` en el host **no sirve** — escribe `~/.netrc` del host, que ningún contenedor monta. La
key tiene que llegar como variable de entorno.

**Si no defines `WANDB_API_KEY` (o la dejas vacía), nada se rompe.** `WandbWriter` detecta la ausencia
de credenciales *antes* de intentar conectarse y baja solo a `mode: offline`: sigue escribiendo el
historial localmente en `runs/<exp>/<name>/wandb/wandb/offline-run-*/` (dentro del mismo mount `-v
"$REPO/runs:/app/runs"` que ya usan todos los scripts), sin tocar la red y sin bloquear el
entrenamiento ni un segundo. Para subir esas corridas después, desde el host con `wandb` instalado:

```bash
pip install wandb && wandb login
wandb sync runs/<exp>/<name>/wandb/wandb/offline-run-*
```

Lo mismo pasa si el paquete `wandb` no está instalado en la imagen (por ejemplo, mientras no se haya
reconstruido y republicado `ayax911/federal-learning:latest` tras este cambio): se loguea un aviso una
vez y el entrenamiento sigue exactamente igual, solo sin datos en W&B.

**GitHub Actions**: definir el secret `WANDB_API_KEY` en el repo (Settings → Secrets → Actions) para
que `run-experiment.yml`/`run-batch.yml`/`run-research-suite.yml` lo pasen automáticamente a los
contenedores de servidor/centralizado.

### Qué ver en la UI

Con la key configurada, entra a `wandb.ai/<tu-entity>/<project>` mientras la corrida está en marcha:
curvas de `train_loss`/`val_loss`/`val_roc_auc`/etc. en tiempo real (actualizan cada época en
centralizado, cada ronda en federado), el config completo del experimento (sin secretos — `WandbConfig`
no tiene ni puede tener un campo de API key, ver su docstring), y en federado un panel por
`node_<id>/…` con las métricas de cada nodo superpuestas.

---

## Gráficas automáticas

Desde la v0.6.0, al terminar cualquier entrenamiento (centralizado o federado, simulación o gRPC) se
generan automáticamente las gráficas en `runs/<exp>/<name>/plots/` — ya no hace falta correr
`scripts/plot_experiment.py` a mano después. Si `matplotlib` no está disponible o el ploteo falla por
cualquier razón, solo se loguea un warning; nunca hace fallar la corrida.

Nuevo en v0.6.0 (antes rotas en federado — ver [CHANGELOG.md](../CHANGELOG.md)):

- `nodes_loss_train_val.png` — train loss y val loss lado a lado, una línea por nodo.
- `node_<N>_loss.png` — un archivo por nodo, train (sólida) + val (discontinua) en el mismo eje, para
  ver de un vistazo si ese nodo concreto está sobreajustando.

Para regenerar manualmente (por ejemplo sobre un run viejo, de antes de este cambio):

```bash
python scripts/plot_experiment.py --run-dir runs/<exp>/<name>
```

---

## Hiperparámetros que deben coincidir entre server y client

Si modificas alguno de estos en `exp07/server.yaml`, cámbialo también en `exp07/client.yaml`:

| Campo | Ubicación |
|---|---|
| `federated.rounds` | server + client |
| `training.local_epochs` | server + client |
| `training.scheduler.t_max` | server + client — **debe ser igual a `local_epochs`** |
| `model.freeze_backbone` | server + client |
| `model.unfreeze_layers` | server + client |
| `model.local_unfreeze_at_epoch` | server + client |
| `training.optimizer.lr_head` | server + client |
| `training.optimizer.lr_backbone` | server + client |
