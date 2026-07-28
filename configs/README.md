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

## Selección de mejor checkpoint (solo centralizado)

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
`val_roc_auc`/`val_loss` no se degradaron antes de confiar en el AUC de test final.

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
