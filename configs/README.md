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
└── legacy/                # Configs anteriores al refactor (no usar en experimentos nuevos)
    ├── fedavg_cbis_ddsm.yaml
    ├── radimagenet_*.yaml
    └── ...
```

---

## Variables de entorno

Definir en la terminal antes de lanzar cualquier contenedor:

```bash
# Raíz del repositorio en el host
REPO="ruta/al/proyecto/Federal-Learning"

# Directorio que contiene Preprocessed_Dataset/ con las imágenes JPG
MAMMO_DATA="/ruta/a/mammo-bench/data"
```

---

## Mounts comunes

Todos los contenedores usan este conjunto base de mounts:

| Mount | Descripción |
|---|---|
| `-v "$REPO/configs:/app/configs:ro"` | YAMLs de configuración |
| `-v "$REPO/manifests:/app/manifests:ro"` | CSVs de splits por dataset |
| `-v "$REPO/weights:/app/weights:ro"` | Checkpoints de pesos preentrenados |
| `-v "$MAMMO_DATA:/app/data:ro"` | Imágenes mamográficas |
| `-v "$REPO/runs:/app/runs"` | Salida de experimentos (escritura) |

El servidor FL no necesita `-v manifests` ni `-v data` (no carga dataset).

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
