# FedMammoBench — Federated Learning for Mammography Classification

Federated learning framework for binary mammography classification (benign / malignant) using [Flower](https://flower.ai/) + PyTorch + Ray.

**Version**: 0.3.0 | **Python**: 3.11 | **License**: MIT

---

## Quick Start (5 minutes)

```bash
# 1. Install
git clone <repo>
cd Federal-Learning
pip install -e .

# 2. Prepare a dataset manifest (see docs/DATA_PREPARATION.md), then run a
#    federated simulation. Point data.manifest_path / data.image_root at your data.
fedmammobench-federated --config configs/fedavg_cbis_ddsm.yaml
```

Artifacts (metrics, TensorBoard logs, config snapshot) are written to `runs/<config-name>/`.

---

## Execution Modes

| Command | Config | Description | Estimated time |
|---------|--------|-------------|----------------|
| `fedmammobench-centralized` | `radimagenet_resnet50_centralized.yaml` | Centralized baseline | depends on data |
| `fedmammobench-federated` | `fedavg_cbis_ddsm.yaml` | FL simulation (Ray) | depends on data |
| `fedmammobench-federated` | `radimagenet_resnet50_fedavg.yaml` | FL simulation with RadImageNet | 10–30 min |
| `fedmammobench-federated` | `fedavg_server_training.yaml` | Hybrid FL (server also trains) | depends on data |
| `python scripts/run_server.py` | `radimagenet_resnet50_grpc_server_v2.yaml` | Real gRPC server | blocks |
| `python scripts/run_client.py` | `radimagenet_resnet50_grpc_client_v2.yaml` | Real gRPC client | blocks |
| `fedmammobench-evaluate` | any | Post-hoc evaluation on a checkpoint | < 1 min |

---

## Configuration

All configs inherit from `configs/base.yaml` via `defaults: base.yaml`.
Override only what changes:

```yaml
# configs/my_experiment.yaml
defaults: base.yaml
name: my_experiment
model:
  name: resnet50
  weight_source: radimagenet
federated:
  rounds: 20
  num_clients: 4
```

**Run it:**
```bash
fedmammobench-federated --config configs/my_experiment.yaml
```

### Key configuration sections

| Section | Key fields | Module |
|---------|-----------|--------|
| `data` | `name`, `manifest_path`, `val_fraction` | `data_config.py` |
| `partitioning` | `scheme` (iid/dirichlet/quantity_skew), `alpha` | `data_config.py` |
| `model` | `name`, `weight_source`, `freeze_backbone`, `unfreeze_at_epoch` | `model_config.py` |
| `training` | `epochs`, `local_epochs`, `optimizer.lr`, `mixed_precision` | `training_config.py` |
| `federated` | `num_clients`, `rounds`, `strategy.name`, `strategy.params` | `federated_config.py` |

### Config validation

Call `ExperimentConfig.validate()` after loading to catch errors early:

```python
from fedmammobench.configs import load_config
cfg = load_config("configs/radimagenet_resnet50_fedavg.yaml")
cfg.validate()  # raises ValueError or emits warnings on bad combinations
```

---

## Federated Strategies

| Strategy | YAML | Description |
|----------|------|-------------|
| FedAvg | `strategy.name: fedavg` | Standard weighted averaging |
| FedProx | `strategy.name: fedprox` + `strategy.params: {mu: 0.01}` | Proximal regularization for non-IID |
| SCAFFOLD | `strategy.name: scaffold` | Variance reduction via control variates |
| FedBN | `strategy.name: fedbn` | Batch-norm layers not aggregated |

---

## Hybrid Server-Side Training

By default the central node only aggregates client updates. Enable
`federated.server_training` to also train on a dataset the server owns: after
each round's aggregation, the server continues training from the aggregated
weights for `local_epochs` and the result becomes the new global model.

```yaml
federated:
  server_training:
    enabled: true
    manifest_path: /path/to/server_manifest.csv   # the server's own data
    image_root: /path/to/server_images
    local_epochs: 1
    server_weight: 1.0   # new = (1-w)*aggregated + w*server_trained
```

This composes with any strategy above. See `configs/fedavg_server_training.yaml`.

---

## Transfer Learning

| `weight_source` | Requires | Best for |
|-----------------|----------|---------|
| `imagenet` | — | RGB natural images; baseline |
| `radimagenet` | `$FEDMAMMOBENCH_RADIMAGENET_DIR` env var | Grayscale medical images |
| `custom` | `checkpoint_path` | Fine-tuning from your own checkpoint |
| `none` | — | Ablation: random init |

```bash
export FEDMAMMOBENCH_RADIMAGENET_DIR=/path/to/radimagenet/checkpoints
fedmammobench-federated --config configs/radimagenet_resnet50_fedavg.yaml
```

See [docs/TRANSFER_LEARNING_GUIDE.md](docs/TRANSFER_LEARNING_GUIDE.md) for setup instructions.

---

## Multi-Node Deployment (gRPC)

```bash
# On the aggregation server:
python scripts/run_server.py \
    --config configs/radimagenet_resnet50_grpc_server_v2.yaml

# On each client node (different machines):
python scripts/run_client.py \
    --config configs/radimagenet_resnet50_grpc_client_v2.yaml \
    --server 192.168.1.10:8080 \
    --client-id 0 \
    --manifest /path/to/node0_manifest.csv \
    --data-dir /path/to/images
```

See [docs/FEDERATED_DEPLOYMENT_GUIDE.md](docs/FEDERATED_DEPLOYMENT_GUIDE.md) for the full topology.

---

## Reproducing Article Results

```bash
# Centralized baseline
fedmammobench-centralized --config configs/radimagenet_resnet50_centralized.yaml

# Federated FedAvg (primary)
fedmammobench-federated --config configs/radimagenet_resnet50_fedavg.yaml

# Federated DenseNet121
fedmammobench-federated --config configs/radimagenet_densenet121_fedavg.yaml
```

All runs use `seed: 42` (set in `base.yaml`).  Effective configs are snapshotted
to `runs/<name>/config.snapshot.yaml`.

---

## Research Experiments (per-experiment configs)

Structured experiments live under `configs/expNN/`. Each directory contains
`server.yaml`, `client.yaml`, and optionally `pretrain.yaml`.

| Exp | Mode | Description |
|-----|------|-------------|
| exp07 | Federated | FedAvg + ResNet50, warm-start from DDSM pretrain, 5 nodes, 15 epochs/round, 7 rounds |
| exp08 | Centralized | Baseline centralizado con todos los datasets (17 139 imgs) |
| exp09 | Federated | FedAvg + ResNet50 sin warm-start (ablación desde RadImageNet) |
| exp10 | Centralized | Variante centralizada de ablación |
| exp12 | Federated | FedAvg + ResNet50, igual a exp07 con métricas por época por nodo y predicciones enriquecidas |

### exp12 — FedAvg + warm-start + métricas extendidas

exp12 replica exp07 (15 épocas locales, 7 rondas, warm-start desde DDSM) y añade:
- **`clients/client_<id>/epoch_metrics.csv`** — train_loss, task_loss, val_loss, val_auc, val_f1, val_sensitivity, val_specificity por cada época local dentro de cada ronda (equivalente al CSV de entrenamiento centralizado).
- **`clients/client_<id>/predictions.csv`** — predicciones enriquecidas con las columnas del manifest de cada nodo (`image_path`, `patient_id`, y columnas extra) para cada muestra del conjunto de validación.

```bash
# Paso 1: pretrain centralizado con DDSM (genera runs/exp12_pretrain_ddsm/final.pt)
python scripts/run_centralized.py --config configs/exp12/pretrain.yaml

# Paso 2: FL distribuido (gRPC)
# En el servidor:
python scripts/run_server.py --config configs/exp12/server.yaml

# En cada nodo (N = 1..5):
python scripts/run_client.py --config configs/exp12/client.yaml \
    --server <SERVER_IP>:8080 \
    --client-id <N> \
    --manifest manifests/node<N>_manifest.csv
```

Para lanzar desde GitHub Actions (reemplaza exp07 por exp12 en el batch):

```
workflow: Run Batch Experiments
  experiments: exp12_fedavg_resnet50
```

---

## Outputs, Per-Node Metrics & Timing

Every federated run writes the following under `runs/<name>/`:

```
runs/<name>/
├── config.snapshot.yaml          # effective config used
├── server_federated_metrics.csv  # GLOBAL model, federated-averaged across nodes
├── server_metrics.csv            # GLOBAL model on the server holdout (if any)
├── server_timing.csv             # per-round wall time, sum/max/mean node fit time
├── timing_summary.csv            # total process time, rounds, avg s/round
├── global_model.pt               # final aggregated GLOBAL model checkpoint
├── tb/                           # server-level TensorBoard
└── clients/
    ├── client_0/
    │   ├── fit_metrics.csv        # per-round train_loss, task_loss, fit_seconds
    │   ├── eval_metrics.csv       # per-round accuracy, f1, roc_auc, eval_seconds
    │   └── tb/                    # per-node TensorBoard
    └── client_1/ ...
```

- **Per-node metrics** (`clients/client_<id>/`) are each node's *local* view:
  training loss and the metrics from evaluating the global model on that node's
  own validation split, with the node's wall-clock `fit_seconds` / `eval_seconds`.
- **Timing**: `server_timing.csv` records, per round, the server wall time plus
  the **sum** (total compute across nodes), **max** (straggler / critical path),
  and **mean** of node training times. `timing_summary.csv` records the overall
  process time. The first round's wall time includes Ray/engine startup.

### Verifying the global model

Per-node metrics measure *local* performance — they are **not** the global
model's generalization. The global model is judged by:

1. `server_federated_metrics.csv` — the weighted average of every node's
   evaluation of the global model (the primary federated verdict).
2. `server_metrics.csv` — the global model on a server-owned holdout, when one is
   configured (`data.name` ≠ `none` with a test split).
3. The saved `global_model.pt`, which you can verify post-hoc on any held-out
   test set:
   ```bash
   fedmammobench-evaluate --config configs/<your>.yaml --checkpoint runs/<name>/global_model.pt
   ```

---

## Data Preparation

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for:
- Downloading and formatting CBIS-DDSM, Mammo-Bench, VinDr-Mammo
- CSV manifest column requirements
- How to verify absence of patient leakage

---

## Scientific Methodology

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for:
- Patient-disjoint split protocol
- Federated evaluation metrics and their limitations
- FedAvg vs FedProx comparison guidelines
- Known approximations to disclose in papers

---

## Audit Checklist

See [docs/EXPERIMENT_AUDIT.md](docs/EXPERIMENT_AUDIT.md) for the pre-publication
checklist (data integrity, metric interpretation, strategy comparisons).

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --tb=short
```

CI runs automatically on push to `main` and `feature/**` branches via `.github/workflows/ci.yml`.

**Extending the framework** — adding a strategy, model, dataset, or config field?
See [docs/EXTENDING.md](docs/EXTENDING.md) for the step-by-step pipeline per module.

---

## Project Structure

```
src/fedmammobench/
├── configs/           Per-section config modules (each with validate())
├── datasets/          Dataset loaders and patient-aware partitioning
├── federated/         Flower client, server, and strategy implementations
├── models/            Architecture builders and weight loaders
├── training/          Unified trainer (centralized + federated)
├── evaluation/        Clinical binary classification metrics
└── utils/             Logging, checkpointing, TensorBoard, seeding
```
