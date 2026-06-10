# FedMammo — Federated Learning for Mammography Classification

Federated learning framework for binary mammography classification (benign / malignant) using [Flower](https://flower.ai/) + PyTorch + Ray.

**Version**: 0.2.0 | **Python**: 3.11 | **License**: MIT

---

## Quick Start (5 minutes)

```bash
# 1. Install
git clone <repo>
cd Federal-Learning
pip install -e .

# 2. Smoke test — synthetic data, CPU only (~1 minute)
fedmammo-federated --config configs/fedavg_synthetic.yaml
```

Artifacts (metrics, TensorBoard logs, config snapshot) are written to `runs/fedavg_synthetic/`.

---

## Execution Modes

| Command | Config | Description | Estimated time |
|---------|--------|-------------|----------------|
| `fedmammo-centralized` | `centralized_synthetic.yaml` | Centralized baseline, synthetic | < 1 min |
| `fedmammo-federated` | `fedavg_synthetic.yaml` | FL simulation (Ray), synthetic | < 1 min |
| `fedmammo-federated` | `radimagenet_resnet50_fedavg.yaml` | FL simulation with RadImageNet | 10–30 min |
| `python scripts/run_server.py` | `radimagenet_resnet50_grpc_server_v2.yaml` | Real gRPC server | blocks |
| `python scripts/run_client.py` | `radimagenet_resnet50_grpc_client_v2.yaml` | Real gRPC client | blocks |
| `fedmammo-evaluate` | any | Post-hoc evaluation on a checkpoint | < 1 min |

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
fedmammo-federated --config configs/my_experiment.yaml
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
from fedmammo.configs import load_config
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

## Transfer Learning

| `weight_source` | Requires | Best for |
|-----------------|----------|---------|
| `imagenet` | — | RGB natural images; baseline |
| `radimagenet` | `$FEDMAMMO_RADIMAGENET_DIR` env var | Grayscale medical images |
| `custom` | `checkpoint_path` | Fine-tuning from your own checkpoint |
| `none` | — | Ablation: random init |

```bash
export FEDMAMMO_RADIMAGENET_DIR=/path/to/radimagenet/checkpoints
fedmammo-federated --config configs/radimagenet_resnet50_fedavg.yaml
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
fedmammo-centralized --config configs/radimagenet_resnet50_centralized.yaml

# Federated FedAvg (primary)
fedmammo-federated --config configs/radimagenet_resnet50_fedavg.yaml

# Federated DenseNet121
fedmammo-federated --config configs/radimagenet_densenet121_fedavg.yaml
```

All runs use `seed: 42` (set in `base.yaml`).  Effective configs are snapshotted
to `runs/<name>/config.snapshot.yaml`.

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

---

## Project Structure

```
src/fedmammo/
├── configs/           Per-section config modules (each with validate())
├── datasets/          Dataset loaders and patient-aware partitioning
├── federated/         Flower client, server, and strategy implementations
├── models/            Architecture builders and weight loaders
├── training/          Unified trainer (centralized + federated)
├── evaluation/        Clinical binary classification metrics
└── utils/             Logging, checkpointing, TensorBoard, seeding
```
