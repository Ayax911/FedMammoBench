# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable + dev tools)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_config_modules.py -v

# Linting / formatting
ruff check src/ tests/
ruff format src/ tests/
black src/ tests/
mypy src/

# Run a federated simulation
fedmammobench-federated --config configs/exp07/server.yaml

# Run a centralized baseline
fedmammobench-centralized --config configs/exp08/centralized.yaml

# Post-hoc evaluation on a checkpoint
fedmammobench-evaluate --config configs/<exp>/server.yaml --checkpoint runs/<name>/global_model.pt

# Multi-device gRPC mode (manual)
python scripts/run_server.py --config configs/exp07/server.yaml   # on the aggregation server
python scripts/run_client.py --config configs/exp07/client.yaml \
  --server 192.168.1.10:8080 --client-id 0 \
  --manifest manifests/node0_manifest.csv --data-dir data/       # on each node

# Docker federated deployment (automated, all in containers)
scripts/docker-deploy-federated.sh exp14                          # Launch server + 5 clients
scripts/docker-deploy-federated.sh exp14 --monitor                # Monitor until Round 1 completes
scripts/docker-deploy-federated.sh exp14 --no-clean               # Skip cleanup of previous containers

# Plot experiment metrics
python scripts/plot_experiment.py runs/<name>/
```

Python 3.11 is required (strict `>=3.11,<3.12`). The venv at `./.venv/` is already configured. Runtime deps are pinned in `requirements.txt` (not `pyproject.toml`, which carries only loose constraints); for GPU boxes install `torch`/`torchvision` from the CUDA wheel index *before* `pip install -r requirements.txt`.

## Architecture

### Execution modes

The CLI exposes three entry points (`pyproject.toml [project.scripts]`):
- `fedmammobench-federated` → `cli/federated.py` → `federated/server.py:run_simulation` (Ray-based, all clients in-process)
- `fedmammobench-centralized` → `cli/centralized.py` (single-node training loop)
- `fedmammobench-evaluate` → `cli/evaluate.py` (loads checkpoint, runs `Evaluator`)

The `federated/server.py` also exposes `run_grpc_server` for real multi-device deployment via `scripts/run_server.py` + `scripts/run_client.py`. The simulation and gRPC paths share the same strategy/logging wiring; the difference is whether Flower spawns virtual clients or waits for physical ones over the network.

### Config system

All YAML configs inherit from `configs/base.yaml` via a `defaults:` key resolved in `configs/loader.py`. The loaded dict is deserialized into typed dataclasses in `configs/schema.py` (which re-exports everything from five sub-modules):

| Section | Module | Key fields |
|---------|--------|-----------|
| `data` / `partitioning` | `data_config.py` | `manifest_path`, `val_fraction`, `scheme` (iid/dirichlet/quantity_skew) |
| `model` | `model_config.py` | `name`, `weight_source`, `freeze_backbone`, `unfreeze_at_epoch`, `local_unfreeze_at_epoch` |
| `training` | `training_config.py` | `local_epochs`, `optimizer`, `scheduler`, `loss` |
| `federated` | `federated_config.py` | `num_clients`, `rounds`, `strategy`, `server_training`, `server_address` |
| experiment + evaluation | `experiment.py` | cross-section validation (freeze reachability, preset↔channels) |

Call `cfg.validate()` after loading to catch bad combinations early. Each section has its own `validate()`; cross-section rules live in `ExperimentConfig.validate()`.

**Important:** `schema.py` re-exports all public names. `loader.py` resolves dataclass types via `vars(schema_mod)`, so any new public symbol must be re-exported there or YAML loading breaks.

### Registry pattern

Three registries follow the same decorator pattern — new entries self-register at import time via side-effect imports in the package `__init__.py`:

- **Strategies** (`federated/strategies/registry.py`): `@register_strategy("name")` on a builder function; reference impl is `fedavg.py`. Add import in `strategies/__init__.py`.
- **Models** (`models/factory.py`): `@register_model("name")` returning an `nn.Module` with correct `in_channels`/`num_classes`/`dropout`. Add name to `Literal` in `model_config.py`. Add import in `models/__init__.py`.
- **Datasets** (`datasets/registry.py`): `@register_dataset("name")` on a builder `(cfg, train_tx, eval_tx) → dict[str, MammographyDataset]`. No `Literal` update needed. Add import in `datasets/__init__.py`.

### FL training loop

Each federated round (simulation or gRPC):
1. **Server** calls `on_fit_config_fn` to broadcast `{current_round, local_epochs, ...}` to selected clients.
2. **Client** (`FedMammoBenchClient.fit`): loads server parameters (strict), runs `apply_freeze_policy`, optionally applies cyclic within-round unfreeze at `local_unfreeze_at_epoch`, trains for `local_epochs`, returns updated weights + metrics.
3. **Strategy** (`aggregate_fit`) averages weights. If `server_training.enabled`, `attach_server_training` wraps `aggregate_fit` to run a server-side training step afterwards (`new_global = (1-w)*aggregated + w*server_trained`).
4. **Clients** evaluate the aggregated model on their local val split; strategy `aggregate_evaluate` weighted-averages → logged to `server_federated_metrics.csv`.
5. **`NodeMetricsRecorder`** (wraps the strategy) captures per-node fit/eval CSVs, per-round timing, and saves `global_model.pt` at the end.

### Outputs

Every run writes under `runs/<name>/`:
- `server_federated_metrics.csv` — primary federated metric (weighted avg across nodes)
- `server_metrics.csv` — centralized evaluation on server holdout (opt-in via `data.name != none`)
- `server_timing.csv` / `timing_summary.csv` — per-round and total wall times
- `global_model.pt` — final aggregated checkpoint
- `clients/client_<id>/fit_metrics.csv`, `eval_metrics.csv` — per-node local view
- `tb/` — TensorBoard event files

### Transfer learning / weight sources

`model.weight_source` controls how pretrained weights are injected (in `models/weight_loaders/`):
- `imagenet` — torchvision defaults
- `radimagenet` — requires `$FEDMAMMOBENCH_RADIMAGENET_DIR` env var pointing to downloaded checkpoints
- `custom` — `model.checkpoint_path` to a `.pt` file (used for warm-start from a pretrain run)
- `none` — random init (ablation)

**⚠️ Checkpoint key-namespace gotcha (causes federated collapse to ~chance).** `save_checkpoint` serializes the **full wrapper** model, so the project's own `.pt` files (`final.pt`, `global_model.pt`) carry keys prefixed `backbone.` (320/320 for resnet50). But the `custom` loader (`weight_loaders/custom.py`) loads into `model.backbone`, which expects **bare** keys (`conv1.weight`, …) — a total mismatch. With `strict_load: false` this fails **silently** (0 tensors loaded) and the model keeps random init, so `custom` warm-start from a pretrain checkpoint is a no-op and the federated global model trains from scratch (~0.5 AUC while centralized `radimagenet` reaches ~0.82). The misleading `LoadReport ... missing=0 unexpected=0` log line is unpopulated and hides it; the real signal is the `Missing keys ['conv1.weight'...]` / `Unexpected keys ['backbone.conv1.weight'...]` warnings just above it. **Guard:** set `strict_load: true` for `custom` warm-start (or fix `custom.py` to normalize the `backbone.` prefix and verify missing/unexpected keys). The `radimagenet`/`imagenet` loaders are unaffected — they consume backbone-only checkpoints and remap keys. Post-hoc `run_evaluation.py` is unaffected too: it re-loads via `load_checkpoint(--checkpoint, model)` into the full wrapper with `strict=True`.

### Experiment configs layout

Per-experiment configs live under `configs/exp<NN>/`. Each experiment directory typically has:
- `server.yaml` — used for the aggregation server (simulation or gRPC)
- `client.yaml` — used on each physical node in gRPC mode
- `pretrain.yaml` — centralized pre-training that generates `final.pt` for warm-start

Legacy flat configs are in `configs/legacy/`.

## Extension checklist

When adding a strategy, model, or dataset, see `docs/EXTENDING.md`. The short version:
1. Write the module with `@register_*` decorator.
2. Add the side-effect import in the package `__init__.py`.
3. For models: update the `Literal` in `model_config.py`.
4. Add a test; run `pytest tests/ -v`.
5. If RNG order, defaults, `Trainer` signature, or aggregation math changed — bump version in `pyproject.toml` and add a `CHANGELOG.md` entry.
