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

# Multi-device gRPC mode
python scripts/run_server.py --config configs/exp07/server.yaml   # on the aggregation server
python scripts/run_client.py --config configs/exp07/client.yaml \
  --server 192.168.1.10:8080 --client-id 0 \
  --manifest manifests/node0_manifest.csv --data-dir data/       # on each node

# Plot experiment metrics
python scripts/plot_experiment.py runs/<name>/
```

Python 3.11 is required (strict `>=3.11,<3.12`). The venv at `./venv/` is already configured.

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
