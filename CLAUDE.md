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
fedmammobench-evaluate --config configs/<exp>/server.yaml --checkpoint runs/<name>/weights/global_model.pt

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

All YAML configs inherit from `configs/base.yaml` via a `defaults:` key resolved in `configs/loader.py`. The loaded dict is deserialized into typed dataclasses in `configs/schema.py` (which re-exports everything from six sub-modules):

| Section | Module | Key fields |
|---------|--------|-----------|
| `data` / `partitioning` | `data_config.py` | `manifest_path`, `val_fraction`, `scheme` (iid/dirichlet/quantity_skew) |
| `model` | `model_config.py` | `name`, `weight_source`, `freeze_backbone`, `unfreeze_at_epoch`, `local_unfreeze_at_epoch` |
| `training` | `training_config.py` | `local_epochs`, `optimizer`, `scheduler`, `loss`, `save_best_checkpoint`, `best_checkpoint_metric`, `early_stopping_patience` |
| `federated` | `federated_config.py` | `num_clients`, `rounds`, `strategy`, `server_training`, `server_address`, `save_best_checkpoint`, `best_checkpoint_metric`, `early_stopping_patience` |
| `wandb` | `wandb_config.py` | `enabled` (default `true`), `project`, `mode` (online/offline/disabled) — one run per experiment, server/centralized side only |
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
5. **`NodeMetricsRecorder`** (wraps the strategy) captures per-node fit/eval CSVs, per-round timing, and saves `global_model.pt` at the end; if `federated.save_best_checkpoint` is set, it also overwrites `weights/global_best.pt` whenever the tracked weighted-average eval metric improves round-over-round. If `federated.early_stopping_patience` is also set, it raises an internal exception (`EarlyStoppingTriggered`) once that many rounds pass with no improvement — Flower has no native "stop early" hook, so `server.py` catches it around `fl.simulation.start_simulation`/`fl.server.start_server` as a clean, expected stop rather than a crash. Same mechanism centrally via `Trainer.fit(early_stopping_patience=...)`, which just `break`s its own loop. Both require `save_best_checkpoint=True` in the same section (config-level validation enforces this).

Two frozen-backbone subtleties in that loop (both fixed 2026-07-08, regression tests in `tests/test_audit_fixes.py`):
- **BatchNorm drift under freeze.** `model.train()` (called every epoch) re-enables *all* modules including frozen-backbone BN layers, which keep updating `running_mean`/`running_var` even though `requires_grad=False` stops γ/β from updating. `Trainer` re-pins BN layers with frozen affine params back to `eval()` after each `train()` call (`_freeze_bn_running_stats`) — otherwise per-client BN stats drift independently and get corrupted on aggregation.
- **Cyclic unfreeze needs a new optimizer param group.** When `local_unfreeze_at_epoch` triggers a mid-round unfreeze, the optimizer was built while those layers were still frozen and won't otherwise receive their params — `FedMammoBenchClient.fit` (`federated/client.py`) diffs newly-`requires_grad` params against the optimizer's existing param IDs and calls `optimizer.add_param_group(...)` for the new ones (using `optimizer.lr_backbone` if set).

### Outputs

**Training runs** write under `runs/<name>/`:
- `server_federated_metrics.csv` — primary federated metric (weighted avg across nodes)
- `server_metrics.csv` — centralized evaluation on server holdout (opt-in via `data.name != none`)
- `server_timing.csv` / `timing_summary.csv` — per-round and total wall times
- `weights/global_model.pt` — final aggregated checkpoint (federated, always written)
- `weights/global_best.pt` — best-round checkpoint (federated, opt-in via `federated.save_best_checkpoint`; server never reloads this itself, evaluate it post-hoc)
- `weights/final.pt` / `weights/best.pt` — last-epoch / best-epoch checkpoint (centralized, `best.pt` opt-in via `training.save_best_checkpoint`; `scripts/run_centralized.py` reloads `best.pt` before test-set evaluation when enabled)
- `clients/client_<id>/fit_metrics.csv`, `eval_metrics.csv` — per-node local view
- `tb/` — TensorBoard event files
- `wandb/` — local W&B run data, only if `wandb.enabled` (default `true`); populated even offline, sync later with `wandb sync`
- `plots/` — PNGs auto-generated at the end of every run via `fedmammobench.plotting.autoplot` (never fails the run; see `scripts/plot_experiment.py` to regenerate by hand)

**Post-hoc evaluation** (via `scripts/run-eval-queue.sh` or `fedmammobench-evaluate --output-dir`) writes under `runs/<exp>/eval/<config_name>/`:
- `run.log` — evaluation logs
- `metrics.json` — summary metrics (accuracy, precision, recall, auc, etc.)
- `predictions.csv` — per-sample predictions (if `--predictions-out` was passed)

**Orchestration logs** (multi-experiment queues) live in centralized directories:
- `runs/_logs/queue/queue_<timestamp>.log` — master log for `scripts/run-queue.sh` (training queue)
- `runs/_logs/eval/eval_<timestamp>.log` — master log for `scripts/run-eval-queue.sh` (evaluation queue)

### Transfer learning / weight sources

`model.weight_source` controls how pretrained weights are injected (in `models/weight_loaders/`):
- `auto` (default) — infers from the legacy `model.pretrained` bool: `True` → `imagenet`, `False` → `none`. Explicit `weight_source` takes precedence.
- `imagenet` — torchvision defaults
- `radimagenet` — requires `$FEDMAMMOBENCH_RADIMAGENET_DIR` env var pointing to downloaded checkpoints
- `custom` — `model.checkpoint_path` to a `.pt` file (used for warm-start from a pretrain run)
- `none` — random init (ablation)

**Checkpoint key-namespace normalization.** `save_checkpoint` serializes the **full wrapper** model, so the project's own `.pt` files (`final.pt`, `global_model.pt`) carry keys prefixed `backbone.` (320/320 for resnet50), while a bare-backbone checkpoint doesn't. The `custom` loader (`weight_loaders/custom.py::_match_state_dict_prefix`) tries the state_dict as-is and under `module.`/`backbone.` transforms, keeping whichever maximizes key overlap with the target module, and raises `RuntimeError` if 0 tensors end up matching. Until 2026-07-08 this normalization didn't exist: the checkpoint loaded straight into `model.backbone` (bare keys expected) with no fallback, so a `backbone.`-prefixed checkpoint silently matched 0/320 tensors under `strict_load: false` and the federated global model trained from random init (~0.5 AUC vs. ~0.82 centralized) — see `tests/test_audit_fixes.py::TestCustomWarmStartLoader` for the regression coverage. The `radimagenet`/`imagenet` loaders were never affected — they consume backbone-only checkpoints and remap keys directly. Post-hoc `run_evaluation.py` was never affected either: it re-loads via `load_checkpoint(--checkpoint, model)` into the full wrapper with `strict=True`.

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
