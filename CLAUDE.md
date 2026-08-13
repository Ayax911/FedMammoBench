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
fedmammobench-federated --config configs/<exp>/server.yaml

# Run a centralized baseline
fedmammobench-centralized --config configs/<exp>/centralized.yaml

# Post-hoc evaluation on a checkpoint
fedmammobench-evaluate --config configs/<exp>/server.yaml --checkpoint runs/<name>/weights/global_model.pt

# Multi-device gRPC mode (manual)
python scripts/run_server.py --config configs/<exp>/server.yaml   # on the aggregation server
python scripts/run_client.py --config configs/<exp>/client.yaml \
  --server 192.168.1.10:8080 --client-id 0 \
  --manifest manifests/node0_manifest.csv --data-dir data/       # on each node

# Resume a crashed run (re-run the SAME command with --resume added)
python scripts/run_centralized.py --config configs/<exp>/centralized.yaml --resume
python scripts/run_federated.py --config configs/<exp>/server.yaml --resume
python scripts/run_server.py --config configs/<exp>/server.yaml --resume

# Docker federated deployment (automated, all in containers)
scripts/docker-deploy-federated.sh <exp>                          # Launch server + 5 clients
scripts/docker-deploy-federated.sh <exp> --monitor                # Monitor until Round 1 completes
scripts/docker-deploy-federated.sh <exp> --no-clean               # Skip cleanup of previous containers

# Docker compose wrapper (server + node0 + node1 on this or separate hosts; reads .env)
./run.sh build --gpu
./run.sh up --gpu                 # full local experiment
./run.sh server --gpu -d          # only the aggregation server, background
SERVER_ADDRESS=192.168.1.10:8080 ./run.sh node 0 --gpu   # a node on another host
./run.sh logs node0 ; ./run.sh ps ; ./run.sh down

# Sequential multi-experiment queues (Docker, one after another; a failure does not stop the queue)
scripts/run-queue.sh exp70 exp71 exp72
scripts/run-queue.sh --file queue.txt        # one exp per line, '#' = comment
scripts/run-eval-queue.sh                    # post-hoc evaluation queue

# Plot experiment metrics
python scripts/plot_experiment.py runs/<name>/
```

Python 3.11 is required (strict `>=3.11,<3.12`). **There is no usable local interpreter on this
machine** — `./venv/` is Python 3.12 and does not have `fedmammobench` installed, and no `python3.11`
is on `PATH`. Real training/evaluation therefore runs in Docker (`Dockerfile` pins
`python:3.11-slim-bookworm` and symlinks `python3.11` to `python`; override `BASE_IMAGE` for CUDA).
Don't assume `pytest` or the `fedmammobench-*` entry points can be invoked directly on the host —
check first, or run them inside the image. Runtime deps are pinned in `requirements.txt` (not
`pyproject.toml`, which carries only loose constraints); for GPU boxes install `torch`/`torchvision`
from the CUDA wheel index *before* `pip install -r requirements.txt`.

CI (`.github/workflows/ci.yml`) runs `pytest tests/ -v --tb=short --cov=fedmammobench` on Python 3.11
plus a config-import smoke check, on every push to `main`/`feature/**` and every PR to `main`. CI covers
`src/` and `tests/` only — nothing validates `configs/**/*.ipynb`.

None of the commands above apply to the notebook experiments (`configs/exp*/*.ipynb`) — those are run
cell-by-cell in Jupyter on the lab workstation's GPU, outside Docker and outside this package. See
"Notebook experiments" below.

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
| `training.augmentation` | `training_config.py` | `horizontal_flip`, `rotate_limit`, `normalize_preset`/`normalize_mean`/`normalize_std`, `resize_mode` (`squash` default / `letterbox`) |
| `federated` | `federated_config.py` | `num_clients`, `rounds`, `strategy`, `server_training`, `server_address`, `save_best_checkpoint`, `best_checkpoint_metric`, `early_stopping_patience` |
| `wandb` | `wandb_config.py` | `enabled` (default `true`), `project`, `mode` (online/offline/disabled) — one run per experiment, server/centralized side only |
| experiment + evaluation | `experiment.py` | cross-section validation (freeze reachability, preset↔channels) |

Call `cfg.validate()` after loading to catch bad combinations early. Each section has its own `validate()`; cross-section rules live in `ExperimentConfig.validate()`.

**Important:** `schema.py` re-exports all public names. `loader.py` resolves dataclass types via `vars(schema_mod)`, so any new public symbol must be re-exported there or YAML loading breaks.

### Registry pattern

Three registries follow the same decorator pattern — new entries self-register at import time via side-effect imports in the package `__init__.py`:

- **Strategies** (`federated/strategies/registry.py`): `@register_strategy("name")` on a builder function; reference impl is `fedavg.py`. Add import in `strategies/__init__.py`.
  Registered: `fedavg`, `fedprox`, `fedbn`, `scaffold`, `fedadam`, `fedyogi`.
- **Models** (`models/factory.py`): `@register_model("name")` returning an `nn.Module` with correct `in_channels`/`num_classes`/`dropout`. Add name to `Literal` in `model_config.py`. Add import in `models/__init__.py`.
  Registered: `resnet18` (default), `resnet50`, `efficientnet_b0`, `densenet121`, `inception_v3`.
- **Datasets** (`datasets/registry.py`): `@register_dataset("name")` on a builder `(cfg, train_tx, eval_tx) → dict[str, MammographyDataset]`. No `Literal` update needed (`data.name` is a plain `str`). Add import in `datasets/__init__.py`.
  Registered: `cbis_ddsm`, `vindr_mammo`, `mammo_bench`, plus the sentinel `none` (server owns no local dataset).

Shared model plumbing lives outside the per-arch modules, so a new backbone only has to produce the
feature extractor:
- `models/_head.py::build_head` — the configurable MLP head from `model.head`; always ends in
  `Linear(*, num_classes)` emitting **raw logits** (softmax/sigmoid belongs to the loss and to
  `Evaluator`, never to the model).
- `models/_adapt.py` — 1- vs 3-channel input stem. `adapt_first_conv` rewrites the module (model
  builders); `adapt_weight_tensor` rewrites the tensor (weight loaders). Both default to
  `sum_preserving` so activation magnitude survives the channel change.

### FL training loop

Each federated round (simulation or gRPC):
1. **Server** calls `on_fit_config_fn` to broadcast `{current_round, local_epochs, ...}` to selected clients.
2. **Client** (`FedMammoBenchClient.fit`): loads server parameters (strict), runs `apply_freeze_policy`, optionally applies cyclic within-round unfreeze at `local_unfreeze_at_epoch`, trains for `local_epochs`, returns updated weights + metrics.
3. **Strategy** (`aggregate_fit`) averages weights. If `server_training.enabled`, `attach_server_training` wraps `aggregate_fit` to run a server-side training step afterwards (`new_global = (1-w)*aggregated + w*server_trained`).
4. **Clients** evaluate the aggregated model on their local val split; strategy `aggregate_evaluate` weighted-averages → logged to `server_federated_metrics.csv`.
5. **`NodeMetricsRecorder`** (wraps the strategy) captures per-node fit/eval CSVs, per-round timing, and saves `global_model.pt` at the end; if `federated.save_best_checkpoint` is set, it also overwrites `weights/global_best.pt` whenever the tracked weighted-average eval metric improves round-over-round. If `federated.early_stopping_patience` is also set, it raises an internal exception (`EarlyStoppingTriggered`) once that many rounds pass with no improvement — Flower has no native "stop early" hook, so `server.py` catches it around `fl.simulation.start_simulation`/`fl.server.start_server` as a clean, expected stop rather than a crash. Same mechanism centrally via `Trainer.fit(early_stopping_patience=...)`, which just `break`s its own loop. Both require `save_best_checkpoint=True` in the same section (config-level validation enforces this).

### Resume after a crash

`--resume` on `run_centralized.py`/`run_federated.py`/`run_server.py` (v0.8.0) — no new config fields.
On a shared workstation, a hard crash (segfault, OOM-killer, CUDA driver reset) skips Python's
`finally` blocks entirely, so the existing end-of-run checkpoint saves don't help; `--resume` instead
overwrites `weights/final.pt` (centralized, + optimizer/scheduler state) or `weights/global_model.pt`
(federated) every epoch/round as a safety net, and reloads from it on the next invocation. `training.epochs`/
`federated.rounds` are the TOTAL budget across the original run + resumes. Federated resume can't ask
Flower to start its round counter above 1, so it runs only the remaining rounds and applies a
`round_offset` everywhere a round number is logged or sent to clients (never to Flower's own strategy
calls). See `configs/README.md`'s "Resume tras un crash" section for the full mechanism and limitations
(GradScaler state, mid-epoch/round crashes, `num_clients`/`partitioning` drift between resumes).

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

**Notebook runs** (`configs/exp*/*.ipynb`, see below) share `runs/` but use a different, flatter layout
— don't expect the files above:
- `metrics.json` — test-set summary (accuracy, precision, recall, f1, roc_auc) + the run's hyperparameters
- `loss_history.csv` — per-epoch train/val loss
- `plots/loss_curve.png`, `plots/test_confusion_matrix.png`, `plots/test_metric_<name>.png`
- `best_model.pth` — early-stopping checkpoint

Unlike package runs, these notebook outputs **are committed** (`.gitignore` only excludes `*.pt`/`*.pth`),
so `runs/` doubles as the results record for the notebook series.

**Post-hoc evaluation** (via `scripts/run-eval-queue.sh` or `fedmammobench-evaluate --output-dir`) writes under `runs/<exp>/eval/<config_name>/`:
- `run.log` — evaluation logs
- `metrics.json` — summary metrics (accuracy, precision, recall, auc, etc.)
- `predictions.csv` — per-sample predictions (if `--predictions-out` was passed)

**Orchestration logs** (multi-experiment queues) live in centralized directories:
- `runs/_logs/queue/queue_<timestamp>.log` — master log for `scripts/run-queue.sh` (training queue)
- `runs/_logs/eval/eval_<timestamp>.log` — master log for `scripts/run-eval-queue.sh` (evaluation queue)

### Preprocessing reproducibility caveat (v0.9.0)

`build_transforms()` (`datasets/transforms.py`) switched its resize interpolation from Albumentations'
default bilinear to `cv2.INTER_AREA`. Every image in this benchmark is larger than `data.image_size`,
so the resize is always a downscale — but this silently changed resized pixel values for *every*
config, past and future. Historical results (exp07–exp61+) stand as a record of what was run, yet
re-running the same YAML today does not reproduce them pixel-for-pixel. Before attributing a metric
delta to a code change, check whether the comparison straddles 0.9.0.

### Transfer learning / weight sources

`model.weight_source` controls how pretrained weights are injected (in `models/weight_loaders/`):
- `auto` (default) — infers from the legacy `model.pretrained` bool: `True` → `imagenet`, `False` → `none`. Explicit `weight_source` takes precedence.
- `imagenet` — torchvision defaults
- `radimagenet` — requires `$FEDMAMMOBENCH_RADIMAGENET_DIR` env var pointing to downloaded checkpoints
- `custom` — `model.checkpoint_path` to a `.pt` file (used for warm-start from a pretrain run)
- `none` — random init (ablation)

**Checkpoint key-namespace normalization.** `save_checkpoint` serializes the **full wrapper** model, so the project's own `.pt` files (`final.pt`, `global_model.pt`) carry keys prefixed `backbone.` (320/320 for resnet50), while a bare-backbone checkpoint doesn't. The `custom` loader (`weight_loaders/custom.py::_match_state_dict_prefix`) tries the state_dict as-is and under `module.`/`backbone.` transforms, keeping whichever maximizes key overlap with the target module, and raises `RuntimeError` if 0 tensors end up matching. Until 2026-07-08 this normalization didn't exist: the checkpoint loaded straight into `model.backbone` (bare keys expected) with no fallback, so a `backbone.`-prefixed checkpoint silently matched 0/320 tensors under `strict_load: false` and the federated global model trained from random init (~0.5 AUC vs. ~0.82 centralized) — see `tests/test_audit_fixes.py::TestCustomWarmStartLoader` for the regression coverage. The `radimagenet`/`imagenet` loaders were never affected — they consume backbone-only checkpoints and remap keys directly. Post-hoc `run_evaluation.py` was never affected either: it re-loads via `load_checkpoint(--checkpoint, model)` into the full wrapper with `strict=True`.

### Experiment configs layout

`configs/exp<NN>/` holds one experiment each, but the directories come in **two unrelated flavors** —
check what's actually inside before assuming:

- **YAML experiments** (package-driven, everything above applies): `server.yaml` for the aggregation
  server (simulation or gRPC), `client.yaml` for each physical node in gRPC mode, `pretrain.yaml` for
  centralized pre-training that generates `final.pt` for warm-start, and optionally `eval/<name>.yaml`
  for post-hoc evaluation targets (see `configs/exp70/eval/mammo_bench.yaml`).
- **Notebook experiments** (see the next section): a single `.ipynb` and nothing else. These bypass
  `fedmammobench` entirely.

Not every config directory is named `exp<NN>` — one-off YAML experiments get arbitrary names
(`configs/caladam01/`), and those are usually copies of a numbered experiment whose *header comments
still name the original* (`caladam01/server.yaml` opens with `# exp34_fedadam_...`). Trust `name:` /
`output_dir:` in the YAML, not the comment block.

Legacy flat configs are in `configs/legacy/`. Older experiment directories get pruned from the repo
once superseded, so check `ls configs/` for what's currently live rather than trusting a specific
number from history or docs — **and note the numbering has been reused**: `configs/exp01`–`exp09`
today are notebooks, unrelated to the exp01–exp09 YAML experiments that `CHANGELOG.md` and `docs/`
still discuss. `configs/README.md` is the authoritative reference for experiment-specific
mechanics (per-node hyperparameter matching, best-checkpoint selection, resume, W&B, autoplot).

**Three config generators live in `scripts/`; neither output should be hand-edited.**
`gen_lr_backbone_grid.py` emits the exp10–exp19 notebook grid (see next section);
`gen_depth_block_grid.py` emits Block 6 (`exp24`–`exp27`, generated from the `exp23` template — see
next section); `gen_federated_grid.py` emits the federated YAML grid (8 files per experiment:
`server.yaml`, `client.yaml`, `eval/{mammo_bench,node1..5_partition}.yaml`), holding everything
constant but `rounds`, `local_epochs` (= `scheduler.t_max`) and the aggregation strategy. Its module
docstring and inline comments carry calibration learnings that are cheaper to read than to rediscover
(e.g. the FedAdam `eta` value, and which manifest each node gets) — read it before writing a new
federated grid by hand.

## Notebook experiments (`configs/exp*/*.ipynb`)

The current line of work on `feature/radimagenet` (**exp01–exp19**, core series) is **standalone
Jupyter notebooks that do not import `fedmammobench` at all** — plain torchvision `resnet50` + a
hand-written `Classifier`, wrapped in a local `FullModel(backbone, classifier)`. Nothing in the
sections above (config schema, registries, strategies, weight loaders, `Trainer`) is in play. They
are all centralized, single-GPU, `Mammo-Bench`-only ablations over which ResNet50 blocks are
unfrozen (`layer4` / `layer4+layer3` / fully frozen) crossed with head type (`linear` / dropout
0.3 / 0.5).

**`configs/exp20`–`exp22` extend the same series past exp19** (same notebook structure —
parameters cell, seeding, `metrics.json` — and have committed `runs/` results) but vary image
size, batch size, and loss (`img384`/`img512`, discriminative LR + BCE) rather than freeze depth
or head type. They postdate `docs/EXPERIMENTOS_CENTRALIZADOS.md`, which as of this writing only
covers exp01–exp19 — don't assume that doc is exhaustive, check `ls configs/exp2*` and `ls runs/`
for what's actually landed.

**`configs/exp23` is both the W&B template for this notebook series and the base of Block 6.**
It started as a straight copy of `exp22` plus a hand-rolled `wandb.init()`/log/`finish()` following the
same degrade-to-offline logic as the package's `WandbWriter` (`src/fedmammobench/utils/wandb_utils.py`):
never blocks or prompts for a key, falls back to offline mode if no credentials are cached. For any
*future* one-off notebook outside a generated block, copy exp23's W&B cells (imports, `WANDB_*` params,
init, per-epoch log inside `train_model`, final test-metrics/image log + `finish()`) rather than
reinventing the pattern. The lab workstation authenticates via a shared team **service account** in
`~/.netrc` (not a personal key) — never `cat` that file or paste a key into a notebook cell (which gets
committed with its outputs); verify credentials with `grep -q "api.wandb.ai" ~/.netrc` instead.

On top of that, `exp23` diverged further from `exp22` and became the base config for **Block 6 —
depth of unfreezing** (`exp23`–`exp27`): `CrossEntropyLoss` instead of `BCEWithLogitsLoss` (the MLP
head's `linear2` outputs 2 raw logits, not 1 + sigmoid), best-checkpoint/early-stopping tracked by
**F1-macro on validation** instead of min val loss (`val_f1_history`, mirrors `val_loss_history`), the
head (`linear1`→`bn1`→ReLU→dropout→`linear2`, 2048→512→2) reinitialized `normal(mean=0, std=0.01)`
instead of `nn.Linear`'s default kaiming-uniform (`bn1` and the RadImageNet backbone are left alone),
and a single `LR=LR_BACKBONE=1e-4` for both the head and backbone param groups — **intentionally not
differential**, which deliberately re-enters the failure mode `scripts/gen_lr_backbone_grid.py`
documents from exp02/03/05/06/08/09 (undifferentiated LR destroys RadImageNet backbone weights in 1-3
epochs); this is now itself an ablation axis for the block, not an oversight. `BLOCK = 6` in all five
notebooks (was a stale `3` copied all the way from `exp20`–`exp22`, unrelated to their actual content —
fixed here). `UNFREEZE_IDX` — a list of `backbone` `nn.Sequential` indices, see the freeze cell for the
index→layer map (`0`=conv1, `1`=bn1, `4`-`7`=layer1-4) — is the only thing that varies between the five:

| Notebook | `UNFREEZE_IDX` | Depth |
|---|---|---|
| `exp24` | `[]` | backbone fully frozen |
| `exp23` | `[7]` | layer4 only |
| `exp25` | `[7, 6]` | layer4 + layer3 |
| `exp26` | `[7, 6, 5]` | layer4 + layer3 + layer2 |
| `exp27` | `[7, 6, 5, 4, 1, 0]` | entire backbone |

The freeze cell and the optimizer cell in `exp23` were generalized to read `UNFREEZE_IDX` instead of
hardcoding `backbone[7]` — the optimizer builds its backbone param group from
`[p for p in model.backbone.parameters() if p.requires_grad]` rather than a fixed index, so those two
cells are byte-identical across all five notebooks; only the parameters cell
(`EXP_ID`/`RUN_NAME`/`UNFROZEN_DESC`/`UNFREEZE_IDX`/`WANDB_TAGS`) differs. `scripts/gen_depth_block_grid.py`
substitutes exactly that cell to produce `exp24`–`exp27` from the `exp23` template — `exp23` itself is
hand-edited, not generated, so don't regenerate over it; edit the generator's `EXPERIMENTS` list and
re-run with `--force` to change `exp24`–`exp27`. None of the five have been executed yet as of this
writing (no `runs/exp2{3..7}...` directories).

**`configs/exp71`–`exp77` are an earlier, superseded draft of this same work — don't treat them as
part of the series.** Tells: the filename carries a description (`exp72_resnet50_layer4only.ipynb`)
instead of being bare `expNN.ipynb`; ~14 cells instead of ~20; **no parameters cell** (no `EXP_ID` /
`RUN_NAME`), no seeding, and no `metrics.json` write — so they produce no `runs/` directory and
there is no committed record of their results. exp01–exp09 re-do these ablations properly. Read them
for history if you like, but derive new work from exp01–exp19.

Structure of each exp01–exp19 notebook, essentially identical across the series:
- A parameters cell near the top (`EXP_ID`, `RUN_NAME`, `BLOCK`, `HEAD`, `LR`, `BATCH_SIZE`,
  `NUM_EPOCHS`, `PATIENCE`, `PROJECT_ROOT`, `DATASET_ROOT`). Deriving a new experiment means copying a
  notebook and editing this cell — that is the whole "config system" here.
- **Absolute paths hardcoded to the lab workstation** (`/media/imagenesmedicas/DATA1/...`) for
  `PROJECT_ROOT`, `DATASET_ROOT`, and the RadImageNet checkpoint. They do not resolve on other machines
  and do not honor `MAMMO_DATA` / `WEIGHTS_DIR` / `FEDMAMMOBENCH_RADIMAGENET_DIR`.
- Data comes from `manifests/fedmammobench.csv` with columns `split` / `classification` /
  `preprocessed_image_path`, read by a local `CSVDataset` (not `datasets/`).
- Comments and printed output are in **Spanish**; keep that when editing them.
- Outputs land in `runs/<RUN_NAME>/` (see Outputs above) and are committed except the checkpoint.

**Blocks 4–5 (exp10–exp19) are generated, not hand-written.** `scripts/gen_lr_backbone_grid.py` takes
`configs/exp09/exp09.ipynb` as the template and substitutes the eight cells that vary (imports, params,
seeding, DataLoaders, head class, head instance, freeze, optimizer, summary); the grid itself is the
`EXPERIMENTS` list at the top of that script. Edit the generator and re-run it with `--force` rather
than editing those ten notebooks by hand — that is the only way the block stays internally consistent.
Blocks 1–3 (exp01–exp09) predate the generator and are hand-written; don't regenerate over them.
Unlike blocks 1–3, blocks 4–5 seed everything (`SEED = 42`, `seed_everything()` after the device cell
and again before the head init) and use discriminative LRs (`LR` for the head, `LR_BACKBONE` for the
unfrozen blocks, as separate AdamW param groups) — so their numbers are **not** directly comparable to
exp01–exp09, which ran unseeded with a single LR. exp10 and exp11 exist as seeded re-runs of exp07 and
exp09 to bridge that gap. Block 6 (`exp23`–`exp27`, see above) is generated the same way, from a
different template and generator (`gen_depth_block_grid.py` off of `exp23`) — it inherits blocks 4–5's
seeding but not their discriminative-LR convention (LR is intentionally undifferentiated in Block 6).

`runs/` is the results index for this series: one directory per `RUN_NAME`, and a notebook with no
`runs/` entry has not been executed on the workstation yet. The names encode the ablation cell
(`exp13_resnet50_radimagenet_mammobench_l4l3_linearhead_lrbb1e-4`), so `ls runs/` is the fastest way
to see which combinations already have numbers before proposing a new one.

Two subtleties the package handles centrally are re-implemented by hand in each notebook — if you fix
one, fix it in the whole series, and keep them consistent with the package versions:
- `load_radimagenet_backbone` remaps `RadImageNet-resnet50.pth`'s `backbone.<N>.*` `nn.Sequential`
  keys onto torchvision names; without it `load_state_dict(strict=False)` matches zero tensors — the
  same failure mode documented under "Checkpoint key-namespace normalization" above.
- `freeze_bn_running_stats` re-`eval()`s BN layers whose affine params are frozen, mirroring
  `Trainer._freeze_bn_running_stats`.

## Documentation map

`docs/` is large and partly historical. Trustworthiness, highest first:

- `configs/README.md` (Spanish) — **authoritative** for how experiments are actually run: env vars,
  Docker mounts, per-node hyperparameter matching, best-checkpoint selection, resume, W&B, autoplot.
- `CHANGELOG.md` — authoritative for behavior changes and *why* they were made; each entry explains
  the failure mode it addresses. Read this before assuming current behavior from old code comments.
- `docs/EXPERIMENTOS_CENTRALIZADOS.md` (Spanish) — the exp01–exp19 notebook series: the 19
  configurations, their results, and the CMMD patient-level label-propagation defect that puts a
  ~0.44 val_loss floor under all of them. Read it before interpreting any number from `runs/exp*`.
- `docs/EXTENDING.md` — adding a strategy/model/dataset.
- `docs/CHECKPOINT_COMPATIBILITY.md`, `docs/RADIMAGENET_IMPLEMENTATION.md`,
  `docs/TRANSFER_LEARNING_GUIDE.md` — weight-loading specifics.
- `docs/DATA_PREPARATION.md`, `docs/METHODOLOGY.md` — manifests and experimental design.
- `docs/DOCKER.md`, `docs/FEDERATED_DEPLOYMENT_GUIDE.md`, `docs/SETUP_6NODES.md`,
  `docs/QUICK_START_6NODES.md`, `docs/NODE_CONFIGURATION_MATRIX.md` — deployment; node counts and
  IPs are snapshots of a particular setup, verify against `configs/` before relying on them.
- `README.md` — **stale**. It advertises version 0.3.0 (actual: `pyproject.toml` 0.9.0) and every
  config path it names (`configs/fedavg_cbis_ddsm.yaml`, `configs/radimagenet_*.yaml`) has since
  moved to `configs/legacy/`. Prefer this file and `configs/README.md`.

## Repo conventions

- **Language:** source code, docstrings, and this file are in English; operational docs and shell
  scripts (`configs/README.md`, `run.sh`, `scripts/run-queue.sh`, `.claude/commands/*`) are in
  Spanish, matching the user. Follow whichever the surrounding file uses.
- **Project slash commands** live in `.claude/commands/` — `/docker-run`, `/docker-queue`,
  `/new-exp`, `/eval-experiments`, `/plot`, `/compare`, `/check-manifest`, `/validate-configs`.
  Prefer these over hand-rolling equivalent shell for routine experiment work.
- **`scripts/` is append-only in practice:** one-off `run-expNN-MM.sh` / `eval-expNN-MM.sh` drivers
  accumulate with hardcoded experiment IDs and checkpoint paths. They document what was run; do not
  edit them to run something new — copy the pattern or use `scripts/run-queue.sh`.
- **Data, weights, and every `*.pt`/`*.pth` are gitignored** and live on external disks referenced
  through `MAMMO_DATA` / `WEIGHTS_DIR` / `FEDMAMMOBENCH_RADIMAGENET_DIR`. Never hardcode those paths in
  package code or YAML (the notebooks already do, and that is a known wart, not a pattern to copy).
  Non-checkpoint run artifacts under `runs/` — `metrics.json`, CSVs, `plots/*.png` — are *not* ignored
  and do get committed.
- **Notebooks are committed with their outputs**, so a re-executed notebook produces a five-figure
  line diff. Expect that; don't try to "clean up" the diff by stripping outputs unless asked — the
  stored outputs are how the experiment's results are archived.

## Extension checklist

When adding a strategy, model, or dataset, see `docs/EXTENDING.md`. The short version:
1. Write the module with `@register_*` decorator.
2. Add the side-effect import in the package `__init__.py`.
3. For models: update the `Literal` in `model_config.py`.
4. Add a test; run `pytest tests/ -v`.
5. If RNG order, defaults, `Trainer` signature, or aggregation math changed — bump version in `pyproject.toml` and add a `CHANGELOG.md` entry.
