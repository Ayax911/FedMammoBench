# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FedMammoBench: a federated learning framework for binary mammography classification (benign/malignant), built on PyTorch (+ Flower/Ray for the federated side, postponed — see below).

## Branch state — read `REFACTOR.md` first

`feature/radimagenet-refactor` (the branch you are almost certainly on) is a **from-scratch rewrite**
of the training package. The previous `src/fedmammobench/` (~9,900 lines: CLI, config system,
registries, strategies, weight loaders, `Trainer`, resume logic) was **deliberately deleted** on this
branch (commits `12cb3d1` + `6cfd5b6`) as a learning exercise — understand every piece, simplify ~59
opaque modules down to ~15–18, reproduce the legacy's results rather than free-rewrite them (the numbers
feed a paper). **Current scope is centralized training only; federated learning is postponed** until
that pipeline is validated.

`REFACTOR.md` at the repo root is the actively-maintained handoff/status document for this rewrite —
verified tree state, what's done vs. not, open decisions, next steps. **Read it before writing any code
under `src/`, and re-read it each session — it changes every commit, faster than this file does.** Where
anything below disagrees with a more recent `REFACTOR.md`, trust `REFACTOR.md`.

Because of the deletion, a lot of *other* checked-in documentation now describes a package that no
longer exists in this working tree and does not run here: the top of `README.md` (Quick Start, Execution
Modes, Configuration, Federated Strategies, Transfer Learning, Research Experiments — all assume
`fedmammobench-*` CLI entry points and `configs/base.yaml` inheritance), `scripts/DOCS.md`,
`docs/SRC_STRUCTURE.md`, and every `scripts/run_{centralized,federated,server,client,evaluation}.py`
(all `import fedmammobench...`, none of which resolves). Treat any doc or script that imports
`fedmammobench` as legacy reference material, not as something to run. The legacy package is still
intact on `main`:
```bash
git show main:src/fedmammobench/training/trainer.py
git show main:src/fedmammobench/models/weight_loaders/custom.py
git worktree add ../fedmammobench-legacy main   # to browse it as a full checkout
```
`CHANGELOG.md` and `run.sh` were deleted alongside it and are recoverable the same way.

## Environment & running code

`.venv/` (Python 3.12.3) is the interpreter to use — **not** a `venv/` directory, and note
`pyproject.toml`'s `requires-python = ">=3.11,<3.12"` technically rejects it (one of several things
`pyproject.toml` still needs fixing for this branch, see `REFACTOR.md` §8). It currently has only
`torch`, `torchvision`, `pandas`, `pillow`, `numpy` installed — enough for `src/datasets` and most of
`src/models`, but **not** `pytest`, `scikit-learn`, `albumentations`, `opencv`, `ray`, `flwr`, `PyYAML`,
or `wandb`. Install whatever a task needs ad hoc; there is no `requirements.txt` in this tree to install
from wholesale (it existed pre-refactor, not anymore).

`pip install -e .` does not work: `[tool.setuptools.packages.find] include = ["fedmammobench*"]` matches
zero packages now that that directory is gone. Import the new code straight from the repo root instead:
```bash
.venv/bin/python -c "from src.datasets import Manifest, Split; print('ok')"
```
`src/__init__.py` makes `src` a package, so `from src.datasets import ...` needs no `PYTHONPATH` tweak.
Avoid `PYTHONPATH=src` + `from datasets import ...` — that top-level name collides with HuggingFace's
`datasets` package if it's ever installed alongside.

There is no working test command right now. `tests/` holds only `__init__.py` and
`test_wandb_writer.py`, and the latter is itself dead — it imports `fedmammobench.utils.wandb_utils`,
which doesn't exist in this tree. `pytest` isn't installed in `.venv` either. `tests/test_dataset.py` /
`tests/test_splits.py` against the new `src/datasets/` are the first tests actually worth writing here
(see `REFACTOR.md` §12).

None of this applies to the notebook experiments (`configs/exp*/*.ipynb`) — see below.

## Current architecture (`src/`, in progress)

Target layout and the one dependency rule to keep (`REFACTOR.md` §5):
```
config → seed/metrics/checkpoint/tracking/datasets/models → train/ → cli.py
```
Nothing imports from `cli.py`. `datasets/` never imports from `models/` or `train/`. `models/weights.py`
never imports from `models/build.py`. A late import inside a function to dodge a cycle is a design smell
here, not an accepted workaround.

Key design decisions (deliberately different from the deleted package):
| Decision | Choice | Why |
|---|---|---|
| Config | Pydantic v2 + YAML | free validation; `extra="forbid"` catches typos |
| Config inheritance | **None** — no `defaults:`/`base.yaml` chain | every experiment config reads start-to-end |
| Decorator registries (`@register_*`) | **Dropped** — plain dict dispatch | keeps "where does this come from" visible |
| One `Dataset` class per source | **No** — a single one | everything is already consolidated into one manifest CSV |

- **`src/datasets/`** — manifest loading/validation, patient-level train/val/test split
  verification (never generation — the manifest's `split` column is already stratified upstream),
  `MammoBenchDataset`, a `TransformBuilder`, and `builder_dataloader()` tying it all together. This
  package works end-to-end against `manifests/fedmammobench.csv` today. Full per-method contracts,
  raises, and gotchas (e.g. `.iloc` vs `.loc` on the split-filtered, non-contiguous-index DataFrames) are
  in **`src/datasets/DOCS.md`** — read that instead of re-deriving it from the source.
- **`src/models/`** — `build_model()` builds a named architecture (registered in a plain dict, e.g.
  `resnet50_radimagenet`), loads/remaps a checkpoint's `state_dict` (`weights.py`), and applies a
  per-architecture `FreezeStrategy` (`freeze.py`) to unfreeze from a given block onward. Details in
  **`src/models/DOCS.md`**. Two currently-broken things to know before you spend time debugging them:
  - `src/models/__init__.py` imports `.load` and `.train` submodules that don't exist yet (only
    `build.py`, `weights.py`, `freeze.py`, `heads.py`, `reports.py`, `mlp_configs/` do) — `import
    src.models` fails outright. Import the submodule you need directly, e.g.
    `from src.models.build import build_model`.
  - The classification-head factory (`heads.py`, `mlp_configs/standard_mlp.py`) both do
    `from ..heads import HeadBuilder`, but no `HeadBuilder` base class is defined anywhere in the repo
    yet — importing either module raises `ModuleNotFoundError`.
- **`src/train/`** — optimizer/scheduler/loss factories (`build.py`), pure `train_one_epoch()` /
  `evaluate()` functions (`loop.py`), and a `Trainer` (`trainer.py`) that runs the epoch loop and saves a
  checkpoint only when the tracked validation metric improves (never the last epoch — `fit()` returns
  the *best* checkpoint's path). This package is newer and moving fast; some of what `loop.py`/
  `trainer.py` import (`src/metrics.py`, `src/checkpoint.py`) may not exist yet depending on when you
  read this. Check `git log`/`git status` and `REFACTOR.md`'s "próximos pasos" for what's actually
  landed before assuming a module is complete.

Legacy bugs that must be **deliberately reproduced**, not just avoided by accident, because the new
`models/`/`train/` code replaces the modules that had them (`REFACTOR.md` §7):
- **BN drift under freeze.** `model.train()` re-enables frozen BatchNorm layers, which keep updating
  `running_mean`/`running_var` even with `requires_grad=False`. Re-`eval()` them right after every
  `model.train()` call (`train/loop.py`'s `_set_frozen_bn_eval`, once written).
- **Silent weight-loading failure.** A `LoadReport` with `matched == 0` must raise, not just report zero
  — that's exactly the state a `backbone.`-prefix mismatch produces, and it silently trains from random
  init.
- **`backbone.` key prefix.** This project's own checkpoints carry it; torchvision constructors don't
  expect it. Normalize before `load_state_dict`.
- **Evaluate the best checkpoint, not the final one.** `Trainer.fit()` must return the best checkpoint's
  path, and test-set evaluation must always use that return value — never a separate "which checkpoint"
  config flag that can drift from what was actually best.

## Notebook experiments (`configs/exp*/*.ipynb`)

Unaffected by the refactor above — these predate it and don't import any package from `src/`.

The current line of work (**exp01–exp19**, core series, plus **exp20–exp32** extending it — see below)
is **standalone Jupyter notebooks that import nothing from this repo's packages**: plain torchvision
`resnet50` + a hand-written `Classifier`, wrapped in a local `FullModel(backbone, classifier)`. They are
all centralized, single-GPU, `Mammo-Bench`-only ablations over which ResNet50 blocks are unfrozen
(`layer4` / `layer4+layer3` / fully frozen) crossed with head type (`linear` / dropout 0.3 / 0.5).

**`configs/exp20`–`exp22` extend the same series past exp19** (same notebook structure —
parameters cell, seeding, `metrics.json` — and have committed `runs/` results) but vary image
size, batch size, and loss (`img384`/`img512`, discriminative LR + BCE) rather than freeze depth
or head type.

**`configs/exp23` is both the W&B template for this notebook series and the base of Block 6.**
It started as a straight copy of `exp22` plus a hand-rolled `wandb.init()`/log/`finish()` following the
same degrade-to-offline logic as the (now-deleted) package's `WandbWriter`: never blocks or prompts for
a key, falls back to offline mode if no credentials are cached. For any *future* one-off notebook
outside a generated block, copy exp23's W&B cells rather than reinventing the pattern. The lab
workstation authenticates via a shared team **service account** in `~/.netrc` (not a personal key) —
never `cat` that file or paste a key into a notebook cell (which gets committed with its outputs);
verify credentials with `grep -q "api.wandb.ai" ~/.netrc` instead.

On top of that, `exp23` diverged further from `exp22` and became the base config for **Block 6 —
depth of unfreezing** (`exp23`–`exp27`): `CrossEntropyLoss` instead of `BCEWithLogitsLoss` (the MLP
head's `linear2` outputs 2 raw logits, not 1 + sigmoid), best-checkpoint/early-stopping tracked by
**F1-macro on validation** instead of min val loss (`val_f1_history`, mirrors `val_loss_history`), the
head (`linear1`→`bn1`→ReLU→dropout→`linear2`, 2048→512→2) reinitialized `normal(mean=0, std=0.01)`
instead of `nn.Linear`'s default kaiming-uniform (`bn1` and the RadImageNet backbone are left alone),
and a single `LR=LR_BACKBONE=1e-4` for both the head and backbone param groups — **intentionally not
differential**, which deliberately re-enters the failure mode `scripts/gen_lr_backbone_grid.py`
documents from exp02/03/05/06/08/09 (undifferentiated LR destroys RadImageNet backbone weights in 1-3
epochs); this is now itself an ablation axis for the block, not an oversight. `UNFREEZE_IDX` — a list of
`backbone` `nn.Sequential` indices (`0`=conv1, `1`=bn1, `4`-`7`=layer1-4) — is the only thing that varies
between the five:

| Notebook | `UNFREEZE_IDX` | Depth |
|---|---|---|
| `exp24` | `[]` | backbone fully frozen |
| `exp23` | `[7]` | layer4 only |
| `exp25` | `[7, 6]` | layer4 + layer3 |
| `exp26` | `[7, 6, 5]` | layer4 + layer3 + layer2 |
| `exp27` | `[7, 6, 5, 4, 1, 0]` | entire backbone |

The freeze cell and the optimizer cell in `exp23` were generalized to read `UNFREEZE_IDX` instead of
hardcoding `backbone[7]`, so those two cells are byte-identical across all five notebooks; only the
parameters cell differs. `scripts/gen_depth_block_grid.py` substitutes exactly that cell to produce
`exp24`–`exp27` from the `exp23` template — `exp23` itself is hand-edited, not generated; edit the
generator's `EXPERIMENTS` list and re-run with `--force` to change `exp24`–`exp27`. All five have
committed `runs/exp2{3..7}...` results.

**Stale manifest reference in `exp23`–`exp27`.** Their params cell points `MANIFEST_PATH` at
`manifests/fedmammobench_tompei.csv`, which was later deleted from the repo (replaced by a corrected
`manifests/fedmammobench.csv`, still carrying the `source_dataset` column). Their committed `runs/`
results stand, but re-running any of `exp23`–`exp27` today fails at the manifest-loading cell until
`MANIFEST_PATH` is repointed at `fedmammobench.csv`. Block 7 (below) already points at the live file —
don't copy the stale path from `exp23`–`exp27` into new work.

**Block 7 — same depth-of-unfreezing grid as Block 6, at 256px instead of 512px, plus a per-database
test breakdown (`exp28`–`exp32`).** `exp28` is a hand-edited copy of `exp23` with three changes:
`IMAGE_SIZE` 512 → 256; `MANIFEST_PATH` repointed at `manifests/fedmammobench.csv`; and a
`source_dataset`-level test breakdown appended after the existing global test cell.

Evaluation is built around one reusable function, `evaluate_model(model, checkpoint_path, dataloader,
...)`: it loads `checkpoint_path` into `model` and runs inference over whatever `DataLoader` it's
handed, returning accuracy/AUC/precision/recall/F1 + confusion matrix (AUC/precision/recall/F1 come
back `None` with a `"warning"` key when the subset has fewer than 2 classes present). The global test
cell calls it once with the `test` `DataLoader`; the per-dataset cell builds one fresh `DataLoader` per
`source_dataset` value and calls it again per dataset, same `BEST_MODEL_PATH` — a deliberate trade for a
single, stateless evaluation function over one that depends on state from an earlier cell.

Outputs land in two places on purpose: `RUN_DIR/plots/` + `RUN_DIR/metrics.json` stay global-test-only,
while `RUN_DIR/per_dataset/metrics_per_dataset.json` + a grouped bar chart + a confusion-matrix grid go
in their own `RUN_DIR/per_dataset/` folder. `wandb_run.finish()` moved to the end of the per-dataset
plotting cell so both granularities land on the same W&B run.

`save_predictions_csv(manifest_df, result, out_path)` appends `y_true`/`y_pred`/`y_prob` (from
`evaluate_model`'s returned lists) to a copy of the evaluated `DataFrame`, assigned by **position**, not
pandas index — this depends on `shuffle=False` + `CSVDataset(dataframe=...)` preserving input row order.
Global goes to `RUN_DIR/predictions.csv`; each per-dataset subset goes to
`RUN_DIR/per_dataset/predictions_<source_dataset>.csv`.

| Notebook | `UNFREEZE_IDX` | Depth | Block 6 counterpart |
|---|---|---|---|
| `exp29` | `[]` | backbone fully frozen | `exp24` |
| `exp28` | `[7]` | layer4 only | `exp23` |
| `exp30` | `[7, 6]` | layer4 + layer3 | `exp25` |
| `exp31` | `[7, 6, 5]` | layer4 + layer3 + layer2 | `exp26` |
| `exp32` | `[7, 6, 5, 4, 1, 0]` | entire backbone | `exp27` |

`scripts/gen_img256_block_grid.py` substitutes the parameters cell to produce `exp29`–`exp32` from the
`exp28` template, mirroring `gen_depth_block_grid.py`. `exp28` itself is hand-edited, not generated. None
of `exp29`–`exp32` have been executed yet (no `runs/exp2{8..9}...`/`exp3{0..2}...` directories) as of
this writing — check `ls runs/` for the current state.

**`configs/exp71`–`exp77` are an earlier, superseded draft of this same work — don't treat them as
part of the series.** Tells: the filename carries a description
(`exp72_resnet50_layer4only.ipynb`) instead of being bare `expNN.ipynb`; no parameters cell, no seeding,
no `metrics.json` write — no committed `runs/` record. exp01–exp09 re-do these ablations properly.

Structure of each exp01–exp19 notebook, essentially identical across the series:
- A parameters cell near the top (`EXP_ID`, `RUN_NAME`, `BLOCK`, `HEAD`, `LR`, `BATCH_SIZE`,
  `NUM_EPOCHS`, `PATIENCE`, `PROJECT_ROOT`, `DATASET_ROOT`). Deriving a new experiment means copying a
  notebook and editing this cell — that is the whole "config system" here.
- **Absolute paths hardcoded to the lab workstation** (`/media/imagenesmedicas/DATA1/...`) for
  `PROJECT_ROOT`, `DATASET_ROOT`, and the RadImageNet checkpoint. They do not resolve on other machines.
- Data comes from `manifests/fedmammobench.csv` with columns `split` / `classification` /
  `preprocessed_image_path`, read by a local `CSVDataset` (not `src/datasets/`).
- Comments and printed output are in **Spanish**; keep that when editing them.
- Outputs land in `runs/<RUN_NAME>/` and are committed except the checkpoint (`.gitignore` only
  excludes `*.pt`/`*.pth`) — `runs/` doubles as the results record for the notebook series. Layout:
  `metrics.json` (test-set summary + hyperparameters), `loss_history.csv`, `plots/loss_curve.png`,
  `plots/test_confusion_matrix.png`, `plots/test_metric_<name>.png`, `best_model.pth`, and for Block 7
  only `predictions.csv` + `per_dataset/`.

**Blocks 4–5 (exp10–exp19) are generated, not hand-written.** `scripts/gen_lr_backbone_grid.py` takes
`configs/exp09/exp09.ipynb` as the template and substitutes the cells that vary; the grid itself is the
`EXPERIMENTS` list at the top of that script. Edit the generator and re-run it with `--force` rather than
editing those notebooks by hand. Blocks 1–3 (exp01–exp09) predate the generator and are hand-written;
don't regenerate over them. Unlike blocks 1–3, blocks 4–5 seed everything (`SEED = 42`) and use
discriminative LRs (`LR` for the head, `LR_BACKBONE` for the unfrozen blocks) — their numbers are **not**
directly comparable to exp01–exp09, which ran unseeded with a single LR. exp10 and exp11 are seeded
re-runs of exp07 and exp09 to bridge that gap.

Two subtleties the deleted package used to handle centrally are re-implemented by hand in each notebook
— if you fix one, fix it in the whole series:
- `load_radimagenet_backbone` remaps `RadImageNet-resnet50.pth`'s `backbone.<N>.*` `nn.Sequential` keys
  onto torchvision names; without it `load_state_dict(strict=False)` matches zero tensors.
- `freeze_bn_running_stats` re-`eval()`s BN layers whose affine params are frozen — the same fix
  `src/train/loop.py` is (re)implementing above.

`runs/` is the results index for this series: one directory per `RUN_NAME`, and a notebook with no
`runs/` entry has not been executed on the workstation yet. `ls runs/` is the fastest way to see which
combinations already have numbers before proposing a new one.

## Documentation map

Trustworthiness for *this branch's actual working tree*, highest first:

- `REFACTOR.md` — **authoritative for `src/` status.** Verified tree state, open decisions, next
  steps; rewritten frequently, so re-read it each session rather than trusting a memory of it.
- `src/datasets/DOCS.md`, `src/models/DOCS.md` — authoritative, current per-method contracts for those
  two packages (inputs, raises, returns). `src/train/` doesn't have one yet.
- `configs/README.md` (Spanish) — describes running experiments via the legacy package/Docker; useful
  for the operational conventions (per-node hyperparameter matching, W&B, autoplot) but the commands
  themselves assume `fedmammobench-*`, which doesn't exist on this branch.
- `docs/EXPERIMENTOS_CENTRALIZADOS.md` (Spanish) — the exp01–exp19 notebook series: the 19
  configurations, their results, and the CMMD patient-level label-propagation defect that puts a
  ~0.44 val_loss floor under all of them. Still current and unaffected by the refactor.
- **Legacy-only, describes the deleted package** — do not use these to understand what's runnable on
  this branch: the top portion of `README.md` (down to "Documentation Architecture"), `scripts/DOCS.md`,
  `docs/SRC_STRUCTURE.md`, `docs/EXTENDING.md`, `docs/CHECKPOINT_COMPATIBILITY.md`,
  `docs/RADIMAGENET_IMPLEMENTATION.md`, `docs/TRANSFER_LEARNING_GUIDE.md`,
  `docs/FEDERATED_DEPLOYMENT_GUIDE.md`, `docs/SETUP_6NODES.md`, `docs/QUICK_START_6NODES.md`,
  `docs/NODE_CONFIGURATION_MATRIX.md`, `docs/DOCKER.md`. They're still worth reading for *what the
  legacy package did* (the new code is meant to reproduce its results), just not for what to run today.
  `README.md`'s bottom section ("Documentation Architecture" / "Project Structure") is the one part of
  that file describing the current refactor, and is accurate.
- `docs/DATA_PREPARATION.md`, `docs/METHODOLOGY.md` — manifest format and experimental design; not
  package-specific, still relevant.

## Repo conventions

- **Language:** new `src/` code mixes English and Spanish docstrings depending on who wrote the module
  (in flux during the rewrite) — match whichever the file you're editing already uses rather than
  imposing one. Notebooks and their comments/output are Spanish. Operational docs and shell scripts
  (`configs/README.md`, `.claude/commands/*`) are Spanish.
- **Project slash commands** live in `.claude/commands/` — `/docker-run`, `/docker-queue`, `/new-exp`,
  `/eval-experiments`, `/plot`, `/compare`, `/check-manifest`, `/validate-configs`. These predate the
  refactor and assume the legacy package/Docker image; verify a command actually applies before
  reaching for it here.
- **`scripts/` is append-only in practice:** one-off `run-expNN-MM.sh` / `eval-expNN-MM.sh` drivers
  accumulate with hardcoded experiment IDs and checkpoint paths. They document what was run; do not
  edit them to run something new.
- **Data, weights, and every `*.pt`/`*.pth` are gitignored.** Non-checkpoint run artifacts under
  `runs/` — `metrics.json`, CSVs, `plots/*.png` — are *not* ignored and do get committed.
- **Notebooks are committed with their outputs**, so a re-executed notebook produces a five-figure line
  diff. Expect that; don't try to "clean up" the diff by stripping outputs unless asked.
