# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FedMammoBench: binary mammography classification (benign / malignant) with a RadImageNet-pretrained
ResNet50 in PyTorch. **Current scope is centralized training only** — the federated half (Flower/Ray)
is postponed until the centralized pipeline is validated, and nothing under `src/` imports `flwr` or
`ray` today.

## The one command

```bash
.venv/bin/python -m src.cli --config configs/exp05_fedmammobench_full_weighted.yaml
```

That is the entire interface. `src/cli.py:run()` does everything end to end — seed → manifest → split
→ transforms → dataloaders → backbone + head → `Trainer.fit()` → re-evaluate the **best** checkpoint on
val *and* test → plots, JSONs, `predictions.csv`, W&B. There is no separate evaluate/plot entry point,
no resume, and no way to score an existing checkpoint without re-running training. Adding one means
writing it.

Cheap sanity check that the tree imports: `.venv/bin/python -c "import src.cli"`.

## Repo state — what is real, what is stale

`main` is the live branch and holds the rewritten `src/` package. A lot of checked-in documentation
predates that and describes things that no longer exist anywhere:

- **The legacy `src/fedmammobench/` package is gone from every branch** (71 files: registries,
  strategies, weight loaders, its own `Trainer`, `configs/base.yaml` inheritance, the
  `fedmammobench-*` console scripts). There is no `pyproject.toml` in this tree at all, so
  `pip install -e .` and every `fedmammobench-*` command are dead by construction.
- **The exp01–exp32 standalone notebook series is also gone from every branch**, deleted in `0e934ec`
  ("remove notebook-era configs/ and runs/"), along with `scripts/gen_*.py` and every
  `scripts/run-expNN-*.sh`/`eval-expNN-*.sh`. `configs/` is now YAML only and `scripts/` holds nothing
  but `DOCS.md`.
- **`ec55408` is the last commit holding both** — the complete legacy package *and* the notebook series
  with its generators. Use it to consult either:
  ```bash
  git show ec55408:src/fedmammobench/training/trainer.py
  git show ec55408:configs/exp28/exp28.ipynb          # the notebook src/ was written to reproduce
  git worktree add ../fedmammobench-legacy ec55408    # to browse it as a full checkout
  ```
- **`REFACTOR.md` is a snapshot from 2026-08-26 and its status sections are wrong** — it claims `src/`
  is 155 lines in 4 files and that `models/`/`train/` are unwritten. Its *rationale* sections are still
  the best explanation of why the code looks the way it does (§4 architecture decisions, §6 the exact
  transform pipeline and index gotcha, §7 legacy bugs to reproduce, §10 manifest statistics); its
  checklists are not.
- **`PHASES.md` is accurate and is the newest design document.** It logs, phase by phase, what was
  ported from the INC project (`inc-project-models-classification-detection-main`, a sibling repo not
  checked in here) into `src/`: F1/precision metrics, `drop_last`, cuDNN determinism, the
  `weight`-as-list loss bug, `FocalLoss`, `EarlyStopping`, test-set evaluation and reporting,
  `ConfigurableMLPHead`, vertical-flip/blur augmentation, periodic checkpoints. Read it before
  touching those areas — most of the odd-looking defaults are "reproduces INC exactly" decisions.
- The `develop` / `phaseN-*` branch convention PHASES.md describes is historical; those branches are
  merged and deleted.

## Environment

`.venv/` (Python **3.12.8**) is the interpreter, and it already has everything in `requirements.txt`:
torch 2.13 + CUDA, torchvision 0.28, torchmetrics, pandas 3.0, pydantic 2.13, PyYAML, tensorboard,
matplotlib, wandb, scikit-learn. Always call it explicitly (`.venv/bin/python`) — there is no
installed package and no activation step in any of the run instructions.

`src/__init__.py` makes `src` a package, so `from src.datasets import ...` works from the repo root
with no `PYTHONPATH` tweak. Do **not** use `PYTHONPATH=src` + `from datasets import ...`; that name
collides with HuggingFace `datasets`.

**There is no test suite and no test runner.** `pytest` is not installed, and `tests/` contains only
`__init__.py` plus `test_wandb_writer.py`, which imports the deleted package and cannot run. Nothing
under `src/` has a test. Verification in this repo has been done the way `PHASES.md` documents it:
drive the real code from a throwaway script with synthetic tensors and assert on the artifacts it
produces. `pyright` is configured (`pyrightconfig.json`, `strict`, `include: ["src"]`) but is not
installed in the venv either — the type annotations and `# pyright: ignore` comments in `src/` exist
to satisfy it, so keep them consistent even though nothing checks them here.

`Dockerfile` still targets Python 3.11 + `pip install -e .` + `scripts/`; it cannot build against this
tree.

## Architecture

One direction of dependency, no exceptions:

```
config.py → seed / metrics / checkpoint / tracking / reporting / datasets / models → train/ → cli.py
```

Nothing imports `cli.py`. `datasets/` never imports `models/` or `train/`. `models/weights.py` never
imports `models/build.py`. A late import inside a function to dodge a cycle is a design smell here,
not an accepted workaround.

Deliberate departures from the deleted package, all still in force:

| Decision | Choice | Why |
|---|---|---|
| Config | Pydantic v2 + YAML, `extra="forbid"` | typos fail validation instead of being ignored |
| Config inheritance | **None** — no `defaults:`/`base.yaml` | every experiment YAML reads start to end |
| Decorator registries | **Dropped** — plain module-level dicts | `_ARCHITECTURES`, `_HEAD_STRATEGIES`, `_OPTIMIZERS`, `_SCHEDULERS`, `_LOSSES` are all greppable in one place |
| One `Dataset` per source | **No** — a single `MammoBenchDataset` | everything is already consolidated into one manifest CSV |

Where the pieces live:

- **`src/datasets/`** — `Manifest` (loads the CSV, requires `preprocessed_image_path` /
  `classification` / `split` / `patient_id`, normalizes labels to `label_norm`, resolves
  `abs_image_path` against `image_root`), `Split` (**verifies** the manifest's existing `split` column
  is patient-disjoint; it never generates a split — stratification happens upstream, outside this
  repo), `MammoBenchDataset`, `TransformBuilder`, and `builder_dataloader()`. Per-method contracts in
  **`src/datasets/DOCS.md`**.
- **`src/models/`** — `build_model(name, weights_path, unfreeze_from, device)` returns
  `(backbone, LoadReport)`: instantiates from `_ARCHITECTURES`, remaps the checkpoint's `backbone.N.`
  keys onto torchvision names, and applies a `FreezeStrategy`. Heads are a separate axis:
  `get_head_strategy(name)` returns an unconstructed `HeadBuilder` subclass (`standard_mlp` — one
  hidden layer, always `BatchNorm1d`; `configurable_mlp` — N hidden layers, selectable activation, no
  BatchNorm by default). Details in **`src/models/DOCS.md`**.
- **`src/train/`** — `build_optimizer`/`build_scheduler`/`build_loss`, the pure `train_one_epoch()` /
  `evaluate()` functions, `EarlyStopping`, `FocalLoss`, `evaluate_checkpoint()`/`predict_on_loader()`,
  and `Trainer`. See **`src/train/DOCS.md`**.
- **`src/reporting.py`** — the four pure artifact writers (`save_metrics_json`,
  `save_predictions_csv`, `plot_confusion_matrix`, `plot_roc_curve`) plus
  `compute_confusion_matrix_metrics()`. Uses **torchmetrics**, not scikit-learn, for the confusion
  matrix and ROC — sklearn is installed but deliberately unused here.
- **`cli.py` owns the assembly**, on purpose: `nn.Sequential(backbone, head.build())`, the W&B run
  (opened before `Trainer` so training and test land on one run), and the `_evaluate_split()` helper
  that runs the identical val and test reporting block.

`LossSpec` (`train/build.py`) is the one abstraction worth understanding before editing the loop: it
pairs the loss function with the correct logits→positive-class-probability conversion, because that
conversion depends on how many logits the head emits (1 for BCE, 2 for CrossEntropy/Focal), not on the
loss's name. `train_one_epoch()`/`evaluate()` call `spec.compute()`/`spec.probs()` with no branching
of their own — the only `if` lives in `build_loss()` and runs once.

## Invariants that will silently ruin a run

These are the legacy failure modes the rewrite exists to prevent. Preserve them.

- **`metric_name: f1` vs `f1_macro`.** `f1` is `BinaryF1Score` — positive class only — and sits at
  exactly `0.0` while the model predicts no malignants, which is the normal state of early epochs on
  the 66/34 `fedmammobench.csv`. `EarlyStopping` requires strict improvement, so that run of zeros
  never resets the patience counter and `fit()` returns the epoch-0 (untrained) checkpoint. Use
  `f1_macro` on that manifest. `f1` is correct only for the INC replicas, whose train split is 84%
  malignant.
- **Evaluate the best checkpoint, never the last.** `Trainer.fit()` returns the best checkpoint path
  and `cli.run()` feeds exactly that into `_evaluate_split()` for both val and test. Never add a
  "which checkpoint" config flag — it can drift from what was actually best.
- **Silent weight-loading failure.** `load_weights()` raises when `matched == 0`. That is precisely the
  state a `backbone.`-prefix mismatch produces, and without the raise the run trains from random init
  and looks merely mediocre.
- **BN drift under freeze.** `model.train()` re-enables frozen BatchNorm layers, whose
  `running_mean`/`running_var` keep updating even at `requires_grad=False`.
  `train/loop.py:_set_frozen_bn_eval()` re-`eval()`s them right after every `model.train()`.
  `freeze_bn_stats: false` turns that off on purpose, to reproduce INC — with a fully frozen backbone
  that is the difference between a backbone that still adapts and one pinned to RadImageNet statistics.
- **`drop_last=True` on the train loader only.** `StandardMLPHead` uses `BatchNorm1d`, which throws on
  a final batch of size 1.
- **Never create `src/data/`.** `.gitignore` has a repo-wide `data/` rule that swallows the whole
  module in silence; that is why the package is `src/datasets/`.
- **`.iloc`, not `.loc`.** `Split`'s DataFrames keep their original non-contiguous indices, so
  positional access is mandatory in `__getitem__`.

## The YAML contract

`ExperimentConfig` (`src/config.py`) is the schema, `extra="forbid"` throughout — an unknown or
misspelled key is a `ValidationError`, not a silent no-op. Sections: `experiment_id`, `architecture`,
`head`, `optimizer`, `scheduler` (optional), `loss`, `data`, `train`. `head`/`optimizer`/`scheduler`/
`loss` are all `NamedComponentConfig` (`name` + free-form `hparams` splatted into the constructor), so
adding a hyperparameter usually means only touching the factory, not the config models.

Two `data` settings encode the **pre-processed float-TIFF pipeline** and must move together:
`Preproccesed/preprocess_images.py` writes 32-bit float single-channel TIFFs (PIL mode `"F"`) already
resized to 224×224 and normalized to `[0,1]` (`norm_0_1/`) or `[-1,1]` (`norm_neg1_1/`), with
`manifests/fedmammobench_norm_{0_1,neg1_1}.csv` pointing at them. Configs consuming those set
`image_size: null` **and** `normalize_mean: null` / `normalize_std: null` — resizing and normalizing
again would be wrong. `MammoBenchDataset.__getitem__` detects mode `"F"` and skips `.convert()`
entirely (PIL clips rather than rescales floats, which would collapse a `[-1,1]` image to near-zero),
replicating to 3 channels on the tensor afterwards instead. Setting `normalize_mean: [0,0,0]` /
`normalize_std: [1,1,1]` is the *other* meaningful value — identity, i.e. plain `[0,1]` pixels, which
is what INC does; the `0.5/0.5` default is not equivalent.

Every config carries **absolute lab-workstation paths** for `weights_path` and `image_root`; they do
not resolve elsewhere. `configs/exp02`, `exp03_*` and `exp04_*` point at
`manifests/dataset_split_formatted.csv`, the INC dataset manifest, which **is not in this repo and
never was** — those three cannot be re-run as-is even on the workstation.

## Run artifacts

`run_dir` (`runs/<experiment_id>/`) gets `config.yaml` (a snapshot of exactly what ran), `metrics.csv`,
TensorBoard events, `plots/` (loss plus one train-vs-val curve per clinical metric), and one folder per
evaluated split — `val/` and `test/`, each with `metrics.json`, `confusion_matrix_metrics.json`,
`predictions.csv`, `confusion_matrix.png`, `roc_curve.png`. Weights go to `checkpoint_dir`
(`runs/<experiment_id>/weights/`) as `best_epoch<N>.pt` plus `best_epoch<N>_backbone.pt` /
`_head.pt` (split out for the eventual federated work, where only the backbone aggregates) and
`epoch<N>.pt` every `save_every` epochs.

`ls runs/` is the fastest way to see which experiments have actually been executed. Checkpoints
(`*.pt`/`*.pth`), `events.out.tfevents.*` and `*.log` are gitignored; `metrics.csv`, `metrics.json`,
`predictions.csv` and `plots/*.png` are committed and serve as the results record.

W&B: `train.wandb_project` (`null` disables it) — the workstation authenticates through a shared team
service account in `~/.netrc`. Never `cat` that file or paste a key anywhere; check credentials with
`grep -q "api.wandb.ai" ~/.netrc`. `MetricsLogger` imports `wandb` lazily and degrades to a no-op, so
a missing key never blocks a run.

## Conventions

- **Language is per-file and mixed on purpose.** `config.py`, `cli.py`, `train/`, `datasets/build.py`
  and every `DOCS.md` are Spanish; `datasets/dataset.py`, `datasets/manifest.py` and `models/` are
  English. Match the file you are editing rather than imposing one. Config YAML comments and
  `PHASES.md`/`REFACTOR.md` are Spanish.
- **Comments carry the *why*, at length.** The existing docstrings and inline comments record which
  bug a line prevents and what the INC project does differently. That density is the house style —
  when you change behavior here, extend that record rather than trimming it.
- **`.claude/commands/`** (`/docker-run`, `/docker-queue`, `/new-exp`, `/eval-experiments`, `/plot`,
  `/compare`, `/check-manifest`, `/validate-configs`) all predate the rewrite and assume the legacy
  package or the Docker image. Verify one actually applies before reaching for it.
- Commit subjects follow `<Verb>: description` (`<Feat>:`, `<Fix>:`, `<Docs>:`, `<add>:`, `<exp>:`),
  with `<exp>:` reserved for committing a run's results.

## Documentation map

Trustworthiness for the current tree, highest first:

- `PHASES.md` — what was ported from INC and why each default is what it is. Current.
- `src/DOCS.md`, `src/datasets/DOCS.md`, `src/models/DOCS.md`, `src/train/DOCS.md` — per-method
  contracts (args, raises, returns) with worked examples. Current; update them alongside code.
- `configs/*.yaml` header comments — the real experiment log. `exp04_inc_strict_replica.yaml` in
  particular documents the four divergences from INC it corrects and the one it deliberately does not.
- `REFACTOR.md` — rationale current, status sections stale (see above).
- `docs/DATA_PREPARATION.md`, `docs/METHODOLOGY.md` — manifest format and experimental design; not
  package-specific, still relevant.
- `docs/EXPERIMENTOS_CENTRALIZADOS.md`, `docs/INFORME_EXP01_22.md` — results of the deleted notebook
  series, including the CMMD patient-level label-propagation defect that puts a ~0.44 val-loss floor
  under all of them. Historical, but the numbers are the ones the current pipeline is compared against.
- **Describes the deleted package — do not use to decide what to run:** `README.md` down to
  "Documentation Architecture", `scripts/DOCS.md`, `docs/SRC_STRUCTURE.md`, `docs/EXTENDING.md`,
  `docs/CHECKPOINT_COMPATIBILITY.md`, `docs/RADIMAGENET_IMPLEMENTATION.md`,
  `docs/TRANSFER_LEARNING_GUIDE.md`, `docs/FEDERATED_DEPLOYMENT_GUIDE.md`, `docs/SETUP_6NODES.md`,
  `docs/QUICK_START_6NODES.md`, `docs/NODE_CONFIGURATION_MATRIX.md`, `docs/DOCKER.md`, `docs/audit/`,
  `docs/audit-plan.md`. Still useful for *what the legacy did* — the new code is meant to reproduce
  its results — just not for what exists today.
