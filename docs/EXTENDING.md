# Extending FedMammoBench

This guide is the **pipeline for adding or modifying each important module** of the
federated infrastructure. Every section is a self-contained recipe: which files to
touch, in what order, and which test to add.

The codebase exposes two extension styles:

| Mechanism | Used by | How a new entry is discovered |
|-----------|---------|-------------------------------|
| **Registry + decorator** | strategies, models, weight loaders, **datasets** | `@register_*` at import time |
| **Dataclass + `validate()`** | configs | new field/section + validation rule |

> Datasets used to be a manual `if/elif` chain in `build_dataset()`; as of v0.3.0
> they use a `@register_dataset` registry like strategies and models, so adding a
> dataset no longer means editing the factory.

> **Golden rule for reproducibility:** any change that alters RNG consumption order,
> default values, the `Trainer` signature, or aggregation math is a *breaking change*.
> Bump the version in `pyproject.toml` and add a `CHANGELOG.md` entry. See
> [METHODOLOGY.md](METHODOLOGY.md) and [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md).

---

## 1. Add a federated strategy

Strategies use the **registry pattern** — the cleanest extension point in the repo.

**Files:** `src/fedmammobench/federated/strategies/<name>.py`, `strategies/__init__.py`

1. **Create the module** `src/fedmammobench/federated/strategies/my_strategy.py`. Inherit
   from `flwr.server.strategy.Strategy` or, more commonly, subclass `FedAvg` so you
   only override what differs (`configure_fit`, `aggregate_fit`, ...):

   ```python
   from typing import Any
   from flwr.server.strategy import FedAvg
   from fedmammobench.federated.strategies.fedavg import _weighted_average
   from fedmammobench.federated.strategies.registry import register_strategy

   class MyStrategy(FedAvg):
       def __init__(self, *, my_param: float = 0.1, **kwargs: Any) -> None:
           kwargs.setdefault("fit_metrics_aggregation_fn", _weighted_average)
           kwargs.setdefault("evaluate_metrics_aggregation_fn", _weighted_average)
           super().__init__(**kwargs)
           self.my_param = float(my_param)

   @register_strategy("my_strategy")
   def build_my_strategy(**kwargs: Any) -> FedAvg:
       return MyStrategy(**kwargs)
   ```

2. **Register the side-effect import** in `strategies/__init__.py` so the decorator
   runs at package import (the registry is only populated by importing the module):

   ```python
   from fedmammobench.federated.strategies import my_strategy  # noqa: F401
   ```

3. **Use it** in YAML — no other code changes:

   ```yaml
   federated:
     strategy:
       name: my_strategy
       params: { my_param: 0.2 }
   ```

   `strategy.params` is forwarded verbatim as `**kwargs` to `build_my_strategy`.

4. **If clients must do extra work** (like FedProx's proximal term), pass data through
   the per-round config: set keys in `configure_fit()` (server side) and read them in
   `FedMammoBenchClient.fit()` via `config.get(...)` (client side). See
   [fedprox.py](../src/fedmammobench/federated/strategies/fedprox.py) as the reference
   end-to-end example.

**Test:** `assert "my_strategy" in list_strategies()` and that `build_strategy("my_strategy", my_param=0.2)` returns your class.

---

## 2. Add a model architecture

Models also use a **registry** (`register_model`), plus a build pipeline that wires in
pretrained weights and the freeze policy automatically.

**Files:** `src/fedmammobench/models/<arch>.py`, `models/__init__.py`, `configs/model_config.py`

1. **Implement the builder** in `src/fedmammobench/models/my_arch.py`. It receives a
   `ModelConfig` and must return an `nn.Module` with the correct `in_channels`,
   `num_classes`, and `dropout`:

   ```python
   from torch import nn
   from fedmammobench.configs.schema import ModelConfig
   from fedmammobench.models.factory import register_model

   @register_model("my_arch")
   def _build_my_arch(cfg: ModelConfig) -> nn.Module:
       model = ...  # build with cfg.in_channels, cfg.num_classes, cfg.dropout
       return model
   ```

   You do **not** load weights or freeze here — `build_model()` calls
   `load_weights()` then `apply_freeze_policy()` for you afterwards.

2. **Side-effect import** in `models/__init__.py`:

   ```python
   from fedmammobench.models.my_arch import MyArchClassifier  # noqa: F401
   ```

3. **Add the name to the config Literal** in
   [model_config.py](../src/fedmammobench/configs/model_config.py) so validation accepts it:

   ```python
   name: Literal[
       "resnet18", "resnet50", "efficientnet_b0", "densenet121", "inception_v3", "my_arch"
   ] = "resnet18"
   ```

4. **(Optional) RadImageNet support.** If you ship RadImageNet weights for this arch,
   add it to `_RADIMAGENET_SUPPORTED` in the same file, and register a weight loader
   (`@register_loader`) if the checkpoint key naming is non-standard. Otherwise
   `weight_source: radimagenet` will be rejected by `ModelConfig.validate()`.

**Test:** `build_model(ModelConfig(name="my_arch", weight_source="none"))` returns a
module; a dummy forward pass with shape `(B, in_channels, H, W)` yields `(B, num_classes)`.

---

## 3. Add a dataset

Datasets use the **registry** (`register_dataset`), symmetric with strategies and
models. Adding one no longer touches `factory.py` — you write a loader module, a
registered builder, and a side-effect import.

**Files:** `src/fedmammobench/datasets/<name>.py`, `datasets/__init__.py`

1. **Subclass `MammographyDataset`** in `src/fedmammobench/datasets/my_dataset.py`. The base
   class handles image IO and transforms; your job is to produce a list of `Sample`
   records. **Critically, populate `patient_id`** on every `Sample` — it is what
   prevents patient leakage during federated partitioning:

   ```python
   from fedmammobench.datasets.base import MammographyDataset, Sample

   class MyDataset(MammographyDataset):
       @classmethod
       def from_manifest(cls, *, manifest_path, image_root, val_fraction,
                         test_fraction, seed, grayscale, transform_train,
                         transform_eval, **kw) -> dict[str, "MyDataset"]:
           # parse manifest -> list[Sample] with patient_id set
           # split at PATIENT level (reuse _stratified_patient_split from cbis_ddsm)
           return {"train": ..., "val": ..., "test": ...}
   ```

   **Do the train/val/test split at the patient level**, never by shuffling images —
   this was bug C1. Reuse `_stratified_patient_split()` from
   [cbis_ddsm.py](../src/fedmammobench/datasets/cbis_ddsm.py) rather than reimplementing it.

2. **Register a builder** in the same module. The builder receives
   `(cfg, train_tx, eval_tx)` and returns the `{"train", "val", "test"}` mapping:

   ```python
   from fedmammobench.datasets.registry import register_dataset

   @register_dataset("my_dataset")
   def _build_my_dataset(cfg, train_tx, eval_tx):
       if not cfg.data.manifest_path or not cfg.data.image_root:
           raise ValueError("data.name=my_dataset requires manifest_path and image_root.")
       return MyDataset.from_manifest(
           manifest_path=cfg.data.manifest_path,
           image_root=cfg.data.image_root,
           val_fraction=cfg.data.val_fraction,
           test_fraction=cfg.data.test_fraction,
           seed=cfg.seed,
           grayscale=cfg.data.grayscale,
           transform_train=train_tx,
           transform_eval=eval_tx,
       )
   ```

3. **Side-effect import** in [datasets/__init__.py](../src/fedmammobench/datasets/__init__.py)
   so the decorator runs at package import (the registry is only populated by importing
   the module). `data.name` is a plain `str` validated against the registry at build
   time — there is **no `Literal` to update**.

4. **Document the manifest format** in [DATA_PREPARATION.md](DATA_PREPARATION.md):
   required columns (`image_path`, `label`, `patient_id`, `split`) and any dataset-specific
   policy fields.

**Test:** `assert "my_dataset" in list_datasets()`, then load a tiny fixture CSV and assert
`set(train_patient_ids) & set(val_patient_ids) == set()` (no leakage), the same check
used for C1.

---

## 4. Add or modify a configuration field / section

Config is split into per-section modules, each owning its own `validate()`. This is
where you enforce invariants *before* an experiment runs — cheaper than a runtime crash
after several rounds.

**Files:** the relevant `configs/<section>_config.py` (+ `experiment.py` for cross-section rules)

| Section | Module | Owns |
|---------|--------|------|
| `data`, `partitioning` | [data_config.py](../src/fedmammobench/configs/data_config.py) | split fractions, NaN patient-id check (C5) |
| `model` | [model_config.py](../src/fedmammobench/configs/model_config.py) | arch↔weights↔channels compatibility |
| `training` | [training_config.py](../src/fedmammobench/configs/training_config.py) | optimizer/loss/AMP rules (FedProx+AMP warning, C2) |
| `federated` | [federated_config.py](../src/fedmammobench/configs/federated_config.py) | client counts, gRPC limits, model-config hash (E1), `server_training` |
| cross-section | [experiment.py](../src/fedmammobench/configs/experiment.py) | preset↔channels, `unfreeze_at_epoch` reachability |

**To add a field:**

1. Add the typed field with a default to the dataclass (defaults keep every existing
   YAML valid — **never remove or rename a field** without a major version bump).
2. Add its validation rule to the same module's `validate()` (raise `ValueError` for
   hard errors, log a warning for "probably wrong but legal" combinations).
3. If the rule spans two sections (e.g. model vs training), put it in
   `ExperimentConfig.validate()` in `experiment.py` instead.
4. Re-export from `schema.py` only if it's a new public symbol — `schema.py` re-exports
   everything for backward compatibility, and `loader.py` resolves types via
   `vars(schema_mod)`, so a missing re-export breaks YAML loading.

**Test:** add a case to [test_config_modules.py](../tests/test_config_modules.py) that
constructs the section in isolation and asserts `validate()` raises (or warns) on the
bad combination. These tests need no datasets or models.

---

## 5. Cross-cutting checklist (do this for every change)

Before opening a PR / generating publication results:

- [ ] **Test added** in `tests/` covering the new branch or rule.
- [ ] `pytest tests/ -v` passes.
- [ ] All configs still load: every `configs/*.yaml` builds and `cfg.validate()` passes.
- [ ] **Reproducibility:** if RNG order, a default value, the `Trainer` signature, or
      aggregation math changed → bump `pyproject.toml` and add a `CHANGELOG.md` entry.
- [ ] **Docs:** new dataset → [DATA_PREPARATION.md](DATA_PREPARATION.md); new
      strategy/metric semantics → [METHODOLOGY.md](METHODOLOGY.md); user-facing flag →
      [README.md](../README.md).
- [ ] CI is green ([.github/workflows/ci.yml](../.github/workflows/ci.yml)).

---

## Appendix — Hybrid server-side training

The central node can train on its own data in addition to aggregating clients.
This is configured (not subclassed) via `federated.server_training`:

- **Config:** `ServerTrainingConfig` in
  [federated_config.py](../src/fedmammobench/configs/federated_config.py)
  (`enabled`, `manifest_path`, `image_root`, `dataset_name`, `local_epochs`,
  `server_weight`).
- **Logic:** [server_training.py](../src/fedmammobench/federated/server_training.py).
  `ServerTrainer` builds the server's dataset/model/optimizer; `attach_server_training`
  wraps any strategy's `aggregate_fit` so the server training step runs each round
  after aggregation. Wiring lives in `_maybe_attach_server_training` in
  [server.py](../src/fedmammobench/federated/server.py) (active in both simulation
  and gRPC paths).
- **Semantics:** `new_global = (1 - server_weight) * aggregated + server_weight *
  server_trained`, where `server_trained` is the result of `local_epochs` of SGD
  starting from the aggregated weights. `server_weight = 1.0` takes the
  server-trained weights outright.

To change the blending rule or add server-side scheduling, edit
`attach_server_training` / `ServerTrainer.train`. Tests live in
[test_server_training.py](../tests/test_server_training.py).
