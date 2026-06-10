# Changelog

## [0.2.0] — 2026-06-10

### Breaking Changes

- **Partitioning RNG order changed**: `_iid_partition_patients` and `_quantity_skew_partition_patients` now consume additional RNG calls when `min_per_client` redistribution is triggered. Experiments run with `0.1.x` using patient-aware IID or quantity-skew partitioning will produce different client splits with the same seed. Re-run baselines after upgrading.
- **`Trainer.train_one_epoch` signature changed**: `global_params` is now `torch.Tensor | None` (flat parameter vector) instead of `list[torch.Tensor] | None`. Any external code calling `train_one_epoch` with FedProx must update accordingly.

### Scientific Methodology Fixes (publication-blocking)

- **C1 — Train↔val patient leakage fixed**: When a manifest CSV contains only `train/test` split labels (no `val`), the validation fallback now uses `_stratified_patient_split()` at the patient level instead of a random image-level shuffle. This prevents the same patient from appearing in both train and val, which would artificially inflate validation AUC, sensitivity, and specificity. Affects `cbis_ddsm.py` and `mammo_bench.py`.
- **C2 — FedProx AMP underflow fixed**: The proximal term `(μ/2)||w - w_global||²` is now computed inside `torch.cuda.amp.autocast(enabled=False)` with explicit `.float()` casting. Previously, with `mixed_precision=True` and small μ, the term underflowed to zero in FP16, silently degrading FedProx to FedAvg. Any FedAvg-vs-FedProx comparison with `mixed_precision: true` from `0.1.x` should be re-run.
- **C3 — Separate task_loss and total_loss**: `Trainer.train_one_epoch` now returns both `task_loss` (cross-entropy only) and `loss` (task + proximal penalty). TensorBoard logs both under `{tag}/task_loss` and `{tag}/train_loss`. `FedMammoClient.fit` now reports `task_loss` in the metrics dict. Use `task_loss` for strategy comparisons; `train_loss` is only meaningful within a single FedProx run.
- **C4 — val_ds partitioned per client**: Each client now receives its own validation subset (IID partition of the shared `val_ds`) instead of the entire shared set. This makes federated validation metrics reflect true local distributions. The fallback to shared `val_ds` remains when `len(val_ds) < num_clients`.
- **C5 — NaN patient_id detection**: The patient_id check in `_materialize_client_partitions` now detects `float('nan')` (pandas CSV missing values) in addition to `None`. Uses `check_patient_ids_for_nan()` from `fedmammo.configs.data_config`.

### Scalability Improvements

- **E3 — FedProx memory optimization**: Global parameters for FedProx are now stored as a single flat `torch.Tensor` via `torch.nn.utils.parameters_to_vector`, reducing Python GC overhead and memory fragmentation vs. a list of ~100 per-layer tensors (ResNet50).
- **E4 — Configurable gRPC message length**: `FederatedConfig` now has `grpc_max_message_length` (default 512 MB). This is passed to `fl.server.start_server()`. Increase for large models or many simultaneous clients.
- **E2 — Round timeout**: `FederatedConfig` now has `round_timeout_seconds` (default 0 = no timeout). Passed to Flower's `ServerConfig.round_timeout`. Prevents indefinite blocking when a client goes offline.
- **E5 — min_per_client enforcement in patient-aware IID and quantity-skew**: `_iid_partition_patients` and `_quantity_skew_partition_patients` now enforce `min_per_client` by redistributing samples from the largest client. A warning is logged if redistribution is not possible.

### Configuration Refactoring

- **R1 — Config modules per section**: `schema.py` (372 lines) split into:
  - `data_config.py` — `DataConfig`, `DataColumnMapping`, `PartitioningConfig`, `check_patient_ids_for_nan()`
  - `model_config.py` — `ModelConfig`, `NORMALIZE_PRESETS`
  - `training_config.py` — `TrainingConfig`, `OptimizerConfig`, `SchedulerConfig`, `AugmentationConfig`, `LossConfig`
  - `federated_config.py` — `FederatedConfig`, `StrategyConfig`
  - `experiment.py` — `ExperimentConfig`, `EvaluationConfig`
  - `schema.py` now re-exports all symbols for backward compatibility.
  - Each section module has a `validate()` method with built-in consistency checks.
  - `ExperimentConfig.validate()` runs all section validators plus cross-section checks (preset↔channels, unfreeze_at_epoch reachability, FedProx+AMP warning).

### Infrastructure

- **E6 — CI pipeline**: Added `.github/workflows/ci.yml` with Python 3.11, `pytest --cov`, and a smoke import check. Runs on push to `main` and `feature/**` branches and on PRs to `main`.

## [0.1.0] — 2026-05-25

- Initial release with FedAvg, FedProx, SCAFFOLD, FedBN strategies.
- RadImageNet weight loading for ResNet50, DenseNet121, InceptionV3, ResNet18.
- Progressive unfreezing support.
- Simulation (Ray) and real gRPC deployment modes.
- Mammo-Bench, CBIS-DDSM, VinDr-Mammo, and synthetic dataset support.
