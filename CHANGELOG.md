# Changelog

## [0.8.0] — 2026-07-28

### Features

- **Resume training after a crash, for both centralized and federated runs.**
  New `--resume` flag on `scripts/run_centralized.py`, `scripts/run_federated.py`,
  and `scripts/run_server.py` — no new `ExperimentConfig`/YAML fields. Why not a
  `finally`-block save: a hard crash on this shared workstation (segfault,
  OOM-killer, CUDA driver reset — the actual failure mode this targets) skips
  Python's exception machinery entirely, so the existing `save_global_model`/
  post-`fit()` `finally` writes never run. The only real protection is a
  progressive, per-epoch/per-round safety-net checkpoint.
  - `--resume` + no existing checkpoint → runs normally, but now also writes
    the safety-net checkpoint every epoch/round (reusing the existing
    `weights/final.pt` / `weights/global_model.pt` files — no new filenames)
    so a later crash can be recovered from.
  - `--resume` + an existing checkpoint → reloads model (+ optimizer +
    scheduler, centralized) and continues until `training.epochs`/
    `federated.rounds` (interpreted as the TOTAL budget across the original
    run + resumes, not "how many more").
  - Without `--resume`: byte-identical to today.
  - Centralized: `Trainer.fit` gained `resume_checkpoint_path`/`resume_state`
    (both optional, default `None` — no signature-compatibility break) and
    `save_checkpoint`/`load_checkpoint` gained a `scheduler=` parameter
    mirroring the existing `optimizer=`. `best_value`/`best_epoch`/
    `rounds_no_improve` are seeded from `resume_state` so a resumed run
    doesn't forget the best epoch seen before a crash or reset the
    early-stopping patience counter.
  - Federated: Flower's round loop has no way to "start counting from round
    N" (its internal counter always restarts at 1 for a new process), so
    resume loads the last checkpoint's weights as the new
    `initial_parameters`, runs only the REMAINING rounds, and applies a
    `round_offset` at every point a round number reaches a log file, a
    checkpoint, or a client (`NodeMetricsRecorder.wrap()`,
    `_make_on_fit_config_fn`/`_make_on_evaluate_config_fn`,
    `_build_evaluate_fn`, `_attach_federated_logging`) — never to the calls
    Flower's own strategy makes/receives, which keep seeing Flower's raw
    round numbers. Without the client-config-fn offset, a resumed run's
    progressive-unfreeze schedule (`model.unfreeze_at_epoch`) would desync
    from what was configured.
  - Known limitations, accepted rather than solved: mixed-precision
    `GradScaler` state isn't persisted (re-adapts within a few iterations,
    same spirit as the already-undocumented-as-guaranteed RNG state);
    per-node cumulative timing (`per_node_timing.csv`) doesn't merge across
    resumes; only whole epochs/rounds are checkpointed, so a crash mid-epoch/
    mid-round loses that partial work; changing `federated.num_clients` or
    `partitioning.*` between crash and resume silently reassigns which data
    each client sees (partitioning is deterministic from `cfg.seed` alone) —
    the YAML should stay otherwise unchanged except `training.epochs`/
    `federated.rounds`.

### Fixes

- The centralized safety-net checkpoint stores the epoch as an **absolute
  0-based index** (matching `best.pt`'s existing convention), not the
  `epochs_run` count `final.pt`'s post-`fit()` save used pre-resume — the two
  only ever coincided because `start_epoch` was always 0 before this feature;
  a resumed run makes them diverge, and using the count would have silently
  miscomputed the next resume point on a second, chained resume.
- `NodeMetricsRecorder.write_timing_summary` now distinguishes "rounds
  completed this session" from "final absolute round reached" when resumed
  (`round_offset > 0`) — previously these would have looked contradictory in
  `final_summary.txt` (e.g. "6 rounds completed" next to "round #20"). Gated
  so the default (non-resumed) case stays byte-identical.
- **`model.unfreeze_at_epoch`/`unfreeze_layers` (progressive backbone
  unfreeze) now actually works for centralized training.** The docstring
  promised "federated round (or centralized epoch)", but `apply_freeze_policy`
  was only ever called once per federated round, by `federated/client.py`
  — `Trainer.fit` never called it at all, so any `centralized.yaml` setting
  `unfreeze_at_epoch` silently trained with the backbone permanently frozen
  for the whole run instead. `Trainer.fit` gained `model_cfg=`/`unfreeze_lr=`
  (both optional, default `None` — no behavior change for the ~56 existing
  configs, none of which set these) and now applies the freeze policy at the
  start of every epoch, mirroring the federated client. Because centralized
  training runs one continuous optimizer across the whole loop (unlike
  federated, which rebuilds a fresh optimizer every round), newly-unfrozen
  params aren't yet owned by it — registered via `optimizer.add_param_group`,
  the same fix already used for the federated *cyclic*
  (`local_unfreeze_at_epoch`) unfreeze. `scripts/run_centralized.py` threads
  `cfg.model` and the resolved `lr_backbone` (or `lr` if unset) through.
  Known limitation: `--resume` combined with a crash that lands *after* the
  unfreeze threshold fails loudly (`optimizer.load_state_dict` param-group
  count mismatch) rather than silently — the freshly-rebuilt optimizer before
  a resume-load only has the original group(s); resuming past that point
  currently means dropping `--resume` and accepting the lost partial epochs.

## [0.7.0] — 2026-07-28

### Features

- **Early stopping (patience) for both centralized and federated training.**
  New `training.early_stopping_patience` / `federated.early_stopping_patience`
  (default `0`, disabled — existing configs run unchanged). When > 0, stops
  after that many consecutive epochs/rounds with no improvement in
  `best_checkpoint_metric`. Requires `save_best_checkpoint: true` in the same
  section (config-level `validate()` rejects the combination otherwise) —
  early stopping needs to know which epoch/round was best both to decide when
  to stop and which weights to keep.
  - Centralized: `Trainer.fit` gained the `early_stopping_patience` parameter
    (new, optional, default `0` — no signature-compatibility break) and now
    `break`s its own epoch loop; returns `epochs_run` (actual epoch count)
    and `early_stopped` (bool) in the result dict unconditionally.
  - Federated: Flower's round loop has no native "stop early" hook, so
    `NodeMetricsRecorder.record_eval` raises a new `EarlyStoppingTriggered`
    exception from the wrapped strategy once patience is exhausted.
    `federated/server.py` catches it around `fl.simulation.start_simulation`
    (which Flower internally re-wraps as `RuntimeError(...) from ex` — caught
    via `isinstance(e.__cause__, EarlyStoppingTriggered)`) and around
    `fl.server.start_server` (propagates raw, no Flower-side wrapping) as a
    clean, expected stop (INFO log), not a crash. The existing `finally`
    blocks still save `global_model.pt`, write the summary, close sinks, and
    autoplot regardless of how the round loop ends.
  - `ExperimentConfig.validate()` warns (doesn't error) if the configured
    patience is `>=` the epoch/round budget, since it could then never fire.

### Fixes

- **`weights/final.pt` (centralized) and `weights/global_model.pt`
  (federated) now record the actual last epoch/round trained in their
  `epoch=` checkpoint metadata, not the configured budget
  (`training.epochs`/`federated.rounds`).** Previously harmless because the
  two always coincided (nothing could stop training early); early stopping
  makes them diverge. `NodeMetricsRecorder.write_timing_summary`'s "Rondas
  completadas" / `avg_seconds_per_round` / `timing_summary.csv`'s
  `num_rounds` have the same fix, derived from the actual number of
  completed rounds (`len(self._round_agg_metrics)`) instead of
  `federated.rounds`. Byte-identical output for any run that isn't
  early-stopped (the actual and configured counts always match there).
- `NodeMetricsRecorder.record_eval`'s best-checkpoint hook now treats a round
  where the tracked metric is entirely absent from the aggregated metrics
  (e.g. an all-NaN round) the same as "no improvement" for early-stopping
  purposes — previously such rounds were silently skipped, which would have
  let a NaN streak freeze the patience counter and defeat early stopping.
  `Trainer.fit`'s centralized equivalent already covered this case similarly.

## [0.6.0] — 2026-07-28

### Features

- **Best-checkpoint selection for federated training**, mirroring 0.5.0's
  centralized version. New `federated.save_best_checkpoint` (default `false`,
  off) and `federated.best_checkpoint_metric` (default `"roc_auc"`, must be
  one of `FederatedConfig.FEDERATED_BEST_CHECKPOINT_METRICS` — same list as
  `TrainingConfig.BEST_CHECKPOINT_METRICS` minus `auc_pr`, which
  `FedMammoBenchClient.evaluate` never reports so it could never trigger a
  save). When enabled, `NodeMetricsRecorder.record_eval` writes
  `weights/global_best.pt` every time the tracked weighted-average validation
  metric improves round-over-round — the aggregated parameters from that
  round's `aggregate_fit`, not a later round's. `weights/global_model.pt`
  (last round) is still always written, unchanged. `write_timing_summary()`
  now reports the tracked metric generically (was hardcoded to `roc_auc`) and
  prints the best-checkpoint path/round when tracking is on. Unlike the
  centralized path, the server does **not** reload `global_best.pt` before
  anything — federated has no terminal test pass to reload for, since
  `evaluate_fn` already runs every round; evaluate `global_best.pt` post-hoc
  with `fedmammobench-evaluate`. Off by default so existing configs/runs are
  unaffected. See `tests/test_federated_best_checkpoint.py`.

- **Weights & Biases integration**, one run per experiment, server-side only
  (federated clients — in-process Ray workers or gRPC nodes — never call
  `wandb.init`). New `wandb:` config section (`WandbConfig`, defaults
  `enabled: true`, `mode: "online"`, `project: "fedmammobench"`) wired into
  every entry point through `fedmammobench.utils.metrics_sink.MetricSink`, a
  fan-out writer that always logs to TensorBoard and additionally to W&B when
  `wandb.enabled`. `Trainer`, `federated/server.py`, and
  `federated/server_training.py` now type their `tb_writer` parameter as the
  `MetricWriter` protocol instead of the concrete `TensorBoardWriter` class —
  **no constructor signature changed**, so this is not a version-triggering
  break by the letter of the rule, but is called out here because it changes
  what `config.snapshot.yaml` records (see Risks). Federated per-node metrics
  are now also pushed to the shared server-run sink under `node_<id>/…` keys
  (`record_fit`/`record_eval` in `node_logging.py`), so per-client curves show
  up as panels inside the one W&B run instead of requiring per-node
  TensorBoard digging.
  `enabled: true` **by default**, so W&B degradation had to be safe with zero
  configuration on every existing Docker deployment, including keyless ones:
  `WandbWriter` (`fedmammobench.utils.wandb_utils`) never raises, never
  blocks past `init_timeout=30`, and never prompts (`WANDB_SILENT`/
  `WANDB_CONSOLE=off` are forced via `setdefault` *before* `import wandb`, so
  it can't hijack the stdout that `docker-deploy-federated.sh`'s readiness
  `grep` depends on). Missing package → offline degrade to online without a
  `WANDB_API_KEY` (checked via `.strip()`, so the empty string from
  `-e WANDB_API_KEY="${WANDB_API_KEY:-}"` counts as absent) → offline;
  `wandb.init` failure → one offline retry, then disabled; a run that fails
  to log 10 times auto-disables itself. Offline runs are not lost — they land
  in `<out_root>/wandb/wandb/offline-run-*/` (inside the `runs/` bind mount)
  and sync later with `wandb sync`. `docker-deploy-federated.sh`,
  `run-queue.sh`, `run-exp{20-22,24-26,28-31,50-55}.sh`, and the
  `run-experiment`/`run-batch`/`run-research-suite` GitHub Actions workflows
  now pass `WANDB_API_KEY` through to server/centralized containers only
  (never to client/node containers). See `tests/test_wandb_writer.py` and the
  "Weights & Biases" section of `configs/README.md` for host/CI setup.

- **Automatic plotting after every run, and a fixed federated per-node plotting
  bug.** `scripts/plot_experiment.py`'s plotting logic moved into
  `fedmammobench.plotting` (importable, testable, no longer re-executed as a
  script via `_load_script_main`); `plot_experiment.py` is now a thin CLI
  wrapper with unchanged flags. Fixed: `_collect_node_dfs` parsed client
  directory names with `.replace("cid_", "").replace("node_", "")` and then
  `int(...)`, but `NodeMetricsRecorder` has only ever created
  `clients/client_<id>/` — `int("client_0")` raised `ValueError`, the
  `except` swallowed it, and **every** federated run silently produced zero
  `nodes_*.png` files (confirmed: none existed anywhere under `runs/` before
  this fix). Directory names are now parsed with a trailing-digits regex, so
  `client_<N>`, `cid_<N>`, and `node_<N>` all work. Two new plots use data
  that already existed but was never charted: `nodes_loss_train_val.png`
  (train loss and val loss side by side, one line per node) and
  `node_<N>_loss.png` per node (train solid + val dashed on the same axes,
  making per-node overfitting visible at a glance — the pattern that
  motivated this whole release, see 0.5.0 and exp50-55). `autoplot(run_dir)`
  now runs automatically at the end of centralized (`run_centralized.py`) and
  federated (`server.py`, both simulation and gRPC `finally` blocks) training,
  wrapped so a plotting failure (including `matplotlib` not being installed)
  never fails the run — only logs a warning. `matplotlib` moved from
  `requirements.txt`-only to also being a declared `pyproject.toml`
  dependency (the two manifests had diverged). See `tests/test_plotting.py`.

### Fixes

- `scripts/run_centralized.py` now closes its metric sink (`TensorBoardWriter`
  previously, `MetricSink` now) inside a `finally` block. Previously, an
  exception during `trainer.fit()` or evaluation would leak the writer —
  harmless for TensorBoard, but would leave a W&B run stuck in the "running"
  state in the UI forever since `run.finish()` was never called.

### Risks / Notes

- `wandb: {...}` now appears in every future `config.snapshot.yaml` (the
  full resolved `ExperimentConfig` is passed to `wandb.init(config=...)` and
  is also what gets snapshotted to disk by `save_config`). `WandbConfig` is
  deliberately kept free of any credential field — see its docstring — so
  this is not a secrets leak, but it will show up in config diffs between
  0.5.x and 0.6.x runs.
- Old `configs/*.yaml` do not set `wandb:` and get the dataclass default
  (`enabled: true`). Anyone re-running a pre-0.6.0 config against a rebuilt
  Docker image will start emitting W&B data (offline, if no
  `WANDB_API_KEY`) unless `wandb.enabled: false` is added explicitly.
- Requires a Docker image rebuild to take effect: `Dockerfile` installs from
  `requirements.txt` (now pinned `wandb>=0.17,<0.22`), and every deploy
  script pulls `ayax911/federal-learning:latest`. Until the image is rebuilt
  and republished, containers fall back to the "wandb not installed" path
  (one INFO log line, no data), which does not affect training.

## [0.5.0] — 2026-07-28

### Features

- **Best-checkpoint selection for centralized training**, via
  `training.save_best_checkpoint` (default `false`, off) and
  `training.best_checkpoint_metric` (default `"roc_auc"`, must be one of
  `TrainingConfig.BEST_CHECKPOINT_METRICS` — all "higher is better":
  `roc_auc`, `f1`, `accuracy`, `auc_pr`, `precision`, `recall`). When enabled,
  `Trainer.fit()` overwrites `weights/best.pt` every time the tracked
  validation metric improves, and returns `best_epoch` /
  `best_val_<metric>` in its metrics dict. `scripts/run_centralized.py`
  reloads `best.pt` before the final test-set evaluation instead of using the
  last epoch's weights, and records which checkpoint was used in
  `test_metrics.csv` (`checkpoint` column: `"final"` or `"best_epoch_<N>"`).
  `weights/final.pt` is still always written (last epoch), unchanged.
  Motivated by exp50-55 (`configs/exp50`-`exp55`): all six linear-probing runs
  reached their best `val_roc_auc` within the first 3-15 of 100 epochs, then
  degraded from overfitting (`train_loss` → ~0, `val_loss` up 3-9x) — the
  pipeline had no way to recover or report that better checkpoint, so the
  reported test AUC (0.834-0.852) understated what each config actually
  achieved (~0.86-0.865). Off by default so existing configs/runs are
  unaffected. See `tests/test_best_checkpoint.py`.

## [0.4.0] — 2026-07-25

### Features

- **Configurable multi-layer classification head** for all five architectures
  (resnet18/50, efficientnet_b0, densenet121, inception_v3), via
  `model.head` (`HeadConfig`): `hidden_dims`, `activation`
  (`relu`/`gelu`/`tanh`/`leaky_relu`), `batch_norm`. Built by the new shared
  `fedmammobench.models._head.build_head()`, used by every model builder
  instead of a hand-rolled `Linear`/`Sequential(Dropout, Linear)`. The final
  output layer always produces raw logits — no activation follows it, since
  softmax/sigmoid is applied by the loss (`CrossEntropyLoss`/
  `BCEWithLogitsLoss`), not the model.
  Default (`hidden_dims: []`) reproduces the previous head exactly — same
  module structure and state_dict keys — so existing configs and checkpoints
  are unaffected. `ModelConfig.config_hash_fields()` now includes the head
  shape so the server↔client consistency check in gRPC mode catches a
  mismatched head config. See `docs/EXTENDING.md` §2 and
  `configs/reference.yaml`.

## [0.3.0] — 2026-06-11

### Breaking Changes

- **Package renamed `fedmammo` → `fedmammobench`**. All imports change
  (`import fedmammo` → `import fedmammobench`), the console scripts become
  `fedmammobench-centralized` / `fedmammobench-federated` / `fedmammobench-evaluate`,
  and the RadImageNet environment variable is now `FEDMAMMOBENCH_RADIMAGENET_DIR`
  (was `FEDMAMMO_RADIMAGENET_DIR`). Update your shell environment and any scripts.
- **Synthetic dataset removed**. `SyntheticMammographyDataset`, the
  `data.name: synthetic` option, and the `data.synthetic_num_samples` field are
  gone, along with the `*_synthetic.yaml` configs. Smoke tests now use a tiny
  on-disk PNG fixture loaded through the CBIS-DDSM loader. Servers without local
  images should use `data.name: none`.
- **`DataConfig.name` is now `str`** (was a closed `Literal`). It is validated
  against the dataset registry at build time, so new datasets need no edit here.
  The default changed from `synthetic` to `cbis_ddsm`.

### Features

- **Dataset registry (scalability)**: datasets now self-register via
  `@register_dataset("name")` (in `fedmammobench.datasets.registry`), mirroring the
  strategy and model registries. Adding a dataset no longer requires editing
  `build_dataset()` or any `Literal` — see `docs/EXTENDING.md` §3.
- **Hybrid server-side training**: the central node can train on its own dataset
  after each round's aggregation via `federated.server_training`
  (`ServerTrainingConfig` + `fedmammobench.federated.server_training`). New global
  weights = `(1 - server_weight) * aggregated + server_weight * server_trained`.
  Composes with any strategy; active in both simulation and gRPC paths. See
  `configs/fedavg_server_training.yaml`.
- **Per-node metrics, timing, and global-model checkpoint**
  (`fedmammobench.federated.node_logging.NodeMetricsRecorder`): each node's
  per-round fit/evaluate metrics are written to `runs/<name>/clients/client_<id>/`
  (CSV + TensorBoard); clients report `fit_seconds` / `eval_seconds`; the server
  writes per-round timing (`server_timing.csv`) and an overall `timing_summary.csv`;
  and the final aggregated **global model** is saved to `runs/<name>/global_model.pt`
  for post-hoc verification with `fedmammobench-evaluate`. All I/O happens in the
  server process (no Ray write contention). See README "Outputs, Per-Node Metrics
  & Timing".

## [0.2.0] — 2026-06-10

### Breaking Changes

- **Partitioning RNG order changed**: `_iid_partition_patients` and `_quantity_skew_partition_patients` now consume additional RNG calls when `min_per_client` redistribution is triggered. Experiments run with `0.1.x` using patient-aware IID or quantity-skew partitioning will produce different client splits with the same seed. Re-run baselines after upgrading.
- **`Trainer.train_one_epoch` signature changed**: `global_params` is now `torch.Tensor | None` (flat parameter vector) instead of `list[torch.Tensor] | None`. Any external code calling `train_one_epoch` with FedProx must update accordingly.

### Scientific Methodology Fixes (publication-blocking)

- **C1 — Train↔val patient leakage fixed**: When a manifest CSV contains only `train/test` split labels (no `val`), the validation fallback now uses `_stratified_patient_split()` at the patient level instead of a random image-level shuffle. This prevents the same patient from appearing in both train and val, which would artificially inflate validation AUC, sensitivity, and specificity. Affects `cbis_ddsm.py` and `mammo_bench.py`.
- **C2 — FedProx AMP underflow fixed**: The proximal term `(μ/2)||w - w_global||²` is now computed inside `torch.cuda.amp.autocast(enabled=False)` with explicit `.float()` casting. Previously, with `mixed_precision=True` and small μ, the term underflowed to zero in FP16, silently degrading FedProx to FedAvg. Any FedAvg-vs-FedProx comparison with `mixed_precision: true` from `0.1.x` should be re-run.
- **C3 — Separate task_loss and total_loss**: `Trainer.train_one_epoch` now returns both `task_loss` (cross-entropy only) and `loss` (task + proximal penalty). TensorBoard logs both under `{tag}/task_loss` and `{tag}/train_loss`. `FedMammoBenchClient.fit` now reports `task_loss` in the metrics dict. Use `task_loss` for strategy comparisons; `train_loss` is only meaningful within a single FedProx run.
- **C4 — val_ds partitioned per client**: Each client now receives its own validation subset (IID partition of the shared `val_ds`) instead of the entire shared set. This makes federated validation metrics reflect true local distributions. The fallback to shared `val_ds` remains when `len(val_ds) < num_clients`.
- **C5 — NaN patient_id detection**: The patient_id check in `_materialize_client_partitions` now detects `float('nan')` (pandas CSV missing values) in addition to `None`. Uses `check_patient_ids_for_nan()` from `fedmammobench.configs.data_config`.

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
