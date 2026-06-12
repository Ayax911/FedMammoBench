# Experiment Audit Guide

Use this checklist before submitting results for publication.

---

## Pre-publication Checklist

### Data integrity

- [ ] Manifest CSVs have a `patient_id` column for every dataset used.
- [ ] No patient appears in more than one split. Verify with:
  ```python
  train_pids = {s.patient_id for s in train_dataset.samples}
  val_pids   = {s.patient_id for s in val_dataset.samples}
  test_pids  = {s.patient_id for s in test_dataset.samples}
  assert train_pids.isdisjoint(val_pids)
  assert train_pids.isdisjoint(test_pids)
  assert val_pids.isdisjoint(test_pids)
  ```
- [ ] If manifest has only `train/test` split labels (no `val`), confirm you
  are on v0.2.0+ which fixes the image-level fallback (C1).
- [ ] Run `ExperimentConfig.validate()` before training to catch config errors.

### Federated evaluation

- [ ] Understand which CSV contains your primary results:
  - `server_federated_metrics.csv` — federated evaluation (weighted client
    metrics). Use for federated comparisons.
  - `server_metrics.csv` — centralized holdout evaluation (only present if
    server has a test split). Use for absolute performance comparison.
- [ ] Report federated AUC as "federated AUC (weighted mean)" to distinguish
  it from pooled AUC.
- [ ] Do not use `train_loss` to compare FedAvg vs FedProx; use `task_loss`.

### Strategy comparisons

| Comparison | Correct metric | Incorrect |
|------------|----------------|-----------|
| FedAvg vs FedProx convergence | `task_loss` | `train_loss` (includes proximal penalty for FedProx) |
| FedAvg vs FedProx performance | val AUC, sensitivity, specificity | — |
| IID vs non-IID | val AUC per round | — |

### Reproducibility

- [ ] Save `config.snapshot.yaml` for every run (done automatically).
- [ ] Record the exact version (`fedmammobench.__version__`) in the paper.
- [ ] Include hardware description (CPU/GPU model, RAM) in supplementary material.
- [ ] Include `pip freeze` output in supplementary material.

### AMP and FedProx

- [ ] If using FedProx with `mixed_precision: true`, confirm v0.2.0+ is used.
  Prior versions silently degraded FedProx to FedAvg. Check the log for:
  ```
  UserWarning: FedProx with mixed_precision=True: ...
  ```
  and consider disabling AMP for the most numerically stable results.

---

## How to Interpret the CSV Files

### `server_federated_metrics.csv`

| Column | Type | Description |
|--------|------|-------------|
| `round` | int | FL round number (1-indexed) |
| `phase` | str | Always `"federated"` |
| `loss` | float | Weighted-mean validation loss across clients |
| `accuracy` | float | Weighted-mean accuracy |
| `sensitivity` | float | Weighted-mean TP/(TP+FN) |
| `specificity` | float | Weighted-mean TN/(TN+FP) |
| `roc_auc` | float | Weighted-mean AUC (approximate — see METHODOLOGY.md) |
| `f1` | float | Weighted-mean F1 |
| `task_loss` | float | Cross-entropy only (use for strategy comparisons) |

### `server_metrics.csv`

| Column | Type | Description |
|--------|------|-------------|
| `round` | int | FL round number |
| `phase` | str | Always `"centralized"` |
| `loss` | float | Centralized test loss |
| `roc_auc` | float | Pooled AUC on server test set |

---

## Verifying Absence of Leakage

Run this script before reporting any results:

```bash
python scripts/run_evaluation.py \
    --config configs/radimagenet_resnet50_fedavg.yaml \
    --checkpoint runs/radimagenet_resnet50_fedavg/final.pt \
    --split test
```

If the test AUC is substantially higher than val AUC on the same dataset,
investigate whether the split was patient-disjoint.

---

## Known Approximations to Document in the Paper

1. **Federated AUC** is a sample-weighted mean of per-client AUCs, not the AUC
   computed over pooled predictions. It is consistent but not identical.
   Label it clearly in tables and figures.

2. **Simulated heterogeneity** via Dirichlet(α) controls label skew but not
   covariate shift (different acquisition protocols, scanner vendors).

3. **Progressive unfreezing uses round count as proxy for epoch count**: the
   server sends `current_round` and clients unfreeze at `unfreeze_at_epoch`
   rounds, regardless of `local_epochs`. For `local_epochs > 1`, the effective
   number of gradient steps before unfreezing is `unfreeze_at_epoch × local_epochs`.
