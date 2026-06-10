# Methodology Reference

This document describes the experimental setup of FedMammo in the terms needed
for a scientific article methods section.  All parameters correspond directly to
YAML configuration fields so that experiments are fully reproducible.

---

## 1. Reproducibility

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| Global seed | `seed` | 42 | Controls RNG for data splits, partitioning, and weight initialization |
| Per-client seed | `seed + client_id + 1` | — | Each client gets a distinct but reproducible state |
| Deterministic mode | `torch.backends.cudnn.deterministic=True` | enabled | Set by `set_global_seed(..., deterministic=True)` |

**To reproduce exactly**: use the same `seed`, same `num_clients`, same
partitioning `scheme`, and the same dataset split. All effective configs are
snapshotted to `runs/<name>/config.snapshot.yaml` at startup.

Note: upgrading from v0.1.x to v0.2.x changes the RNG consumption order for
IID and quantity-skew patient-aware partitioning. Re-run baselines after
upgrading (see `CHANGELOG.md`).

---

## 2. Dataset Splits

FedMammo enforces **patient-disjoint** splits: all images from the same patient
are assigned to the same split.  This prevents data leakage that would
artificially inflate validation and test metrics.

### Split generation

**Manifest with explicit `split` column (train / val / test)**

The `split` column is respected directly.  If `val` is absent but
`val_fraction > 0`, validation samples are carved from the training set at the
*patient level* using `_stratified_patient_split()` (not at the image level).
This is the C1 fix — prior to v0.2.0 this fallback used an image-level shuffle
which could place images from the same patient in both train and val.

**Manifest without `split` column**

`_stratified_patient_split()` partitions patients into train / val / test
according to `val_fraction` and `test_fraction`, then approximates class
balance across splits by sorting patients by their majority label.

### Verification

```python
train_pids = {samples[i].patient_id for i in splits["train"]}
val_pids   = {samples[i].patient_id for i in splits["val"]}
assert train_pids.isdisjoint(val_pids)
```

---

## 3. Federated Partitioning

The training set is distributed across clients using one of three schemes.

### IID (baseline)

Patients are shuffled uniformly and divided into `num_clients` equal chunks.
All clients receive approximately the same class distribution.

### Dirichlet (label skew)

For each class, patient proportions per client are drawn from
Dir(α).  Lower α → stronger heterogeneity.  Typical values: α = 0.1 (high
non-IID), α = 0.5 (moderate), α = 1.0 (near-IID).  Partitioning is retried up
to `max_retries` times if any client receives fewer than `min_per_client`
samples.

### Quantity Skew

Patients are divided with IID labels but unequal counts drawn from a log-normal
distribution with standard deviation `quantity_skew_sigma`.  σ = 0 reduces to
IID equal counts.

### Patient-awareness

All three schemes operate at the **patient** level when `patient_ids` are
provided (which is the default for all real datasets).  A patient's images are
always assigned to the same client, eliminating cross-client leakage.

---

## 4. Transfer Learning Protocol

### Weight sources

| `weight_source` | Description |
|-----------------|-------------|
| `imagenet` | Torchvision ImageNet-1k pretrained weights |
| `radimagenet` | RadImageNet checkpoint (Mei et al., 2022) — medical-domain pretraining |
| `custom` | Arbitrary `.pth` checkpoint provided by `checkpoint_path` |
| `none` | Random initialization |

### Progressive unfreezing

When `freeze_backbone: true` and `unfreeze_at_epoch: N`:

- **Rounds 0 … N-1**: backbone frozen, only the classification head is trained.
- **Rounds N …**: backbone unfrozen, full model fine-tuning.

This two-phase approach stabilizes early training when the head is randomly
initialized while the backbone carries domain-specific features.

Typical schedule for RadImageNet:

```yaml
freeze_backbone: true
unfreeze_at_epoch: 3   # federated
# or unfreeze_at_epoch: 5 for centralized
```

### Learning rate justification

| Setting | LR |
|---------|-----|
| Centralized with ImageNet | 1e-4 |
| Federated with RadImageNet | 5e-5 |

The lower federated LR accounts for the non-IID gradient noise and the fact
that each client trains on a small local dataset per round.

---

## 5. Evaluation Protocol

### Federated evaluation (default)

After each aggregation round, a subset of clients (controlled by
`fraction_evaluate`) evaluates the global model on their **local** validation
split and returns per-client metrics.  The server computes a sample-weighted
mean:

```
metric_global = Σ_i (metric_i × n_i) / Σ_i n_i
```

**Exact metrics** (linear at fixed threshold): accuracy, sensitivity,
specificity, precision, F1, loss.

**Approximate metrics** (ranking): ROC-AUC.  Weighted-averaged AUC is a
consistent estimator of pooled AUC but not identical to it; the gap is small
when client distributions are similar.

As of v0.2.0 each client validates on a **locally-distinct** subset (IID
partition of the shared val set) rather than the identical shared set.  This
makes federated validation metrics reflect local distributions.

### Centralized holdout (optional)

Set `data.name` to a real dataset with `test_fraction > 0` on the server.  The
server evaluates the global model after each round and logs to
`server_metrics.csv`.  This requires the server operator to hold a local test
set and is only available in gRPC mode.

### Metrics

```python
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float           # = sensitivity
    f1: float
    roc_auc: float
    sensitivity: float      # TP / (TP + FN)
    specificity: float      # TN / (TN + FP)
```

Positive class: 1 (malignant).  Default threshold: 0.5 (configurable via
`evaluation.threshold`).

---

## 6. Strategy Comparison Guidelines

### FedAvg vs FedProx

When comparing strategies, use `task_loss` (cross-entropy only), **not**
`train_loss` (which includes the proximal penalty for FedProx and is therefore
not comparable across strategies).

Both `task_loss` and `train_loss` are logged separately in TensorBoard and
`server_federated_metrics.csv` as of v0.2.0.

### AMP and FedProx

As of v0.2.0, the proximal term is computed in FP32 even when
`mixed_precision: true` is set, preventing FP16 underflow.  Prior to v0.2.0,
FedProx with `mixed_precision: true` silently degraded to FedAvg.  All
FedProx results from v0.1.x with `mixed_precision: true` should be discarded.

---

## 7. Limitations

1. **Weighted AUC ≠ pooled AUC**: The federated AUC is a sample-weighted mean
   of per-client AUC values.  It is a consistent approximation but differs from
   the AUC that would be computed on the pooled predictions.  Report as
   "federated AUC (weighted)" in the paper.

2. **Synchronous FL only**: All clients must respond before a round advances.
   Straggler mitigation is not implemented.

3. **No differential privacy**: Parameter updates are aggregated in plaintext.
   Do not claim privacy guarantees beyond the basic federated setup.

4. **No secure aggregation**: The server sees each client's parameter update.

5. **Heterogeneity is simulated**: Dirichlet and quantity-skew simulate
   distribution shift on a single dataset.  In a real hospital network,
   additional domain shift (scanner type, protocol, ethnicity) would be present.
