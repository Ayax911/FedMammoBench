# Data Preparation Guide

---

## Manifest CSV Format

All real datasets are loaded via a CSV manifest.  The required columns are:

| Column | Default name | Required | Description |
|--------|-------------|----------|-------------|
| `image_path` | `image_path` | Yes | Path to image file (relative to `image_root` or absolute) |
| `label` / `pathology` | `pathology` | Yes | Class label string (see dataset-specific notes below) |
| `patient_id` | `patient_id` | Strongly recommended | Patient identifier for leakage-free splits |
| `split` | `split` | Optional | `train`, `val`, or `test` (if absent, auto-split is used) |

Column names can be overridden in YAML:

```yaml
data:
  columns:
    image_path: filepath
    label: classification
    patient_id: pid
    split: fold
```

### Label mapping

Labels are mapped to binary (0 = benign, 1 = malignant) by each dataset loader.
Rows with unrecognised labels are **dropped** with a logged warning.

---

## CBIS-DDSM

**Download**: [TCIA CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)

Expected manifest columns:

| Column | Values |
|--------|--------|
| `pathology` | `MALIGNANT`, `BENIGN`, `BENIGN_WITHOUT_CALLBACK` |
| `patient_id` | Patient ID string (e.g. `P_00001`) |
| `split` | `train` / `test` (no val — the loader auto-creates val from train at patient level) |

```yaml
data:
  name: cbis_ddsm
  manifest_path: manifests/cbis_ddsm_manifest.csv
  image_root: data/cbis_ddsm/images
  val_fraction: 0.15
  test_fraction: 0.0   # test split is from manifest 'split' column
```

---

## Mammo-Bench

**Source**: Mammo-Bench aggregates multiple mammography datasets. Use
`scripts/partition_mammobench.py` to generate per-node manifests for gRPC
deployment.

```bash
python scripts/partition_mammobench.py \
    --manifest manifests/mammobench_full.csv \
    --num-nodes 2 \
    --output-dir manifests/
```

This creates `manifests/node0_manifest.csv`, `manifests/node1_manifest.csv`,
etc.  Patient IDs are used to ensure no patient appears in multiple nodes.

Expected manifest columns:

| Column | Values |
|--------|--------|
| `_label` (internal) | `0` (benign) / `1` (malignant) after `normal_policy` mapping |
| `patient_id` | Patient ID string |

```yaml
data:
  name: mammo_bench
  manifest_path: manifests/node0_manifest.csv
  image_root: data/mammobench/images
  normal_policy: benign   # or 'drop' to exclude normal cases
  val_fraction: 0.15
```

---

## VinDr-Mammo

**Download**: [PhysioNet VinDr-Mammo](https://physionet.org/content/vindr-mammo/)

Expected files:
- `finding_annotations.csv` — breast-level annotations with BI-RADS scores
- DICOM images in `images/` directory

```yaml
data:
  name: vindr_mammo
  annotations_path: data/vindr_mammo/finding_annotations.csv
  image_root: data/vindr_mammo/images
  image_format: dicom
  birads_3_policy: drop   # drop BI-RADS 3 (ambiguous)
  val_fraction: 0.15
  test_fraction: 0.1
```

---

## Synthetic (smoke test only)

Used for fast CPU validation (smoke tests, CI). Does not require any real data.

```yaml
data:
  name: synthetic
  synthetic_num_samples: 256
  image_size: 96
```

---

## Manifest Validation

Before training, verify your manifest has no leakage:

```python
import pandas as pd

df = pd.read_csv("manifests/node0_manifest.csv")

# 1. Check for missing patient IDs
assert df["patient_id"].notna().all(), "Missing patient_id values!"

# 2. Check patient disjointness across splits (if split column present)
if "split" in df.columns:
    grouped = df.groupby("patient_id")["split"].nunique()
    leaky = grouped[grouped > 1]
    assert len(leaky) == 0, f"Patients in multiple splits: {leaky.index.tolist()}"

# 3. Check class balance
print(df["pathology"].value_counts())
```

The config-level validation `ExperimentConfig.validate()` also checks
`val_fraction + test_fraction < 1.0` and emits a warning if patient IDs contain
NaN values.
