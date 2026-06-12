# Transfer Learning Guide

This guide covers using pretrained weights (ImageNet or RadImageNet) for
mammography classification in fedmammobench.

---

## 1 — Obtaining RadImageNet Weights

RadImageNet weights are published by the BMEII-AI lab.  Download them from the
official repository:

> **BMEII-AI/RadImageNet** on GitHub

The checkpoints you need are named:

| Architecture | File |
|---|---|
| ResNet50 | `RadImageNet-resnet50.pth` |
| DenseNet121 | `RadImageNet-densenet121.pth` |
| InceptionV3 | `RadImageNet-inception_v3.pth` |

EfficientNet-B0 and ResNet18 are **not** available from RadImageNet; use
`weight_source: imagenet` for those.

### Placing the checkpoints

Choose one of two options:

**Option A — per-config path**

```yaml
model:
  weight_source: radimagenet
  checkpoint_path: /absolute/path/to/RadImageNet-resnet50.pth
```

**Option B — directory via env var (recommended for shared clusters)**

```bash
export FEDMAMMOBENCH_RADIMAGENET_DIR=/data/checkpoints/radimagenet
```

Place all `.pth` files in that directory.  fedmammobench will look for
`$FEDMAMMOBENCH_RADIMAGENET_DIR/RadImageNet-{arch}.pth`.

---

## 2 — Recommended Fine-tuning Workflow

### Phase 1 — Head-only warm-up (rounds 0 – N-1)

Freeze the backbone so only the new classification head trains.  This prevents
catastrophic forgetting of the RadImageNet feature representations.

```yaml
model:
  name: resnet50
  weight_source: radimagenet
  freeze_backbone: true
  unfreeze_at_epoch: 5      # end of Phase 1

training:
  optimizer:
    lr: 1.0e-4              # head is randomly initialized → higher LR is fine
  augmentation:
    normalize_preset: radimagenet_gray
```

### Phase 2 — Full fine-tuning (rounds N onwards)

Once `current_round >= unfreeze_at_epoch`, the server-injected round counter
triggers `apply_freeze_policy` to release all parameters.  Switch to a lower
LR to protect the pretrained representations.

```yaml
model:
  freeze_backbone: true
  unfreeze_at_epoch: 5      # flip to trainable at round 5

training:
  optimizer:
    lr: 5.0e-5              # lower after unfreezing
```

### FedProx + RadImageNet

When using FedProx, start with a moderate proximal mu to anchor clients close
to the RadImageNet initialization:

```yaml
federated:
  strategy:
    name: fedprox
    params:
      proximal_mu: 0.01
```

Reduce `proximal_mu` after unfreezing to allow more local adaptation.

---

## 3 — Normalize Presets

Use the preset that matches your weight source and channel count.

| Preset | Mean | Std | Use when |
|---|---|---|---|
| `radimagenet_gray` | `(0.5,)` | `(0.5,)` | RadImageNet + grayscale (most common) |
| `radimagenet_rgb` | `(0.5, 0.5, 0.5)` | `(0.5, 0.5, 0.5)` | RadImageNet + RGB input |
| `imagenet_gray` | `(0.449,)` | `(0.226,)` | ImageNet + grayscale |
| `imagenet_rgb` | `(0.485, 0.456, 0.406)` | `(0.229, 0.224, 0.225)` | ImageNet + RGB |
| `mammo_default` | `(0.5,)` | `(0.25,)` | legacy / no pretrained weights |

Set the preset in your config:

```yaml
training:
  augmentation:
    normalize_preset: radimagenet_gray
```

If `normalize_preset` is set, the scalar `normalize_mean` / `normalize_std`
fields are ignored.  Pass a list for per-channel control without a preset:

```yaml
training:
  augmentation:
    normalize_mean: [0.485, 0.456, 0.406]
    normalize_std: [0.229, 0.224, 0.225]
```

---

## 4 — Risks and Mitigations

### Catastrophic forgetting (non-IID data)

Federated non-IID settings (e.g. Dirichlet α=0.1) can cause aggressive local
drift away from the RadImageNet representations.

**Mitigations:**
- Use `freeze_backbone: true` for the first few rounds (warm-up).
- Use FedProx with `proximal_mu ≥ 0.01`.
- Keep `local_epochs: 1` — more local steps = more drift.

### BatchNorm stats drift during frozen training

`apply_freeze_policy` sets frozen BatchNorm modules to `eval()` mode to prevent
running-stat updates.  However, `model.train()` (called at the start of each
training epoch inside `Trainer`) resets all BatchNorm to training mode, which
resumes stat updates for frozen layers.

**Effect:** BatchNorm γ/β weights are frozen (correct), but running
`mean`/`var` buffers may drift slightly during local training.  For most
federated fine-tuning scenarios this is acceptable.  Strict stat freezing
requires the trainer to re-apply `_set_bn_eval` after each `model.train()` call.

### Normalize mismatch

Using ImageNet normalization (`imagenet_gray`) with RadImageNet weights — or
vice versa — can degrade convergence.  Always match the preset to the
checkpoint source.

---

## 5 — Memory / Compute Reference

| Architecture | Params | Payload per round\* | Recommended batch |
|---|---|---|---|
| ResNet18 | ~11M | ~44 MB | 32 |
| ResNet50 | ~25M | ~100 MB | 16 |
| DenseNet121 | ~8M | ~32 MB | 16 |
| InceptionV3 | ~27M | ~108 MB | 12 |
| EfficientNet-B0 | ~5M | ~20 MB | 32 |

\* Full state_dict (float32), one client.  Total round payload = value × `num_clients`.

InceptionV3 requires `data.image_size >= 299`.  Smaller images will produce
incorrect activations due to hard-coded pooling strides in the Inception modules.
