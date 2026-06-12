# RadImageNet Implementation Architecture

This document describes the weight-loading architecture introduced to support
RadImageNet pretrained transfer learning in fedmammobench.

---

## Architecture Overview

```
ExperimentConfig
    └─ ModelConfig (weight_source, freeze_backbone, ...)
          │
          ▼
    build_model(cfg)   [models/factory.py]
          │
          ├─ 1. builder(cfg)          → nn.Module (random weights, correct shape)
          ├─ 2. load_weights(model, cfg)     → LoadReport
          └─ 3. apply_freeze_policy(model, cfg) → {trainable, total, frozen}
```

Each step is independent and testable in isolation.

---

## WeightLoader Protocol

All loaders implement the `WeightLoader` protocol (`weight_loaders/base.py`):

```python
class WeightLoader(Protocol):
    def load(self, model: nn.Module, cfg: ModelConfig) -> LoadReport: ...
```

`LoadReport` captures:
- `source` — loader name (e.g. `"radimagenet"`)
- `arch` — architecture name
- `missing_keys`, `unexpected_keys` — from `load_state_dict(strict=False)`
- `remapped_keys` — number of keys renamed (e.g. DataParallel prefix strip)
- `shape_mismatches` — keys where shapes didn't match
- `checkpoint_uri` — absolute path or `"torchvision://..."` identifier

---

## Registered Loaders

| Source | Class | Location |
|---|---|---|
| `imagenet` | `ImageNetLoader` | `weight_loaders/imagenet.py` |
| `radimagenet` | `RadImageNetLoader` | `weight_loaders/radimagenet.py` |
| `custom` | `CustomCheckpointLoader` | `weight_loaders/custom.py` |
| `none` | `NoneLoader` | `weight_loaders/none.py` |

### Resolving `weight_source="auto"` (BC)

`resolve_source(cfg)` handles the legacy `pretrained: bool` flag:

| `weight_source` | `pretrained` | Effective source |
|---|---|---|
| `"auto"` | `True` | `"imagenet"` |
| `"auto"` | `False` | `"none"` |
| anything else | any | that value (explicit wins) |

---

## Adding a New Loader

1. Create `weight_loaders/my_loader.py` implementing `WeightLoader`.
2. Register it:

```python
from fedmammobench.models.weight_loaders import register_loader
from my_package import MyLoader

register_loader("my_source", MyLoader())
```

3. Use it in a YAML config: `model.weight_source: my_source`.

---

## Adding a New Backbone

1. Create `models/my_arch.py` following the existing pattern:
   - Build with `weights=None` (factory injects weights via `load_weights`).
   - Adapt first conv with `adapt_first_conv(conv, cfg.in_channels)`.
   - Replace head with `Dropout → Linear`.
   - Expose backbone as `self.backbone`.

2. Register the builder:

```python
@register_model("my_arch")
def _build_my_arch(cfg: ModelConfig) -> nn.Module:
    return MyArchClassifier(cfg)
```

3. Import in `models/__init__.py` to trigger registration.

4. If RadImageNet weights are available:
   - Add to `_keymaps.SUPPORTED_ARCHS`.
   - Add the first-conv key to `_keymaps.FIRST_CONV_KEY`.
   - Add head prefixes to `_keymaps._HEAD_PREFIXES_BY_ARCH`.

---

## Supported Combinations

| Architecture | ImageNet | RadImageNet | Notes |
|---|---|---|---|
| `resnet18` | ✓ | ✗ | Not published by RadImageNet |
| `resnet50` | ✓ | ✓ | Primary RadImageNet backbone |
| `efficientnet_b0` | ✓ | ✗ | Not published by RadImageNet |
| `densenet121` | ✓ | ✓ | Dense connections; more mem |
| `inception_v3` | ✓ | ✓ | Requires image_size ≥ 299 |

---

## Channel Adaptation

When `cfg.in_channels != 3` (e.g. grayscale input with 3-channel pretrained
weights), the first conv is adapted using the **sum-preserving** strategy:

```
averaged = W.mean(dim=channel)        # [out, 1, k, k]
scale = src_in / target_in
new_W = averaged.repeat(target_in) * scale
```

This preserves the expected activation magnitude assuming similar pixel
statistics across channels.

The `"legacy_mean"` strategy (pre-FASE 3) omits the scale factor — use it only
to reproduce runs created before this refactor.

See `models/_adapt.py` for the implementation.
