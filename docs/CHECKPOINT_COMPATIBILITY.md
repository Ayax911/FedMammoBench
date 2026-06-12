# Checkpoint Compatibility Reference

## RadImageNet Key Remapping

RadImageNet checkpoints may have the following differences from torchvision
state_dicts:

| Transformation | Description |
|---|---|
| `module.` prefix strip | DataParallel training wraps keys with `module.` |
| Head key removal | `fc.*`, `classifier.*`, `AuxLogits.*` are always dropped |
| No other renames | Keys are otherwise identical to torchvision |

### Per-architecture details

**ResNet50** (`weight_source: radimagenet`):
- First conv key: `conv1.weight`
- Head keys dropped: `fc.weight`, `fc.bias`
- Expected missing after load: `fc.*` (replaced head; normal)

**DenseNet121** (`weight_source: radimagenet`):
- First conv key: `features.conv0.weight`
- Head keys dropped: `classifier.weight`, `classifier.bias`
- Expected missing after load: `classifier.*` (replaced head; normal)

**InceptionV3** (`weight_source: radimagenet`):
- First conv key: `Conv2d_1a_3x3.conv.weight`
- Head keys dropped: `fc.*`, `AuxLogits.*`
- Built with `aux_logits=False` → no `AuxLogits` in model state_dict
- Expected missing after load: `fc.*` (replaced head; normal)
- **Requires image_size ≥ 299**

---

## Interpreting LoadReport

After every `load_weights()` call, a `LoadReport` is logged at INFO level.

```
LoadReport source='radimagenet' arch='resnet50'
  missing=2 unexpected=0 remapped=24 shape_mismatches=0
  uri='/data/ckpts/RadImageNet-resnet50.pth'
```

| Field | Normal value | Investigate if |
|---|---|---|
| `missing_keys` | Head keys only (fc.*, classifier.*) | Non-head keys are missing |
| `unexpected_keys` | 0 | > 0 AND non-head keys |
| `remapped_keys` | 0 (torchvision) or > 0 (RadImageNet DataParallel) | Unexpected value |
| `shape_mismatches` | 0 | > 0 (architecture mismatch) |

---

## Channel Adaptation Breaking Change (sum-preserving vs. legacy_mean)

Before FASE 3 of the RadImageNet integration, `adapt_first_conv` used the
`legacy_mean` strategy: averaged pretrained 3-channel weights without a scale
factor.  For 3→1 adaptation this produces activations **3× smaller** than
the sum-preserving strategy.

If you need to reproduce runs created before this refactor, set:

```python
# In custom code only; not exposed via YAML
from fedmammobench.models._adapt import adapt_weight_tensor
adapted = adapt_weight_tensor(w, target_in_channels=1, strategy="legacy_mean")
```

---

## Flower Serialization Compatibility

Frozen parameters are **included** in the state_dict serialized by
`state_dict_to_ndarrays` and sent to the server each round.  This is correct:

- Frozen params don't change locally → FedAvg averages identical values → no-op.
- BatchNorm running stats are buffers (not parameters) and are also serialized.
- State_dict shape is identical whether or not freeze is active → no protocol change.

The payload size per round is therefore the same regardless of
`freeze_backbone` setting.
