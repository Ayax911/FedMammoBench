"""ImageNet weight loader using torchvision's Weights API."""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

from fedmammo.configs.schema import ModelConfig
from fedmammo.models._adapt import adapt_weight_tensor
from fedmammo.models.weight_loaders.base import LoadReport
from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# Maps architecture name → (torchvision module, Weights class name)
_TORCHVISION_WEIGHTS: dict[str, tuple[str, str]] = {
    "resnet18":      ("torchvision.models", "ResNet18_Weights"),
    "resnet50":      ("torchvision.models", "ResNet50_Weights"),
    "efficientnet_b0": ("torchvision.models", "EfficientNet_B0_Weights"),
    "densenet121":   ("torchvision.models", "DenseNet121_Weights"),
    "inception_v3":  ("torchvision.models", "Inception_V3_Weights"),
}

# Architecture → key of the first conv's weight in the state_dict.
# Used to adapt the pretrained RGB conv to the model's in_channels.
_FIRST_CONV_KEY: dict[str, str] = {
    "resnet18":       "conv1.weight",
    "resnet50":       "conv1.weight",
    "efficientnet_b0": "features.0.0.weight",
    "densenet121":    "features.conv0.weight",
    "inception_v3":  "Conv2d_1a_3x3.conv.weight",
}

# Keys whose absence is expected because the head is always replaced.
_HEAD_PREFIXES = ("fc.", "classifier.", "AuxLogits.")


class ImageNetLoader:
    """Load torchvision ImageNet defaults into a model's backbone."""

    def load(self, model: nn.Module, cfg: ModelConfig) -> LoadReport:
        arch = cfg.name.lower()
        if arch not in _TORCHVISION_WEIGHTS:
            raise ValueError(
                f"No ImageNet weights registered for arch {arch!r}. "
                f"Available: {sorted(_TORCHVISION_WEIGHTS)}"
            )

        mod_name, weights_cls_name = _TORCHVISION_WEIGHTS[arch]
        mod = importlib.import_module(mod_name)
        weights_cls = getattr(mod, weights_cls_name)
        weights_obj = weights_cls.DEFAULT
        state = dict(weights_obj.get_state_dict(progress=True))

        state = _adapt_first_conv(state, arch, cfg.in_channels)

        backbone = getattr(model, "backbone", model)
        missing, unexpected = backbone.load_state_dict(state, strict=False)

        # Filter out expected missing keys (replaced head) for cleaner warnings.
        non_head_missing = [k for k in missing if not _is_head_key(k)]
        if non_head_missing:
            _logger.warning(
                "ImageNet load [%s]: unexpected missing keys: %s", arch, non_head_missing
            )
        # Unexpected keys are the old head's tensors — log at debug only.
        if unexpected:
            _logger.debug(
                "ImageNet load [%s]: discarded keys (old head): %s", arch, unexpected
            )

        _logger.info("Loaded ImageNet weights for %s (in_channels=%d)", arch, cfg.in_channels)
        return LoadReport(
            source="imagenet",
            arch=arch,
            missing_keys=missing,
            unexpected_keys=unexpected,
            checkpoint_uri=f"torchvision://{weights_cls_name}.DEFAULT",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_head_key(key: str) -> bool:
    return any(key.startswith(p) for p in _HEAD_PREFIXES)


def _adapt_first_conv(
    state: dict[str, Any], arch: str, target_in_channels: int
) -> dict[str, Any]:
    """Adapt the first-conv weight in ``state`` to ``target_in_channels``.

    Uses the ``"sum_preserving"`` strategy from :mod:`fedmammo.models._adapt`.
    Returns ``state`` unchanged when ``target_in_channels == 3`` (torchvision
    weights already have 3 channels).
    """
    if target_in_channels == 3:
        return state
    key = _FIRST_CONV_KEY.get(arch)
    if key is None or key not in state:
        return state
    w = state[key]
    adapted = adapt_weight_tensor(w, target_in_channels, strategy="sum_preserving")
    state = dict(state)   # don't mutate the cached torchvision dict
    state[key] = adapted
    return state


__all__ = ["ImageNetLoader"]
