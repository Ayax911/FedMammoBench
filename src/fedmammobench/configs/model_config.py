"""Model architecture and weight-loading configuration with built-in validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Normalization presets
# ---------------------------------------------------------------------------

NORMALIZE_PRESETS: dict[str, dict[str, tuple[float, ...]]] = {
    # Standard ImageNet RGB stats
    "imagenet_rgb": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    # Luminance-weighted single-channel equivalent of ImageNet RGB
    "imagenet_gray": {"mean": (0.449,), "std": (0.226,)},
    # RadImageNet publishes (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5) for RGB
    "radimagenet_rgb": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    # Single-channel RadImageNet (grayscale mammography)
    "radimagenet_gray": {"mean": (0.5,), "std": (0.5,)},
    # Legacy default used in earlier fedmammobench configs
    "mammo_default": {"mean": (0.5,), "std": (0.25,)},
}

# Architectures that support RadImageNet weights
_RADIMAGENET_SUPPORTED = frozenset({"resnet18", "resnet50", "densenet121", "inception_v3"})

# Presets that require grayscale (1-channel) input
_GRAY_PRESETS = frozenset({"imagenet_gray", "radimagenet_gray", "mammo_default"})

# Presets that require RGB (3-channel) input
_RGB_PRESETS = frozenset({"imagenet_rgb", "radimagenet_rgb"})


@dataclass
class ModelConfig:
    """Model architecture and head settings.

    Attributes:
        name: Model identifier registered in :mod:`fedmammobench.models.factory`.
        pretrained: Legacy flag — kept for backward compatibility. When
            ``weight_source`` is ``"auto"`` (default), this flag is consulted:
            ``True`` → ``"imagenet"``, ``False`` → ``"none"``. Explicit
            ``weight_source`` values take precedence.
        weight_source: Where to load pretrained weights from.
            ``"auto"`` infers from the legacy ``pretrained`` flag.
            ``"imagenet"`` uses torchvision ImageNet defaults.
            ``"radimagenet"`` loads a RadImageNet PyTorch checkpoint.
            ``"custom"`` loads an arbitrary local checkpoint (requires
            ``checkpoint_path``).
            ``"none"`` keeps random initialization.
        checkpoint_path: Absolute (or ``~``-expanded) path to a ``.pth``
            checkpoint file. Required when ``weight_source="custom"``.
            For ``"radimagenet"`` this overrides the ``FEDMAMMOBENCH_RADIMAGENET_DIR``
            environment variable lookup.
        pretrained_num_classes: Number of output classes in the source
            checkpoint's head. Used for shape validation; the loaded head is
            always discarded in favor of a fresh head sized to ``num_classes``.
        strict_load: If ``False`` (default), missing and unexpected keys during
            checkpoint loading are logged as warnings rather than errors.
            ``True`` makes ``load_state_dict`` strict.
        dropout: Dropout probability applied at the classification head.
        num_classes: Should match :attr:`DataConfig.num_classes`.
        in_channels: 1 if grayscale; 3 otherwise. The model factory adapts
            the first conv layer accordingly.
        freeze_backbone: Freeze all backbone parameters (requires_grad=False)
            and set BatchNorm layers to eval mode to prevent running-stat drift.
        freeze_head: Freeze the classification head parameters.
        unfreeze_at_epoch: If set, backbone freezing is lifted when the
            federated round (or centralized epoch) reaches this value,
            enabling progressive unfreezing.
    """

    name: Literal[
        "resnet18", "resnet50", "efficientnet_b0", "densenet121", "inception_v3"
    ] = "resnet18"
    pretrained: bool = True
    weight_source: Literal["imagenet", "radimagenet", "custom", "none", "auto"] = "auto"
    checkpoint_path: str | None = None
    pretrained_num_classes: int | None = None
    strict_load: bool = False

    dropout: float = 0.2
    num_classes: int = 2
    in_channels: int = 1

    freeze_backbone: bool = False
    freeze_head: bool = False
    unfreeze_at_epoch: int | None = None

    def validate(self, normalize_preset: str | None = None) -> None:
        """Raise ValueError for invalid model configurations.

        Args:
            normalize_preset: The augmentation normalize_preset from
                :class:`~fedmammobench.configs.training_config.AugmentationConfig`,
                used to cross-check channel counts.
        """
        # RadImageNet weight source requires a supported architecture
        effective_source = self._effective_weight_source()
        if effective_source == "radimagenet" and self.name not in _RADIMAGENET_SUPPORTED:
            raise ValueError(
                f"weight_source='radimagenet' is not supported for architecture '{self.name}'. "
                f"Supported: {sorted(_RADIMAGENET_SUPPORTED)}"
            )

        # Custom checkpoint requires checkpoint_path
        if effective_source == "custom" and not self.checkpoint_path:
            raise ValueError(
                "weight_source='custom' requires checkpoint_path to be set."
            )

        # Normalize preset ↔ channel count consistency
        if normalize_preset is not None:
            if normalize_preset in _GRAY_PRESETS and self.in_channels != 1:
                raise ValueError(
                    f"normalize_preset='{normalize_preset}' expects 1-channel (grayscale) input "
                    f"but in_channels={self.in_channels}. Set in_channels=1 or change the preset."
                )
            if normalize_preset in _RGB_PRESETS and self.in_channels != 3:
                raise ValueError(
                    f"normalize_preset='{normalize_preset}' expects 3-channel (RGB) input "
                    f"but in_channels={self.in_channels}. Set in_channels=3 or change the preset."
                )

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")

        if self.in_channels not in (1, 3):
            raise ValueError(f"in_channels must be 1 or 3, got {self.in_channels}")

    def _effective_weight_source(self) -> str:
        """Resolve 'auto' using the legacy pretrained flag."""
        if self.weight_source != "auto":
            return self.weight_source
        return "imagenet" if self.pretrained else "none"

    def config_hash_fields(self) -> dict[str, Any]:
        """Return a dict of fields relevant for server↔client consistency checks."""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "dropout": self.dropout,
        }


__all__ = [
    "ModelConfig",
    "NORMALIZE_PRESETS",
]
