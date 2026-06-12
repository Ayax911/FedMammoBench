"""EfficientNet-B0 classifier for binary mammography classification."""

from __future__ import annotations

import torch
from torch import nn

from fedmammo.configs.schema import ModelConfig
from fedmammo.models._adapt import adapt_first_conv
from fedmammo.models.factory import register_model


class EfficientNetB0Classifier(nn.Module):
    """Thin wrapper around ``torchvision.models.efficientnet_b0``.

    - First conv adapted to ``in_channels`` (default 1 for grayscale).
    - Classification head: ``Dropout(p) -> Linear(in_features, num_classes)``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # weights=None always: the factory calls load_weights() after building.
        from torchvision.models import efficientnet_b0

        backbone = efficientnet_b0(weights=None)

        # First conv lives at backbone.features[0][0].
        first_conv = backbone.features[0][0]
        if not isinstance(first_conv, nn.Conv2d):  # pragma: no cover - upstream layout change
            raise RuntimeError(
                "Unexpected EfficientNet-B0 layout: features[0][0] is not Conv2d. "
                "Adapt this code to the installed torchvision version."
            )
        if cfg.in_channels != 3:
            backbone.features[0][0] = adapt_first_conv(first_conv, cfg.in_channels)

        # Replace the classifier head.
        # backbone.classifier is Sequential(Dropout, Linear); we substitute both.
        if not isinstance(backbone.classifier, nn.Sequential) or len(backbone.classifier) < 2:
            raise RuntimeError(  # pragma: no cover
                "Unexpected EfficientNet-B0 classifier layout."
            )
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout, inplace=True),
            nn.Linear(in_features, cfg.num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@register_model("efficientnet_b0")
def _build_efficientnet_b0(cfg: ModelConfig) -> nn.Module:
    return EfficientNetB0Classifier(cfg)


__all__ = ["EfficientNetB0Classifier"]
