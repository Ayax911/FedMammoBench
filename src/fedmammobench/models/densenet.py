"""DenseNet121 classifier for mammography classification."""

from __future__ import annotations

import torch
from torch import nn

from fedmammobench.configs.schema import ModelConfig
from fedmammobench.models._adapt import adapt_first_conv
from fedmammobench.models.factory import register_model


class DenseNet121Classifier(nn.Module):
    """Thin wrapper around ``torchvision.models.densenet121``.

    - First conv (``features.conv0``) adapted to ``in_channels``.
    - Classification head replaced with ``Dropout(p) -> Linear(in_features, num_classes)``.

    DenseNet121 is one of the three architectures with published RadImageNet
    weights.  The underlying DenseNet is reachable as ``self.backbone``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        from torchvision.models import densenet121

        backbone = densenet121(weights=None)

        if cfg.in_channels != 3:
            backbone.features.conv0 = adapt_first_conv(
                backbone.features.conv0, cfg.in_channels
            )

        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_features, cfg.num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@register_model("densenet121")
def _build_densenet121(cfg: ModelConfig) -> nn.Module:
    return DenseNet121Classifier(cfg)


__all__ = ["DenseNet121Classifier"]
