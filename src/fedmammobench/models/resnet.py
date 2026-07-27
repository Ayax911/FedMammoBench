"""ResNet18 and ResNet50 classifiers for mammography classification."""

from __future__ import annotations

import torch
from torch import nn

from fedmammobench.configs.schema import ModelConfig
from fedmammobench.models._adapt import adapt_first_conv
from fedmammobench.models._head import build_head
from fedmammobench.models.factory import register_model


def _build_resnet_classifier(backbone: nn.Module, cfg: ModelConfig) -> nn.Module:
    """Shared setup: adapt first conv + replace FC head.

    The head is built by :func:`build_head` — a plain ``nn.Linear`` (or
    ``Sequential(Dropout, Linear)`` with dropout) when ``cfg.head.hidden_dims``
    is empty, or a multi-layer MLP otherwise.
    """
    if cfg.in_channels != 3:
        backbone.conv1 = adapt_first_conv(backbone.conv1, cfg.in_channels)
    in_features = backbone.fc.in_features
    backbone.fc = build_head(in_features, cfg)
    return backbone


class ResNet18Classifier(nn.Module):
    """Thin wrapper around ``torchvision.models.resnet18``.

    - First conv adapted to ``in_channels`` (default 1 for grayscale mammograms).
    - Final FC replaced with ``Linear(in_features, num_classes)`` when
      ``dropout=0``, or ``Dropout(p) -> Linear(in_features, num_classes)``
      when ``dropout > 0``.

    The underlying ResNet is reachable as ``self.backbone`` for layer-wise LR
    schedules and weight-loader access.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = _build_resnet_classifier(resnet18(weights=None), cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet50Classifier(nn.Module):
    """Thin wrapper around ``torchvision.models.resnet50``.

    Same interface as :class:`ResNet18Classifier`.  ResNet50 is the primary
    backbone published by the RadImageNet project.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        from torchvision.models import resnet50
        self.backbone = _build_resnet_classifier(resnet50(weights=None), cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@register_model("resnet18")
def _build_resnet18(cfg: ModelConfig) -> nn.Module:
    return ResNet18Classifier(cfg)


@register_model("resnet50")
def _build_resnet50(cfg: ModelConfig) -> nn.Module:
    return ResNet50Classifier(cfg)


__all__ = ["ResNet18Classifier", "ResNet50Classifier"]
