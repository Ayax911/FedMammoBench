"""InceptionV3 classifier for mammography classification.

.. warning::
    InceptionV3 requires a minimum input resolution of **299 × 299**.
    Using smaller images (e.g. ``image_size=224``) produces incorrect activations
    due to hard-coded pooling strides inside the Inception modules.
    Always set ``data.image_size >= 299`` when using this backbone.
"""

from __future__ import annotations

import torch
from torch import nn

from fedmammo.configs.schema import ModelConfig
from fedmammo.models._adapt import adapt_first_conv
from fedmammo.models.factory import register_model


class InceptionV3Classifier(nn.Module):
    """Thin wrapper around ``torchvision.models.inception_v3``.

    - ``aux_logits=False`` to avoid the auxiliary loss path (not needed for
      federated fine-tuning; simplifies state_dict layout).
    - First conv (``Conv2d_1a_3x3.conv``) adapted to ``in_channels``.
    - Final FC replaced with ``Dropout(p) -> Linear(in_features, num_classes)``.

    The underlying Inception model is reachable as ``self.backbone``.

    .. note::
        InceptionV3 is one of the three architectures published by RadImageNet.
        It is supported only for ``weight_source="radimagenet"`` or
        ``weight_source="imagenet"``; EfficientNet-B0 and ResNet18/50 are better
        choices when RadImageNet weights are unavailable.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        from torchvision.models import inception_v3

        # aux_logits=False removes the AuxLogits branch entirely, keeping the
        # state_dict flat and compatible with standard Flower serialization.
        backbone = inception_v3(weights=None, aux_logits=False)

        # The first conv is wrapped inside a BasicConv2d container.
        # BasicConv2d exposes its nn.Conv2d as .conv.
        first_conv = backbone.Conv2d_1a_3x3.conv
        if not isinstance(first_conv, nn.Conv2d):  # pragma: no cover
            raise RuntimeError(
                "Unexpected InceptionV3 layout: Conv2d_1a_3x3.conv is not Conv2d. "
                "Adapt this code to the installed torchvision version."
            )
        if cfg.in_channels != 3:
            backbone.Conv2d_1a_3x3.conv = adapt_first_conv(first_conv, cfg.in_channels)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_features, cfg.num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@register_model("inception_v3")
def _build_inception_v3(cfg: ModelConfig) -> nn.Module:
    return InceptionV3Classifier(cfg)


__all__ = ["InceptionV3Classifier"]
