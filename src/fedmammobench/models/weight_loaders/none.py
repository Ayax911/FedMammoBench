"""No-op weight loader — keeps random initialization."""

from __future__ import annotations

from torch import nn

from fedmammo.configs.schema import ModelConfig
from fedmammo.models.weight_loaders.base import LoadReport


class NoneLoader:
    """Retains the model's random initialization without loading any weights."""

    def load(self, model: nn.Module, cfg: ModelConfig) -> LoadReport:
        return LoadReport(
            source="none",
            arch=cfg.name,
            checkpoint_uri=None,
        )


__all__ = ["NoneLoader"]
