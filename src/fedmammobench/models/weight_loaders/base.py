"""Base types for the weight-loading abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import nn

if TYPE_CHECKING:
    from fedmammo.configs.schema import ModelConfig


@dataclass
class LoadReport:
    """Structured summary of a weight-loading operation.

    Attributes:
        source: The weight source used (``"imagenet"``, ``"radimagenet"``,
            ``"custom"``, ``"none"``).
        arch: Architecture name from :attr:`ModelConfig.name`.
        missing_keys: Keys present in the model but absent from the checkpoint
            (excluding the replaced classification head).
        unexpected_keys: Keys present in the checkpoint but absent from the
            model (typically the original head weights that were discarded).
        remapped_keys: Number of checkpoint keys renamed before loading
            (e.g. stripping ``module.`` prefixes for DataParallel checkpoints).
        shape_mismatches: List of ``(key, ckpt_shape, model_shape)`` tuples
            for tensors that could not be loaded due to shape incompatibility.
        checkpoint_uri: Human-readable pointer to the source, e.g.
            ``"torchvision://ResNet18_Weights.DEFAULT"`` or an absolute path.
    """

    source: str
    arch: str
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    remapped_keys: int = 0
    shape_mismatches: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    checkpoint_uri: str | None = None


@runtime_checkable
class WeightLoader(Protocol):
    """Protocol every concrete loader must satisfy."""

    def load(self, model: nn.Module, cfg: "ModelConfig") -> LoadReport: ...


__all__ = ["LoadReport", "WeightLoader"]
