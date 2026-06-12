"""Custom checkpoint loader (arbitrary local .pth files)."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from fedmammobench.configs.schema import ModelConfig
from fedmammobench.models.weight_loaders.base import LoadReport
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)


class CustomCheckpointLoader:
    """Load an arbitrary local checkpoint in fedmammobench format.

    The checkpoint must be a ``.pth`` file produced by
    :func:`fedmammobench.utils.checkpoint.save_checkpoint`, i.e. a dict with at
    least a ``"state_dict"`` key.  The state_dict is loaded into the model
    backbone with ``strict=cfg.strict_load``.

    ``cfg.checkpoint_path`` must be set when ``weight_source="custom"``.
    """

    def load(self, model: nn.Module, cfg: ModelConfig) -> LoadReport:
        if not cfg.checkpoint_path:
            raise ValueError(
                "weight_source='custom' requires checkpoint_path to be set. "
                "Set model.checkpoint_path in your YAML config."
            )

        # Reuse the existing fedmammobench checkpoint helper.
        from fedmammobench.utils.checkpoint import load_checkpoint

        src = Path(cfg.checkpoint_path).expanduser().resolve()
        backbone = getattr(model, "backbone", model)
        load_checkpoint(src, backbone, strict=cfg.strict_load)

        _logger.info("Loaded custom checkpoint from %s", src)
        return LoadReport(
            source="custom",
            arch=cfg.name,
            checkpoint_uri=str(src),
        )


__all__ = ["CustomCheckpointLoader"]
