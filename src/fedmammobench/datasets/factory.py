"""Dataset factory.

:func:`build_dataset` consumes an :class:`ExperimentConfig` and returns a
``{"train", "val", "test"}`` mapping of :class:`MammographyDataset` instances
ready to be wrapped in DataLoaders.

Datasets are resolved through the :mod:`fedmammobench.datasets.registry`: each
loader self-registers a builder with ``@register_dataset``, so adding a new
dataset does not require editing this file. The only special-cased name is
``none`` (a sentinel that constructs no dataset — used by gRPC servers that
keep no local holdout).
"""

from __future__ import annotations

from fedmammobench.configs.schema import ExperimentConfig
from fedmammobench.datasets.base import MammographyDataset
from fedmammobench.datasets.registry import build_registered_dataset
from fedmammobench.datasets.transforms import build_transforms
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def build_dataset(cfg: ExperimentConfig) -> dict[str, MammographyDataset]:
    """Construct {"train", "val", "test"} datasets from a config.

    Args:
        cfg: A fully populated :class:`ExperimentConfig`. Only ``cfg.data``
            and ``cfg.training.augmentation`` are consumed.

    Raises:
        ValueError: if the dataset name is unknown, or required paths for a
            file-backed dataset are missing.
        FileNotFoundError: if the manifest / annotations file is absent.
    """
    name = cfg.data.name
    if name == "none":
        # Sentinel: the caller (typically a gRPC server with no holdout) does
        # not want any local dataset constructed. Returning an empty mapping
        # signals downstream code to disable any dataset-dependent path
        # (centralized evaluation, loss-from-labels heuristics, etc.).
        _logger.info("data.name='none' — skipping dataset construction.")
        return {}

    train_tx, eval_tx = build_transforms(
        image_size=cfg.data.image_size,
        augmentation=cfg.training.augmentation,
        in_channels=cfg.model.in_channels,
    )
    return build_registered_dataset(name, cfg, train_tx, eval_tx)


__all__ = ["build_dataset"]
