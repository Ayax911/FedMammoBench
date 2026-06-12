"""Dataset registry — the scalable extension point for adding datasets.

A dataset is added by writing a *builder* function and decorating it with
``@register_dataset("name")``. The builder receives the fully populated
:class:`ExperimentConfig` plus the train/eval transforms and returns the
``{"train", "val", "test"}`` mapping of :class:`MammographyDataset` instances.

This mirrors the strategy and model registries (``register_strategy`` /
``register_model``): new datasets self-register at import time, so adding one no
longer requires editing :func:`fedmammobench.datasets.factory.build_dataset`.

The registry is populated by side-effect imports in
:mod:`fedmammobench.datasets` (see its ``__init__``). To add a dataset, register
your builder in your loader module and make sure that module is imported there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedmammobench.configs.schema import ExperimentConfig
    from fedmammobench.datasets.base import MammographyDataset

# A builder takes (cfg, train_tx, eval_tx) and returns the split mapping.
DatasetBuilder = Callable[..., "dict[str, MammographyDataset]"]

_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(name: str) -> Callable[[DatasetBuilder], DatasetBuilder]:
    """Register a dataset builder under ``name``.

    Raises:
        ValueError: if ``name`` is already registered (duplicate registration is
            almost always an accidental copy-paste and would silently shadow the
            earlier builder).
    """

    def _decorator(builder: DatasetBuilder) -> DatasetBuilder:
        if name in _REGISTRY:
            raise ValueError(
                f"Dataset {name!r} is already registered by "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}."
            )
        _REGISTRY[name] = builder
        return builder

    return _decorator


def build_registered_dataset(
    name: str,
    cfg: "ExperimentConfig",
    train_tx,  # noqa: ANN001
    eval_tx,  # noqa: ANN001
) -> "dict[str, MammographyDataset]":
    """Look up ``name`` and invoke its builder, or raise a helpful error."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset name: {name!r}. Registered datasets: {list_datasets()}. "
            "Add one with @register_dataset in your loader module."
        )
    return _REGISTRY[name](cfg, train_tx, eval_tx)


def list_datasets() -> list[str]:
    """Return the sorted names of all registered datasets."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    """Return True if ``name`` has a registered builder."""
    return name in _REGISTRY


__all__ = [
    "register_dataset",
    "build_registered_dataset",
    "list_datasets",
    "is_registered",
    "DatasetBuilder",
]
