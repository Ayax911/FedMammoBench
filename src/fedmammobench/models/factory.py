"""Model factory + registry.

Models register themselves with :func:`register_model` at import time, and
:func:`build_model` is the single switch point used by training/FL code.

Build pipeline per call::

    architecture = builder(cfg)      # random weights, correct shape
    load_weights(architecture, cfg)  # inject pretrained weights
    apply_freeze_policy(architecture, cfg)

Adding a model:

    @register_model("my_arch")
    def _build_my_arch(cfg: ModelConfig) -> nn.Module:
        ...
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from fedmammo.configs.schema import ModelConfig

_BuildFn = Callable[[ModelConfig], nn.Module]
_REGISTRY: dict[str, _BuildFn] = {}


def register_model(name: str) -> Callable[[_BuildFn], _BuildFn]:
    """Decorator that registers a builder under ``name``."""

    def _decorator(fn: _BuildFn) -> _BuildFn:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Model name already registered: {name!r}")
        _REGISTRY[key] = fn
        return fn

    return _decorator


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate a model from its config.

    Steps:
    1. Build the architecture (random weights, correct shapes).
    2. Load pretrained weights via :func:`~fedmammo.models.weight_loaders.load_weights`.
    3. Apply freeze policy via :func:`~fedmammo.models.weight_loaders.apply_freeze_policy`.
    """
    # Late import avoids circular deps at module load time.
    from fedmammo.models.weight_loaders import apply_freeze_policy, load_weights

    key = cfg.name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model name: {cfg.name!r}. Registered: {sorted(_REGISTRY.keys())}"
        )
    model = _REGISTRY[key](cfg)
    load_weights(model, cfg)
    apply_freeze_policy(model, cfg)
    return model


def list_models() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


__all__ = ["build_model", "register_model", "list_models"]
