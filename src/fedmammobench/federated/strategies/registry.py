"""Federated strategy registry.

Single source of truth for the ``strategy.name`` -> builder mapping. Strategy
modules import this and call ``@register_strategy("name")`` at import time.
"""

from __future__ import annotations

from typing import Any, Callable

from flwr.server.strategy import Strategy

_BuildFn = Callable[..., Strategy]
_REGISTRY: dict[str, _BuildFn] = {}


def register_strategy(name: str) -> Callable[[_BuildFn], _BuildFn]:
    """Decorator that registers a strategy builder under ``name``."""

    def _decorator(fn: _BuildFn) -> _BuildFn:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Strategy name already registered: {name!r}")
        _REGISTRY[key] = fn
        return fn

    return _decorator


def build_strategy(name: str, **kwargs: Any) -> Strategy:
    """Instantiate a strategy by registered name with keyword arguments."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name!r}. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](**kwargs)


def list_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = ["register_strategy", "build_strategy", "list_strategies"]
