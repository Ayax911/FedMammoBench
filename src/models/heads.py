"""Classification head factory registry and lookup utilities."""

from typing import Type

from ..heads import HeadBuilder
from .standard_mlp import StandardMLPHead

# Internal registry mapping head strategy names to their uninstantiated builder classes
_HEAD_STRATEGIES: dict[str, Type[HeadBuilder]] = {
    "standard_mlp": StandardMLPHead,
}


def get_head_strategy(name: str) -> Type[HeadBuilder]:
    """Retrieves the uninstantiated classification head builder class by registered name.

    Args:
        name: Name key of the head strategy registered in `_HEAD_STRATEGIES`
            (e.g., `"standard_mlp"`).

    Returns:
        Type[HeadBuilder]: The class (builder factory) of the matching head strategy.

    Raises:
        ValueError: If the requested head strategy name is not registered in `_HEAD_STRATEGIES`.
    """
    if name not in _HEAD_STRATEGIES:
        raise ValueError(
            f"Unknown head strategy: {name!r}. Registered options: {sorted(_HEAD_STRATEGIES)}"
        )
    return _HEAD_STRATEGIES[name]