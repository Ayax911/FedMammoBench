"""Classification head factory registry and lookup utilities.

Provides central lookup and registration for classification head strategy classes.

Example:
    >>> from src.models.heads import get_head_strategy
    >>> HeadClass = get_head_strategy("standard_mlp")
    >>> head = HeadClass(in_features=2048, hidden_dim=512).build()
"""

from typing import Type

from .head_builder import HeadBuilder
from .mlp_configs.standard_mlp import StandardMLPHead

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

    Example:
        >>> StrategyCls = get_head_strategy("standard_mlp")
        >>> builder = StrategyCls(in_features=2048, num_classes=1)
        >>> module = builder.build()
    """
    if name not in _HEAD_STRATEGIES:
        raise ValueError(
            f"Unknown head strategy: {name!r}. Registered options: {sorted(_HEAD_STRATEGIES)}"
        )
    return _HEAD_STRATEGIES[name]