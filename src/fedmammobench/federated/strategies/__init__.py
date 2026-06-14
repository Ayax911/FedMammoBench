"""Federated strategy registry.

Strategies are registered via the ``@register_strategy("name")`` decorator and
instantiated through ``build_strategy(name, **kwargs)``. Adding a new strategy:

1. Implement a class inheriting from ``flwr.server.strategy.Strategy``
   (or a subclass such as ``FedAvg``).
2. Decorate with ``@register_strategy("my_name")``.
3. Reference the name in YAML under ``federated.strategy.name``.
"""

from fedmammobench.federated.strategies.registry import (
    build_strategy,
    list_strategies,
    register_strategy,
)

# Side-effect imports populate the registry.
from fedmammobench.federated.strategies import fedavg  # noqa: F401
from fedmammobench.federated.strategies import fedprox  # noqa: F401
from fedmammobench.federated.strategies import fedyogi  # noqa: F401
from fedmammobench.federated.strategies import fedadam  # noqa: F401
from fedmammobench.federated.strategies import scaffold  # noqa: F401
from fedmammobench.federated.strategies import fedbn  # noqa: F401

__all__ = ["build_strategy", "list_strategies", "register_strategy"]
