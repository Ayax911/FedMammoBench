"""Flower-based federated learning runtime."""

from fedmammobench.federated.client import FedMammoBenchClient, client_fn_factory
from fedmammobench.federated.server import run_simulation
from fedmammobench.federated.strategies import build_strategy, list_strategies, register_strategy

__all__ = [
    "FedMammoBenchClient",
    "client_fn_factory",
    "run_simulation",
    "build_strategy",
    "list_strategies",
    "register_strategy",
]
