"""FedYogi strategy (Reddi et al., 2021).

FedYogi applies a Yogi adaptive update on the server side:
  x_{t+1} = x_t + eta * m_t / (sqrt(v_t) + tau)
  v_t = v_{t-1} + sign(delta^2 - v_{t-1}) * delta^2   (Yogi update rule)

Clients train with standard SGD/Adam locally; the server applies the adaptive
step using the aggregated pseudo-gradient (delta = x_t - aggregated_params).
No changes to client.py are required.

Hyperparameters are passed via federated.strategy.params in the YAML:
  eta:    server-side learning rate      (default 0.01)
  eta_l:  client-side LR normalizer      (default 0.0316)
  beta_1: momentum decay                 (default 0.9)
  beta_2: second moment decay            (default 0.99)
  tau:    adaptivity / numerical stability (default 0.001)
"""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedYogi

from fedmammobench.federated.strategies.fedavg import _default_fit_config, _weighted_average
from fedmammobench.federated.strategies.registry import register_strategy
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@register_strategy("fedyogi")
def build_fedyogi(**kwargs: Any) -> FedYogi:
    """Build a FedYogi strategy with metric aggregation pre-wired.

    All kwargs are forwarded to :class:`flwr.server.strategy.FedYogi`.
    ``initial_parameters`` is required and injected by the server before
    calling this builder (see server.py strategy_kwargs).
    """
    kwargs.setdefault("fit_metrics_aggregation_fn", _weighted_average)
    kwargs.setdefault("evaluate_metrics_aggregation_fn", _weighted_average)
    kwargs.setdefault("on_fit_config_fn", _default_fit_config)
    _logger.info("Building FedYogi strategy with kwargs: %s", sorted(kwargs.keys()))
    return FedYogi(**kwargs)


__all__ = ["build_fedyogi"]
