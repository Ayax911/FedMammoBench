"""FedAdam strategy (Reddi et al., 2021).

FedAdam applies a server-side Adam adaptive update:
  m_t = beta_1 * m_{t-1} + (1 - beta_1) * delta
  v_t = beta_2 * v_{t-1} + (1 - beta_2) * delta^2   (Adam update rule)
  x_{t+1} = x_t + eta * m_t / (sqrt(v_t) + tau)

Clients train with standard SGD/Adam locally; the server applies the adaptive
step using the aggregated pseudo-gradient (delta = x_t - aggregated_params).
No changes to client.py are required.

Hyperparameters are passed via federated.strategy.params in the YAML:
  eta:    server-side learning rate      (default 0.01)
  eta_l:  client-side LR normalizer      (default 0.0316)
  beta_1: first moment decay             (default 0.9)
  beta_2: second moment decay            (default 0.99)
  tau:    adaptivity / numerical stability (default 0.001)
"""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedAdam

from fedmammobench.federated.strategies.fedavg import _default_fit_config, _weighted_average
from fedmammobench.federated.strategies.registry import register_strategy
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@register_strategy("fedadam")
def build_fedadam(**kwargs: Any) -> FedAdam:
    """Build a FedAdam strategy with metric aggregation pre-wired.

    All kwargs are forwarded to :class:`flwr.server.strategy.FedAdam`.
    ``initial_parameters`` is required and injected by the server before
    calling this builder (see server.py strategy_kwargs).
    """
    kwargs.setdefault("fit_metrics_aggregation_fn", _weighted_average)
    kwargs.setdefault("evaluate_metrics_aggregation_fn", _weighted_average)
    kwargs.setdefault("on_fit_config_fn", _default_fit_config)
    _logger.info("Building FedAdam strategy with kwargs: %s", sorted(kwargs.keys()))
    return FedAdam(**kwargs)


__all__ = ["build_fedadam"]
