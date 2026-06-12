"""FedAvg strategy with weighted metric aggregation.

Extends :class:`flwr.server.strategy.FedAvg` to also aggregate the per-client
metric dicts returned by ``fit`` and ``evaluate`` using sample-count weights.
The default Flower FedAvg drops these metrics unless you provide an
``fit_metrics_aggregation_fn`` / ``evaluate_metrics_aggregation_fn``.
"""

from __future__ import annotations

from typing import Any

from flwr.common import Metrics, Scalar
from flwr.server.strategy import FedAvg

from fedmammo.federated.strategies.registry import register_strategy
from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def _weighted_average(metrics: list[tuple[int, Metrics]]) -> dict[str, Scalar]:
    """Sample-count weighted mean of every numeric key the clients reported."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total <= 0:
        return {}
    keys = set()
    for _, m in metrics:
        keys.update(m.keys())
    agg: dict[str, Scalar] = {}
    for k in keys:
        weighted = 0.0
        weight = 0
        for n, m in metrics:
            v = m.get(k, None)
            if isinstance(v, (int, float)) and v == v:  # filter NaN
                weighted += float(v) * n
                weight += n
        if weight > 0:
            agg[k] = weighted / weight
    return agg


def _default_fit_config(server_round: int) -> dict[str, Scalar]:
    """Default per-round config injected into every client's FitIns.

    Clients read ``current_round`` to apply progressive unfreezing via
    :func:`~fedmammo.models.weight_loaders.apply_freeze_policy`.
    """
    return {"current_round": server_round}


@register_strategy("fedavg")
def build_fedavg(**kwargs: Any) -> FedAvg:
    """Build a FedAvg strategy with metric aggregation pre-wired.

    All kwargs are forwarded to :class:`flwr.server.strategy.FedAvg`. The
    factory injects sensible defaults for metric aggregation and per-round
    config; callers can override them.
    """
    kwargs.setdefault("fit_metrics_aggregation_fn", _weighted_average)
    kwargs.setdefault("evaluate_metrics_aggregation_fn", _weighted_average)
    kwargs.setdefault("on_fit_config_fn", _default_fit_config)
    _logger.info("Building FedAvg strategy with kwargs: %s", sorted(kwargs.keys()))
    return FedAvg(**kwargs)


__all__ = ["build_fedavg"]
