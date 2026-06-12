"""SCAFFOLD strategy — STUB.

SCAFFOLD (Karimireddy et al., 2020) introduces *control variates* on both
server and client. The aggregation involves transmitting and updating client
control variates ``c_i`` alongside the parameters, which changes the
:class:`flwr.common.FitIns`/``FitRes`` payload shapes non-trivially.

Implementation plan when filled in:

1. Subclass :class:`flwr.server.strategy.Strategy` directly (FedAvg
   aggregation is *not* a drop-in here).
2. Maintain server-side global control variate ``c``; pack ``(params, c)``
   into ``FitIns`` via a side-channel (e.g. ``configure_fit`` sends them as
   serialized scalars or stuffs them into ``parameters`` by concatenation).
3. On the client, the local update follows:

       y_i = x - eta_l * (g_i(y_i) - c_i + c)
       c_i^+ = c_i - c + (1 / (K * eta_l)) * (x - y_i)

   where ``K`` is the number of local steps.
4. The server then updates ``x`` and ``c`` from per-client deltas.

This stub registers the name and raises a NotImplementedError when used so
configs fail fast and audibly.
"""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import Strategy

from fedmammo.federated.strategies.registry import register_strategy


class ScaffoldNotImplemented(Strategy):
    """Placeholder that signals SCAFFOLD has not been implemented."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - failing on use is the point
        raise NotImplementedError(
            "SCAFFOLD is a scaffold class in this build. Implement Strategy "
            "methods or pick another strategy in your YAML."
        )


@register_strategy("scaffold")
def build_scaffold(**_kwargs: Any) -> Strategy:
    raise NotImplementedError(
        "SCAFFOLD is registered but not implemented in this build. "
        "Switch federated.strategy.name to 'fedavg' (or implement the algorithm "
        "in src/fedmammo/federated/strategies/scaffold.py)."
    )


__all__ = ["build_scaffold", "ScaffoldNotImplemented"]
