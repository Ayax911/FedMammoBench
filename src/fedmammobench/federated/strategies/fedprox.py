"""FedProx strategy (Li et al., 2020).

FedProx adds a proximal term ``(mu / 2) * ||w - w_global||^2`` to the
client's local objective, discouraging large drifts from the global model
between communication rounds.

Server side: injects ``proximal_mu`` into each client's ``FitIns.config``
via :meth:`configure_fit`. Aggregation is identical to FedAvg.

Client side: :class:`fedmammo.federated.client.FedMammoClient` reads
``config["proximal_mu"]`` in :meth:`fit`, captures the global parameters
before any local update, and adds the proximal penalty each batch via
:meth:`fedmammo.training.Trainer.train_one_epoch`.
"""

from __future__ import annotations

from typing import Any

from flwr.common import FitIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedmammo.federated.strategies.fedavg import _weighted_average
from fedmammo.federated.strategies.registry import register_strategy
from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)


class FedProx(FedAvg):
    """FedProx server strategy (scaffold).

    Injects ``proximal_mu`` into the per-round client config so the client
    training loop can add the regularizer. Aggregation logic is identical to
    FedAvg.

    Args:
        proximal_mu: Strength of the proximal regularizer (Li et al., 2020).
    """

    def __init__(self, *, proximal_mu: float = 0.01, **kwargs: Any) -> None:
        kwargs.setdefault("fit_metrics_aggregation_fn", _weighted_average)
        kwargs.setdefault("evaluate_metrics_aggregation_fn", _weighted_average)
        super().__init__(**kwargs)
        self.proximal_mu = float(proximal_mu)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        instructions = super().configure_fit(server_round, parameters, client_manager)
        out: list[tuple[ClientProxy, FitIns]] = []
        for client, fit_ins in instructions:
            cfg: dict[str, Scalar] = dict(fit_ins.config)
            cfg["proximal_mu"] = self.proximal_mu
            cfg.setdefault("current_round", server_round)
            out.append((client, FitIns(fit_ins.parameters, cfg)))
        return out


@register_strategy("fedprox")
def build_fedprox(**kwargs: Any) -> FedAvg:
    return FedProx(**kwargs)


__all__ = ["FedProx", "build_fedprox"]
