"""Cyclic weight transfer (FLamby): un cliente entrena por ronda, el modelo pasa al siguiente.

No es una agregación simultánea: `configure_fit()` elige UN solo cliente por
ronda y `aggregate_fit()` no promedia nada -- los parámetros que devuelve ese
cliente SON los nuevos parámetros globales (N=1, sin ponderación). La
evaluación federada (`configure_evaluate`/`aggregate_evaluate`, heredadas de
`FedAvg`) sigue corriendo sobre TODOS los nodos: Cyclic solo cambia el fit.
El cliente (`client.py`) no necesita ningún cambio -- ve un fit normal.

Orden de visita:
- `order="fixed"`: los clientes conectados ordenados por `cid`, siempre el mismo
  ciclo.
- `order="shuffled"`: al comenzar cada vuelta completa se re-sortea una
  permutación con `numpy.random.default_rng(seed + n_vuelta)` (reproducible).

CUIDADO (documentado en el informe): en flwr 1.31 sobre gRPC el `cid` lo asigna
el servidor al conectarse, así que "quién es el primero" del orden fijo es
estable DENTRO de una corrida pero no necesariamente entre corridas. Para el
paper, una "ronda" de Cyclic es una vuelta completa de los K clientes; acá una
ronda de flwr == UN cliente, así que una vuelta completa = `num_nodes` rondas.
"""

from typing import Any, Literal

import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class CyclicStrategy(FedAvg):
    """Estrategia Cyclic sobre la API `Strategy` de flwr (hereda de `FedAvg`).

    Args:
        order: `"fixed"` o `"shuffled"` (ver docstring del módulo).
        seed: semilla base de la permutación en `"shuffled"`.
        **kwargs: se pasan a `FedAvg` (`min_available_clients`,
            `on_fit_config_fn`, `initial_parameters`, ...).

    Example:
        >>> strategy = CyclicStrategy(order="shuffled", min_available_clients=6)
    """

    def __init__(
        self, *, order: Literal["fixed", "shuffled"] = "fixed", seed: int = 0, **kwargs: Any
    ) -> None:
        if order not in ("fixed", "shuffled"):
            raise ValueError(f"order debe ser 'fixed' o 'shuffled', recibió {order!r}")
        super().__init__(**kwargs)
        self.order = order
        self.seed = seed
        self._visit_count = 0
        self._current_cycle: list[str] = []

    def __repr__(self) -> str:
        return f"CyclicStrategy(order={self.order!r})"

    def _next_cid(self, available: list[str]) -> str:
        """Siguiente cid a entrenar; recompone el ciclo al empezar cada vuelta."""
        n = len(available)
        if self._visit_count % n == 0 or set(self._current_cycle) != set(available):
            cycle_idx = self._visit_count // n
            ordered = sorted(available)
            if self.order == "shuffled":
                rng = np.random.default_rng(self.seed + cycle_idx)
                ordered = [ordered[i] for i in rng.permutation(n)]
            self._current_cycle = ordered
            self._visit_count = cycle_idx * n  # re-alinea si cambió el conjunto
        cid = self._current_cycle[self._visit_count % n]
        self._visit_count += 1
        return cid

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Un solo cliente por ronda, con los parámetros globales actuales."""
        config: dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        client_manager.wait_for(num_clients=self.min_available_clients)
        clients = client_manager.all()
        cid = self._next_cid(list(clients.keys()))
        return [(clients[cid], fit_ins)]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Los parámetros del único cliente entrenado son los nuevos globales."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        assert len(results) == 1, "Cyclic entrena un solo cliente por ronda"
        _, fit_res = results[0]

        metrics: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn([(fit_res.num_examples, fit_res.metrics)])
        return fit_res.parameters, metrics
