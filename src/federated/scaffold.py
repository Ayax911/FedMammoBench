"""SCAFFOLD (Karimireddy et al. 2020, opción II) como estrategia de servidor.

No hay clase stock en `flwr.server.strategy` (verificado en flwr 1.31.0). El
servidor guarda el control variate global `c` (estado de instancia) y los
nodos guardan su `c_i` persistente (ver `client.py`).

Transporte (el `config` de flwr solo admite escalares, no tensores): los
arrays viajan en la lista de `Parameters`, con `c` concatenado tras los del
modelo.

- servidor -> nodo (fit): `[*x_t (n arrays), *c (n arrays)]`, `n = len(state_dict)`
  del alcance activo, en el orden de `get_model_ndarrays`. `config` lleva
  `"scaffold_active": True`.
- nodo -> servidor (fit): `[*delta_y_i (n), *delta_c_i (n)]`.
- evaluate: solo `x_{t+1}` (n arrays), sin `c`. Los parámetros "globales" que ve
  `TrackedStrategy` (checkpoints) son siempre los n arrays del modelo.

`c`/`c_i` se transportan sobre TODAS las entradas del state_dict (incluidos
buffers de BN) por simplicidad del corte posicional; solo las entradas que son
parámetros entrenables del nodo tienen valores no nulos -- las demás
son siempre cero (los buffers no tienen gradiente que corregir).

Agregación (participación completa, |S| = N):
    x_{t+1} = x_t + server_lr * mean_i(delta_y_i)
    c_{t+1} = c   + mean_i(delta_c_i)
"""

from typing import Any

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class ScaffoldStrategy(FedAvg):
    """Estrategia SCAFFOLD (hereda de `FedAvg` solo el muestreo/evaluación).

    Args:
        server_lr: paso del servidor (`x += server_lr * mean(delta_y)`); 1.0
            reproduce el paper para Fed-ISIC2019.
        **kwargs: se pasan a `FedAvg`.

    Example:
        >>> strategy = ScaffoldStrategy(server_lr=1.0, min_available_clients=6)
    """

    def __init__(self, *, server_lr: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_lr = server_lr
        self.c: list[np.ndarray] | None = None  # control variate global
        self._x_t: list[np.ndarray] | None = None

    def __repr__(self) -> str:
        return f"ScaffoldStrategy(server_lr={self.server_lr})"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Envía `[*x_t, *c]` y marca `scaffold_active` en el config."""
        x_t = parameters_to_ndarrays(parameters)
        if self.c is None:
            self.c = [np.zeros(a.shape, dtype=np.float32) for a in x_t]
        self._x_t = x_t
        extended = ndarrays_to_parameters([*x_t, *self.c])
        instructions = super().configure_fit(server_round, extended, client_manager)
        return [
            (client, FitIns(ins.parameters, {**ins.config, "scaffold_active": True}))
            for client, ins in instructions
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Actualiza `x` y `c` con la media simple de los `(delta_y, delta_c)`."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        assert self._x_t is not None and self.c is not None
        n = len(self._x_t)

        delta_ys: list[list[np.ndarray]] = []
        delta_cs: list[list[np.ndarray]] = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) != 2 * n:
                raise ValueError(
                    f"Scaffold: un nodo devolvió {len(arrays)} arrays, se esperaban {2 * n} "
                    "([*delta_y, *delta_c]). ¿Nodo sin soporte de scaffold_active?"
                )
            delta_ys.append(arrays[:n])
            delta_cs.append(arrays[n:])

        new_x: list[np.ndarray] = []
        new_c: list[np.ndarray] = []
        for i in range(n):
            mean_dy = np.mean([dy[i].astype(np.float64) for dy in delta_ys], axis=0)
            updated = self._x_t[i].astype(np.float64) + self.server_lr * mean_dy
            if np.issubdtype(self._x_t[i].dtype, np.integer):
                updated = np.rint(updated)
            new_x.append(updated.astype(self._x_t[i].dtype))
            mean_dc = np.mean([dc[i].astype(np.float64) for dc in delta_cs], axis=0)
            new_c.append((self.c[i].astype(np.float64) + mean_dc).astype(np.float32))
        self.c = new_c

        metrics: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn(
                [(r.num_examples, r.metrics) for _, r in results]
            )
        return ndarrays_to_parameters(new_x), metrics
