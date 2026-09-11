"""`TrackedStrategy`: logging + mejor-ronda por COMPOSICIÓN, no por monkey-patch.

El legacy le pegaba comportamiento a una estrategia stock reasignando sus
métodos bound (`strategy.aggregate_evaluate = wrapped`) en un orden
apilado a mano y frágil (tres wrappers, uno de ellos condicional, con
comentarios explicando en qué orden había que aplicarlos para que el
`server training` viera los parámetros correctos). Acá `TrackedStrategy`
implementa la interfaz de 6 métodos de `Strategy` DELEGANDO a la estrategia
interna (`self.inner`, la clase stock de `strategies.build_strategy`) para
la matemática de agregación, y solo envuelve `aggregate_fit`/
`aggregate_evaluate` con lo que este proyecto necesita encima: registrar
la ronda en `metrics.csv`, decidir si es la mejor ronda vista, y
persistir el checkpoint global cuando lo es.

Reemplaza también el early stopping por excepción del legacy (que se
des-envolvía distinto en simulación que en gRPC — una verruga documentada
en su propia auditoría): acá las rondas son fijas
(`FederationConfig.rounds`) y lo único que hay es un TRACKER de mejor
ronda (`EarlyStopping(patience=None)`, que nunca activa `should_stop`),
cumpliendo el invariante del repo "evaluar el mejor checkpoint, nunca el
último" sin ninguna excepción de control de flujo.
"""

import json
from pathlib import Path
from typing import Any

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from ..checkpoint import save_checkpoint
from ..tracking import MetricsLogger
from ..train.early_stopping import EarlyStopping
from .config import AggregationScope
from .param_utils import set_model_ndarrays


class TrackedStrategy(Strategy):
    """Envuelve una estrategia stock con logging por ronda + mejor-modelo.

    Attributes expuestos para que `server.py` arme plots/resumen al
    terminar: `round_history` (una entrada por ronda con TODAS las
    claves — `train_*`, `val_*` — de esa ronda) y `best_round`/
    `best_metric_value`/`best_checkpoint_path`.
    """

    def __init__(
        self,
        inner: Strategy,
        *,
        logger: MetricsLogger,
        template_model: Any,  # nn.Sequential — Any para no importar torch acá arriba
        scope: AggregationScope,
        model_hash: str,
        best_metric_name: str,
        best_metric_mode: str,
        checkpoint_dir: Path,
        run_dir: Path,
        save_every_rounds: int | None = None,
    ) -> None:
        self.inner = inner
        self.logger = logger
        self.template_model = template_model
        self.scope = scope
        self.model_hash = model_hash
        self.best_metric_name = best_metric_name
        self.tracker = EarlyStopping(patience=None, mode=best_metric_mode)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_dir = Path(run_dir)
        self.save_every_rounds = save_every_rounds

        self.round_history: list[dict[str, float]] = []
        self.best_round: int | None = None
        self.best_checkpoint_path: Path | None = None
        self._latest_ndarrays: list | None = None  # de aggregate_fit, para checkpointear
        self._latest_fit_metrics: dict[str, float] = {}  # de aggregate_fit, para la fila de la ronda

    # -- passthrough puro: la matemática de agregación es toda del inner -- #

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.inner.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        return self.inner.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return self.inner.configure_evaluate(server_round, parameters, client_manager)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        # El servidor no tiene datos (requisito de diseño): sin evaluate_fn,
        # esto es None siempre -- toda la evaluación es la federada de
        # aggregate_evaluate. Delegar igual, por si algún día se cablea un
        # evaluate_fn centralizado en la estrategia interna.
        return self.inner.evaluate(server_round, parameters)

    # -- envueltos: acá vive lo que este proyecto agrega -- #

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Delega la agregación de pesos+métricas y guarda los ndarrays.

        Los ndarrays quedan en `self._latest_ndarrays` para que
        `aggregate_evaluate()` de ESTA MISMA ronda (que Flower llama justo
        después, sobre los mismos parámetros recién agregados) los pueda
        checkpointear si resulta ser la mejor ronda -- sin volver a pedirle
        los pesos a nadie.
        """
        parameters, metrics = self.inner.aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self._latest_ndarrays = parameters_to_ndarrays(parameters)
        self._latest_fit_metrics = dict(metrics)
        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Delega la agregación de la evaluación federada, luego registra la ronda.

        `metrics` que devuelve `self.inner.aggregate_evaluate` ya viene del
        `evaluate_metrics_aggregation_fn = weighted_average` cableado en
        `strategies.build_strategy()` -- acá solo se combina con las
        métricas de fit de la MISMA ronda (guardadas por
        `aggregate_fit()`), se registra en `metrics.csv`/TensorBoard/W&B, y
        se decide si esta ronda es la mejor vista.

        CUIDADO: el promedio ponderado de AUCs por nodo que llega en
        `metrics[f"val_{name}"]` NO es el AUC del pool combinado de
        predicciones -- ver docs/FEDERATED_DESIGN.md.
        """
        loss, eval_metrics = self.inner.aggregate_evaluate(server_round, results, failures)

        row: dict[str, float] = {**self._latest_fit_metrics, **eval_metrics}
        if loss is not None:
            row.setdefault("val_loss", float(loss))
        self.round_history.append({"round": float(server_round), **row})
        self.logger.log(server_round, row)
        self._append_server_log(
            f"ronda {server_round}: "
            + " ".join(f"{k}={v:.4f}" for k, v in sorted(row.items()))
        )

        metric_key = f"val_{self.best_metric_name}"
        if metric_key in row and self._latest_ndarrays is not None:
            improved = self.tracker.step(row[metric_key])
            if improved:
                self._save_best(server_round, row[metric_key])
        if self.save_every_rounds and server_round % self.save_every_rounds == 0:
            self._save_periodic(server_round)

        return loss, eval_metrics

    # -- helpers internos -- #

    def _append_server_log(self, text: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "server.log", "a") as f:
            f.write(text + "\n")
        print(f"[server] {text}", flush=True)

    def _save_best(self, server_round: int, metric_value: float) -> None:
        assert self._latest_ndarrays is not None
        set_model_ndarrays(self.template_model, self._latest_ndarrays, self.scope)
        checkpoint_path = self.checkpoint_dir / f"best_round{server_round}.pt"
        if self.scope == "full":
            save_checkpoint(
                self.template_model,
                checkpoint_path,
                epoch=server_round,
                metric_value=metric_value,
                extra_state_dicts={
                    "backbone": self.template_model[0],
                    "head": self.template_model[1],
                },
            )
        else:
            # backbone: el módulo cuyo state_dict viaja YA es model[0] --
            # guardarlo directo, sin extra_state_dicts (no hay cabeza que
            # partir: la cabeza global no existe en este alcance).
            save_checkpoint(
                self.template_model[0],
                checkpoint_path,
                epoch=server_round,
                metric_value=metric_value,
            )
        self.best_round = server_round
        self.best_checkpoint_path = checkpoint_path
        best_json = {
            "best_round": server_round,
            "metric_name": self.best_metric_name,
            "metric_value": metric_value,
            "checkpoint": str(checkpoint_path),
            "aggregation_scope": self.scope,
            "model_hash": self.model_hash,
        }
        (self.run_dir / "best.json").write_text(json.dumps(best_json, indent=2))
        self._append_server_log(
            f"ronda {server_round}: NUEVO MEJOR modelo global "
            f"({self.best_metric_name}={metric_value:.4f}) -> {checkpoint_path}"
        )

    def _save_periodic(self, server_round: int) -> None:
        assert self._latest_ndarrays is not None
        set_model_ndarrays(self.template_model, self._latest_ndarrays, self.scope)
        checkpoint_path = self.checkpoint_dir / f"round{server_round}.pt"
        module = self.template_model if self.scope == "full" else self.template_model[0]
        save_checkpoint(module, checkpoint_path, epoch=server_round, metric_value=float("nan"))
