"""Registro de estrategias de agregación + agregador de métricas ponderado.

Sin matemática de agregación propia: las cuatro estrategias son las clases
stock de `flwr.server.strategy`, elegidas por nombre desde el YAML
(`strategy.name` + `strategy.hparams`, mismo patrón `NamedComponentConfig`
que optimizer/loss). El registro es un dict plano a nivel de módulo —
convención del repo (ver la tabla de decisiones de CLAUDE.md): greppeable
en un solo lugar, sin decoradores.

Lo único bespoke es `weighted_average()`, el agregador de métricas por
muestras que el legacy ya tenía bien (con filtrado de NaN por clave) y que
acá se reimplementa limpio, y `make_on_config()`, que cablea el handshake
por hash en el config de CADA ronda — la pieza que el legacy escribió y
nunca conectó.
"""

from typing import Any, Callable

from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAdam, FedAvg, FedProx, FedYogi, Strategy

from .config import AggregationScope

# Dict plano a propósito, como _ARCHITECTURES/_OPTIMIZERS/_LOSSES. Las
# cuatro clases son stock de flwr; sus hparams de YAML se splatean tal cual
# al constructor (fedprox: proximal_mu; fedadam/fedyogi: eta, eta_l,
# beta_1, beta_2, tau). fedbn/scaffold NO están: el legacy los registraba
# como stubs que levantaban NotImplementedError al construir — registrar lo
# que no existe solo pospone el error.
_STRATEGIES: dict[str, type[Strategy]] = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedadam": FedAdam,
    "fedyogi": FedYogi,
}


def weighted_average(metrics: list[tuple[int, Metrics]]) -> dict[str, Scalar]:
    """Promedio ponderado por muestras de cada métrica numérica reportada.

    Ponderación POR CLAVE: cada métrica se promedia sobre los nodos que la
    reportaron con un valor numérico no-NaN — así un nodo que omite (o
    NaN-ea) `auc` porque su val local quedó monoclase no arrastra a cero
    las demás métricas ni al resto de los nodos. Un reporte con
    `num_examples == 0` queda excluido de todo (el mecanismo documentado
    para que un nodo se auto-excluya de la agregación de métricas). Valores
    no numéricos (ej. `node_name`) se ignoran.

    CUIDADO al leer el resultado: el promedio ponderado de AUCs por nodo NO
    es el AUC del pool combinado de predicciones (media de métricas ≠
    métrica de la unión; documentado también en docs/FEDERATED_DESIGN.md).
    Para la métrica pooled del checkpoint global, evaluarlo post-hoc con
    `src.evaluate` sobre el manifest completo.

    Args:
        metrics: lista `(num_examples, dict_de_métricas)` — la forma en que
            flwr entrega los reportes de los nodos a
            `fit_metrics_aggregation_fn`/`evaluate_metrics_aggregation_fn`.

    Returns:
        dict[str, Scalar]: una entrada por métrica que al menos un nodo
        reportó numérica y no-NaN. Vacío si no hubo ninguna.

    Example:
        >>> weighted_average([(10, {"auc": 0.9}), (30, {"auc": 0.7})])
        {'auc': 0.75}
    """
    aggregated: dict[str, Scalar] = {}
    keys: set[str] = set()
    for _, m in metrics:
        keys.update(m.keys())
    for key in sorted(keys):
        weighted_sum = 0.0
        weight = 0
        for num_examples, m in metrics:
            if num_examples <= 0:
                continue
            value = m.get(key)
            # bool es subclase de int -- se acepta como 0/1, igual que el
            # legacy. NaN se filtra con la comparación v == v.
            if isinstance(value, (int, float)) and value == value:
                weighted_sum += float(value) * num_examples
                weight += num_examples
        if weight > 0:
            aggregated[key] = weighted_sum / weight
    return aggregated


def make_on_config(
    model_hash: str, local_epochs: int, aggregation_scope: AggregationScope
) -> Callable[[int], dict[str, Scalar]]:
    """Fábrica del `on_fit_config_fn`/`on_evaluate_config_fn` de la estrategia.

    El dict que devuelve la closure viaja a cada nodo en CADA ronda, para
    fit Y para evaluate. Es el canal del handshake: el nodo compara
    `model_hash` contra el hash de su propio YAML antes de tocar los
    parámetros recibidos (ver `handshake.py` y `client.py`). También lleva
    `local_epochs` (el nodo NO tiene campo epochs propio — el servidor
    manda) y `current_round`/`aggregation_scope` como contexto.

    Nota: la clase stock `FedProx` además inyecta `proximal_mu` en el
    config de fit por su cuenta (su `configure_fit` lo agrega) — el nodo lo
    lee de ahí, no de acá.

    Args:
        model_hash: `handshake.model_config_hash(...)` del config del
            SERVIDOR.
        local_epochs: `federation.local_epochs` del server.yaml.
        aggregation_scope: `federation.aggregation_scope` del server.yaml.

    Returns:
        Callable: `server_round -> config dict` que flwr invoca por ronda.

    Example:
        >>> fn = make_on_config("9e75249b5f52d729", 2, "full")
        >>> fn(3)["current_round"]
        3
    """

    def on_config(server_round: int) -> dict[str, Scalar]:
        return {
            "current_round": server_round,
            "local_epochs": local_epochs,
            "model_hash": model_hash,
            "aggregation_scope": aggregation_scope,
        }

    return on_config


def build_strategy(
    name: str,
    *,
    initial_parameters: Parameters,
    num_nodes: int,
    on_config: Callable[[int], dict[str, Scalar]],
    accept_failures: bool,
    **hparams: Any,
) -> Strategy:
    """Construye una estrategia stock de flwr por nombre, pre-cableada.

    Fija la participación total (`fraction_* = 1.0`, `min_* = num_nodes`:
    con un nodo por base de datos, una ronda sin todos los nodos no es el
    experimento diseñado), el agregador de métricas (`weighted_average`
    para fit y evaluate) y el canal del handshake (`on_config` como
    `on_fit_config_fn` Y `on_evaluate_config_fn`). `hparams` va splateado
    al constructor stock — `proximal_mu` para fedprox; `eta`, `eta_l`,
    `beta_1`, `beta_2`, `tau` para fedadam/fedyogi.

    `initial_parameters` es SIEMPRE del servidor (su modelo plantilla, en
    el alcance activo): obligatorio para fedadam/fedyogi (sus momentos
    necesitan un punto de partida) y deseable para fedavg/fedprox (sin él,
    flwr le pide los pesos iniciales a un nodo arbitrario y la corrida
    dejaría de estar anclada a los pesos preentrenados del servidor).

    Args:
        name: clave en `_STRATEGIES`.
        initial_parameters: parámetros iniciales ya serializados
            (`ndarrays_to_parameters(get_model_ndarrays(template, scope))`).
        num_nodes: `federation.num_nodes`.
        on_config: closure de `make_on_config()`.
        accept_failures: `federation.accept_failures` — con `False`, un
            nodo caído o incoherente aborta la corrida en vez de seguir con
            menos nodos.
        **hparams: `strategy.hparams` del YAML, al constructor stock.

    Returns:
        Strategy: la estrategia lista para envolver en
            `round_tracking.TrackedStrategy`.

    Raises:
        ValueError: nombre no registrado.
        TypeError: hparam no soportado por el constructor stock (mismo
            contrato que las demás fábricas del repo: el typo revienta en
            la construcción, no en silencio).

    Example:
        >>> strategy = build_strategy("fedprox", initial_parameters=params,
        ...     num_nodes=4, on_config=fn, accept_failures=False, proximal_mu=0.1)
    """
    if name not in _STRATEGIES:
        raise ValueError(f"Estrategia desconocida: {name!r}. Opciones: {sorted(_STRATEGIES)}")
    return _STRATEGIES[name](
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_nodes,
        min_evaluate_clients=num_nodes,
        min_available_clients=num_nodes,
        accept_failures=accept_failures,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_config,
        on_evaluate_config_fn=on_config,
        **hparams,
    )
