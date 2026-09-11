"""Handshake de coherencia servidor<->nodos por hash de configuración.

El paquete legacy escribió `model_config_hash()` para exactamente esto y
NUNCA lo cableó: la coherencia entre server.yaml y los node_*.yaml quedaba
custodiada por comentarios ("DEBE coincidir con server.yaml"), que no
detienen nada. Consecuencia posible: un nodo con otra cabeza u otro
`unfreeze_from` entrena un modelo con otra forma y la agregación o bien
revienta tarde (largo de ndarrays distinto) o bien — peor — pasa y produce
un Frankenstein silencioso.

Acá el hash SÍ viaja: el servidor lo mete en el config de CADA ronda
(fit y evaluate, vía `on_fit_config_fn`/`on_evaluate_config_fn`, ver
`strategies.make_on_config`) y cada nodo compara contra el suyo antes de
tocar los parámetros recibidos, levantando `RuntimeError` con ambos hashes
en el mensaje al primer desacuerdo. Con `accept_failures: false` (default)
eso aborta la corrida completa en la ronda 1.

Qué entra al hash y qué no — el criterio es "define la FORMA del modelo o
la semántica de la agregación":

- SÍ: `experiment_id` (un nodo lanzado con el YAML de otro experimento debe
  fallar), `architecture.name` y `architecture.unfreeze_from` (qué se
  entrena), `head.name` + `head.hparams` (capas y `num_classes` -> 1 vs 2
  logits), `aggregation_scope` (un nodo en `full` contra un servidor en
  `backbone` enviaría una lista de largo distinto).
- NO: `architecture.weights_path` (ruta local, puede diferir por máquina;
  los pesos efectivos llegan del servidor en la ronda 1), `loss` (el
  `pos_weight` se calcula sobre el train split de CADA nodo y es
  legítimamente distinto entre nodos), `optimizer`/`scheduler`/`data`
  (locales por definición).
"""

import hashlib
import json

from ..config import ArchitectureConfig, NamedComponentConfig

# Largo del hash expuesto. 16 hex = 64 bits: de sobra para detectar deriva
# de configuración (no es un uso criptográfico adversarial) y corto para
# leerse completo en un mensaje de error o en best.json.
_HASH_LEN = 16


def model_config_hash(
    experiment_id: str,
    architecture: ArchitectureConfig,
    head: NamedComponentConfig,
    aggregation_scope: str,
) -> str:
    """Hash estable de la parte de la config que define la forma del modelo.

    SHA-256 sobre un JSON canónico (`sort_keys=True`) de los campos que
    servidor y nodos DEBEN compartir, truncado a 16 hex. Determinista entre
    procesos y máquinas: solo depende de los valores del YAML, no de rutas
    ni del entorno.

    Args:
        experiment_id: identificador del experimento (server y nodos deben
            usar el mismo).
        architecture: sección `architecture` del YAML. Solo entran `name` y
            `unfreeze_from` — `weights_path` queda fuera a propósito (ver
            docstring del módulo).
        head: sección `head` del YAML — entran `name` y `hparams`
            completos.
        aggregation_scope: `"full"` o `"backbone"`.

    Returns:
        str: 16 caracteres hexadecimales.

    Example:
        >>> h = model_config_hash(cfg.experiment_id, cfg.architecture,
        ...                       cfg.head, cfg.aggregation_scope)
        >>> len(h)
        16
    """
    payload = {
        "experiment_id": experiment_id,
        "architecture_name": architecture.name,
        "unfreeze_from": architecture.unfreeze_from,
        "head_name": head.name,
        "head_hparams": head.hparams,
        "aggregation_scope": aggregation_scope,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_HASH_LEN]
