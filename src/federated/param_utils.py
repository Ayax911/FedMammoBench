"""Conversión state_dict <-> lista de ndarrays, consciente del alcance de agregación.

Flower transporta parámetros como `list[np.ndarray]` sin nombres: el orden
ES el contrato. Acá el orden es siempre el de `state_dict()` del módulo del
alcance activo (`OrderedDict`, determinista para una misma definición de
modelo), así que servidor y nodos coinciden por construcción MIENTRAS
construyan el mismo modelo — que es exactamente lo que el handshake por
hash garantiza antes de que estas funciones toquen nada. Los chequeos de
largo y shape de `set_model_ndarrays` son la segunda línea de defensa, no
la primera.

Dos decisiones que vienen del legacy y de su auditoría:

- El state_dict COMPLETO viaja, buffers de BatchNorm incluidos
  (`running_mean`/`running_var`/`num_batches_tracked`). FedAvg promediando
  buffers es el comportamiento stock de Flower y del legacy — se conserva
  por comparabilidad y se documenta (docs/FEDERATED_DESIGN.md), incluida la
  consecuencia con FedAdam/FedYogi (momentos adaptativos sobre buffers).
- `num_batches_tracked` es int64. La agregación del servidor promedia en
  float y devuelve float64; sin el cast de vuelta al dtype del tensor
  destino, `load_state_dict(strict=True)` falla o — peor según la versión
  de torch — degrada en silencio. `set_model_ndarrays` castea SIEMPRE al
  dtype del destino, por clave.
"""

import numpy as np
import torch
import torch.nn as nn

from .config import AggregationScope


def scope_module(model: nn.Sequential, scope: AggregationScope) -> nn.Module:
    """Devuelve el submódulo que viaja por la red según el alcance.

    Args:
        model: el `nn.Sequential(backbone, head)` ensamblado (2 elementos —
            misma suposición que `Trainer._split_state_dicts()`).
        scope: `"full"` -> el modelo completo; `"backbone"` -> `model[0]`.

    Returns:
        nn.Module: el módulo cuyo `state_dict()` define qué se serializa.

    Raises:
        ValueError: scope no reconocido (imposible viniendo de un config
            validado — el Literal de Pydantic ya lo atrapó — pero estas
            funciones también se usan desde scripts de verificación).

    Example:
        >>> scope_module(model, "backbone") is model[0]
        True
    """
    if scope == "full":
        return model
    if scope == "backbone":
        return model[0]
    raise ValueError(f"aggregation_scope desconocido: {scope!r}. Opciones: full, backbone")


def get_model_ndarrays(model: nn.Sequential, scope: AggregationScope) -> list[np.ndarray]:
    """Serializa el alcance activo del modelo como lista de ndarrays.

    Orden: el de `state_dict()` del módulo del alcance. `detach().cpu()`
    antes de `.numpy()` — los nodos entrenan en CUDA y numpy no ve
    tensores de GPU.

    Args:
        model: el `nn.Sequential(backbone, head)` ensamblado.
        scope: ver `scope_module()`.

    Returns:
        list[np.ndarray]: un ndarray por entrada del state_dict, dtypes
        originales (float32 para pesos, int64 para `num_batches_tracked`).

    Example:
        >>> arrays = get_model_ndarrays(model, "full")
        >>> len(arrays) == len(model.state_dict())
        True
    """
    module = scope_module(model, scope)
    return [v.detach().cpu().numpy() for v in module.state_dict().values()]


def set_model_ndarrays(
    model: nn.Sequential, arrays: list[np.ndarray], scope: AggregationScope
) -> None:
    """Carga una lista de ndarrays en el alcance activo del modelo, in-place.

    Estricta a propósito: verifica largo y shape por clave ANTES de tocar
    el modelo, y castea cada array al dtype del tensor destino (el fix de
    `num_batches_tracked` — ver el docstring del módulo). La carga final es
    `load_state_dict(strict=True)`: ninguna clave puede faltar ni sobrar.

    Args:
        model: el `nn.Sequential(backbone, head)` a mutar.
        arrays: parámetros recibidos de Flower, en el orden de
            `state_dict()` del alcance (el mismo que produce
            `get_model_ndarrays`).
        scope: ver `scope_module()`.

    Raises:
        ValueError: largo de `arrays` distinto del state_dict del alcance
            (el síntoma clásico de un nodo en `full` contra un servidor en
            `backbone`, si el handshake no corrió), o shape que no calza en
            alguna clave.

    Example:
        >>> set_model_ndarrays(model, arrays, "backbone")  # muta model[0]
    """
    module = scope_module(model, scope)
    state = module.state_dict()
    if len(arrays) != len(state):
        raise ValueError(
            f"Largo de parámetros no calza para scope={scope!r}: el modelo "
            f"tiene {len(state)} tensores, llegaron {len(arrays)} ndarrays. "
            "¿Servidor y nodo con aggregation_scope distinto?"
        )
    new_state: dict[str, torch.Tensor] = {}
    for (key, target), array in zip(state.items(), arrays):
        tensor = torch.as_tensor(array)
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"Shape no calza en {key!r} (scope={scope!r}): el modelo "
                f"espera {tuple(target.shape)}, llegó {tuple(tensor.shape)}."
            )
        # Cast SIEMPRE al dtype destino: la agregación promedia en float y
        # devuelve float64 hasta para buffers int64 (num_batches_tracked).
        new_state[key] = tensor.to(dtype=target.dtype)
    module.load_state_dict(new_state, strict=True)
