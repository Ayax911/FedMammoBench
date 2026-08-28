from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn as nn

from .reports import LoadReport


def load_weights(
    model_factory: Callable[[], nn.Module],
    weights_path: str | Path,
    key_remap: dict[str, str],
    valid_prefixes: tuple[str, ...],
    device: str = "cpu",
) -> tuple[nn.Sequential, LoadReport]:
    """Carga un checkpoint en cualquier arquitectura construida por
    model_factory, remapeando y filtrando sus claves según los parámetros
    dados, y devuelve el backbone truncado + reporte.

    Args:
        model_factory: función o clase invocable sin argumentos que
            construye el modelo base.
        weights_path: ruta al checkpoint.
        key_remap: mapeo de prefijos {prefijo_original: prefijo_real}.
        valid_prefixes: prefijos que definen qué claves pertenecen al backbone.
        device: dispositivo destino para los tensores cargados.

    Returns:
        (backbone, report)

    Raises:
        RuntimeError: el state_dict remapeado/filtrado quedó vacío, o
            matched == 0 (ver LoadReport).
    """
    model = model_factory()

    checkpoint: dict[str, Any] = torch.load(weights_path, map_location=device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint  # pyright: ignore[reportUnnecessaryIsInstance]
        else checkpoint
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    remapped: dict[str, Any] = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in key_remap.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                break
        remapped[new_k] = v

    state_dict: dict[str, Any] = {
        k: v for k, v in remapped.items() if k.startswith(valid_prefixes)
    }

    if len(state_dict) == 0:
        raise RuntimeError(
            "0 tensors survived remapping/filtering — "
            "check the checkpoint's key format before continuing."
        )

    missing, unexpected = cast(
        "tuple[list[str], list[str]]",
        model.load_state_dict(state_dict, strict=False),
    )
    report = LoadReport(
        matched=len(state_dict) - len(unexpected),
        missing=list(missing),
        unexpected=list(unexpected),
    )

    encoder_layers = list(model.children())
    backbone = nn.Sequential(*encoder_layers[:9])

    return backbone, report