from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module,
                    path: str | Path,
                    *, 
                    epoch: int,
                     metric_value: float) -> None:
    """Guarda el state_dict del modelo junto con metadata de la corrida.

    Args:
        model: modelo a guardar.
        path: ruta destino del archivo .pt.
        epoch: número de época en que se guardó.
        metric_value: valor de la métrica de validación en esa época
            (para poder auditar después por qué se eligió este checkpoint).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "metric_value": metric_value},
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: str = "cpu"
) -> dict[str, Any]:
    """Carga un checkpoint guardado con save_checkpoint, aplicando los
    pesos directamente sobre `model` (in-place).

    Args:
        model: modelo sobre el que se cargan los pesos.
        path: ruta al archivo .pt guardado con save_checkpoint.
        device: dispositivo destino de los tensores cargados.

    Returns:
        El diccionario completo del checkpoint (incluye epoch, metric_value),
        por si se necesita auditar de dónde vino.

    Raises:
        FileNotFoundError: path no existe.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")

    checkpoint: dict[str, Any] = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint