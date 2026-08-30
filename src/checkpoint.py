"""Gestión de checkpoints PyTorch con preservación de metadatos de auditoría.

Proporciona funciones puras para serializar y des-serializar el `state_dict`
de un modelo junto con la época y el valor de métrica asociada.

Ejemplo de uso:
    >>> import torch.nn as nn
    >>> from src.checkpoint import save_checkpoint, load_checkpoint
    >>> model = nn.Linear(10, 1)
    >>> save_checkpoint(model, "runs/exp01/best.pt", epoch=5, metric_value=0.89)
    >>> metadata = load_checkpoint(model, "runs/exp01/best.pt", device="cpu")
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    epoch: int,
    metric_value: float,
) -> None:
    """Guarda el state_dict del modelo junto con metadata de la corrida en formato PyTorch.

    Args:
        model: Modelo PyTorch cuyo `state_dict` se va a guardar.
        path: Ruta destino del archivo de pesos `.pt`. Se crea el directorio padre si no existe.
        epoch: Número de época en que se generó el checkpoint.
        metric_value: Valor numérico de la métrica de validación en esa época (para auditoría).

    Example:
        >>> save_checkpoint(model, "weights/best.pt", epoch=10, metric_value=0.912)
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
    device: str = "cpu",
) -> dict[str, Any]:
    """Carga un checkpoint guardado con `save_checkpoint`, aplicando los pesos in-place sobre `model`.

    Args:
        model: Modelo instanciado sobre el que se cargará el `state_dict`.
        path: Ruta al archivo `.pt` generado por `save_checkpoint`.
        device: Dispositivo de destino (`"cpu"`, `"cuda"`).

    Returns:
        dict[str, Any]: El diccionario completo guardado en el archivo `.pt`
        (incluyendo `"model_state_dict"`, `"epoch"` y `"metric_value"`).

    Raises:
        FileNotFoundError: Si la ruta especificada en `path` no existe.

    Example:
        >>> metadata = load_checkpoint(model, "weights/best.pt", device="cuda")
        >>> print(f"Cargada época {metadata['epoch']} con AUC {metadata['metric_value']:.4f}")
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")

    checkpoint: dict[str, Any] = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint