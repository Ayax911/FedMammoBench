from typing import Callable
from .reports import LoadReport   
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, cast


def load_weights(
    model_factory: Callable[[], nn.Module],
    weights_path: str | Path,
) -> tuple[nn.Sequential, LoadReport]:
    """Carga pesos de RadImageNet en cualquier arquitectura construida por
    model_factory, y devuelve el backbone truncado (sin cabeza) + reporte.

    Args:
        model_factory: función o clase invocable sin argumentos que
            construye el modelo base — ej. `resnet50` (de torchvision),
            o `lambda: resnet50(weights=None)`.
        weights_path: ruta al checkpoint de RadImageNet.
        device: dispositivo destino para los tensores cargados.

    Returns:
        (backbone, report)

    Raises:
        RuntimeError: el state_dict remapeado quedó vacío, o matched == 0.
    """
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_factory()

    checkpoint: dict[str, Any] = torch.load(weights_path, map_location=device) 
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint # pyright: ignore[reportUnnecessaryIsInstance]
        else checkpoint
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    backbone_remap = {
        "backbone.0.": "conv1.", "backbone.1.": "bn1.",
        "backbone.4.": "layer1.", "backbone.5.": "layer2.",
        "backbone.6.": "layer3.", "backbone.7.": "layer4.",
    }
    remapped: dict[str, Any] = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in backbone_remap.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                break
        remapped[new_k] = v
    state_dict:dict[str, Any] = {
        k: v for k, v in remapped.items()
        if k.startswith(("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"))
    }

    if len(state_dict) == 0:
        raise RuntimeError(
            "0 RadImageNet tensors survived remapping/filtering — "
            "check the checkpoint's key format before continuing."
        )

    missing, unexpected = cast(
    "tuple[list[str], list[str]]",model.load_state_dict(state_dict, strict=False))
    report = LoadReport(
        matched=len(state_dict) - len(unexpected),
        missing=list(missing),
        unexpected=list(unexpected),
    )

    encoder_layers = list(model.children())
    backbone = nn.Sequential(*encoder_layers[:9])

    return backbone, report