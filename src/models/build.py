from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torchvision.models import resnet50  # pyright: ignore[reportMissingTypeStubs]
from .weights import load_weights, LoadReport
from .freeze import FreezeStrategy, ResNetFreezeStrategy


@dataclass
class ArchitectureSpec:
    """Todo lo que varía por arquitectura """
    key_remap: dict[str, str]
    valid_prefixes: tuple[str, ...]
    freeze_strategy: FreezeStrategy

# Factory de modelos
_ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "resnet50_radimagenet": ArchitectureSpec(
        model_factory=lambda: resnet50(weights=None),
        # Mapeo de claves de RadImageNet a las claves de PyTorch en el state_dict
        key_remap={
            "backbone.0.": "conv1.", "backbone.1.": "bn1.",
            "backbone.4.": "layer1.", "backbone.5.": "layer2.",
            "backbone.6.": "layer3.", "backbone.7.": "layer4.",
        },
        valid_prefixes=("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"),
        freeze_strategy=ResNetFreezeStrategy(),
    ),
}

# Función que construye el modelo
def build_model(
    name: str,
    weights_path: str,
    *,
    unfreeze_from: str = "none",
    device: str = "cpu",
) -> tuple[nn.Sequential, LoadReport]:
    """Construye un backbone completo: arquitectura + pesos + freeze.

    Args:
        name: clave en _ARCHITECTURES, ej. "resnet50_radimagenet".
        weights_path: ruta al checkpoint de pesos preentrenados.
        unfreeze_from: pasado directo a FreezeStrategy.apply().
        device: dispositivo destino.

    Returns:
        (backbone, report) — el backbone ya con freeze aplicado.

    Raises:
        ValueError: name no está en _ARCHITECTURES.
    """
    if name not in _ARCHITECTURES:
        raise ValueError(f"Arquitectura desconocida: {name!r}. Opciones: {sorted(_ARCHITECTURES)}")

    spec = _ARCHITECTURES[name]

    # Carga de pesos preentrenados con mapeo de claves
    backbone, report = load_weights(
        model_factory=spec.model_factory,
        weights_path=weights_path,
        key_remap=spec.key_remap,
        valid_prefixes=spec.valid_prefixes,
        device=device,
    )

    # Congela las capas iniciales del backbone según el parámetro unfreeze_from
    spec.freeze_strategy.apply(backbone, unfreeze_from=unfreeze_from)

    return backbone, report