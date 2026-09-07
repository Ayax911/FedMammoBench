"""Checkpoint weight loading, state_dict cleaning, key remapping, and model truncation utilities.

Provides low-level functions to inspect and map pretrained weights onto base PyTorch vision models.

Example:
    >>> from torchvision.models import resnet50
    >>> from src.models.weights import load_weights
    >>> model_factory = lambda: resnet50(weights=None)
    >>> backbone, report = load_weights(
    ...     model_factory=model_factory,
    ...     weights_path="checkpoints/RadImageNet-ResNet50_notop.pth",
    ...     key_remap={"backbone.0.": "conv1."},
    ...     valid_prefixes=("conv1", "bn1", "layer1")
    ... )
"""

from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn as nn

from .reports import LoadReport


def truncate_backbone(model: nn.Module, n: int = 9) -> nn.Sequential:
    """Trunca un modelo a sus primeros `n` submódulos hijos directos.

    Único lugar donde vive el número mágico "9" -- el índice que separa el
    encoder (`conv1..avgpool`) de la cabeza de clasificación (`fc`) en un
    `resnet50` de torchvision. `load_weights()` lo llama después de cargar un
    checkpoint externo; `build_model()` (`models/build.py`) lo vuelve a llamar
    para las arquitecturas con `ArchitectureSpec.weights_from_factory=True`
    (ImageNet vía torchvision, ej. `resnet50_imagenet_v2`), que nunca pasan
    por `load_weights()` porque `model_factory()` ya les entrega los pesos
    cargados -- sin este helper compartido, ese "9" tendría que repetirse en
    `build.py` y podría divergir si algún día cambia acá.

    Args:
        model: modelo completo (con `fc` incluido) del que extraer el encoder.
        n: cantidad de hijos directos a conservar, en orden. Default 9, que
            para `resnet50` es exactamente `conv1, bn1, relu, maxpool,
            layer1, layer2, layer3, layer4, avgpool` -- descarta `fc`.

    Returns:
        nn.Sequential: los primeros `n` submódulos, en el mismo orden.

    Example:
        >>> from torchvision.models import resnet50
        >>> backbone = truncate_backbone(resnet50(weights=None))
        >>> len(backbone)
        9
    """
    return nn.Sequential(*list(model.children())[:n])


def load_weights(
    model_factory: Callable[[], nn.Module],
    weights_path: str | Path,
    key_remap: dict[str, str],
    valid_prefixes: tuple[str, ...],
    device: str = "cpu",
) -> tuple[nn.Sequential, LoadReport]:
    """Loads a pretrained checkpoint into a model instance created by model_factory.

    Performs state_dict cleaning (stripping `module.` wrapper prefixes from PyTorch DataParallel),
    prefix remapping (matching custom checkpoint keys to PyTorch target module names), prefix
    filtering (retaining only backbone tensors), non-strict parameter loading, and backbone truncation.

    Args:
        model_factory: Callable or factory function returning an uninitialized base model.
        weights_path: File system path to the weight checkpoint (`.pth` or `.pt`).
        key_remap: Dictionary mapping original checkpoint key prefixes to target layer prefixes.
            Example: `{"backbone.0.": "conv1."}`.
        valid_prefixes: Tuple of valid tensor name prefixes belonging to the encoder backbone.
        device: Target compute device for loading state_dict tensors (`"cpu"` or `"cuda"`).

    Returns:
        tuple[nn.Sequential, LoadReport]: A tuple containing:
            1. Truncated `nn.Sequential` encoder backbone containing the first 9 layer blocks.
            2. `LoadReport` dataclass detailing matched, missing, and unexpected tensor keys.

    Raises:
        RuntimeError: If zero tensors survive key remapping/filtering, or if no parameters match.

    Example:
        >>> backbone, report = load_weights(
        ...     model_factory=lambda: resnet50(weights=None),
        ...     weights_path="checkpoints/model.pth",
        ...     key_remap={"backbone.0.": "conv1."},
        ...     valid_prefixes=("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
        ... )
        >>> print(f"Matched tensors: {report.matched}")
    """
    model = model_factory()

    # Load checkpoint state dictionary onto target compute device
    checkpoint: dict[str, Any] = torch.load(weights_path, map_location=device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint  # pyright: ignore[reportUnnecessaryIsInstance]
        else checkpoint
    )
    # Strip PyTorch DataParallel 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Remap layer tensor key prefixes to match PyTorch model architecture names
    remapped: dict[str, Any] = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in key_remap.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                break
        remapped[new_k] = v

    # Filter state_dict to keep only valid backbone encoder layers
    state_dict: dict[str, Any] = {
        k: v for k, v in remapped.items() if k.startswith(valid_prefixes)
    }

    # Validate that state_dict is not empty prior to parameter loading
    if len(state_dict) == 0:
        raise RuntimeError(
            "0 tensors survived remapping/filtering — "
            "check the checkpoint's key format before continuing."
        )

    # Load parameters into model without strict matching (head weights may be missing)
    missing, unexpected = cast(
        "tuple[list[str], list[str]]",
        model.load_state_dict(state_dict, strict=False),
    )
    report = LoadReport(
        matched=len(state_dict) - len(unexpected),
        missing=list(missing),
        unexpected=list(unexpected),
    )

    # Truncate model up to layer index 9 (extracting standard ResNet encoder layers)
    backbone = truncate_backbone(model)

    return backbone, report