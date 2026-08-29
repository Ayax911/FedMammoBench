"""Factory orchestration for model architecture creation, weight loading, and block freezing."""

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torchvision.models import resnet50  # pyright: ignore[reportMissingTypeStubs]

from .freeze import FreezeStrategy, ResNetFreezeStrategy
from .weights import load_weights, LoadReport



@dataclass
class ArchitectureSpec:
    """Encapsulates model-specific architecture parameters, weight remapping, and freeze rules.

    Attributes:
        model_factory: Callable returning an uninitialized base PyTorch module instance.
        key_remap: Dictionary mapping custom checkpoint tensor key prefixes to standard PyTorch names.
        valid_prefixes: Tuple of layer name prefixes belonging to the encoder backbone.
        freeze_strategy: Strategy implementation handling block freezing and gradient unfreezing.
    """
    model_factory: Callable[[], nn.Module]
    key_remap: dict[str, str]
    valid_prefixes: tuple[str, ...]
    freeze_strategy: FreezeStrategy


# Internal registry of supported model specifications
_ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "resnet50_radimagenet": ArchitectureSpec(
        model_factory=lambda: resnet50(weights=None),
        # Tensor key substitutions to map RadImageNet state_dict keys to PyTorch ResNet50 layer names
        key_remap={
            "backbone.0.": "conv1.", "backbone.1.": "bn1.",
            "backbone.4.": "layer1.", "backbone.5.": "layer2.",
            "backbone.6.": "layer3.", "backbone.7.": "layer4.",
        },
        valid_prefixes=("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"),
        freeze_strategy=ResNetFreezeStrategy(),
    ),
}


def build_model(
    name: str,
    weights_path: str,
    *,
    unfreeze_from: str = "none",
    device: str = "cpu",
) -> tuple[nn.Sequential, LoadReport]:
    """Constructs a complete vision backbone: instantiates architecture, loads weights, and applies freeze rules.

    Args:
        name: Name identifier registered in `_ARCHITECTURES` (e.g., `"resnet50_radimagenet"`).
        weights_path: File path to the pretrained weights checkpoint (`.pth` / `.pt`).
        unfreeze_from: Layer block name from which parameters are unfrozen for fine-tuning.
            Passed directly to `FreezeStrategy.apply()`.
        device: Target compute device for model initialization (`"cpu"` or `"cuda"`).

    Returns:
        tuple[nn.Sequential, LoadReport]: A tuple containing:
            1. Truncated `nn.Sequential` backbone module with parameters frozen/unfrozen.
            2. `LoadReport` dataclass summarizing tensor matching statistics.

    Raises:
        ValueError: If `name` is not registered in `_ARCHITECTURES`.
    """
    if name not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name!r}. Registered options: {sorted(_ARCHITECTURES)}")

    spec = _ARCHITECTURES[name]

    # Instantiate base model, clean state_dict, remap keys, and filter encoder parameters
    backbone, report = load_weights(
        model_factory=spec.model_factory,
        weights_path=weights_path,
        key_remap=spec.key_remap,
        valid_prefixes=spec.valid_prefixes,
        device=device,
    )

    # Apply parameter freezing strategy starting from specified unfreeze_from layer block
    spec.freeze_strategy.apply(backbone, unfreeze_from=unfreeze_from)

    return backbone, report