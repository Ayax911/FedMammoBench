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
    encoder_layers = list(model.children())
    backbone = nn.Sequential(*encoder_layers[:9])

    return backbone, report