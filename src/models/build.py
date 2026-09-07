"""Factory orchestration for model architecture creation, weight loading, and block freezing.

Provides the primary entrypoint `build_model()` to instantiate encoder backbones,
remap pretrained checkpoint keys, filter valid layers, and apply block freeze strategies.

Example:
    >>> from src.models.build import build_model
    >>> backbone, report = build_model(
    ...     name="resnet50_radimagenet",
    ...     weights_path="checkpoints/RadImageNet-ResNet50_notop.pth",
    ...     unfreeze_from="layer3",
    ...     device="cpu"
    ... )

Pesos de ImageNet (`resnet50_imagenet_v1` / `resnet50_imagenet_v2`): a
diferencia de RadImageNet, torchvision ya trae esos pesos embebidos en su
propio `model_factory` (`resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`,
descargados/cacheados por torchvision la primera vez, sin checkpoint local
que gestionar) -- ver `ArchitectureSpec.weights_from_factory`. No hace falta
`key_remap` porque las claves del `state_dict` resultante ya son los nombres
estándar de `resnet50` (`conv1.weight`, `layer1.0.conv1.weight`, ...), los
mismos que produce `model_factory()` -- por eso las entradas de
`_ARCHITECTURES` para ImageNet usan `key_remap={}`.

    >>> backbone, report = build_model(name="resnet50_imagenet_v2", unfreeze_from="layer4")
"""

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights  # pyright: ignore[reportMissingTypeStubs]

from .freeze import FreezeStrategy, ResNetFreezeStrategy
from .weights import load_weights, truncate_backbone, LoadReport


@dataclass
class ArchitectureSpec:
    """Encapsulates model-specific architecture parameters, weight remapping, and freeze rules.

    Attributes:
        model_factory: Callable returning a base PyTorch module instance. For
            `weights_from_factory=False` it must return an UNinitialized model
            (`weights=None`) -- `build_model()` loads an external checkpoint
            into it via `load_weights()`. For `weights_from_factory=True` it
            must return the model ALREADY carrying its pretrained weights
            (e.g. torchvision's own `weights=ResNet50_Weights.IMAGENET1K_V2`)
            -- `build_model()` skips `load_weights()`/`torch.load()` entirely
            for these and just truncates the returned model.
        key_remap: Dictionary mapping custom checkpoint tensor key prefixes to standard PyTorch names.
            Ignored when `weights_from_factory=True` (nothing to remap: the
            checkpoint IS the model's own state_dict, keys already match).
        valid_prefixes: Tuple of layer name prefixes belonging to the encoder backbone.
            Ignored when `weights_from_factory=True` for the same reason --
            still declared so every registry entry has a uniform shape,
            greppable in one place (see CLAUDE.md's registry-dict decision).
        freeze_strategy: Strategy implementation handling block freezing and gradient unfreezing.
        weights_from_factory: `True` when `model_factory()` already returns a
            model with pretrained weights loaded -- no external checkpoint
            file to read, so `weights_path` is not required for this
            architecture and `build_model()` never calls `load_weights()` for
            it. Default `False` (the RadImageNet case: `model_factory()`
            returns a bare, randomly-initialized `resnet50`, and the real
            weights come from a `.pth` file on disk via `weights_path`).
            Adding a new architecture whose weights ship inside torchvision
            (or any other library) means setting this to `True` and nothing
            else in `build_model()` needs to change -- that's the scalability
            this flag buys: one boolean per registry entry, not a second
            code path per architecture.

    Example:
        >>> spec = ArchitectureSpec(
        ...     model_factory=lambda: resnet50(weights=None),
        ...     key_remap={"backbone.0.": "conv1."},
        ...     valid_prefixes=("conv1", "bn1"),
        ...     freeze_strategy=ResNetFreezeStrategy()
        ... )
    """
    model_factory: Callable[[], nn.Module]
    key_remap: dict[str, str]
    valid_prefixes: tuple[str, ...]
    freeze_strategy: FreezeStrategy
    weights_from_factory: bool = False


# Registro interno de especificaciones de modelo soportadas. Cada entrada es
# autocontenida y greppable -- agregar una arquitectura nueva es agregar una
# entrada acá, nunca tocar la lógica de build_model() de abajo (ver docstring
# de ArchitectureSpec.weights_from_factory).
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
    # Pesos de ImageNet de torchvision -- dos entradas, una por receta de
    # entrenamiento, en vez de una sola con un default silencioso: quién lea
    # el config YAML ve exactamente cuál corrió sin tener que ir a mirar acá.
    # IMAGENET1K_V1 son los pesos originales de la arquitectura ResNet50
    # (~76.1% top-1); IMAGENET1K_V2 es la receta de entrenamiento más nueva de
    # torchvision (TrivialAugment + más épocas + LR warmup, ~80.9% top-1) --
    # mismo backbone, mismos nombres de capa, solo cambian los pesos.
    # weights_from_factory=True: no hay .pth que gestionar, torchvision
    # descarga/cachea el checkpoint (~/.cache/torch/hub/checkpoints/) la
    # primera vez que se instancia -- requiere acceso a internet esa primera
    # vez únicamente.
    "resnet50_imagenet_v1": ArchitectureSpec(
        model_factory=lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        key_remap={},
        valid_prefixes=("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"),
        freeze_strategy=ResNetFreezeStrategy(),
        weights_from_factory=True,
    ),
    "resnet50_imagenet_v2": ArchitectureSpec(
        model_factory=lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        key_remap={},
        valid_prefixes=("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"),
        freeze_strategy=ResNetFreezeStrategy(),
        weights_from_factory=True,
    ),
}


def build_model(
    name: str,
    weights_path: str | None = None,
    *,
    unfreeze_from: str = "none",
    device: str = "cpu",
) -> tuple[nn.Sequential, LoadReport]:
    """Constructs a complete vision backbone: instantiates architecture, loads weights, and applies freeze rules.

    Args:
        name: Name identifier registered in `_ARCHITECTURES` (e.g., `"resnet50_radimagenet"`,
            `"resnet50_imagenet_v1"`, `"resnet50_imagenet_v2"`).
        weights_path: File path to the pretrained weights checkpoint (`.pth` / `.pt`).
            Required when `_ARCHITECTURES[name].weights_from_factory` is
            `False` (e.g. `resnet50_radimagenet`) -- raises if omitted.
            Ignored (may be left `None`) when it's `True` (the ImageNet
            entries): those already carry their weights from `model_factory()`.
        unfreeze_from: Layer block name from which parameters are unfrozen for fine-tuning.
            Passed directly to `FreezeStrategy.apply()`.
        device: Target compute device for model initialization (`"cpu"` or `"cuda"`).

    Returns:
        tuple[nn.Sequential, LoadReport]: A tuple containing:
            1. Truncated `nn.Sequential` backbone module with parameters frozen/unfrozen.
            2. `LoadReport` dataclass summarizing tensor matching statistics.

    Raises:
        ValueError: If `name` is not registered in `_ARCHITECTURES`, or if
            `weights_path` is `None` for an architecture that requires one
            (`weights_from_factory=False`).

    Example:
        >>> backbone, report = build_model(
        ...     name="resnet50_radimagenet",
        ...     weights_path="checkpoints/RadImageNet-ResNet50_notop.pth",
        ...     unfreeze_from="layer4",
        ...     device="cpu"
        ... )
        >>> print(report.matched)

        >>> # ImageNet: sin weights_path, torchvision ya trae los pesos.
        >>> backbone, report = build_model(name="resnet50_imagenet_v2", unfreeze_from="layer4")
    """
    if name not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name!r}. Registered options: {sorted(_ARCHITECTURES)}")

    spec = _ARCHITECTURES[name]

    if spec.weights_from_factory:
        # model_factory() ya devuelve el modelo con sus pesos preentrenados
        # cargados (torchvision descarga/cachea el checkpoint de ImageNet
        # bajo el capó) -- no hay archivo externo que leer, así que el
        # camino de load_weights()/torch.load() se salta por completo.
        # truncate_backbone() es el mismo helper que usa load_weights(), para
        # no duplicar el "9" que separa encoder de `fc` (ver su docstring en
        # weights.py).
        model = spec.model_factory()
        backbone = truncate_backbone(model)
        # No hay missing/unexpected posible acá: el state_dict truncado
        # proviene de la propia arquitectura de torchvision, que por
        # construcción calza 1:1 consigo misma. LoadReport se construye de
        # todas formas para que build_model() tenga un tipo de retorno
        # uniforme sin importar la arquitectura, y para que el print de
        # cli.py ("Pesos cargados: N tensores") siga siendo veraz.
        report = LoadReport(matched=len(backbone.state_dict()), missing=[], unexpected=[])
    else:
        if weights_path is None:
            raise ValueError(
                f"architecture {name!r} requires weights_path (no built-in pretrained "
                "weights) -- see ArchitectureSpec.weights_from_factory."
            )
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