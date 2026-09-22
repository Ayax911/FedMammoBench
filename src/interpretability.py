"""Módulo de interpretabilidad y mapas de calor Grad-CAM para mamografías.

Proporciona funciones puras para calcular mapas de activación de clase ponderados
por gradiente (Grad-CAM) sobre capas intermedias de un modelo convolucional y
superponerlos sobre imágenes base en formato PIL.

Ejemplo de uso:
    >>> import torch
    >>> from PIL import Image
    >>> from src.interpretability import compute_gradcam, overlay_heatmap
    >>> result = compute_gradcam(model, image_tensor.unsqueeze(0), loss_spec, device="cpu")
    >>> composite = overlay_heatmap(display_image, result.heatmap)
    >>> composite.save("gradcam.png")
"""

from dataclasses import dataclass
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .train.build import LossSpec


@dataclass(frozen=True)
class GradCAMResult:
    """Resultado de un Grad-CAM sobre una sola imagen.

    Attributes:
        heatmap: array `[H, W]` float32 en `[0, 1]`, ya upsampleado a la
            resolución de `image` (bilinear, `align_corners=False`).
        target_score: el escalar de `loss_spec.probs()` sobre el que se hizo
            `.backward()` -- la probabilidad de malignant que el mapa explica.
    """

    heatmap: np.ndarray
    target_score: float


def compute_gradcam(
    model: nn.Sequential,
    image: torch.Tensor,
    loss_spec: LossSpec,
    device: str = "cpu",
    target_layer_index: int = 7,
) -> GradCAMResult:
    """Grad-CAM de una imagen sobre `model[0][target_layer_index]`.

    Args:
        model: `nn.Sequential(backbone, head)` con pesos ya cargados
            (mismo contrato que `evaluate()`/`predict_on_loader()`) --
            `model[0]` debe tener al menos `target_layer_index + 1` hijos.
        image: tensor `[1, C, H, W]` YA pasado por `eval_transform_builder`
            (sin augmentación) -- una sola imagen, no un batch. `C` es 3
            siempre (`MammoBenchDataset` ya expande TIFFs mode "F" a 3
            canales antes de esto, ver `src/datasets/dataset.py:89-106`).
        loss_spec: de `build_loss()` -- unifica BCE/CE.
        device: debe coincidir con el device de `model`.
        target_layer_index: índice dentro de `model[0]` (el backbone). 7 =
            `layer4`. NO usar 8 (`avgpool`, sin resolución espacial).

    Returns:
        GradCAMResult.

    Raises:
        ValueError: Si la imagen no tiene dimensiones `[1, C, H, W]` o `[C, H, W]`.
        IndexError: Si `target_layer_index` está fuera de rango para `model[0]`.

    Note:
        Si el gradiente de las activaciones es todo cero (ReLU muerta o score
        saturado), `heatmap` es un array de ceros sin lanzar excepción.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"image must have shape [1, C, H, W] or [C, H, W], got {image.shape}")

    target_layer = model[0][target_layer_index]

    activations: torch.Tensor | None = None

    def forward_hook(module: nn.Module, input_: Any, output: torch.Tensor) -> None:
        nonlocal activations
        activations = output
        if output.requires_grad:
            output.retain_grad()

    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            img = image.to(device).detach().requires_grad_(True)
            logits = model(img)
            probs = loss_spec.probs(logits)
            score = probs[0]
            score.backward()
    finally:
        hook_handle.remove()

    if activations is None:
        raise RuntimeError("Target layer forward hook was not called during forward pass.")

    h, w = image.shape[-2:]
    grad = activations.grad

    if grad is None or (grad == 0).all():
        heatmap = np.zeros((h, w), dtype=np.float32)
        return GradCAMResult(heatmap=heatmap, target_score=float(score.item()))

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

    cam_min = cam.min()
    cam_max = cam.max()
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    heatmap = cam_norm.detach().cpu().squeeze().numpy().astype(np.float32)
    if heatmap.ndim != 2:
        heatmap = heatmap.reshape((h, w))

    return GradCAMResult(heatmap=heatmap, target_score=float(score.item()))


def overlay_heatmap(
    base_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Superpone `heatmap` (colormap 'jet') sobre `base_image`.

    Args:
        base_image: imagen PIL YA en modo RGB y YA en un rango visualizable
            (`uint8`, `0-255`) -- responsabilidad de quien llama (ver
            `src/gradcam.py`, sección "imagen para mostrar ≠ imagen que ve
            el modelo"). Esta función NO reescala ni reinterpreta el rango
            de `base_image`.
        heatmap: `[H, W]` float32 en `[0, 1]` (la salida de
            `compute_gradcam().heatmap`), mismo `(H, W)` que `base_image`.
        alpha: peso del heatmap en la mezcla (`0` = solo `base_image`, `1` =
            solo el heatmap).

    Returns:
        Image.Image: composición RGB, mismo tamaño que `base_image`.

    Example:
        >>> composite = overlay_heatmap(original_pil.convert("RGB"), result.heatmap)
        >>> composite.save("gradcam_example.png")
    """
    if base_image.mode != "RGB":
        base_image = base_image.convert("RGB")

    base_arr = np.array(base_image, dtype=np.float32)
    cmap = matplotlib.colormaps["jet"]
    heatmap_rgba = cmap(heatmap)
    heatmap_rgb = (heatmap_rgba[..., :3] * 255.0).astype(np.float32)

    composite = (1.0 - alpha) * base_arr + alpha * heatmap_rgb
    composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)

    return Image.fromarray(composite_uint8, mode="RGB")
