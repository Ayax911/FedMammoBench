"""Backbone convolucional custom (4 bloques Conv-BN-ReLU x2 + GAP), entrenado desde cero.

Diseño provisto por el usuario: 4 bloques de `Conv2d(k=3, padding="same") -> BatchNorm2d
-> ReLU` repetido 2 veces, con `MaxPool2d(2)` al final de los bloques 1-3 (el bloque 4 no
reduce resolución), canales 32 -> 64 -> 128 -> 256, seguido de `AdaptiveAvgPool2d(1)`
(GlobalAveragePooling2D). Sin pesos preentrenados -- ver `ArchitectureSpec.weights_from_factory`
en `models/build.py`, que registra este backbone con ese flag en `True`.

Ejemplo:
    >>> from src.models.custom_cnn import CustomCNNBackbone
    >>> backbone = CustomCNNBackbone()
    >>> out = backbone(torch.randn(2, 3, 256, 256))
    >>> out.shape
    torch.Size([2, 256, 1, 1])
"""

import torch
import torch.nn as nn


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    """Bloque `Conv2d(3x3, padding="same") -> BatchNorm2d -> ReLU`, la unidad repetida 2x por bloque.

    Args:
        in_channels: canales de entrada.
        out_channels: canales de salida (= canales de los siguientes `Conv2d`/`BatchNorm2d`).

    Returns:
        nn.Sequential: las tres capas en orden.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CustomCNNBackbone(nn.Module):
    """Backbone CNN custom de 4 bloques + GlobalAveragePooling2D, sin pesos preentrenados.

    Entrada esperada: `[B, 3, H, W]` -- 3 canales para calzar con el resto del pipeline de
    datos de este repo (`datasets/dataset.py` ya replica el TIFF de 1 canal a 3 ANTES del
    transform, igual que todos los demás backbones registrados en `models/build.py`; no hay
    pérdida de información real, los 3 canales son la misma imagen). Al ser fully-convolutional
    hasta el GAP, `H`/`W` son libres -- no hace falta que sean 256.

    Salida: `[B, 256, 1, 1]` tras el GAP. `ConfigurableMLPHead.build()` (`models/mlp_configs/
    configurable_mlp.py`) empieza con `nn.Flatten()`, así que la acepta sin cambios.

    `truncate_backbone()` (`models/weights.py`, `n=9` default) conserva los 5 hijos directos
    (`block1..block4, gap`) sin cambios -- 5 < 9.

    Example:
        >>> backbone = CustomCNNBackbone()
        >>> out = backbone(torch.randn(2, 3, 256, 256))
        >>> out.shape
        torch.Size([2, 256, 1, 1])
    """

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(_conv_bn_relu(3, 32), _conv_bn_relu(32, 32), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(_conv_bn_relu(32, 64), _conv_bn_relu(64, 64), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(_conv_bn_relu(64, 128), _conv_bn_relu(128, 128), nn.MaxPool2d(2))
        # Bloque 4 sin MaxPool2d -- por diseño, termina en 32x32x256 (para input 256x256), no reduce más.
        self.block4 = nn.Sequential(_conv_bn_relu(128, 256), _conv_bn_relu(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica los 4 bloques convolucionales y el GAP en orden.

        Args:
            x: tensor `[B, 3, H, W]`.

        Returns:
            torch.Tensor: `[B, 256, 1, 1]`.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return x
