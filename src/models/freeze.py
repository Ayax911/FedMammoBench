from abc import ABC, abstractmethod

import torch.nn as nn


class FreezeStrategy(ABC):
    """Define cómo congelar/descongelar un backbone según sus bloques, en
    el orden en que aparecen dentro del nn.Sequential ya truncado. Cada
    arquitectura tiene su propia subclase, porque cada una organiza sus
    bloques distinto."""

    @property
    @abstractmethod
    def block_order(self) -> list[str]:
        """Nombres descriptivos de los bloques, en el mismo orden posicional
        en que quedan dentro del nn.Sequential truncado (ver
        load_radimagenet_backbone: nn.Sequential(*encoder_layers[:9]))."""
        ...

    def apply(self, backbone: nn.Sequential, *, unfreeze_from: str) -> dict[str, int]:
        
        if unfreeze_from != "none" and unfreeze_from not in self.block_order:
            raise ValueError(
                f"unfreeze_from={unfreeze_from!r} no reconocido para "
                f"{type(self).__name__}. Opciones: 'none', {self.block_order}"
            )

        for param in backbone.parameters():
            param.requires_grad = False

        if unfreeze_from != "none":
            start_idx = self.block_order.index(unfreeze_from)
            blocks_to_unfreeze = list(backbone.children())[start_idx:]

            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in backbone.parameters())
        return {"trainable": trainable, "total": total}


class ResNetFreezeStrategy(FreezeStrategy):
    @property
    def block_order(self) -> list[str]:
        return ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]