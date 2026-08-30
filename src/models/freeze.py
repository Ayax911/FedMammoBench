"""Abstract base class and architecture-specific implementations for layer freezing strategies.

Provides positional block-based parameter freezing and unfreezing for fine-tuning vision backbones.

Example:
    >>> import torch.nn as nn
    >>> from src.models.freeze import ResNetFreezeStrategy
    >>> strategy = ResNetFreezeStrategy()
    >>> backbone = nn.Sequential() # PyTorch backbone container
    >>> counts = strategy.apply(backbone, unfreeze_from="layer4")
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class FreezeStrategy(ABC):
    """Abstract base class managing layer freezing and unfreezing in vision backbones.

    Provides positional block-based unfreezing control for fine-tuning pre-trained models.
    Subclasses define architecture-specific block ordering matching layer sequences in `nn.Sequential`.

    Example:
        >>> strategy = ResNetFreezeStrategy()
        >>> print(strategy.block_order)
    """

    @property
    @abstractmethod
    def block_order(self) -> list[str]:
        """Ordered list of block layer names matching the positional order in `nn.Sequential`.

        Returns:
            list[str]: Sequential block name strings (e.g. `["conv1", "bn1", ..., "layer4"]`).
        """
        ...

    def apply(self, backbone: nn.Sequential, *, unfreeze_from: str) -> dict[str, int]:
        """Applies parameter freezing to all backbone layers and unfreezes blocks from unfreeze_from.

        Args:
            backbone: Truncated `nn.Sequential` PyTorch model container.
            unfreeze_from: Block name string from which parameter gradients are enabled.
                If `"none"`, all backbone parameters remain frozen (`requires_grad = False`).

        Returns:
            dict[str, int]: Dictionary containing param count statistics: `{"trainable": int, "total": int}`.

        Raises:
            ValueError: If `unfreeze_from` is not `"none"` and not present in `self.block_order`.

        Example:
            >>> strategy = ResNetFreezeStrategy()
            >>> summary = strategy.apply(backbone, unfreeze_from="layer3")
            >>> print(summary["trainable"], summary["total"])
        """
        if unfreeze_from != "none" and unfreeze_from not in self.block_order:
            raise ValueError(
                f"unfreeze_from={unfreeze_from!r} is unrecognized for "
                f"{type(self).__name__}. Valid options: 'none', {self.block_order}"
            )

        # Freeze all backbone parameter gradients by default
        for param in backbone.parameters():
            param.requires_grad = False

        # Selectively unfreeze layer blocks from unfreeze_from index to the end
        if unfreeze_from != "none":
            start_idx = self.block_order.index(unfreeze_from)
            blocks_to_unfreeze = list(backbone.children())[start_idx:]

            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True

        # Calculate parameter count summary
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in backbone.parameters())
        return {"trainable": trainable, "total": total}


class ResNetFreezeStrategy(FreezeStrategy):
    """Concrete FreezeStrategy implementation for ResNet50 vision architectures.

    Example:
        >>> strategy = ResNetFreezeStrategy()
        >>> print(strategy.block_order[0])
    """

    @property
    def block_order(self) -> list[str]:
        """Ordered list of ResNet50 positional block names.

        Returns:
            list[str]: Standard ResNet50 block sequence names.
        """
        return ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]