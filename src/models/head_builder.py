"""Abstract interface every classification head builder must implement.

Provides the Strategy pattern base interface for vision classification heads.

Example:
    >>> from src.models.head_builder import HeadBuilder
    >>> # Subclasses implement .build() returning nn.Sequential
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class HeadBuilder(ABC):
    """Strategy interface for classification head builders.

    Assembling a full model (backbone + head) happens outside this module —
    a `HeadBuilder` only knows how to construct its own classification layers.

    Example:
        >>> class MyHead(HeadBuilder):
        ...     def build(self) -> nn.Sequential:
        ...         return nn.Sequential(nn.Linear(2048, 1))
    """

    @abstractmethod
    def build(self) -> nn.Sequential:
        """Assembles this head's layers into an executable PyTorch module container.

        Returns:
            nn.Sequential: The classification head's layers, ready to receive backbone output.
        """
        ...
