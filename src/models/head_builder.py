"""Abstract interface every classification head builder must implement."""

from abc import ABC, abstractmethod

import torch.nn as nn


class HeadBuilder(ABC):
    """Strategy interface for classification heads.

    Assembling a full model (backbone + head) happens outside this module —
    a HeadBuilder only knows how to build its own layers.
    """

    @abstractmethod
    def build(self) -> nn.Sequential:
        """Assembles this head's layers into an executable module.

        Returns:
            nn.Sequential: the head's layers, ready to receive backbone output.
        """
        ...
