"""Training hyperparameter configuration with built-in validation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class OptimizerConfig:
    name: Literal["sgd", "adam", "adamw"] = "adamw"
    lr: float = 1e-4
    # Discriminative LRs: when both are set, the optimizer uses separate param
    # groups for the classification head and the backbone feature extractor,
    # mirroring the centralizada setup (lr_head=1e-3, lr_backbone=1e-4).
    # Falls back to `lr` for any group whose specific rate is None.
    lr_head: float | None = None
    lr_backbone: float | None = None
    weight_decay: float = 1e-4
    momentum: float = 0.9  # only used by SGD


@dataclass
class SchedulerConfig:
    name: Literal["none", "cosine", "step", "plateau"] = "none"
    step_size: int = 10
    gamma: float = 0.1
    t_max: int = 50  # cosine


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotate_limit: int = 15
    brightness_contrast: bool = True
    elastic: bool = False
    normalize_preset: str | None = None
    # Accept scalar (replicated across channels) or per-channel list.
    # YAML sequences are loaded as list[float]; scalars as float.
    normalize_mean: Any = 0.5
    normalize_std: Any = 0.25


@dataclass
class LossConfig:
    """Loss function selection.

    ``ce`` is class-weighted cross-entropy (weights derived from train counts
    if :attr:`auto_class_weights` is True). ``focal`` uses focal loss with
    :attr:`focal_gamma`.
    """

    name: Literal["ce", "focal", "bce"] = "ce"
    auto_class_weights: bool = True
    focal_gamma: float = 2.0


@dataclass
class TrainingConfig:
    """Centralized / per-client local training hyperparameters.

    Attributes:
        epochs: Number of epochs for centralized training (run_centralized.py).
        local_epochs: Number of local epochs per Flower round per client.
        grad_clip_norm: Optional gradient-norm clipping value (0 disables).
        mixed_precision: Use ``torch.cuda.amp`` autocast when CUDA is available.
            Note: FedProx with mixed_precision=True can cause the proximal term
            to underflow in FP16. This is handled automatically in Trainer, but
            be aware when comparing FedProx vs FedAvg convergence curves.
    """

    epochs: int = 20
    local_epochs: int = 1
    grad_clip_norm: float = 0.0
    mixed_precision: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate(self, strategy_name: str = "fedavg", proximal_mu: float = 0.0) -> None:
        """Raise ValueError or emit warnings for invalid training settings.

        Args:
            strategy_name: Name of the FL strategy (from StrategyConfig.name).
                Used to warn about FedProx + mixed_precision interactions.
            proximal_mu: The proximal regularization strength (from
                StrategyConfig.params.get('mu', 0.0)).
        """
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {self.local_epochs}")
        if self.optimizer.lr <= 0.0:
            raise ValueError(f"optimizer.lr must be > 0, got {self.optimizer.lr}")
        if self.optimizer.lr_head is not None and self.optimizer.lr_head <= 0.0:
            raise ValueError(f"optimizer.lr_head must be > 0, got {self.optimizer.lr_head}")
        if self.optimizer.lr_backbone is not None and self.optimizer.lr_backbone <= 0.0:
            raise ValueError(f"optimizer.lr_backbone must be > 0, got {self.optimizer.lr_backbone}")

        # Warn about FedProx + AMP: the proximal term is computed in FP32 inside
        # Trainer.train_one_epoch so underflow is prevented, but AMP reduces
        # gradient precision for the rest of the backward pass.
        if strategy_name == "fedprox" and proximal_mu > 0.0 and self.mixed_precision:
            warnings.warn(
                "FedProx with mixed_precision=True: the proximal term is computed in FP32 "
                "(underflow-safe), but the rest of the backward pass uses FP16. "
                "Disable mixed_precision for the most numerically stable FedProx behavior.",
                UserWarning,
                stacklevel=2,
            )


__all__ = [
    "AugmentationConfig",
    "LossConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
]
