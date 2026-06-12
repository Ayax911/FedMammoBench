"""Loss functions for binary mammography classification.

Provides:

- weighted cross-entropy (the default; weights derived from train counts or
  user-supplied)
- focal loss (Lin et al., 2017) for severe class imbalance
- BCE with logits (``bce``) — identical to the centralized model's
  ``BCEWithLogitsLoss(pos_weight=...)``; use with ``num_classes=1``
- :func:`build_loss` consumes :class:`LossConfig` plus a label vector and
  returns the configured ``nn.Module``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fedmammobench.configs.schema import LossConfig


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    ``alpha`` (per-class weight tensor of shape ``[num_classes]``) and
    ``gamma`` (focusing parameter) match the original paper. Implementation
    follows the standard CE-based formulation:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        *,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Gather the target-class log-prob and prob.
        target_long = target.long()
        log_pt = log_probs.gather(1, target_long.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target_long.unsqueeze(1)).squeeze(1)

        focal = (1.0 - pt).clamp(min=0.0).pow(self.gamma) * (-log_pt)
        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, target_long)
            focal = at * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class BCEWithLogitsLossWrapper(nn.Module):
    """``BCEWithLogitsLoss`` that accepts long-int targets and squeezed logits.

    The Trainer casts targets to ``.long()`` before passing them to the loss.
    This wrapper converts them back to float and squeezes single-logit outputs
    ``[batch, 1] → [batch]``, so the Trainer needs no changes when switching
    from CE to BCE.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        return self._bce(logits, targets.float())


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so the mean weight is 1.

    Stable in the presence of absent classes (gives them weight 1.0 to avoid
    division by zero).
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (counts * num_classes)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def build_loss(
    cfg: LossConfig,
    *,
    train_labels: np.ndarray | None = None,
    num_classes: int = 2,
) -> nn.Module:
    """Construct the configured loss.

    Args:
        cfg: LossConfig instance.
        train_labels: Labels of the training set; used when
            ``cfg.auto_class_weights`` is True. If None and auto-weighting is
            requested, weights default to uniform.
        num_classes: Total number of classes (2 for binary).
    """
    weight: torch.Tensor | None = None
    if cfg.auto_class_weights:
        if train_labels is None:
            weight = torch.ones(num_classes, dtype=torch.float32)
        else:
            weight = compute_class_weights(train_labels, num_classes)

    if cfg.name == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if cfg.name == "focal":
        return FocalLoss(alpha=weight, gamma=cfg.focal_gamma)
    if cfg.name == "bce":
        pos_weight: torch.Tensor | None = None
        if cfg.auto_class_weights and train_labels is not None:
            n_pos = float((train_labels == 1).sum())
            n_neg = float((train_labels == 0).sum())
            if n_pos > 0:
                pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        return BCEWithLogitsLossWrapper(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss name: {cfg.name!r}")


__all__ = ["BCEWithLogitsLossWrapper", "FocalLoss", "build_loss", "compute_class_weights"]
