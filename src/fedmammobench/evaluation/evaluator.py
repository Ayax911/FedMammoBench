"""Evaluator: runs a model over a DataLoader and returns metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedmammobench.evaluation.metrics import BinaryClassificationMetrics, compute_metrics


class Evaluator:
    """Stateless evaluator.

    Args:
        model: Module producing logits of shape ``(B, num_classes)``.
        device: Device to run on.
        threshold: Decision threshold for the positive class.
    """

    def __init__(self, model: nn.Module, device: torch.device, *, threshold: float = 0.5) -> None:
        self.model = model
        self.device = device
        self.threshold = float(threshold)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        *,
        criterion: nn.Module | None = None,
        return_predictions: bool = False,
    ) -> dict[str, Any]:
        """Run the model over ``loader`` and return a metrics dict.

        Returns:
            A dict with keys from :class:`BinaryClassificationMetrics` plus
            ``loss`` (mean loss, if ``criterion`` is provided) and
            ``num_batches``. When ``return_predictions`` is True, also
            includes ``y_true``, ``y_prob`` arrays for downstream analysis.
        """
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        loss_sum = 0.0
        n_batches = 0
        n_samples = 0

        for batch in loader:
            images, targets = batch
            images = images.to(self.device, non_blocking=True)
            targets_dev = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            if criterion is not None:
                loss = criterion(logits, targets_dev.long())
                loss_sum += float(loss.item()) * targets_dev.size(0)
                n_samples += int(targets_dev.size(0))
            probs = torch.softmax(logits, dim=-1)
            # Positive class is index 1; if num_classes > 2 the metrics module
            # will treat anything above threshold as positive, but for binary
            # mammography the positive class is malignant == 1.
            pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            all_probs.append(pos_probs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            n_batches += 1

        if not all_probs:
            return {
                "loss": 0.0,
                "num_batches": 0,
                **BinaryClassificationMetrics(
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    roc_auc=float("nan"),
                    sensitivity=0.0,
                    specificity=0.0,
                    support=0,
                    positives=0,
                    negatives=0,
                ).to_dict(),
            }

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        metrics = compute_metrics(y_true, y_prob, threshold=self.threshold)
        out: dict[str, Any] = {
            **metrics.to_dict(),
            "num_batches": n_batches,
            "loss": (loss_sum / n_samples) if (criterion is not None and n_samples > 0) else 0.0,
        }
        if return_predictions:
            out["y_true"] = y_true
            out["y_prob"] = y_prob
        return out


__all__ = ["Evaluator"]
