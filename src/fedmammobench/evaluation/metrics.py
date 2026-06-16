"""Clinical-style binary classification metrics.

The positive class is ``1`` (malignant). Sensitivity = recall on positives,
specificity = true negative rate on benign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """A snapshot of metrics for one evaluation pass."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    auc_pr: float       # Average Precision — primary metric in centralized training
    sensitivity: float
    specificity: float
    support: int
    positives: int
    negatives: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_prob_positive: np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """Compute metrics from ground truth and positive-class probability.

    Args:
        y_true: Integer labels (0 / 1).
        y_prob_positive: Probability of the positive class. Shape (N,).
        threshold: Decision threshold for binarization.

    Returns:
        A :class:`BinaryClassificationMetrics` instance.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob_positive).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, y_prob_positive={y_prob.shape}"
        )

    y_pred = (y_prob >= float(threshold)).astype(int)

    accuracy = float((y_pred == y_true).mean()) if y_true.size > 0 else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    # ROC-AUC and AUC-PR both need both classes present.
    if np.unique(y_true).size < 2:
        roc_auc = float("nan")
        auc_pr = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        auc_pr = float(average_precision_score(y_true, y_prob))

    # Confusion matrix with explicit labels for stable shape.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        auc_pr=auc_pr,
        sensitivity=sensitivity,
        specificity=specificity,
        support=int(y_true.size),
        positives=int((y_true == 1).sum()),
        negatives=int((y_true == 0).sum()),
    )


__all__ = ["BinaryClassificationMetrics", "compute_metrics"]
