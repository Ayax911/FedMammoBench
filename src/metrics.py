"""Cálculo de métricas de clasificación binaria usando torchmetrics.

Estos objetos acumulan estado batch a batch: se instancian una vez por
evaluación, se actualizan por cada batch con `.update()`, y se consultan
al final con `.compute()`.

Métricas incluidas:
    - Accuracy (`BinaryAccuracy`)
    - AUROC (`BinaryAUROC`)
    - Sensitivity / Recall (`BinaryRecall`)
    - Specificity (`BinarySpecificity`)
    - F1-Score (`BinaryF1Score`) -- métrica de selección de mejor checkpoint
      del proyecto INC (early_stopping sobre "Val_F1-Score"); ver
      `train/early_stopping.py` y PHASES.md fase 1/3.
    - Precision (`BinaryPrecision`) -- equivalente al "VPP" (TP/(TP+FP))
      calculado a mano en el proyecto INC
      (classification_images/metrics.py:vpp).

Ejemplo de uso:
    >>> import torch
    >>> from src.metrics import build_metric_collection
    >>> metrics = build_metric_collection(device="cpu")
    >>> metrics.update(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    >>> results = metrics.compute()
    >>> print(results["auc"].item())
"""

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,  # sensibilidad = recall de la clase positiva
    BinarySpecificity,
)


def build_metric_collection(device: str = "cpu") -> MetricCollection:
    """Crea el conjunto de métricas a trackear durante la evaluación del modelo.

    Args:
        device: Dispositivo al que mover las métricas (`"cpu"`, `"cuda"`). Debe coincidir
            con el dispositivo de los tensores pasados a `.update()`.

    Returns:
        MetricCollection: Colección de `BinaryAccuracy`, `BinaryAUROC`, `BinaryRecall` (Sensibilidad),
        `BinarySpecificity`, `BinaryF1Score` y `BinaryPrecision` reubicada en el dispositivo especificado.

    Example:
        >>> metrics = build_metric_collection(device="cuda")
        >>> metrics.update(probs, labels)
        >>> print(metrics.compute())
    """
    return MetricCollection({
        "accuracy": BinaryAccuracy(),
        "auc": BinaryAUROC(),
        "sensitivity": BinaryRecall(),
        "specificity": BinarySpecificity(),
        "f1": BinaryF1Score(),
        "precision": BinaryPrecision(),
    }).to(device)
