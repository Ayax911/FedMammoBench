"""Cálculo de métricas de clasificación binaria usando torchmetrics.
Estos objetos acumulan estado batch a batch: se instancian una vez por
evaluación, se actualizan por cada batch con .update(), y se consultan
al final con .compute()."""
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryRecall,  # sensibilidad = recall de la clase positiva
    BinarySpecificity,
)


def build_metric_collection(device: str = "cpu") -> MetricCollection:
    """Crea el conjunto de métricas a trackear durante una evaluación.

    Args:
        device: dispositivo al que mover las métricas (debe coincidir con
            el dispositivo del modelo y los tensores que se les pasen).

    Returns:
        MetricCollection con accuracy, auc, sensitivity y specificity.
    """
    return MetricCollection({
        "accuracy": BinaryAccuracy(),
        "auc": BinaryAUROC(),
        "sensitivity": BinaryRecall(),
        "specificity": BinarySpecificity(),
    }).to(device)