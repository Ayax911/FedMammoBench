"""Cálculo de métricas de clasificación binaria usando torchmetrics.

Estos objetos acumulan estado batch a batch: se instancian una vez por
evaluación, se actualizan por cada batch con `.update()`, y se consultan
al final con `.compute()`.

Métricas incluidas:
    - Accuracy (`BinaryAccuracy`)
    - AUROC (`BinaryAUROC`)
    - Sensitivity / Recall (`BinaryRecall`)
    - Specificity (`BinarySpecificity`)
    - F1-Score (`BinaryF1Score`) -- SOLO la clase positiva (maligno).
    - F1-macro (`BinaryMacroF1Score`) -- promedio del F1 de ambas clases;
      es el criterio de mejor checkpoint de la serie de notebooks
      (`f1_score(..., average="macro")`) y el que debe usarse para
      reproducir sus resultados. Ver la nota de abajo.
    - Precision (`BinaryPrecision`) -- equivalente al "VPP" (TP/(TP+FP))
      calculado a mano en el proyecto INC
      (classification_images/metrics.py:vpp).

`"f1"` vs `"f1_macro"` -- cuál elegir en `train.metric_name`:
    `BinaryF1Score` mide únicamente la clase positiva, así que vale
    exactamente `0.0` en cuanto el modelo no predice ningún maligno --
    el estado normal de las primeras épocas con el backbone congelado y
    el desbalance ~66/34 del manifest. Como `EarlyStopping` exige mejora
    estricta, una racha de ceros no resetea el contador: con
    `patience: 10` la corrida se detiene en la época 10 y `fit()` devuelve
    el checkpoint de la época 0, es decir un modelo sin entrenar. El
    F1-macro nunca vale 0 en ese escenario (la clase mayoritaria aporta
    ~0.40) y varía suavemente, que es justo lo que un criterio de parada
    necesita. `"f1"` se mantiene disponible por comparabilidad, pero
    `"f1_macro"` es el default correcto para esta serie.

Ejemplo de uso:
    >>> import torch
    >>> from src.metrics import build_metric_collection
    >>> metrics = build_metric_collection(device="cpu")
    >>> metrics.update(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    >>> results = metrics.compute()
    >>> print(results["auc"].item())
"""

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,  # sensibilidad = recall de la clase positiva
    BinarySpecificity,
)


class BinaryMacroF1Score(Metric):
    """F1-macro binario: promedio simple del F1 de la clase negativa y el de la positiva.

    Equivale a `sklearn.metrics.f1_score(y_true, y_pred, average="macro")`,
    que es lo que usan los notebooks de la serie (`val_f1_history`) para
    elegir el mejor checkpoint. torchmetrics no trae un equivalente binario
    directo: `MulticlassF1Score(num_classes=2, average="macro")` sí lo
    calcula, pero espera `[B, C]` o etiquetas enteras `[B]`, mientras que
    `evaluate()` (`train/loop.py`) alimenta toda la colección con la
    probabilidad de la clase positiva `[B]` -- de ahí esta implementación.

    Acumula `tp`/`fp`/`fn` por clase (vectores de 2) en vez de envolver dos
    `BinaryF1Score`: evita anidar métricas dentro de una `MetricCollection`
    (que tendría que propagar `reset()`/`to()` a mano) y deja el cálculo a
    la vista.

    Attributes:
        threshold: umbral sobre la probabilidad de la clase positiva para
            binarizar las predicciones. Default `0.5`, igual que el resto de
            métricas binarias de torchmetrics.

    Example:
        >>> import torch
        >>> metric = BinaryMacroF1Score()
        >>> metric.update(torch.tensor([0.9, 0.2, 0.7]), torch.tensor([1, 0, 0]))
        >>> print(metric.compute().item())
    """

    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False

    tp: Tensor
    fp: Tensor
    fn: Tensor

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        """Inicializa los acumuladores por clase.

        Args:
            threshold: umbral de binarización sobre la probabilidad positiva.
            **kwargs: argumentos propios de `torchmetrics.Metric`.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("tp", default=torch.zeros(2), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(2), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(2), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Acumula el conteo de tp/fp/fn de cada clase para este batch.

        Args:
            preds: probabilidad de la clase positiva `[B]` (float), o
                etiquetas ya binarizadas `[B]` (int).
            target: etiquetas reales `[B]` (0 = benigno, 1 = maligno).
        """
        # Binarizar UNA sola vez, con el mismo criterio que torchmetrics
        # (`>` estricto), y derivar la clase negativa de esa misma
        # predicción. Invertir la probabilidad (`1 - preds > 0.5`) daría un
        # resultado distinto justo en preds == 0.5, donde ambas ramas
        # marcarían la muestra como suya.
        pred_labels = (preds > self.threshold).long() if preds.is_floating_point() else preds.long()
        target = target.long()

        for class_idx in (0, 1):
            predicted = pred_labels == class_idx
            actual = target == class_idx
            self.tp[class_idx] += (predicted & actual).sum()
            self.fp[class_idx] += (predicted & ~actual).sum()
            self.fn[class_idx] += (~predicted & actual).sum()

    def compute(self) -> Tensor:
        """Promedia el F1 de ambas clases.

        Returns:
            Tensor: escalar con el F1-macro. Una clase sin ninguna
            predicción ni etiqueta aporta `0.0`, igual que el
            `zero_division=0` de scikit-learn.
        """
        denominator = 2 * self.tp + self.fp + self.fn
        f1_per_class = torch.where(
            denominator > 0, 2 * self.tp / denominator, torch.zeros_like(denominator)
        )
        return f1_per_class.mean()


def build_metric_collection(device: str = "cpu") -> MetricCollection:
    """Crea el conjunto de métricas a trackear durante la evaluación del modelo.

    Args:
        device: Dispositivo al que mover las métricas (`"cpu"`, `"cuda"`). Debe coincidir
            con el dispositivo de los tensores pasados a `.update()`.

    Returns:
        MetricCollection: Colección de `BinaryAccuracy`, `BinaryAUROC`, `BinaryRecall` (Sensibilidad),
        `BinarySpecificity`, `BinaryF1Score`, `BinaryMacroF1Score` y `BinaryPrecision`
        reubicada en el dispositivo especificado.

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
        "f1": BinaryF1Score(),          # solo la clase positiva
        "f1_macro": BinaryMacroF1Score(),  # promedio de ambas -- ver nota del módulo
        "precision": BinaryPrecision(),
    }).to(device)
