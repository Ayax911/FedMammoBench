"""Evaluación de un checkpoint guardado sobre un `DataLoader` — típicamente
el checkpoint que devuelve `Trainer.fit()` (el mejor, nunca el último;
ver `Trainer.fit()`) sobre `loaders["test"]`.

Antes de este módulo, `src/cli.py` terminaba en `trainer.fit()` y nunca
tocaba `loaders["test"]` — el pipeline no podía producir ni una sola métrica
de test. Reutiliza `checkpoint.load_checkpoint()` y `train/loop.py:evaluate()`
en vez de reimplementar el forward pass, a diferencia del `test_model()` del
proyecto INC (`classification_images/training.py`), que duplica el loop de
evaluación completo dentro de una clase de 426 líneas.

Ejemplo de uso:
    >>> from src.train.evaluation import evaluate_checkpoint, predict_on_loader
    >>> best_path = trainer.fit(loaders["train"], loaders["val"], epochs=10)
    >>> test_metrics = evaluate_checkpoint(model, best_path, loaders["test"], loss_spec, device)
    >>> y_true, y_pred, y_prob = predict_on_loader(model, loaders["test"], loss_spec, device)
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..checkpoint import load_checkpoint
from .build import LossSpec
from .loop import evaluate


def evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    loader: DataLoader[tuple[torch.Tensor, int]],
    loss_spec: LossSpec,
    device: str = "cpu",
) -> dict[str, float]:
    """Carga `checkpoint_path` en `model` (in-place) y evalúa sobre `loader`.

    Args:
        model: modelo completo (backbone + cabeza). Sus pesos actuales se
            sobreescriben con los del checkpoint — no importa en qué estado
            llegue.
        checkpoint_path: ruta a un checkpoint guardado por
            `checkpoint.save_checkpoint()` (típicamente el retorno de
            `Trainer.fit()`).
        loader: DataLoader a evaluar, típicamente `loaders["test"]`.
        loss_spec: especificación de pérdida construida con `build_loss()`
            — debe ser la misma que se usó para entrenar ese checkpoint.
        device: dispositivo de cómputo (`"cpu"`, `"cuda"`).

    Returns:
        dict[str, float]: el mismo dict que devuelve `evaluate()` —
            `"loss"` + métricas clínicas (`accuracy`, `auc`, `sensitivity`,
            `specificity`, `f1`, `precision`).

    Example:
        >>> best_path = trainer.fit(loaders["train"], loaders["val"], epochs=10)
        >>> test_metrics = evaluate_checkpoint(model, best_path, loaders["test"], loss_spec, "cuda")
        >>> print(test_metrics["auc"])
    """
    load_checkpoint(model, checkpoint_path, device=device)
    return evaluate(model, loader, loss_spec, device)


@torch.no_grad()
def predict_on_loader(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    loss_spec: LossSpec,
    device: str = "cpu",
) -> tuple[list[int], list[int], list[float]]:
    """Corre inferencia sobre todo `loader` y devuelve etiquetas/predicciones/probabilidades planas.

    Separado de `evaluate()` (`train/loop.py`) a propósito: `evaluate()` se
    llama una vez por época durante entrenamiento y solo necesita el resumen
    agregado (rápido, sin acumular nada en memoria). Esta función paga el
    costo de acumular todo el split en memoria — necesario para la matriz de
    confusión, la curva ROC y `predictions.csv` (`src/reporting.py`) — pero
    solo se llama una vez, al final, sobre test.

    Args:
        model: modelo con pesos ya cargados (ver `evaluate_checkpoint()` —
            llamarla primero deja `model` listo para esta función).
        loader: DataLoader a recorrer, típicamente `loaders["test"]`.
        loss_spec: usado solo por `.probs()`, para convertir logits en
            probabilidad de la clase positiva sin importar el esquema de
            pérdida (`bce` de 1 logit o `cross_entropy`/`focal` de 2) — ver
            `train/build.py`.
        device: dispositivo de cómputo.

    Returns:
        tuple[list[int], list[int], list[float]]: `(y_true, y_pred, y_prob)`,
            alineados por posición con el orden de iteración de `loader`.
            Requiere `shuffle=False`, que es como `builder_dataloader()`
            construye los loaders de `"val"`/`"test"`. `y_pred` usa un
            umbral fijo de 0.5 sobre `y_prob`.

    Example:
        >>> y_true, y_pred, y_prob = predict_on_loader(model, loaders["test"], loss_spec, "cuda")
        >>> len(y_true) == len(loaders["test"].dataset)
        True
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = loss_spec.probs(outputs)

        y_true.extend(labels.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())
        y_pred.extend((probs >= 0.5).long().detach().cpu().tolist())

    return y_true, y_pred, y_prob
