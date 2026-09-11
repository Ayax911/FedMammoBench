"""Funciones puras de entrenamiento y evaluación por época.

Proporciona `train_one_epoch()` y `evaluate()`, respetando el congelamiento
estadístico de capas BatchNorm congeladas y aislando el cálculo de pérdida y
probabilidades mediante `LossSpec`. Ambas devuelven exactamente las mismas
claves (`"loss"` + las siete de `build_metric_collection()`) -- así
`Trainer.fit()` puede combinarlas con prefijo `train_`/`val_` en un solo
dict por época sin casos especiales, y graficarse en pares train-vs-val
(ver `reporting.py:plot_metric_curve()`).

Ejemplo de uso:
    >>> from src.train.loop import train_one_epoch, evaluate
    >>> train_metrics = train_one_epoch(model, train_loader, optimizer, loss_spec, device="cpu")
    >>> val_metrics = evaluate(model, val_loader, loss_spec, device="cpu")
"""

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..metrics import build_metric_collection
from .build import LossSpec


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: Optimizer,
    loss_spec: LossSpec,
    device: str,
    freeze_bn_stats: bool = True,
    regularizer: Callable[[], Tensor] | None = None,
) -> dict[str, float]:
    """Entrena el modelo durante una época completa.

    Pone el modelo en `.train()`, mantiene congeladas las capas BatchNorm cuyas
    variables no requieren gradiente (`_set_frozen_bn_eval`), realiza forward,
    calcula la pérdida con `loss_spec.compute`, ejecuta backward y actualiza pesos.

    Args:
        model: modelo completo (backbone + cabeza).
        loader: DataLoader de entrenamiento.
        optimizer: optimizador PyTorch (ej. AdamW).
        loss_spec: especificación de pérdida construida con `build_loss()`.
        device: dispositivo de cómputo (`"cpu"`, `"cuda"`).
        freeze_bn_stats: si `True` (default) llama a `_set_frozen_bn_eval()`
            tras `model.train()`, dejando las BN congeladas en `eval()` para
            que no actualicen `running_mean`/`running_var`. `False` omite esa
            llamada y reproduce el comportamiento del proyecto INC, donde las
            estadísticas de BN sí derivan durante el entrenamiento aunque los
            pesos estén congelados.
        regularizer: término aditivo opcional de pérdida, `() -> Tensor`
            escalar, evaluado una vez por batch DESPUÉS del forward. Existe
            para el término proximal de FedProx (`federated/client.py`):
            `LossSpec.compute` no ve el modelo, así que la penalización
            (mu/2)·||w - w_global||² no puede vivir dentro del `LossSpec` —
            entra acá como closure sobre los parámetros del modelo. Con
            regularizer, el backward corre sobre `task_loss + regularizer()`
            pero la clave `"loss"` devuelta sigue siendo SOLO la task loss
            (misma semántica que `evaluate()` y comparable entre estrategias
            — el legacy reportaba la suma y contaminaba toda comparación
            FedAvg-vs-FedProx, su auditoría N8); el término penal se reporta
            aparte en `"prox_loss"`. `None` (default) = comportamiento
            byte-idéntico a antes de que este parámetro existiera: ni un
            `if` extra en la ruta del backward ni clave nueva en el dict.

    Returns:
        dict[str, float]: mismas claves que `evaluate()` -- `"loss"` +
        métricas clínicas (`"accuracy"`, `"auc"`, `"sensitivity"`,
        `"specificity"`, `"f1"`, `"f1_macro"`, `"precision"`) -- acumuladas
        sobre las predicciones de entrenamiento de esta época, no solo la
        pérdida. Antes de esto, `train_one_epoch()` solo devolvía `"loss"`;
        `Trainer.fit()` ya prefija cada clave con `train_` al combinarlas en
        `trainer.history` (ver `Trainer.fit()`), así que agregar claves acá
        las hace aparecer automáticamente ahí, en `metrics.csv`, TensorBoard
        y W&B (`MetricsLogger.log()`) sin tocar nada más. Con
        `regularizer` fijado se suma la clave `"prox_loss"` (media del
        término penal por batch) — SOLO en ese caso, para que el keyset de
        `MetricsLogger.log()` sea estable dentro de una corrida dada.

    Example:
        >>> train_metrics = train_one_epoch(model, train_loader, optimizer, loss_spec, "cuda")
        >>> print(f"Train loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
    """
    model.train()
    if freeze_bn_stats:
        _set_frozen_bn_eval(model)

    metrics = build_metric_collection(device)
    total_loss = 0.0
    total_reg = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_spec.compute(outputs, labels)
        if regularizer is not None:
            # El backward corre sobre la suma, pero `loss` (task) y `reg`
            # se acumulan por separado -- ver el docstring de `regularizer`.
            reg = regularizer()
            (loss + reg).backward()  # pyright: ignore[reportUnknownMemberType]
            total_reg += reg.item()
        else:
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()

        # Sin gradiente -- el paso de optimización ya terminó, esto solo
        # acumula estado para las métricas (mismo patrón que evaluate()).
        # outputs sigue con su grafo de autograd intacto en este punto;
        # .detach() antes de loss_spec.probs() evita retenerlo un batch de
        # más innecesariamente.
        with torch.no_grad():
            probs = loss_spec.probs(outputs.detach())
            metrics.update(probs, labels)

        total_loss += loss.item()
        n_batches += 1

    result = {k: v.item() for k, v in metrics.compute().items()}
    result["loss"] = total_loss / n_batches
    if regularizer is not None:
        result["prox_loss"] = total_reg / n_batches
    return result


def _set_frozen_bn_eval(model: nn.Module) -> None:
    """Mantiene en modo eval las capas BatchNorm cuyos parámetros están
    congelados (requires_grad=False), incluso después de model.train().

    Sin esto, BN congelado sigue actualizando running_mean/running_var con
    datos de entrenamiento — el bug legacy documentado del proyecto.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    loss_spec: LossSpec,
    device: str,
) -> dict[str, float]:
    """Evalúa el modelo sobre cualquier loader (val o test, indistintamente).

    Ejecuta el paso forward en modo `@torch.no_grad()`, actualiza el acumulador de
    `torchmetrics` (Accuracy, AUC, Sensitivity, Specificity) y calcula la pérdida promedio.

    Args:
        model: modelo completo (backbone + cabeza).
        loader: DataLoader de validación o test.
        loss_spec: especificación de pérdida construida con `build_loss()`.
        device: dispositivo de cómputo (`"cpu"`, `"cuda"`).

    Returns:
        dict[str, float]: Diccionario con `"loss"` y métricas clínicas (`"accuracy"`,
        `"auc"`, `"sensitivity"`, `"specificity"`).

    Example:
        >>> val_metrics = evaluate(model, val_loader, loss_spec, "cuda")
        >>> print(f"AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    """
    model.eval()
    metrics = build_metric_collection(device)

    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_spec.compute(outputs, labels)
        probs = loss_spec.probs(outputs)

        metrics.update(probs, labels)
        total_loss += loss.item()
        n_batches += 1

    result = {k: v.item() for k, v in metrics.compute().items()}
    result["loss"] = total_loss / n_batches
    return result