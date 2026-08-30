# train/build.py
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT


_OPTIMIZERS: dict[str, Callable[..., Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    
}


def build_optimizer(params: ParamsT, name: str, **hparams: Any) -> Optimizer:
    """Construye un optimizer por nombre, con los hiperparámetros que le pases.

    Args:
        params: parámetros del modelo a optimizar. Acepta lo mismo que
            Optimizer acepta nativamente — model.parameters() (un iterable
            plano), o una lista de param groups (dicts con su propio "lr",
            para LR discriminativo cabeza/backbone) — no solo el primer caso.
        name: clave en _OPTIMIZERS, ej. "adam".
        **hparams: hiperparámetros propios de ese optimizer (lr, weight_decay, etc.).

    Raises:
        ValueError: name no reconocido.
    """
    if name not in _OPTIMIZERS:
        raise ValueError(f"Optimizer desconocido: {name!r}. Opciones: {sorted(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](params, **hparams)

_SCHEDULERS: dict[str, Callable[..., LRScheduler]] = {
    "reduceonplateu": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}

def build_scheduler(optimizer: Optimizer, name: str, **hparams: Any) -> LRScheduler:
    """Construye un scheduler por nombre, con los hiperparámetros que le pases.

    Args:
        optimizer: optimizer al que aplicar el scheduler.
        name: clave en _SCHEDULERS, ej. "reduceonplateu".
        **hparams: hiperparámetros propios de ese scheduler (patience, mode, etc.).

    Returns:
        LRScheduler: el scheduler ya construido (instancia, no una fábrica) —
            listo para pasarle a Trainer(scheduler=...).

    Raises:
        ValueError: name no reconocido.
    """
    if name not in _SCHEDULERS:
        raise ValueError(f"Scheduler desconocido: {name!r}. Opciones: {sorted(_SCHEDULERS)}")
    return _SCHEDULERS[name](optimizer, **hparams)

@dataclass(frozen=True)
class LossSpec:
    """Une, para un esquema de salida dado, la función de pérdida con la
    forma correcta de convertir logits crudos en la probabilidad de la clase
    positiva. Existe porque esa conversión depende de cuántos logits emite
    la cabeza (1 con BCE, 2 con CrossEntropy) — nunca del nombre de la loss
    ni de un flag aparte que se pueda desincronizar del modelo real.

    `train_one_epoch()`/`evaluate()` (train/loop.py) reciben un `LossSpec`
    en vez de una `nn.Module` suelta y llaman `spec.compute(...)` /
    `spec.probs(...)` sin ningún `if` propio — el único `if` de todo este
    mecanismo vive en `build_loss()`, se evalúa una vez al construir el
    `LossSpec`, no una vez por batch.

    Atributos:
        compute: `(outputs, labels) -> loss escalar`. Ya sabe si tiene que
            castear/squeeze `outputs`/`labels` para el esquema elegido.
        probs: `outputs -> probabilidad de la clase positiva`, shape `[B]`.
            Usado por `evaluate()` para alimentar las métricas (AUC, etc.),
            nunca para entrenar.
    """

    compute: Callable[[Tensor, Tensor], Tensor]
    probs: Callable[[Tensor], Tensor]


def _make_bce(**hparams: Any) -> LossSpec:
    """Esquema de 1 logit: BCEWithLogitsLoss + sigmoid.

    La cabeza debe emitir `[B, 1]`. `squeeze(1)` lo deja en `[B]` para que
    calce con `labels` (que llega `[B]`, `Long` desde el DataLoader);
    BCEWithLogitsLoss además exige `labels` en `float`, de ahí el `.float()`.
    """
    loss_fn = nn.BCEWithLogitsLoss(**hparams)
    return LossSpec(
        compute=lambda outputs, labels: loss_fn(outputs.squeeze(1), labels.float()),
        probs=lambda outputs: torch.sigmoid(outputs.squeeze(1)),
    )


def _make_cross_entropy(**hparams: Any) -> LossSpec:
    """Esquema de 2 logits: CrossEntropyLoss + softmax.

    La cabeza debe emitir `[B, 2]`. CrossEntropyLoss ya espera `labels`
    como índice de clase `Long` `[B]` — que es justo lo que entrega el
    DataLoader por defecto, sin casteo. `probs` toma la columna 1
    (`malignant`, ver `Manifest.normalize_labels`) del softmax.
    """
    loss_fn = nn.CrossEntropyLoss(**hparams)
    return LossSpec(
        compute=lambda outputs, labels: loss_fn(outputs, labels),
        probs=lambda outputs: torch.softmax(outputs, dim=1)[:, 1],
    )


_LOSSES: dict[str, Callable[..., LossSpec]] = {
    "bce": _make_bce,
    "cross_entropy": _make_cross_entropy,
}


def build_loss(name: str, **hparams: Any) -> LossSpec:
    """Construye un LossSpec por nombre, con los hiperparámetros que le pases.

    El nombre elegido debe ser consistente con `num_classes` de la cabeza
    del modelo: "bce" espera una cabeza de 1 logit, "cross_entropy" una de 2.
    Esta función no puede verificar eso (no ve el modelo) — el mismatch se
    manifiesta como un error de shape/dtype de PyTorch dentro de
    `LossSpec.compute()`, no aquí.

    Args:
        name: clave en _LOSSES, "bce" o "cross_entropy".
        **hparams: hiperparámetros propios de la función de pérdida (weight, reduction, etc.).

    Returns:
        LossSpec: el par (compute, probs) ya cerrado sobre la loss construida.

    Raises:
        ValueError: name no reconocido.
    """
    if name not in _LOSSES:
        raise ValueError(f"Loss desconocida: {name!r}. Opciones: {sorted(_LOSSES)}")
    return _LOSSES[name](**hparams)
