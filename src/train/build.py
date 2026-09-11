"""Fábricas de optimizadores, schedulers y especificadores de funciones de pérdida (LossSpec).

Ejemplo de uso:
    >>> import torch.nn as nn
    >>> from src.train.build import build_optimizer, build_scheduler, build_loss
    >>> model = nn.Linear(10, 1)
    >>> optimizer = build_optimizer(model.parameters(), name="adamw", lr=1e-4)
    >>> scheduler = build_scheduler(optimizer, name="cosine", T_max=10)
    >>> loss_spec = build_loss(name="bce")
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT

from .focal_loss import FocalLoss


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

    Returns:
        Optimizer: Instancia configurada del optimizador PyTorch.

    Raises:
        ValueError: name no reconocido.

    Example:
        >>> optimizer = build_optimizer(model.parameters(), "adamw", lr=0.001)
    """
    if name not in _OPTIMIZERS:
        raise ValueError(f"Optimizer desconocido: {name!r}. Opciones: {sorted(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](params, **hparams)


def build_param_groups(
    model: nn.Sequential, optimizer_hparams: dict[str, Any]
) -> tuple[ParamsT, dict[str, Any]]:
    """Arma los param groups para LR discriminativo backbone/cabeza.

    Extraído de `cli.py` (donde vivía inline) para que TODO ensamblador de
    optimizador — `cli.run()` y el cliente federado (`federated/client.py`),
    que reconstruye su optimizer fresco cada ronda — pase por el mismo
    código: si esta lógica viviera solo en `cli.py`, un cliente federado
    con `backbone_lr` en su YAML la perdería en silencio y entrenaría con
    un solo LR plano.

    `backbone_lr`, si viene en `optimizer_hparams`, separa `model[0]` (el
    backbone) en su propio param group con ese LR, dejando `model[1]` (la
    cabeza) en el LR general del resto de los hparams. Filtra por
    `requires_grad` para no meterle al estado del optimizer parámetros del
    backbone que `unfreeze_from` dejó congelados (un freeze parcial, ej.
    `layer4`, no debería aportarle momentum/weight_decay a conv1-layer3).
    Sin `backbone_lr`, comportamiento idéntico a no llamar esta función:
    un solo grupo con `model.parameters()` completo.

    Args:
        model: el `nn.Sequential(backbone, head)` ensamblado — se indexa
            `model[0]`/`model[1]`, así que debe tener exactamente esa forma.
        optimizer_hparams: los `hparams` del YAML del optimizador. NO se
            muta: se copia, y `backbone_lr` se extrae de la copia.

    Returns:
        tuple: `(params, hparams_restantes)` — `params` es lo que se le
        pasa a `build_optimizer()` (param groups o `model.parameters()`), y
        `hparams_restantes` los hparams sin `backbone_lr` (que no es un
        argumento válido del constructor de ningún optimizer de PyTorch).

    Example:
        >>> params, hparams = build_param_groups(model, {"lr": 1e-3, "backbone_lr": 1e-4})
        >>> optimizer = build_optimizer(params, "adamw", **hparams)
    """
    hparams = dict(optimizer_hparams)
    backbone_lr = hparams.pop("backbone_lr", None)
    if backbone_lr is not None:
        params: ParamsT = [
            {"params": [p for p in model[0].parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": model[1].parameters()},
        ]
    else:
        params = model.parameters()
    return params, hparams


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

    Example:
        >>> scheduler = build_scheduler(optimizer, "cosine", T_max=20)
    """
    if name not in _SCHEDULERS:
        raise ValueError(f"Scheduler desconocido: {name!r}. Opciones: {sorted(_SCHEDULERS)}")
    return _SCHEDULERS[name](optimizer, **hparams)


@dataclass(frozen=True)
class LossSpec:
    """Une, para un esquema de salida dado, la función de pérdida con la
    forma correcta de convertir logits crudos en la probabilidad de la clase
    positiva. Existe porque esa conversión depende de cuántos logits emite
    la cabeza (1 con BCE, 2 con CrossEntropy/Focal) — nunca del nombre de la
    loss ni de un flag aparte que se pueda desincronizar del modelo real.

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

    Example:
        >>> loss_spec = build_loss("bce")
        >>> loss_val = loss_spec.compute(logits, labels)
        >>> probs_val = loss_spec.probs(logits)
    """

    compute: Callable[[Tensor, Tensor], Tensor]
    probs: Callable[[Tensor], Tensor]


def _make_bce(*, device: str = "cpu", **hparams: Any) -> LossSpec:
    """Esquema de 1 logit: BCEWithLogitsLoss + sigmoid.

    La cabeza debe emitir `[B, 1]`. `squeeze(1)` lo deja en `[B]` para que
    calce con `labels` (que llega `[B]`, `Long` desde el DataLoader);
    BCEWithLogitsLoss además exige `labels` en `float`, de ahí el `.float()`.

    `pos_weight`, si se pasa, llega desde YAML como una lista de Python (no
    hay tensores en YAML) — se convierte a `torch.Tensor` en el device
    correcto antes de construir la loss. Sin esta conversión,
    `BCEWithLogitsLoss(pos_weight=[...])` lanza `TypeError` al intentar
    registrar una lista como buffer (ver PHASES.md fase 2).
    """
    if "pos_weight" in hparams:
        hparams = {**hparams, "pos_weight": torch.as_tensor(hparams["pos_weight"], dtype=torch.float32, device=device)}

    loss_fn = nn.BCEWithLogitsLoss(**hparams).to(device)
    return LossSpec(
        compute=lambda outputs, labels: loss_fn(outputs.squeeze(1), labels.float()),
        probs=lambda outputs: torch.sigmoid(outputs.squeeze(1)),
    )


def _make_cross_entropy(*, device: str = "cpu", **hparams: Any) -> LossSpec:
    """Esquema de 2 logits: CrossEntropyLoss + softmax.

    La cabeza debe emitir `[B, 2]`. CrossEntropyLoss ya espera `labels`
    como índice de clase `Long` `[B]` — que es justo lo que entrega el
    DataLoader por defecto, sin casteo. `probs` toma la columna 1
    (`malignant`, ver `Manifest.normalize_labels`) del softmax.

    `weight` (pesos por clase `[negativo, positivo]`, ver el `class_balance`
    del proyecto INC), si se pasa, llega desde YAML como lista de Python y
    se convierte a `torch.Tensor` en el device correcto antes de construir
    la loss — mismo motivo que `pos_weight` en `_make_bce` (PHASES.md fase 2).
    """
    if "weight" in hparams:
        hparams = {**hparams, "weight": torch.as_tensor(hparams["weight"], dtype=torch.float32, device=device)}

    loss_fn = nn.CrossEntropyLoss(**hparams).to(device)
    return LossSpec(
        compute=lambda outputs, labels: loss_fn(outputs, labels),
        probs=lambda outputs: torch.softmax(outputs, dim=1)[:, 1],
    )


def _make_focal(*, device: str = "cpu", **hparams: Any) -> LossSpec:
    """Esquema de 2 logits con Focal Loss (ver `train/focal_loss.py`).

    Portado del proyecto INC — reduce el peso de ejemplos ya bien
    clasificados y admite ponderación por clase vía `alpha`, útil para el
    desbalance ~66/34 (benigno/maligno) de `fedmammobench.csv`. `alpha`,
    igual que `weight`/`pos_weight` en las otras dos losses, llega desde
    YAML como lista y se convierte a tensor en el device correcto.
    """
    if "alpha" in hparams:
        hparams = {**hparams, "alpha": torch.as_tensor(hparams["alpha"], dtype=torch.float32, device=device)}

    loss_fn = FocalLoss(**hparams).to(device)
    return LossSpec(
        compute=lambda outputs, labels: loss_fn(outputs, labels),
        probs=lambda outputs: torch.softmax(outputs, dim=1)[:, 1],
    )


_LOSSES: dict[str, Callable[..., LossSpec]] = {
    "bce": _make_bce,
    "cross_entropy": _make_cross_entropy,
    "focal": _make_focal,
}


def build_loss(name: str, *, device: str = "cpu", **hparams: Any) -> LossSpec:
    """Construye un LossSpec por nombre, con los hiperparámetros que le pases.

    El nombre elegido debe ser consistente con `num_classes` de la cabeza
    del modelo: "bce" espera una cabeza de 1 logit, "cross_entropy"/"focal"
    una de 2. Esta función no puede verificar eso (no ve el modelo) — el
    mismatch se manifiesta como un error de shape/dtype de PyTorch dentro de
    `LossSpec.compute()`, no aquí.

    Args:
        name: clave en _LOSSES: "bce", "cross_entropy" o "focal".
        device: dispositivo al que mover la loss y cualquier hparam que
            llegue como tensor de pesos (`weight`, `pos_weight`, `alpha`).
            Debe coincidir con el device del modelo — mismo patrón que
            `build_metric_collection(device=...)` en `src/metrics.py`.
        **hparams: hiperparámetros propios de la función de pérdida (weight,
            pos_weight, alpha, gamma, reduction, etc. según el esquema).

    Returns:
        LossSpec: el par (compute, probs) ya cerrado sobre la loss construida.

    Raises:
        ValueError: name no reconocido.

    Example:
        >>> loss_spec = build_loss("bce", device="cuda")
        >>> loss_spec = build_loss("cross_entropy", weight=[2.0, 1.0], device="cuda")
    """
    if name not in _LOSSES:
        raise ValueError(f"Loss desconocida: {name!r}. Opciones: {sorted(_LOSSES)}")
    return _LOSSES[name](device=device, **hparams)
