import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..metrics import build_metric_collection
from .build import LossSpec


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_spec: LossSpec,
    device: str,
) -> dict[str, float]:
    """Entrena el modelo durante una época completa.

    Args:
        model: modelo completo (backbone + cabeza).
        loader: DataLoader de entrenamiento.
        optimizer: construido vía train/build.py.
        loss_spec: construido vía train/build.py (build_loss). Encapsula el
            esquema de salida (1 logit BCE vs 2 logits CrossEntropy) — esta
            función nunca necesita saber cuál de los dos es.
        device: dispositivo de entrenamiento.

    Returns:
        {"loss": promedio de la pérdida sobre todos los batches}
    """
    model.train()
    _set_frozen_bn_eval(model)

    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_spec.compute(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


def _set_frozen_bn_eval(model: nn.Module) -> None:
    """Mantiene en modo eval las capas BatchNorm cuyos parámetros están
    congelados (requires_grad=False), incluso después de model.train().

    Sin esto, BN congelado sigue actualizando running_mean/running_var con
    datos de entrenamiento — el bug legacy documentado del proyecto."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_spec: LossSpec,
    device: str,
) -> dict[str, float]:
    """Evalúa el modelo sobre cualquier loader (val o test, indistintamente).
    No calcula gradientes ni actualiza pesos.

    Args:
        model: modelo completo (backbone + cabeza).
        loader: DataLoader de validación o test.
        loss_spec: construido vía train/build.py (build_loss). `loss_spec.probs`
            reemplaza el softmax/sigmoid hardcodeado que solía vivir acá —
            ver LossSpec en train/build.py para el porqué.
        device: dispositivo de evaluación.

    Returns:
        Dict con "loss" + accuracy, auc, sensitivity, specificity
        (de build_metric_collection).
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