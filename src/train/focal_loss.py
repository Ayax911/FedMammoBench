"""Focal Loss para clasificación binaria con cabeza de 2 logits.

Portado desde el proyecto INC
(`classification_images/focal_loss.py`) para reducir el peso de ejemplos ya
bien clasificados y enfocar el gradiente en los difíciles — útil en un
dataset desbalanceado ~66/34 benigno/maligno como `fedmammobench.csv`.

Se registra como una tercera entrada de `_LOSSES` en `train/build.py`
("focal"), con el mismo contrato `LossSpec` que "bce"/"cross_entropy" — ver
ese módulo para el mecanismo de dispatch.

A diferencia de la versión del INC (que guarda `alpha` como atributo plano y
asume `torch.device('cuda')` hardcodeado), aquí `alpha` se registra como
buffer de `nn.Module` — así `loss_fn.to(device)` lo mueve junto con el resto
del módulo, sin asumir ningún dispositivo fijo.

Ejemplo de uso:
    >>> import torch
    >>> from src.train.focal_loss import FocalLoss
    >>> alpha = torch.tensor([2.0, 1.0])  # pesa más la clase negativa
    >>> loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    >>> logits = torch.randn(4, 2)
    >>> targets = torch.tensor([0, 1, 1, 0])
    >>> loss = loss_fn(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss para clasificación binaria con logits `[B, 2]`.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduce el peso relativo de ejemplos fáciles (`p_t` alto) frente a
    BCE/CrossEntropy estándar, y opcionalmente aplica pesos por clase
    (`alpha`) además del reweighting dinámico por `gamma`.

    Attributes:
        alpha: tensor `[2]` con el peso por clase (`[negativo, positivo]`),
            o `None` para no ponderar por clase. Registrado como buffer, no
            como atributo plano, para que se mueva de dispositivo junto con
            el módulo.
        gamma: exponente de enfoque — `0` reduce esto a cross-entropy
            ponderada por `alpha`.
        reduction: `"mean"` | `"sum"` | `"none"`.

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> loss_fn(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Inicializa la loss con los pesos por clase y el exponente de enfoque.

        Args:
            alpha: tensor `[2]` `[peso_clase_0, peso_clase_1]`, o `None`.
            gamma: exponente de enfoque (ver docstring de la clase).
            reduction: cómo reducir la pérdida por batch — `"mean"`, `"sum"`
                o `"none"` (sin reducir, devuelve `[B]`).
        """
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcula la focal loss sobre un batch.

        Args:
            logits: tensor `[B, 2]` de logits crudos (sin softmax aplicado).
            targets: tensor `[B]` `Long` con el índice de clase (0 o 1).

        Returns:
            torch.Tensor: escalar si `reduction != "none"`; si no, `[B]`.
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        idx = torch.arange(len(targets), device=logits.device)
        pt = probs[idx, targets]
        ce_loss = -log_probs[idx, targets]
        focal_factor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            loss = self.alpha[targets] * focal_factor * ce_loss
        else:
            loss = focal_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
