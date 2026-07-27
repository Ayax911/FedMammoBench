"""Shared classification-head builder used by every model architecture.

The head always ends in ``Linear(*, num_classes)`` producing raw logits — no
activation or normalization follows the final layer. Softmax/sigmoid is
applied by the loss function (``CrossEntropyLoss``/``BCEWithLogitsLoss``) and
by the evaluator at inference time, never by the model itself.
"""

from __future__ import annotations

from torch import nn

from fedmammobench.configs.schema import ModelConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def build_head(in_features: int, cfg: ModelConfig) -> nn.Module:
    """Build a classification head from ``in_features`` to ``cfg.num_classes``.

    With ``cfg.head.hidden_dims`` empty (default), this reproduces the
    original head exactly: ``Linear(in_features, num_classes)`` when
    ``cfg.dropout == 0``, or ``Sequential(Dropout, Linear)`` otherwise — same
    module indices, same state_dict keys as before this feature existed.

    With hidden dims set, each hidden layer is
    ``[Dropout?] -> Linear -> [BatchNorm1d?] -> Activation``, followed by a
    final ``Linear(last_dim, num_classes)`` with nothing after it.
    """
    hidden_dims = cfg.head.hidden_dims

    if not hidden_dims:
        if cfg.dropout > 0.0:
            return nn.Sequential(
                nn.Dropout(p=cfg.dropout),
                nn.Linear(in_features, cfg.num_classes),
            )
        return nn.Linear(in_features, cfg.num_classes)

    activation_cls = _ACTIVATIONS[cfg.head.activation]

    layers: list[nn.Module] = []
    prev_dim = in_features
    for dim in hidden_dims:
        if cfg.dropout > 0.0:
            layers.append(nn.Dropout(p=cfg.dropout))
        layers.append(nn.Linear(prev_dim, dim))
        if cfg.head.batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(activation_cls())
        prev_dim = dim

    layers.append(nn.Linear(prev_dim, cfg.num_classes))
    return nn.Sequential(*layers)


__all__ = ["build_head"]
