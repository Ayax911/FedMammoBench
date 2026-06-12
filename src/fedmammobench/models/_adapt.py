"""Utilities for adapting the first conv of a pretrained backbone.

Two entry points:

* :func:`adapt_first_conv` — replaces a ``nn.Conv2d`` module in-place, used
  by model builders to set up the correct input-channel architecture.
* :func:`adapt_weight_tensor` — pure tensor operation, used by weight loaders
  to adapt a pretrained weight tensor before ``load_state_dict``.

Strategies
----------
``"sum_preserving"`` (default)
    Average the source channels, then repeat and scale so the expected
    activation magnitude is preserved when input pixel statistics are similar
    across channels.

    For src_in→target_in:
      ``scale = src_in / target_in``
      ``new_weight[..., c, ...] = mean(W, dim=channel) * scale``

    Special cases:
      3→1:  sum of the three RGB filters → single grayscale filter.
            An all-equal-pixel image produces the same activation as with
            the original RGB conv.
      1→3:  original weight / 3 replicated on each RGB channel.
      N→M:  generalisation of the above.

``"legacy_mean"``
    Original fedmammobench behaviour (pre-FASE 3). Averages channels and repeats
    *without* the ``src_in/target_in`` scale factor. Kept for reproducibility
    of older runs.  Produces activations that are ``target_in`` times larger
    than ``sum_preserving`` when target_in > src_in.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)

_Strategy = Literal["sum_preserving", "legacy_mean"]


# ---------------------------------------------------------------------------
# Pure tensor operation (used by weight loaders)
# ---------------------------------------------------------------------------

def adapt_weight_tensor(
    weight: torch.Tensor,
    target_in_channels: int,
    strategy: _Strategy = "sum_preserving",
) -> torch.Tensor:
    """Adapt a conv weight tensor ``[out, src_in, k, k]`` to
    ``[out, target_in_channels, k, k]``.

    Args:
        weight: The source weight tensor.
        target_in_channels: Desired number of input channels.
        strategy: Adaptation strategy.  ``"sum_preserving"`` preserves
            expected activation magnitude; ``"legacy_mean"`` matches pre-FASE 3
            behaviour.

    Returns:
        New tensor with shape ``[out, target_in_channels, *spatial]``.
        Returns ``weight`` unchanged if ``src_in == target_in_channels``.
    """
    src_in = weight.shape[1]
    if src_in == target_in_channels:
        return weight

    averaged = weight.mean(dim=1, keepdim=True)  # [out, 1, k, k]
    if strategy == "sum_preserving":
        scale = src_in / target_in_channels
        adapted = averaged.repeat(1, target_in_channels, 1, 1) * scale
    else:  # legacy_mean
        adapted = averaged.repeat(1, target_in_channels, 1, 1)

    _logger.debug(
        "adapt_weight_tensor: %s → %s channels (strategy=%r, scale=%.3f)",
        src_in,
        target_in_channels,
        strategy,
        (src_in / target_in_channels) if strategy == "sum_preserving" else 1.0,
    )
    return adapted


# ---------------------------------------------------------------------------
# Module-level adapter (used by model builders)
# ---------------------------------------------------------------------------

def adapt_first_conv(
    conv: nn.Conv2d,
    in_channels: int,
    strategy: _Strategy = "sum_preserving",
) -> nn.Conv2d:
    """Return a new ``nn.Conv2d`` with ``in_channels`` input channels.

    The weight is adapted from ``conv`` using ``strategy``.  Bias (if present)
    is copied unchanged.

    Args:
        conv: The source Conv2d layer (e.g. ``backbone.conv1``).
        in_channels: Desired number of input channels.
        strategy: Weight adaptation strategy.  Defaults to
            ``"sum_preserving"``; use ``"legacy_mean"`` to reproduce runs
            created before FASE 3.

    Returns:
        The input ``conv`` unchanged when ``conv.in_channels == in_channels``;
        otherwise a new Conv2d with adapted weights.
    """
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,       # type: ignore[arg-type]
        stride=conv.stride,                 # type: ignore[arg-type]
        padding=conv.padding,               # type: ignore[arg-type]
        dilation=conv.dilation,             # type: ignore[arg-type]
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight.copy_(
            adapt_weight_tensor(conv.weight, in_channels, strategy=strategy)
        )
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    _logger.info(
        "adapt_first_conv: %d→%d channels (strategy=%r)",
        conv.in_channels,
        in_channels,
        strategy,
    )
    return new_conv


__all__ = ["adapt_first_conv", "adapt_weight_tensor"]
