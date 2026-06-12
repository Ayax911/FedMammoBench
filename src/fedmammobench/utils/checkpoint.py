"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    epoch: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint.

    The checkpoint dict has stable keys: ``state_dict``, optionally
    ``optimizer``, ``epoch``, ``extra``.
    """
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload["extra"] = extra
    torch.save(payload, out_path)
    _logger.info("Saved checkpoint to %s", out_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights (and optionally optimizer state) into ``model`` in place.

    Returns the full checkpoint payload so callers can inspect ``epoch`` /
    ``extra``.
    """
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"No checkpoint at {src}")
    payload = torch.load(src, map_location=map_location)
    if "state_dict" not in payload:
        raise KeyError(f"Checkpoint at {src} has no 'state_dict' key")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=strict)
    if missing:
        _logger.warning("Missing keys when loading %s: %s", src, missing)
    if unexpected:
        _logger.warning("Unexpected keys when loading %s: %s", src, unexpected)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    _logger.info("Loaded checkpoint from %s", src)
    return payload


__all__ = ["save_checkpoint", "load_checkpoint"]
