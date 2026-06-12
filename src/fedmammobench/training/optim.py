"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fedmammo.configs.schema import OptimizerConfig, SchedulerConfig


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> Optimizer:
    params: Iterable[torch.nn.Parameter] = (p for p in model.parameters() if p.requires_grad)
    name = cfg.name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.name!r}")


def build_scheduler(
    optimizer: Optimizer, cfg: SchedulerConfig
) -> LRScheduler | None:
    name = cfg.name.lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.t_max)
    if name == "plateau":
        # ReduceLROnPlateau is technically not an LRScheduler subclass in older
        # torch versions; callers must invoke .step(metric).
        return torch.optim.lr_scheduler.ReduceLROnPlateau(  # type: ignore[return-value]
            optimizer, mode="max", factor=cfg.gamma, patience=cfg.step_size
        )
    raise ValueError(f"Unknown scheduler: {cfg.name!r}")


__all__ = ["build_optimizer", "build_scheduler"]
