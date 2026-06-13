"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fedmammobench.configs.schema import OptimizerConfig, SchedulerConfig


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> Optimizer:
    if cfg.lr_head is not None or cfg.lr_backbone is not None:
        params: Iterable[torch.nn.Parameter] | list[dict] = _discriminative_param_groups(model, cfg)
    else:
        params = (p for p in model.parameters() if p.requires_grad)
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


def _discriminative_param_groups(model: nn.Module, cfg: OptimizerConfig) -> list[dict]:
    """Return AdamW-style param groups with separate LRs for backbone and head.

    Mirrors the centralizada approach: head gets lr_head, backbone gets
    lr_backbone. Falls back to cfg.lr for any group whose specific rate is None.
    """
    backbone = getattr(model, "backbone", model)
    head = None
    for attr in ("fc", "classifier"):
        head = getattr(backbone, attr, None)
        if head is not None:
            break

    head_ids = {id(p) for p in head.parameters()} if head is not None else set()
    backbone_params = [p for p in backbone.parameters() if p.requires_grad and id(p) not in head_ids]
    head_params = [p for p in head.parameters() if p.requires_grad] if head is not None else []

    lr_bb = cfg.lr_backbone if cfg.lr_backbone is not None else cfg.lr
    lr_hd = cfg.lr_head if cfg.lr_head is not None else cfg.lr

    groups: list[dict] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_bb})
    if head_params:
        groups.append({"params": head_params, "lr": lr_hd})
    if not groups:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr}]
    return groups


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
