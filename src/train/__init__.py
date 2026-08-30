"""Módulo de entrenamiento, optimización y evaluación para Federal-Learning.

Reexporta las clases y funciones principales para entrenamiento centralizado y local:
    - Trainer: Orquestador multi-época con guardado del mejor checkpoint y tracking.
    - LossSpec, build_loss: Especificación y fábrica de funciones de pérdida (BCE / CrossEntropy).
    - build_optimizer, build_scheduler: Fábricas de optimizadores y schedulers de PyTorch.
    - train_one_epoch, evaluate: Funciones puras de entrenamiento y evaluación por época.
"""

from .build import LossSpec, build_loss, build_optimizer, build_scheduler
from .loop import evaluate, train_one_epoch
from .trainer import Trainer

__all__ = [
    "Trainer",
    "LossSpec",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
]
