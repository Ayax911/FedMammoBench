# train/trainer.py
"""Orquestador del loop de entrenamiento. Es la única pieza de train/ con
memoria entre épocas — compara la métrica de validación contra la mejor
vista hasta ahora, y guarda el checkpoint solo cuando mejora."""
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint
from ..tracking import MetricsLogger
from .build import LossSpec
from .loop import evaluate, train_one_epoch


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_spec: LossSpec,
        checkpoint_dir: str | Path,
        run_dir: str | Path,
        device: str = "cpu",
        scheduler: LRScheduler | None = None,
        metric_name: str = "auc",
    ) -> None:
        """
        Args:
            model: modelo completo (backbone + cabeza ya unidos).
            optimizer: construido vía train/build.py (build_optimizer).
            loss_spec: construido vía train/build.py (build_loss). Encapsula
                tanto el cálculo de la pérdida como la conversión de logits a
                probabilidad de clase positiva, para que ni Trainer ni
                train/loop.py necesiten un `if` sobre si el esquema de salida
                es BCE (1 logit) o CrossEntropy (2 logits).
            checkpoint_dir: carpeta donde se guardan los checkpoints.
            run_dir: carpeta donde MetricsLogger escribe metrics.csv y los
                eventos de TensorBoard de esta corrida. Separado de
                checkpoint_dir a propósito — son dos destinos distintos
                aunque hoy coincidan en la práctica (runs/<RUN_NAME>/).
            device: dispositivo de entrenamiento ("cpu" o "cuda").
            scheduler: opcional — si se pasa, se llama scheduler.step() al
                final de cada época.
            metric_name: clave del dict que devuelve evaluate() a maximizar
                para decidir el mejor checkpoint. Default "auc".
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_spec = loss_spec
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_dir = Path(run_dir)
        self.device = device
        self.scheduler = scheduler
        self.metric_name = metric_name

        self.best_metric: float = float("-inf")
        self.best_checkpoint_path: Path | None = None

    def fit(self, train_loader: DataLoader[tuple[torch.Tensor, int]], 
            val_loader: DataLoader[tuple[torch.Tensor, int]], epochs: int) -> Path:
        """Corre el loop completo de épocas: entrena, valida, y guarda el
        checkpoint solo cuando la métrica de validación mejora.

        Args:
            train_loader: DataLoader de entrenamiento (shuffle=True).
            val_loader: DataLoader de validación (shuffle=False).
            epochs: cantidad de épocas a correr.

        Returns:
            Ruta al mejor checkpoint según self.metric_name — NUNCA el de
            la última época. Este es el único valor que debe usarse para
            la evaluación final en test.

        Raises:
            RuntimeError: ninguna época produjo un checkpoint válido (por
                ejemplo, si epochs == 0).
        """
        with MetricsLogger(self.run_dir) as logger:
            for epoch in range(epochs):
                train_metrics = train_one_epoch(
                    self.model, train_loader, self.optimizer, self.loss_spec, self.device
                )
                val_metrics = evaluate(self.model, val_loader, self.loss_spec, self.device)

                if self.scheduler is not None:
                    self.scheduler.step()

                current_metric = val_metrics[self.metric_name]
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_checkpoint_path = self.checkpoint_dir / f"best_epoch{epoch}.pt"
                    save_checkpoint(
                        self.model,
                        self.best_checkpoint_path,
                        epoch=epoch,
                        metric_value=current_metric,
                    )

                # MetricsLogger no distingue splits — combinar acá, con
                # prefijo, en un solo dict por época (ver tracking.py).
                epoch_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                logger.log(epoch, epoch_metrics)

                print(
                    f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} "
                    f"val_{self.metric_name}={current_metric:.4f} (best={self.best_metric:.4f})"
                )

        if self.best_checkpoint_path is None:
            raise RuntimeError(
                "Ninguna época produjo un checkpoint válido — revisar "
                "epochs > 0 y que val_loader no esté vacío."
            )

        return self.best_checkpoint_path