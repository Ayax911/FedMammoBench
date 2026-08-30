"""Orquestador del loop de entrenamiento.

Es la única pieza de `train/` con memoria entre épocas — compara la métrica
de validación contra la mejor vista hasta ahora, y guarda el checkpoint solo
cuando mejora.

Ejemplo de uso:
    >>> from src.train.trainer import Trainer
    >>> trainer = Trainer(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     loss_spec=loss_spec,
    ...     checkpoint_dir="runs/exp01/weights",
    ...     run_dir="runs/exp01",
    ...     device="cuda"
    ... )
    >>> best_ckpt = trainer.fit(train_loader, val_loader, epochs=10)
"""

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
    """Orquestador multi-época para entrenamiento centralizado y local.

    Guarda automáticamente el mejor checkpoint de pesos (`.pt`) según la métrica
    de validación indicada (`metric_name`), e integra `MetricsLogger` para persisitir
    métricas batch por batch en CSV, TensorBoard y opcionalmente W&B.

    Attributes:
        model: Modelo PyTorch completo (backbone + cabeza).
        optimizer: Optimizador PyTorch.
        loss_spec: Especificador de función de pérdida (`LossSpec`).
        checkpoint_dir: Directorio para guardar checkpoints `.pt`.
        run_dir: Directorio para guardar logs (`metrics.csv`, TensorBoard).
        device: Dispositivo de cómputo (`"cpu"`, `"cuda"`).
        scheduler: Scheduler opcional de learning rate.
        metric_name: Nombre de la métrica a maximizar para guardar checkpoints (default `"auc"`).
        wandb_project: Nombre opcional de proyecto en W&B.
        wandb_run_name: Nombre opcional de corrida en W&B.

    Example:
        >>> trainer = Trainer(model, optimizer, loss_spec, "weights/", "runs/", "cuda")
        >>> best_path = trainer.fit(train_loader, val_loader, epochs=5)
    """

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
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        """Inicializa los componentes de entrenamiento y estado de mejor checkpoint.

        Args:
            model: modelo completo (backbone + cabeza ya unidos).
            optimizer: construido vía train/build.py (build_optimizer).
            loss_spec: construido vía train/build.py (build_loss). Encapsula
                tanto el cálculo de la pérdida como la conversión de logits a
                probabilidad de clase positiva.
            checkpoint_dir: carpeta donde se guardan los checkpoints.
            run_dir: carpeta donde MetricsLogger escribe metrics.csv y los
                eventos de TensorBoard de esta corrida.
            device: dispositivo de entrenamiento ("cpu" o "cuda").
            scheduler: opcional — si se pasa, se llama scheduler.step() al
                final de cada época.
            metric_name: clave del dict que devuelve evaluate() a maximizar
                para decidir el mejor checkpoint. Default "auc".
            wandb_project: opcional — pasado directo a MetricsLogger. None
                (default) desactiva W&B por completo.
            wandb_run_name: opcional — nombre de esta corrida en W&B.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_spec = loss_spec
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_dir = Path(run_dir)
        self.device = device
        self.scheduler = scheduler
        self.metric_name = metric_name
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.best_metric: float = float("-inf")
        self.best_checkpoint_path: Path | None = None

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
        val_loader: DataLoader[tuple[torch.Tensor, int]],
        epochs: int,
    ) -> Path:
        """Corre el loop completo de épocas: entrena, valida, y guarda el
        checkpoint solo cuando la métrica de validación mejora.

        Args:
            train_loader: DataLoader de entrenamiento (shuffle=True).
            val_loader: DataLoader de validación (shuffle=False).
            epochs: cantidad de épocas a correr.

        Returns:
            Path: Ruta al mejor checkpoint según self.metric_name — NUNCA el de
            la última época. Este es el único valor que debe usarse para
            la evaluación final en test.

        Raises:
            RuntimeError: ninguna época produjo un checkpoint válido (por
                ejemplo, si epochs == 0).

        Example:
            >>> best_path = trainer.fit(train_loader, val_loader, epochs=10)
            >>> print(best_path)
        """
        with MetricsLogger(
            self.run_dir, wandb_project=self.wandb_project, wandb_run_name=self.wandb_run_name
        ) as logger:
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