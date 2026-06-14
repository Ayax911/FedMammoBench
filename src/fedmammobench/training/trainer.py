"""Reusable training loop.

The Trainer is intentionally minimal: one method to fit for N epochs, one
method to run a single epoch. The same class is used by both the centralized
script and the FL client (which calls ``train_one_epoch`` once per local
epoch per round).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from fedmammobench.evaluation.evaluator import Evaluator
from fedmammobench.utils.csv_logger import CSVLogger
from fedmammobench.utils.logging_utils import get_logger
from fedmammobench.utils.tensorboard_utils import TensorBoardWriter

_logger = get_logger(__name__)


class Trainer:
    """Reusable epoch-level trainer.

    Args:
        model: The network to train (kept on ``device``).
        optimizer: A torch Optimizer pre-bound to ``model``'s parameters.
        criterion: Loss module producing a scalar.
        device: torch.device for tensors.
        scheduler: Optional LR scheduler. Stepped once per epoch (or per call
            to :meth:`train_one_epoch`).
        grad_clip_norm: If > 0, applies ``clip_grad_norm_`` after backward.
        mixed_precision: If True and ``device.type == 'cuda'``, runs the
            forward in autocast and scales the loss with GradScaler.
        tb_writer: Optional TensorBoardWriter for scalar logging.
        csv_logger: Optional CSVLogger for per-epoch rows.
        log_tag: Prefix for logs (useful in FL, e.g. ``"client_2"``).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        *,
        scheduler: LRScheduler | None = None,
        grad_clip_norm: float = 0.0,
        mixed_precision: bool = False,
        tb_writer: TensorBoardWriter | None = None,
        csv_logger: CSVLogger | None = None,
        log_tag: str = "centralized",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip_norm = float(grad_clip_norm)
        self.mixed_precision = bool(mixed_precision) and device.type == "cuda"
        self.tb_writer = tb_writer
        self.csv_logger = csv_logger
        self.log_tag = log_tag
        self._scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda") if self.mixed_precision else None
        )
        self._global_step = 0

    # ------------------------------------------------------------------

    def train_one_epoch(
        self,
        loader: DataLoader,
        *,
        epoch: int,
        proximal_mu: float = 0.0,
        global_params: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run a single epoch and return mean loss + sample count.

        Args:
            loader: Training DataLoader.
            epoch: Current epoch index (used for logging).
            proximal_mu: FedProx regularization strength. When > 0 and
                ``global_params`` is provided, adds
                ``(mu/2) * ||w - w_global||²`` to the task loss each batch.
            global_params: Flat parameter vector (from
                ``torch.nn.utils.parameters_to_vector``) captured from the
                global model at the start of the round. Required when
                ``proximal_mu > 0``; ignored otherwise. Using a single flat
                tensor instead of a list of per-layer tensors reduces Python
                GC overhead and memory fragmentation.
        """
        self.model.train()
        total_task_loss = 0.0
        total_loss = 0.0
        n_samples = 0
        use_prox = proximal_mu > 0.0 and global_params is not None
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(set_to_none=True)
            if self._scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    task_loss = self.criterion(logits, targets)
                loss = task_loss
                if use_prox:
                    # Proximal term computed in FP32 (via flat vector) to prevent
                    # FP16 underflow when drift is small (early rounds, low mu).
                    with torch.amp.autocast("cuda", enabled=False):
                        current = torch.nn.utils.parameters_to_vector(
                            self.model.parameters()
                        ).float()
                        prox = ((current - global_params.float()) ** 2).sum()
                    loss = task_loss + (proximal_mu / 2.0) * prox
                self._scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                logits = self.model(images)
                task_loss = self.criterion(logits, targets)
                loss = task_loss
                if use_prox:
                    current = torch.nn.utils.parameters_to_vector(self.model.parameters())
                    prox = ((current - global_params) ** 2).sum()
                    loss = task_loss + (proximal_mu / 2.0) * prox
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            batch_n = targets.size(0)
            total_task_loss += float(task_loss.item()) * batch_n
            total_loss += float(loss.item()) * batch_n
            n_samples += int(batch_n)
            self._global_step += 1

        mean_task_loss = total_task_loss / max(n_samples, 1)
        mean_loss = total_loss / max(n_samples, 1)
        if self.scheduler is not None and not isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step()
        if self.tb_writer is not None:
            # task_loss: cross-entropy only (comparable across FedAvg and FedProx)
            # train_loss: task_loss + proximal penalty (equals task_loss for FedAvg)
            self.tb_writer.log_scalar(f"{self.log_tag}/task_loss", mean_task_loss, epoch)
            self.tb_writer.log_scalar(f"{self.log_tag}/train_loss", mean_loss, epoch)
            self.tb_writer.log_scalar(
                f"{self.log_tag}/lr",
                self.optimizer.param_groups[0]["lr"],
                epoch,
            )
        _logger.info(
            "[%s] epoch %d: task_loss=%.4f train_loss=%.4f (samples=%d)",
            self.log_tag,
            epoch,
            mean_task_loss,
            mean_loss,
            n_samples,
        )
        return {"loss": mean_loss, "task_loss": mean_task_loss, "samples": n_samples}

    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        *,
        epochs: int,
        val_loader: DataLoader | None = None,
        evaluator: Evaluator | None = None,
        start_epoch: int = 0,
    ) -> dict[str, Any]:
        """Train for ``epochs`` epochs. Returns the last epoch's metrics."""
        last_metrics: dict[str, Any] = {}
        for epoch in range(start_epoch, start_epoch + epochs):
            train_stats = self.train_one_epoch(train_loader, epoch=epoch)
            last_metrics = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}

            if val_loader is not None and evaluator is not None:
                val_metrics = evaluator.evaluate(val_loader, criterion=self.criterion)
                last_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                if self.tb_writer is not None:
                    self.tb_writer.log_scalars(
                        f"{self.log_tag}/val",
                        {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
                        epoch,
                    )
                _logger.info(
                    "[%s] epoch %d: val_loss=%.4f val_auc=%.4f val_f1=%.4f",
                    self.log_tag,
                    epoch,
                    val_metrics.get("loss", float("nan")),
                    val_metrics.get("roc_auc", float("nan")),
                    val_metrics.get("f1", float("nan")),
                )
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("roc_auc", val_metrics.get("f1", 0.0)))

            if self.csv_logger is not None:
                row = {k: v for k, v in last_metrics.items() if isinstance(v, (int, float, str))}
                self.csv_logger.append(row)

        return last_metrics


__all__ = ["Trainer"]
