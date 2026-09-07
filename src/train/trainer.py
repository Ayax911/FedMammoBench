"""Orquestador del loop de entrenamiento.

Es la única pieza de `train/` con memoria entre épocas — delega en
`EarlyStopping` (`train/early_stopping.py`) la comparación de la métrica de
validación contra la mejor vista hasta ahora, y guarda el checkpoint solo
cuando esa comparación dice que mejoró.

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

import contextlib
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint
from ..tracking import MetricsLogger
from .build import LossSpec
from .early_stopping import EarlyStopping
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
        metric_name: Nombre de la métrica a trackear para guardar checkpoints (default `"auc"`).
        tracker: `EarlyStopping` que decide, cada época, si `metric_name`
            mejoró (según `metric_mode`) y si hay que detener el entrenamiento.
        save_every: si no es `None`, cada cuántas épocas guardar un
            checkpoint periódico además del mejor.
        best_epoch: índice (base 0) de la época que produjo el mejor
            checkpoint, o `None` si `fit()` todavía no corrió.
        history: una entrada por época con las mismas claves que se
            escriben en `metrics.csv` (`train_*`, `val_*`,
            `duration_seconds`). Poblado por `fit()`.

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
        metric_mode: str = "max",
        patience: int | None = None,
        min_delta: float = 0.0,
        save_every: int | None = None,
        freeze_bn_stats: bool = True,
        logger: MetricsLogger | None = None,
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
            metric_name: clave del dict que devuelve evaluate() a trackear
                para decidir el mejor checkpoint. Default "auc".
            metric_mode: "max" si más `metric_name` es mejor (auc, f1,
                accuracy, sensitivity, specificity, precision — el caso
                común), "min" si menos es mejor (loss). Antes de este
                parámetro, `Trainer` siempre maximizaba sin importar la
                métrica — ver PHASES.md fase 3.
            patience: épocas sin mejora antes de detener el entrenamiento
                temprano. `None` (default) desactiva la parada temprana y
                corre las `epochs` completas, igual que antes de esta
                opción existir.
            min_delta: mejora mínima para que una época cuente como mejora
                real, tanto para guardar checkpoint como para resetear el
                contador de `patience`. Default `0.0`. El proyecto INC usa
                `0.005` sobre F1 de validación.
            save_every: si no es `None`, guarda un checkpoint periódico
                (`epoch{N}.pt`, independiente del mejor) cada `save_every`
                épocas. `None` (default) no guarda ninguno.
            freeze_bn_stats: se pasa tal cual a `train_one_epoch()` en cada
                época. `True` (default) mantiene las BN congeladas en
                `eval()`; `False` deja que sus estadísticas deriven, como
                hace el proyecto INC. Ver `train/loop.py`.
            logger: `MetricsLogger` ya abierto por quien llama (típicamente
                `cli.run()`, para que una sola corrida de W&B cubra
                entrenamiento + evaluación de test). `Trainer` NO la cierra
                -- quien la inyectó es quien la abrió y debe cerrarla. `None`
                (default) hace que `fit()` cree y cierre su propia
                `MetricsLogger(run_dir)` sin W&B, para uso standalone (ver
                ejemplo de módulo).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_spec = loss_spec
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_dir = Path(run_dir)
        self.device = device
        self.scheduler = scheduler
        self.metric_name = metric_name
        self.save_every = save_every
        self.freeze_bn_stats = freeze_bn_stats
        self._injected_logger = logger

        self.tracker = EarlyStopping(patience=patience, min_delta=min_delta, mode=metric_mode)
        self.best_checkpoint_path: Path | None = None
        # Historial en memoria además de metrics.csv: `src/cli.py` lo necesita
        # para graficar la curva de pérdida sin volver a parsear el CSV que
        # MetricsLogger acaba de escribir.
        self.best_epoch: int | None = None
        self.history: list[dict[str, float]] = []

    @property
    def best_metric(self) -> float:
        """Mejor valor de `metric_name` visto hasta ahora (delegado a `self.tracker`)."""
        return self.tracker.best_value

    def _split_state_dicts(self) -> dict[str, nn.Module] | None:
        """Backbone/cabeza por separado si `self.model` es `nn.Sequential(backbone, head)`.

        Devuelve `None` para cualquier otra forma de modelo — el desglose es
        una conveniencia (por ejemplo, para lo federado, donde solo el
        backbone se agrega entre nodos), no un requisito. Ver
        `checkpoint.save_checkpoint`'s `extra_state_dicts`.
        """
        if isinstance(self.model, nn.Sequential) and len(self.model) == 2:
            return {"backbone": self.model[0], "head": self.model[1]}
        return None

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
        val_loader: DataLoader[tuple[torch.Tensor, int]],
        epochs: int,
    ) -> Path:
        """Corre el loop completo de épocas: entrena, valida, y guarda el
        checkpoint solo cuando la métrica de validación mejora. Si se
        configuró `patience`, puede terminar antes de `epochs` por early
        stopping (ver `self.tracker`, `train/early_stopping.py`).

        Args:
            train_loader: DataLoader de entrenamiento (shuffle=True).
            val_loader: DataLoader de validación (shuffle=False).
            epochs: cantidad máxima de épocas a correr.

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
        # Si cli.run() ya abrió una MetricsLogger (para que cubra también la
        # evaluación de test que viene después de fit()), usarla sin
        # cerrarla acá -- nullcontext no llama __exit__. Sin una inyectada
        # (uso standalone, ver docstring del módulo), Trainer crea y cierra
        # la suya propia, sin W&B.
        logger_ctx = (
            contextlib.nullcontext(self._injected_logger)
            if self._injected_logger is not None
            else MetricsLogger(self.run_dir)
        )
        with logger_ctx as logger:
            for epoch in range(epochs):
                epoch_start = time.time()

                train_metrics = train_one_epoch(
                    self.model,
                    train_loader,
                    self.optimizer,
                    self.loss_spec,
                    self.device,
                    freeze_bn_stats=self.freeze_bn_stats,
                )
                val_metrics = evaluate(self.model, val_loader, self.loss_spec, self.device)

                current_metric = val_metrics[self.metric_name]

                # ReduceLROnPlateau necesita la métrica para decidir si bajar el
                # LR; el resto de schedulers avanza solo por número de época y
                # rechaza el argumento. Sin este reparto, `scheduler.step()` a
                # secas rompía con TypeError toda config que eligiera
                # "reduceonplateu" (ver train/build.py:_SCHEDULERS).
                # Ojo: el `mode` de ReduceLROnPlateau se fija en el YAML y debe
                # coincidir con `metric_mode` — nada los sincroniza.
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                elif self.scheduler is not None:
                    self.scheduler.step()

                improved = self.tracker.step(current_metric)
                if improved:
                    self.best_epoch = epoch
                    self.best_checkpoint_path = self.checkpoint_dir / f"best_epoch{epoch}.pt"
                    save_checkpoint(
                        self.model,
                        self.best_checkpoint_path,
                        epoch=epoch,
                        metric_value=current_metric,
                        extra_state_dicts=self._split_state_dicts(),
                    )

                if self.save_every is not None and epoch % self.save_every == 0:
                    save_checkpoint(
                        self.model,
                        self.checkpoint_dir / f"epoch{epoch}.pt",
                        epoch=epoch,
                        metric_value=current_metric,
                    )

                # MetricsLogger no distingue splits — combinar acá, con
                # prefijo, en un solo dict por época (ver tracking.py).
                epoch_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                epoch_metrics["duration_seconds"] = round(time.time() - epoch_start, 2)
                self.history.append(epoch_metrics)
                logger.log(epoch, epoch_metrics)

                # f1_macro va siempre en el log (aunque metric_name sea otra) porque
                # es la métrica de referencia del proyecto en manifests desbalanceados
                # (ver CLAUDE.md, "f1 vs f1_macro") -- verla por época, incluso cuando
                # se hace early stopping sobre otra métrica, ayuda a detectar el
                # estancamiento en 0.0 típico de las primeras épocas.
                print(
                    f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"f1_macro={val_metrics['f1_macro']:.4f} "
                    f"val_{self.metric_name}={current_metric:.4f} (best={self.tracker.best_value:.4f}) "
                    f"[{epoch_metrics['duration_seconds']:.1f}s]"
                )

                if self.tracker.should_stop:
                    print(
                        f"Early stopping activado en época {epoch} "
                        f"(sin mejora en {self.tracker.patience} épocas)"
                    )
                    break

        if self.best_checkpoint_path is None:
            raise RuntimeError(
                "Ninguna época produjo un checkpoint válido — revisar "
                "epochs > 0 y que val_loader no esté vacío."
            )

        return self.best_checkpoint_path
