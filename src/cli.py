"""Punto de entrada principal CLI: integra config -> seed -> datasets -> models -> train.

Es el único módulo que conoce todos los demás — nada depende de él.

Ejemplo de ejecución desde CLI:
    $ python -m src.cli --config configs/exp01.yaml

Ejemplo de uso programático en Python:
    >>> from src.config import load_config
    >>> from src.cli import run
    >>> cfg = load_config("configs/exp01.yaml")
    >>> run(cfg)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config, save_config
from .datasets.build import builder_dataloader
from .datasets.manifest import Manifest
from .datasets.split import Split
from .datasets.transform import TransformBuilder
from .models.build import build_model
from .models.heads import get_head_strategy
from .reporting import (
    compute_confusion_matrix_metrics,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_metric_curve,
    plot_roc_curve,
    save_metrics_json,
    save_predictions_csv,
)
from .seed import set_global_seed
from .tracking import MetricsLogger
from .train.build import LossSpec, build_loss, build_optimizer, build_scheduler
from .train.evaluation import evaluate_checkpoint, predict_on_loader
from .train.trainer import Trainer


def _evaluate_split(
    split_name: str,
    model: nn.Module,
    best_checkpoint: Path,
    loader: DataLoader[tuple[torch.Tensor, int]],
    loss_spec: LossSpec,
    device: str,
    run_dir: Path,
    logger: MetricsLogger,
) -> dict[str, float]:
    """Evalúa `best_checkpoint` sobre un split (val o test) y persiste/loguea todo, idéntico para ambos.

    Antes de que existiera val/, este bloque estaba escrito a mano una sola
    vez para test dentro de `run()`. Reusarlo para val en vez de duplicarlo
    evita repetir a mano un bug real que agregar val hizo evidente:
    `compute_confusion_matrix_metrics()` devuelve las mismas claves
    (`cm_mcc`, `cm_npv`, ...) sin importar el split, así que loguearlas sin
    prefijo al summary PLANO de W&B (como hacía la primera versión de este
    bloque, solo para test) haría que val sobreescriba a test en cuanto se
    agregara. El prefijo `{split_name}_` evita la colisión.

    `evaluate_checkpoint()` recarga `best_checkpoint` en `model` -- SIEMPRE
    el mejor, nunca el estado en el que haya quedado `model` al final del
    entrenamiento (ver docstring de `Trainer.fit()`). Esto importa tanto
    para val como para test: el `evaluate()` que corre dentro del loop de
    entrenamiento evalúa el modelo tal como está ESA época, no releído desde
    el mejor checkpoint -- por eso este bloque vuelve a evaluar val aunque
    ya se haya evaluado, por época, durante `fit()`.

    Args:
        split_name: `"val"` o `"test"` -- prefijo de las claves logueadas a
            W&B y nombre de la subcarpeta bajo `run_dir`.
        model: modelo completo; sus pesos se sobreescriben in-place con los
            de `best_checkpoint`.
        best_checkpoint: ruta devuelta por `Trainer.fit()`.
        loader: `loaders["val"]` o `loaders["test"]`.
        loss_spec: mismo `LossSpec` usado para entrenar ese checkpoint.
        device: dispositivo de cómputo.
        run_dir: `config.train.run_dir` -- la subcarpeta `run_dir/{split_name}/`
            se crea si no existe.
        logger: `MetricsLogger` ya abierto por `run()`.

    Returns:
        dict[str, float]: el mismo dict que devuelve `evaluate_checkpoint()`
            (`"loss"` + métricas clínicas), sin prefijo -- para imprimir o
            inspeccionar en `run()`.
    """
    split_metrics = evaluate_checkpoint(model, best_checkpoint, loader, loss_spec, device)
    logger.log_summary({f"{split_name}_{k}": v for k, v in split_metrics.items()})

    y_true, y_pred, y_prob = predict_on_loader(model, loader, loss_spec, device)

    # Todas las métricas calculables desde la matriz de confusión -- no solo
    # las siete que evaluate_checkpoint() ya trackea -- ver docstring de
    # compute_confusion_matrix_metrics() para la lista completa (NPV, MCC,
    # kappa, likelihood ratios, etc.).
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    logger.log_summary({f"{split_name}_{k}": v for k, v in cm_metrics.items()})

    split_dir = run_dir / split_name
    save_metrics_json(split_metrics, split_dir / "metrics.json")
    save_metrics_json(cm_metrics, split_dir / "confusion_matrix_metrics.json")

    predictions_path = split_dir / "predictions.csv"
    save_predictions_csv(y_true, y_pred, y_prob, predictions_path)
    logger.log_table(f"{split_name}/predictions", predictions_path)

    confusion_matrix_path = split_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, confusion_matrix_path)
    logger.log_image(f"{split_name}/confusion_matrix", confusion_matrix_path)

    roc_curve_path = split_dir / "roc_curve.png"
    plot_roc_curve(y_true, y_prob, roc_curve_path)
    logger.log_image(f"{split_name}/roc_curve", roc_curve_path)

    return split_metrics


def run(config: ExperimentConfig) -> None:
    """Corre un experimento completo de punta a punta a partir de un ExperimentConfig.

    Args:
        config: Objeto `ExperimentConfig` ya validado desde YAML.

    Example:
        >>> config = load_config("configs/exp01.yaml")
        >>> run(config)
    """
    # Primero que nada, antes de construir CUALQUIER otra cosa (datasets,
    # modelo, dataloaders) — ver seed.py.
    set_global_seed(config.data.seed)

    manifest = Manifest(manifest_path=config.data.manifest_path, image_root=config.data.image_root)
    split = Split(manifest=manifest)

    # use_rotation se deriva de rotation_degrees > 0 en vez de ser un flag
    # aparte -- un solo lugar decide si rota, no dos que puedan divergir.
    aug = config.data.augmentation
    train_transform_builder = TransformBuilder(
        image_size=config.data.image_size,
        use_horizontal_flip=aug.horizontal_flip,
        horizontal_flip_p=aug.horizontal_flip_p,
        use_rotation=aug.rotation_degrees > 0,
        rotation_degrees=aug.rotation_degrees,
        use_vertical_flip=aug.vertical_flip,
        vertical_flip_p=aug.vertical_flip_p,
        use_blur=aug.blur,
        blur_p=aug.blur_p,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    # La normalización va también acá: es preprocesamiento, no augmentación —
    # si train y eval normalizaran distinto, el modelo vería dos
    # distribuciones de entrada diferentes.
    eval_transform_builder = TransformBuilder(
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    loaders = builder_dataloader(
        split,
        train_transform_builder,
        eval_transform_builder,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        seed=config.data.seed,
    )

    backbone, load_report = build_model(
        config.architecture.name,
        weights_path=str(config.architecture.weights_path),
        unfreeze_from=config.architecture.unfreeze_from,
        device=config.train.device,
    )
    print(
        f"Pesos cargados: {load_report.matched} tensores "
        f"(missing={len(load_report.missing)}, unexpected={len(load_report.unexpected)})"
    )

    # head_cls es Type[HeadBuilder] — la ABC base, no la subclase concreta
    # que resulte en runtime. Pyright solo puede tipar el constructor de
    # HeadBuilder (no tiene uno propio), así que kwargs fijos nunca calzan
    # acá para ninguna subclase; **hparams (dict[str, Any]) es lo correcto,
    # no un workaround — mismo patrón que optimizer/scheduler/loss abajo.
    head_cls = get_head_strategy(config.head.name)
    head = head_cls(**config.head.hparams)
    # Ensamblado backbone+cabeza: a propósito acá, no en models/ — ver decisión
    # de diseño en CLAUDE.md (freeze y cabeza son ejes de experimentación
    # independientes).
    model = nn.Sequential(backbone, head.build())
    # build_model() ya cargó el backbone en config.train.device, pero la
    # cabeza se acaba de crear en CPU — moverlo ANTES de construir el
    # optimizer, para que apunte a los parámetros ya ubicados en destino.
    model = model.to(config.train.device)

    optimizer = build_optimizer(model.parameters(), config.optimizer.name, **config.optimizer.hparams)
    scheduler = (
        build_scheduler(optimizer, config.scheduler.name, **config.scheduler.hparams)
        if config.scheduler is not None
        else None
    )
    loss_spec = build_loss(config.loss.name, device=config.train.device, **config.loss.hparams)

    # La corrida de W&B (si config.train.wandb_project no es None) cubre
    # entrenamiento Y evaluación de test bajo un solo `with` -- antes,
    # MetricsLogger se abría y cerraba dentro de trainer.fit(), así que
    # wandb.finish() corría antes de que existiera un solo plot o métrica de
    # test que subir. `config=` deja el YAML del experimento adjunto a la
    # corrida en W&B, para poder filtrar/agrupar corridas por hiperparámetro
    # en vez de solo por nombre.
    with MetricsLogger(
        config.train.run_dir,
        wandb_project=config.train.wandb_project,
        wandb_run_name=config.experiment_id,
        config=config.model_dump(mode="json"),
    ) as logger:
        trainer = Trainer(
            model,
            optimizer,
            loss_spec,
            checkpoint_dir=config.train.checkpoint_dir,
            run_dir=config.train.run_dir,
            device=config.train.device,
            scheduler=scheduler,
            metric_name=config.train.metric_name,
            metric_mode=config.train.metric_mode,
            patience=config.train.patience,
            min_delta=config.train.min_delta,
            save_every=config.train.save_every,
            freeze_bn_stats=config.train.freeze_bn_stats,
            logger=logger,
        )

        # Dejar registrado, junto al resto de run_dir, exactamente qué config
        # produjo esta corrida.
        save_config(config, config.train.run_dir / "config.yaml")

        best_checkpoint = trainer.fit(loaders["train"], loaders["val"], epochs=config.train.epochs)
        print(f"Mejor checkpoint: {best_checkpoint}")

        # Curvas de entrenamiento: se grafican desde trainer.history (mismo
        # contenido que metrics.csv) antes de tocar val/test, porque solo
        # dependen del entrenamiento.
        loss_curve_path = config.train.run_dir / "plots" / "loss_curve.png"
        plot_loss_curve(
            [epoch_metrics["train_loss"] for epoch_metrics in trainer.history],
            [epoch_metrics["val_loss"] for epoch_metrics in trainer.history],
            loss_curve_path,
            best_epoch=trainer.best_epoch,
        )
        logger.log_image("plots/loss_curve", loss_curve_path)

        # Una curva train-vs-val por cada métrica clínica -- ahora que
        # train_one_epoch() (train/loop.py) las calcula igual que evaluate(),
        # no solo loss. Mismo nombre de display para el eje Y/leyenda que
        # las claves de trainer.history sin el prefijo train_/val_.
        metric_display_names = {
            "accuracy": "Accuracy",
            "auc": "AUC",
            "sensitivity": "Sensibilidad",
            "specificity": "Especificidad",
            "f1": "F1",
            "f1_macro": "F1-macro",
            "precision": "Precisión",
        }
        for metric_key, display_name in metric_display_names.items():
            metric_curve_path = config.train.run_dir / "plots" / f"{metric_key}_curve.png"
            plot_metric_curve(
                [epoch_metrics[f"train_{metric_key}"] for epoch_metrics in trainer.history],
                [epoch_metrics[f"val_{metric_key}"] for epoch_metrics in trainer.history],
                metric_curve_path,
                metric_name=display_name,
                best_epoch=trainer.best_epoch,
            )
            logger.log_image(f"plots/{metric_key}_curve", metric_curve_path)

        # Evaluación final en val y test -- SIEMPRE con el mejor checkpoint
        # que acaba de devolver fit(), nunca con el estado final del modelo
        # ni con una bandera de config aparte que pueda desincronizarse de
        # cuál fue realmente el mejor (ver docstring de Trainer.fit()). val
        # ya se evaluó una vez por época durante fit(), pero sobre el modelo
        # tal como estaba ESA época -- esto la vuelve a evaluar releyendo el
        # mejor checkpoint, para tener matriz de confusión/ROC/métricas
        # derivadas consistentes con lo que realmente se reporta como
        # resultado (ver docstring de _evaluate_split()).
        for split_name, loader in (("val", loaders["val"]), ("test", loaders["test"])):
            split_metrics = _evaluate_split(
                split_name,
                model,
                best_checkpoint,
                loader,
                loss_spec,
                config.train.device,
                config.train.run_dir,
                logger,
            )
            print(
                f"{split_name.capitalize()} "
                f"({config.train.metric_name}={split_metrics[config.train.metric_name]:.4f}): "
                f"{split_metrics}"
            )


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Objeto con el argumento `--config` parseado.
    """
    parser = argparse.ArgumentParser(description="Corre un experimento centralizado de punta a punta.")
    parser.add_argument("--config", required=True, type=str, help="Ruta al YAML del experimento.")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada invocable al ejecutar el script directamente."""
    args = parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
