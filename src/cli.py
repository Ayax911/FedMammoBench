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

import torch.nn as nn

from .config import ExperimentConfig, load_config, save_config
from .datasets.build import builder_dataloader
from .datasets.manifest import Manifest
from .datasets.split import Split
from .datasets.transform import TransformBuilder
from .models.build import build_model
from .models.heads import get_head_strategy
from .reporting import (
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics_json,
    save_predictions_csv,
)
from .seed import set_global_seed
from .train.build import build_loss, build_optimizer, build_scheduler
from .train.evaluation import evaluate_checkpoint, predict_on_loader
from .train.trainer import Trainer


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
    )
    eval_transform_builder = TransformBuilder(image_size=config.data.image_size)
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
        wandb_project=config.train.wandb_project,
        wandb_run_name=config.experiment_id,
    )

    # Dejar registrado, junto al resto de run_dir, exactamente qué config
    # produjo esta corrida.
    save_config(config, config.train.run_dir / "config.yaml")

    best_checkpoint = trainer.fit(loaders["train"], loaders["val"], epochs=config.train.epochs)
    print(f"Mejor checkpoint: {best_checkpoint}")

    # Evaluación final en test -- SIEMPRE con el mejor checkpoint que acaba
    # de devolver fit(), nunca con el estado final del modelo ni con una
    # bandera de config aparte que pueda desincronizarse de cuál fue
    # realmente el mejor (ver docstring de Trainer.fit()).
    test_metrics = evaluate_checkpoint(
        model, best_checkpoint, loaders["test"], loss_spec, config.train.device
    )
    print(f"Test ({config.train.metric_name}={test_metrics[config.train.metric_name]:.4f}): {test_metrics}")

    y_true, y_pred, y_prob = predict_on_loader(model, loaders["test"], loss_spec, config.train.device)

    save_metrics_json(test_metrics, config.train.run_dir / "metrics.json")
    save_predictions_csv(y_true, y_pred, y_prob, config.train.run_dir / "predictions.csv")
    plot_confusion_matrix(y_true, y_pred, config.train.run_dir / "plots" / "confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, config.train.run_dir / "plots" / "roc_curve.png")


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
