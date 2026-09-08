"""Punto de entrada principal CLI: integra config -> seed -> datasets -> models -> train.

Es el único módulo que conoce todos los demás — nada depende de él.
`evaluate.py` (re-evaluación de un checkpoint ya entrenado, sin reentrenar)
NO importa de acá -- ambos importan de `eval_pipeline.py`, que es donde
viven `evaluate_split()`/`evaluate_by_database()` (ver su docstring de
módulo para el porqué de la extracción).

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
from .eval_pipeline import METRIC_DISPLAY_NAMES, evaluate_by_database, evaluate_split
from .models.build import build_model
from .models.heads import get_head_strategy
from .reporting import plot_loss_curve, plot_metric_curve
from .seed import set_global_seed
from .tracking import MetricsLogger
from .train.build import build_loss, build_optimizer, build_scheduler
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
        # None cuando la arquitectura ya trae sus pesos desde el
        # model_factory (ej. resnet50_imagenet_v1/_v2, ver
        # ArchitectureSpec.weights_from_factory en models/build.py) -- str()
        # sobre None daría la ruta literal "None", que build_model()
        # confundiría con un path real en vez de "no hay checkpoint externo".
        weights_path=(
            str(config.architecture.weights_path) if config.architecture.weights_path is not None else None
        ),
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

    # LR discriminativo backbone/cabeza: `backbone_lr`, si viene en
    # optimizer.hparams, separa model[0] (backbone) en su propio param group
    # con ese LR, dejando model[1] (cabeza) en el LR general de `hparams`.
    # build_optimizer() ya acepta param groups tal cual (ver su docstring en
    # train/build.py) -- lo único que faltaba era este ensamblado. Filtra
    # por requires_grad para no meterle al estado de AdamW parámetros del
    # backbone que unfreeze_from dejó congelados (un freeze parcial, ej.
    # layer4, no debería aportarle momentum/weight_decay a conv1-layer3).
    # Sin `backbone_lr`, comportamiento idéntico a antes de que existiera:
    # un solo param group con model.parameters() completo.
    optimizer_hparams = dict(config.optimizer.hparams)
    backbone_lr = optimizer_hparams.pop("backbone_lr", None)
    if backbone_lr is not None:
        params = [
            {"params": [p for p in model[0].parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": model[1].parameters()},
        ]
    else:
        params = model.parameters()
    optimizer = build_optimizer(params, config.optimizer.name, **optimizer_hparams)
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
        wandb_group=config.train.wandb_group,
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
        # las claves de trainer.history sin el prefijo train_/val_. Nombres
        # centralizados en METRIC_DISPLAY_NAMES (eval_pipeline.py) para que
        # evaluate_by_database() use exactamente los mismos.
        for metric_key, display_name in METRIC_DISPLAY_NAMES.items():
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
        # resultado (ver docstring de evaluate_split() en eval_pipeline.py).
        for split_name, loader in (("val", loaders["val"]), ("test", loaders["test"])):
            split_metrics = evaluate_split(
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

        # Desglose de test por base de datos -- opt-in, ver
        # DataConfig.by_database_manifests (src/config.py). None/{} (default)
        # deja el comportamiento idéntico a antes de que este campo
        # existiera: ni un Manifest ni un plot de más.
        if config.data.by_database_manifests:
            evaluate_by_database(
                config.data.by_database_manifests,
                config.data.image_root,
                eval_transform_builder,
                model,
                best_checkpoint,
                loss_spec,
                config.data.batch_size,
                config.data.num_workers,
                config.train.device,
                config.train.run_dir,
                logger,
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
