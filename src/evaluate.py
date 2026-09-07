"""Re-evalúa un checkpoint YA entrenado (val, test, desglose por base de datos) sin reentrenar.

`cli.run()` no tiene -- a propósito, ver CLAUDE.md -- ningún modo "solo
evaluar": es `Trainer.fit()` el que produce el mejor checkpoint y lo evalúa
en el mismo proceso. Eso deja sin forma de reproducir `run_dir/test/` para
un checkpoint que ya se entrenó (ej. una corrida vieja, o una corrida cuyo
`config.data.by_database_manifests` no existía todavía cuando se entrenó --
exactamente el caso de exp05..exp16, entrenados antes de que ese campo
existiera). Este módulo es esa pieza que faltaba, escrita aparte de
`cli.py` -- que ningún otro módulo debe importar (ver su docstring) -- y
apoyada en `eval_pipeline.py`, que ya tenía toda la lógica de evaluación
factorizada en funciones puras reusables por cualquiera de los dos
entrypoints.

Reconstruye exactamente lo que `cli.run()` reconstruye antes de entrenar
(manifest, split, transforms, dataloaders, backbone+cabeza) a partir del
MISMO YAML que se usó para entrenar -- por eso pide `--config`, no una lista
suelta de hiperparámetros -- y evalúa el checkpoint indicado por `--checkpoint`
en vez de llamar a `Trainer.fit()`. Nunca reentrena, nunca sobreescribe
`config.yaml`/`metrics.csv`/`plots/` (esos son artefactos del entrenamiento
original) -- solo `run_dir/val/`, `run_dir/test/` y, si `config.data.
by_database_manifests` está fijado, el desglose por base de datos dentro de
`run_dir/test/`, exactamente como los deja `cli.run()`.

Uso:
    $ python -m src.evaluate --config configs/exp05_fedmammobench_full_weighted.yaml \\
        --checkpoint runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt

Ejemplo de uso programático en Python:
    >>> from src.config import load_config
    >>> from src.evaluate import run_evaluation
    >>> config = load_config("configs/exp05_fedmammobench_full_weighted.yaml")
    >>> run_evaluation(config, "runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt")
"""

import argparse
from pathlib import Path

import torch.nn as nn

from .config import ExperimentConfig, load_config
from .datasets.build import builder_dataloader
from .datasets.manifest import Manifest
from .datasets.split import Split
from .datasets.transform import TransformBuilder
from .eval_pipeline import evaluate_by_database, evaluate_split
from .models.build import build_model
from .models.heads import get_head_strategy
from .seed import set_global_seed
from .tracking import MetricsLogger
from .train.build import build_loss


def run_evaluation(config: ExperimentConfig, checkpoint_path: str | Path) -> None:
    """Evalúa `checkpoint_path` sobre val/test (+ desglose por base de datos) sin entrenar nada.

    Args:
        config: `ExperimentConfig` -- el MISMO YAML con el que se entrenó
            `checkpoint_path` (misma arquitectura, misma cabeza, mismo
            manifest/normalización). Nada acá lo valida -- pasar un config
            que no corresponda al checkpoint no rompe la carga (`strict=False`
            en `load_checkpoint()`), pero las métricas resultantes no
            significan nada.
        checkpoint_path: ruta a un `.pt` guardado por
            `checkpoint.save_checkpoint()` -- típicamente un
            `best_epoch<N>.pt` de `config.train.checkpoint_dir`.

    Example:
        >>> config = load_config("configs/exp05_fedmammobench_full_weighted.yaml")
        >>> run_evaluation(config, "runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt")
    """
    checkpoint_path = Path(checkpoint_path)

    # Mismo orden que cli.run(): semilla antes que cualquier dataset/modelo.
    set_global_seed(config.data.seed)

    manifest = Manifest(manifest_path=config.data.manifest_path, image_root=config.data.image_root)
    split = Split(manifest=manifest)

    # train_transform_builder solo existe para que builder_dataloader() (que
    # siempre arma los 3 loaders) tenga con qué construir "train" -- ese
    # loader nunca se itera acá (no hay Trainer.fit()), así que da igual que
    # incluya augmentación; construirlo cuesta nada (un DataLoader no carga
    # datos hasta que se itera).
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

    # Arquitectura sin pesos reales todavía -- evaluate_split() los carga
    # in-place desde checkpoint_path vía evaluate_checkpoint(). Pasarle
    # weights_path acá sería trabajo tirado (RadImageNet/ImageNet) que el
    # checkpoint completo pisa igual dos líneas después.
    backbone, _load_report = build_model(
        config.architecture.name,
        weights_path=(
            str(config.architecture.weights_path) if config.architecture.weights_path is not None else None
        ),
        unfreeze_from=config.architecture.unfreeze_from,
        device=config.train.device,
    )
    head_cls = get_head_strategy(config.head.name)
    head = head_cls(**config.head.hparams)
    model = nn.Sequential(backbone, head.build()).to(config.train.device)

    loss_spec = build_loss(config.loss.name, device=config.train.device, **config.loss.hparams)

    # Mismo run_dir que usó el entrenamiento original -- val/test.png y
    # metrics.json se sobreescriben con esta re-evaluación (por diseño: son
    # artefactos derivados del checkpoint, no del proceso de entrenamiento),
    # pero config.yaml/metrics.csv/plots/ (curvas de entrenamiento) quedan
    # intactos, porque nada acá los toca.
    with MetricsLogger(
        config.train.run_dir,
        wandb_project=config.train.wandb_project,
        wandb_run_name=f"{config.experiment_id}-eval",
        wandb_group=config.train.wandb_group,
        config=config.model_dump(mode="json"),
    ) as logger:
        for split_name, loader in (("val", loaders["val"]), ("test", loaders["test"])):
            split_metrics = evaluate_split(
                split_name, model, checkpoint_path, loader, loss_spec, config.train.device,
                config.train.run_dir, logger,
            )
            print(
                f"{split_name.capitalize()} "
                f"({config.train.metric_name}={split_metrics[config.train.metric_name]:.4f}): "
                f"{split_metrics}"
            )

        if config.data.by_database_manifests:
            evaluate_by_database(
                config.data.by_database_manifests,
                config.data.image_root,
                eval_transform_builder,
                model,
                checkpoint_path,
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
        argparse.Namespace: objeto con `--config` y `--checkpoint` parseados.
    """
    parser = argparse.ArgumentParser(
        description="Re-evalúa un checkpoint ya entrenado (val/test + desglose por base de datos), sin reentrenar."
    )
    parser.add_argument("--config", required=True, type=str, help="Ruta al YAML del experimento (el mismo con el que se entrenó).")
    parser.add_argument("--checkpoint", required=True, type=str, help="Ruta al checkpoint .pt a evaluar.")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada invocable al ejecutar el script directamente."""
    args = parse_args()
    config = load_config(args.config)
    run_evaluation(config, args.checkpoint)


if __name__ == "__main__":
    main()
