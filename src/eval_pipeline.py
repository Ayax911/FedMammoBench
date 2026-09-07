"""Orquestación de evaluación post-entrenamiento (val, test, desglose por base de datos).

Extraído de `cli.py` (que originalmente definía `_evaluate_split()`/
`_evaluate_by_database()` como funciones privadas, usadas solo por `run()`)
para que `evaluate.py` -- el entrypoint que re-evalúa un checkpoint YA
entrenado, sin reentrenar -- pueda reusarlas sin importar `cli.py`, que
CLAUDE.md fija como el único módulo que no debe tener importadores ("Nada
importa a `cli.py`"). Este módulo vive en la cadena de dependencias justo
antes de `cli.py` (después de `train/`, del que depende vía
`train.evaluation.evaluate_checkpoint`/`predict_on_loader`): `cli.py` y
`evaluate.py` importan de acá, nunca al revés.

Ninguna función de acá sabe si la llamó un entrenamiento recién terminado
(`cli.run()`) o una re-evaluación aislada (`evaluate.run_evaluation()`) --
ambas le pasan exactamente lo mismo: un `model` con la arquitectura correcta
(sin pesos cargados todavía), una ruta a checkpoint, un `loss_spec` y un
`run_dir`. Esa simetría es intencional: los artefactos que terminan en
`run_dir/val/` y `run_dir/test/` deben ser idénticos sin importar cuál de
los dos entrypoints los produjo.

Ejemplo de uso:
    >>> from src.eval_pipeline import evaluate_split, evaluate_by_database, METRIC_DISPLAY_NAMES
    >>> for split_name, loader in (("val", loaders["val"]), ("test", loaders["test"])):
    ...     evaluate_split(split_name, model, best_checkpoint, loader, loss_spec, "cuda", run_dir, logger)
    >>> if config.data.by_database_manifests:
    ...     evaluate_by_database(
    ...         config.data.by_database_manifests, config.data.image_root, eval_transform_builder,
    ...         model, best_checkpoint, loss_spec, 64, 4, "cuda", run_dir, logger,
    ...     )
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets.dataset import MammoBenchDataset
from .datasets.manifest import Manifest
from .datasets.split import Split
from .datasets.transform import TransformBuilder
from .reporting import (
    compute_confusion_matrix_metrics,
    plot_confusion_matrix,
    plot_confusion_matrix_by_database,
    plot_metrics_by_database,
    plot_roc_curve,
    save_metrics_by_database_json,
    save_metrics_json,
    save_predictions_csv,
)
from .tracking import MetricsLogger
from .train.build import LossSpec
from .train.evaluation import evaluate_checkpoint, predict_on_loader

# Nombres a mostrar de las métricas clínicas -- compartido entre las curvas
# train-vs-val de cli.run() y el desglose por base de datos de
# evaluate_by_database(), para que ambos lugares le digan lo mismo a la
# misma métrica (ej. "F1-macro", nunca "f1_macro" crudo, en ningún eje ni
# leyenda de run_dir/).
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "sensitivity": "Sensibilidad",
    "specificity": "Especificidad",
    "f1": "F1",
    "f1_macro": "F1-macro",
    "precision": "Precisión",
}


def evaluate_split(
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
    vez para test dentro de `cli.run()`. Reusarlo para val en vez de
    duplicarlo evita repetir a mano un bug real que agregar val hizo
    evidente: `compute_confusion_matrix_metrics()` devuelve las mismas
    claves (`cm_mcc`, `cm_npv`, ...) sin importar el split, así que
    loguearlas sin prefijo al summary PLANO de W&B (como hacía la primera
    versión de este bloque, solo para test) haría que val sobreescriba a
    test en cuanto se agregara. El prefijo `{split_name}_` evita la
    colisión.

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
        best_checkpoint: ruta a un checkpoint guardado por
            `checkpoint.save_checkpoint()` -- típicamente el retorno de
            `Trainer.fit()` (`cli.run()`) o un checkpoint viejo pasado a
            mano (`evaluate.run_evaluation()`).
        loader: `loaders["val"]` o `loaders["test"]`.
        loss_spec: mismo `LossSpec` usado para entrenar ese checkpoint.
        device: dispositivo de cómputo.
        run_dir: `config.train.run_dir` -- la subcarpeta `run_dir/{split_name}/`
            se crea si no existe.
        logger: `MetricsLogger` ya abierto por el caller.

    Returns:
        dict[str, float]: el mismo dict que devuelve `evaluate_checkpoint()`
            (`"loss"` + métricas clínicas), sin prefijo -- para imprimir o
            inspeccionar en el caller.
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


def evaluate_by_database(
    by_database_manifests: dict[str, Path],
    image_root: Path,
    eval_transform_builder: TransformBuilder,
    model: nn.Module,
    best_checkpoint: Path,
    loss_spec: LossSpec,
    batch_size: int,
    num_workers: int,
    device: str,
    run_dir: Path,
    logger: MetricsLogger,
) -> None:
    """Evalúa `best_checkpoint` sobre el test split de cada base de datos y grafica el desglose.

    Opt-in vía `DataConfig.by_database_manifests` (`None` por defecto ->
    el caller ni siquiera llama a esta función). Reconstruye, para cada
    entrada del dict, el mismo camino `Manifest -> Split -> MammoBenchDataset
    -> DataLoader` que usa `builder_dataloader()` para el split de test
    principal -- no filtra en memoria las predicciones de test ya calculadas
    por una columna, porque construir un `Manifest`/`Split` real por base de
    datos reutiliza tal cual la validación anti-leakage de `Split` (ver
    `datasets/split.py`) y dado que cada manifest es un CSV independiente en
    disco (`manifests/by_database/`, generados por
    `scripts/split_manifest_by_database.py`), queda como artefacto auditable
    por separado -- ver `DataConfig.by_database_manifests` en `src/config.py`
    para el razonamiento completo.

    Reutiliza `evaluate_checkpoint()`/`predict_on_loader()`
    (`train/evaluation.py`) igual que `evaluate_split()`, así que el modelo
    siempre corre con los pesos del mejor checkpoint -- nunca con los del
    entrenamiento en curso -- y las métricas de cada base de datos son
    directamente comparables entre sí y con el `test/metrics.json` global.

    Escribe en `run_dir/test/` (la misma carpeta que `evaluate_split()` usa
    para el split de test combinado, no una carpeta aparte):
        - `metrics_by_database.json`: métricas de `evaluate_checkpoint()` +
          `compute_confusion_matrix_metrics()`, por base de datos.
        - `confusion_matrix_by_database.png`: panel `1xN` de matrices de
          confusión, una por base de datos.
        - `metrics_by_database.png`: barras agrupadas por base de datos y
          por métrica, eje Y fijo `[0, 1]`.

    Args:
        by_database_manifests: `config.data.by_database_manifests` -- dict
            `nombre_base_de_datos -> manifest_path`.
        image_root: `config.data.image_root` -- mismo root para las tres,
            los manifests por base de datos son solo un subconjunto de filas
            del manifest combinado, con las mismas rutas relativas.
        eval_transform_builder: el mismo `TransformBuilder` de evaluación
            (sin augmentación) que usa el split de test principal -- misma
            normalización, mismo tamaño de imagen.
        model: modelo completo (backbone + cabeza); sus pesos se
            sobreescriben in-place con los de `best_checkpoint`, una vez por
            base de datos (mismo patrón que `evaluate_split()`).
        best_checkpoint: ruta a un checkpoint guardado por `checkpoint.save_checkpoint()`.
        loss_spec: mismo `LossSpec` usado para entrenar ese checkpoint.
        batch_size: `config.data.batch_size` -- se reusa tal cual; ninguna
            base de datos individual necesita `drop_last` porque estos
            loaders nunca entrenan (ver nota de `drop_last` en
            `datasets/build.py`, específica al loader de train).
        num_workers: `config.data.num_workers`.
        device: `config.train.device`.
        run_dir: `config.train.run_dir` -- se escribe bajo `run_dir/test/`.
        logger: `MetricsLogger` ya abierto por el caller.

    Example:
        >>> evaluate_by_database(
        ...     config.data.by_database_manifests, config.data.image_root,
        ...     eval_transform_builder, model, best_checkpoint, loss_spec,
        ...     config.data.batch_size, config.data.num_workers,
        ...     config.train.device, config.train.run_dir, logger,
        ... )
    """
    eval_transform = eval_transform_builder.build()

    metrics_by_db: dict[str, dict[str, float]] = {}
    cm_by_db: dict[str, tuple[list[int], list[int]]] = {}

    for db_name, manifest_path in by_database_manifests.items():
        db_manifest = Manifest(manifest_path=manifest_path, image_root=image_root)
        db_split = Split(manifest=db_manifest)
        db_test_df = db_split.test_df()

        if db_test_df.empty:
            # Defensivo -- no ocurre con los 4 manifests que genera
            # scripts/split_manifest_by_database.py (todas tienen filas de
            # test), pero un manifest por base de datos armado a mano podría
            # no tenerlas. No tiene sentido evaluar un DataLoader vacío.
            print(f"Aviso: '{db_name}' no tiene filas de test en {manifest_path} -- se omite.")
            continue

        db_loader = DataLoader(
            MammoBenchDataset(df=db_test_df, transform=eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        db_metrics = evaluate_checkpoint(model, best_checkpoint, db_loader, loss_spec, device)
        # y_prob no se usa acá -- ni la matriz de confusión ni
        # compute_confusion_matrix_metrics() lo necesitan (a diferencia de
        # evaluate_split(), que sí lo pasa a plot_roc_curve() para la curva
        # ROC de test combinado; este desglose no grafica una ROC por base
        # de datos).
        y_true, y_pred, _y_prob = predict_on_loader(model, db_loader, loss_spec, device)
        cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)

        logger.log_summary({f"test_by_database_{db_name}_{k}": v for k, v in db_metrics.items()})
        logger.log_summary({f"test_by_database_{db_name}_{k}": v for k, v in cm_metrics.items()})

        metrics_by_db[db_name] = {**db_metrics, **cm_metrics}
        cm_by_db[db_name] = (y_true, y_pred)

    if not metrics_by_db:
        # Todas las bases de datos se omitieron arriba (test vacío) -- no
        # hay nada que graficar ni guardar.
        return

    test_dir = run_dir / "test"
    save_metrics_by_database_json(metrics_by_db, test_dir / "metrics_by_database.json")

    confusion_matrix_by_db_path = test_dir / "confusion_matrix_by_database.png"
    plot_confusion_matrix_by_database(cm_by_db, confusion_matrix_by_db_path)
    logger.log_image("test/confusion_matrix_by_database", confusion_matrix_by_db_path)

    metrics_by_db_path = test_dir / "metrics_by_database.png"
    plot_metrics_by_database(metrics_by_db, metrics_by_db_path, metric_display_names=METRIC_DISPLAY_NAMES)
    logger.log_image("test/metrics_by_database", metrics_by_db_path)
