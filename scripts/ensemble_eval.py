"""Promedia las probabilidades de varias corridas ya evaluadas en un ensemble, sin reentrenar nada.

Motivación (diagnóstico 2026-09-08, serie exp28-32): las cuatro variantes
ImageNet de la receta anti-overfitting (exp28-31) convergen a val AUC
~0.91/test AUC ~0.88 con mejores épocas entre 3 y 12 -- la señal individual
parece agotada rápido. Promediar sus `y_prob` (elegido por val, sin tocar
test hasta reportar) sube test AUC a ~0.90: la diversidad entre corridas con
regularizadores distintos (weight_decay fuerte, label smoothing, sin
input_dropout) captura algo que ningún modelo individual solo.

No reentrena ni promedia pesos -- promedia PROBABILIDADES de corridas ya
evaluadas (`predictions.csv` en `run_dir/val/` y `run_dir/test/`, ver
`save_predictions_csv()` en `src/reporting.py`), asumiendo que todas
recorrieron el mismo split en el mismo orden (mismo manifest/seed/loader sin
shuffle -- val/test nunca barajan, ver `datasets/build.py`). Se verifica
explícitamente comparando `y_true` fila a fila entre las corridas de
entrada; una discrepancia (splits distintos, orden distinto) aborta con un
error claro en vez de promediar probabilidades que no corresponden a la
misma imagen.

Escribe un `output_dir` con la MISMA forma que un `run_dir` de una corrida
real -- para que `scripts/calibrate_threshold.py` (global o
`--by-database`) se le pueda correr encima sin ningún cambio:
    output_dir/val/predictions.csv
    output_dir/val/confusion_matrix_metrics.json
    output_dir/val/metrics.json          ({"auc": ...} -- ver nota abajo)
    output_dir/test/predictions.csv
    output_dir/test/confusion_matrix_metrics.json
    output_dir/test/metrics.json
    output_dir/{val,test}/predictions_{db_name}.csv        (si todas las
    output_dir/{val,test}/confusion_matrix_metrics_{db_name}.json  corridas de
    output_dir/{val,test}/metrics_{db_name}.json                   entrada lo tienen)

`metrics.json` acá NO es el esquema completo de `evaluate_checkpoint()`
(`accuracy/auc/f1/.../loss`): un ensemble de probabilidades promediadas no
tiene una sola función de pérdida que evaluar, así que solo lleva `"auc"`
-- el resto de métricas al umbral 0.5 vive en `confusion_matrix_metrics.json`,
igual que en una corrida real.

Uso (con `-m`, no como ruta de archivo -- mismo motivo que
scripts/calibrate_threshold.py, ver su docstring):
    .venv/bin/python -m scripts.ensemble_eval \
        --run-dirs runs/exp28_antioverfit_base runs/exp29_antioverfit_wd1e2 \
                   runs/exp30_antioverfit_labelsmooth runs/exp31_antioverfit_no_inputdrop \
        --output-dir runs/ensemble_exp28_29_30_31
    .venv/bin/python -m scripts.calibrate_threshold --run-dir runs/ensemble_exp28_29_30_31 --by-database
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import BinaryAUROC

from src.reporting import compute_confusion_matrix_metrics, save_predictions_csv

_SPLIT_NAMES: tuple[str, ...] = ("val", "test")


def _binary_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC vía torchmetrics -- no sklearn, no está instalado en todos los entornos del proyecto."""
    metric = BinaryAUROC()
    return float(metric(torch.tensor(y_prob, dtype=torch.float32), torch.tensor(y_true)))


def _load_and_average(run_dirs: list[Path], split_name: str, db_name: str | None = None) -> pd.DataFrame:
    """Lee `predictions[_{db_name}].csv` de cada `run_dir` y promedia `y_prob`.

    Args:
        run_dirs: corridas de entrada, en el orden que el caller quiera
            (el promedio no depende del orden).
        split_name: `"val"` o `"test"`.
        db_name: si se pasa, lee `predictions_{db_name}.csv` en vez de
            `predictions.csv` (desglose por base de datos).

    Returns:
        pd.DataFrame: columnas `y_true` (de la primera corrida, ya
            verificado idéntico en todas), `y_pred` (umbral 0.5 sobre el
            promedio -- el mismo default que usa `evaluate_split()`),
            `y_prob` (promedio simple entre corridas).

    Raises:
        FileNotFoundError: si a alguna corrida le falta el CSV esperado.
        ValueError: si el `y_true` de alguna corrida no coincide, fila a
            fila, con el de la primera -- señal de que las corridas de
            entrada no comparten split/orden y promediarlas no tiene sentido.
    """
    filename = f"predictions_{db_name}.csv" if db_name is not None else "predictions.csv"
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / split_name / filename
        if not path.is_file():
            raise FileNotFoundError(f"No existe {path} -- ¿todas las corridas de --run-dirs llegaron a evaluate_split()/evaluate_by_database()?")
        frames.append(pd.read_csv(path))

    base_y_true = frames[0]["y_true"].to_numpy()
    for run_dir, frame in zip(run_dirs, frames):
        if not (frame["y_true"].to_numpy() == base_y_true).all():
            raise ValueError(
                f"{run_dir / split_name / filename} tiene un y_true distinto, fila a fila, al de "
                f"{run_dirs[0] / split_name / filename} -- el ensemble asume que todas las corridas "
                "de entrada evaluaron el mismo split en el mismo orden (mismo manifest/seed, val/test "
                "sin shuffle). Promediar probabilidades de splits distintos no tiene sentido."
            )

    y_true = base_y_true
    y_prob = np.mean([frame["y_prob"].to_numpy() for frame in frames], axis=0)
    y_pred = (y_prob >= 0.5).astype(int)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})


def _write_split(ensemble_df: pd.DataFrame, split_dir: Path, suffix: str = "") -> None:
    """Persiste `predictions{suffix}.csv`, `confusion_matrix_metrics{suffix}.json` y `metrics{suffix}.json`."""
    save_predictions_csv(
        ensemble_df["y_true"].tolist(),
        ensemble_df["y_pred"].tolist(),
        ensemble_df["y_prob"].tolist(),
        split_dir / f"predictions{suffix}.csv",
    )
    cm_metrics = compute_confusion_matrix_metrics(ensemble_df["y_true"].tolist(), ensemble_df["y_pred"].tolist())
    (split_dir / f"confusion_matrix_metrics{suffix}.json").write_text(json.dumps(cm_metrics, indent=2))
    auc = _binary_auroc(ensemble_df["y_true"].to_numpy(), ensemble_df["y_prob"].to_numpy())
    (split_dir / f"metrics{suffix}.json").write_text(json.dumps({"auc": auc}, indent=2))


def build_ensemble(run_dirs: list[Path], output_dir: Path) -> dict[str, float]:
    """Construye el ensemble completo (global + por base de datos si aplica) en `output_dir`.

    Args:
        run_dirs: al menos 2 corridas ya evaluadas.
        output_dir: carpeta destino, con la misma forma que un `run_dir` real.

    Returns:
        dict[str, float]: `{"val_auc": ..., "test_auc": ...}` del ensemble global -- lo que
            `main()` imprime; el resto de métricas vive en los JSON escritos.
    """
    auc_summary: dict[str, float] = {}
    for split_name in _SPLIT_NAMES:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        ensemble_df = _load_and_average(run_dirs, split_name)
        _write_split(ensemble_df, split_dir)
        auc_summary[f"{split_name}_auc"] = _binary_auroc(
            ensemble_df["y_true"].to_numpy(), ensemble_df["y_prob"].to_numpy()
        )

    # Desglose por base de datos -- solo si TODAS las corridas de entrada lo
    # tienen en AMBOS splits (evaluate_by_database() con by_database_manifests
    # fijado, y ya con val por base -- ver src/eval_pipeline.py). Una corrida
    # de entrada sin esto simplemente hace que se omita el desglose entero,
    # no un error -- el ensemble global de arriba sigue siendo válido.
    db_names = sorted(p.stem.removeprefix("predictions_") for p in (run_dirs[0] / "test").glob("predictions_*.csv"))
    for db_name in db_names:
        has_all = all(
            (run_dir / split_name / f"predictions_{db_name}.csv").is_file()
            for run_dir in run_dirs
            for split_name in _SPLIT_NAMES
        )
        if not has_all:
            print(
                f"Aviso: '{db_name}' no tiene predictions_{db_name}.csv en val Y test para las "
                f"{len(run_dirs)} corridas de entrada -- se omite del ensemble por base de datos."
            )
            continue
        for split_name in _SPLIT_NAMES:
            ensemble_df = _load_and_average(run_dirs, split_name, db_name=db_name)
            _write_split(ensemble_df, output_dir / split_name, suffix=f"_{db_name}")

    return auc_summary


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Promedia las probabilidades de varias corridas ya evaluadas en un ensemble."
    )
    parser.add_argument(
        "--run-dirs",
        required=True,
        nargs="+",
        type=Path,
        help="Al menos 2 carpetas runs/<experiment_id> de entrada, ya evaluadas (val/ y test/ con predictions.csv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Carpeta destino -- se le puede correr scripts.calibrate_threshold encima como a cualquier run_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.run_dirs) < 2:
        raise ValueError("Un ensemble necesita al menos 2 --run-dirs.")
    auc_summary = build_ensemble(args.run_dirs, args.output_dir)
    print(f"Ensemble de {len(args.run_dirs)} corridas -> {args.output_dir}")
    print(f"  val  AUC={auc_summary['val_auc']:.4f}")
    print(f"  test AUC={auc_summary['test_auc']:.4f}")


if __name__ == "__main__":
    main()
