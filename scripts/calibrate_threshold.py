"""Calibra el umbral de decisión de un `run_dir` ya evaluado, sin reentrenar nada.

`evaluate_split()` (`src/eval_pipeline.py`) fija el umbral de binarización en
0.5 -- correcto para AUC (que no depende del umbral) pero no para el punto de
operación real: con clases muy desbalanceadas por base de datos (ej.
KAU-BCMD, ~4.6% maligno en train), 0.5 puede dejar la sensibilidad de test en
0.000 aunque el AUC diga que el modelo ordena bien (ver diagnóstico
2026-09-08, KAU-BCMD AUC=0.902 pero sensibilidad=0.000 en exp05/exp20/exp22).

Este script no toca `metrics.csv`, checkpoints ni ningún artefacto ya
escrito: lee `run_dir/val/predictions.csv` (columnas `y_true,y_pred,y_prob`,
ver `save_predictions_csv()` en `src/reporting.py`), barre umbrales sobre los
`y_prob` únicos de validación para maximizar `--objective`, y aplica ESE
umbral -- elegido sin haber tocado test -- a `run_dir/test/predictions.csv`.
Reusa `compute_confusion_matrix_metrics()` (`src/reporting.py`) para que las
métricas recalculadas usen exactamente las mismas fórmulas (mismo `f1_macro`,
mismo `_safe_div`) que ya viven en `test/confusion_matrix_metrics.json` --
comparar ambos archivos es válido punto a punto.

Escribe `run_dir/test/metrics_calibrated.json`:
    {
      "threshold": <umbral elegido en val>,
      "objective": "<f1_macro|youden>",
      "val": {métricas de compute_confusion_matrix_metrics() en val, a ese umbral},
      "test_baseline_0.5": {métricas de test a umbral fijo 0.5},
      "test_calibrated": {métricas de test al umbral calibrado},
    }

`--by-database` repite exactamente lo mismo pero una vez POR BASE DE DATOS,
sobre `val/predictions_{db}.csv` / `test/predictions_{db}.csv` -- requiere
que la corrida se haya evaluado con la versión de `evaluate_by_database()`
(`src/eval_pipeline.py`) que también persiste val por base (ver su
docstring); una corrida vieja que solo tenga `predictions_{db}.csv` en
`test/` no alcanza. Necesario porque un umbral global (el modo por defecto,
sin esta bandera) se elige sobre el val COMBINADO, y el desbalance real
varía muchísimo de una base a otra (CMMD ~49% maligno vs KAU-BCMD ~4.6%) --
el umbral que le sirve a CMMD puede ser inútil para KAU-BCMD. Escribe
`run_dir/test/metrics_calibrated_by_database.json`:
    { "<db_name>": {mismo esquema que metrics_calibrated.json de arriba}, ... }

Uso (re-ejecutable en cualquier momento; sobreescribe sin preguntar) -- con
`-m`, no como ruta de archivo: el script importa `src.reporting`, y
`python scripts/calibrate_threshold.py` deja fuera de `sys.path` la raíz del
repo (donde vive el paquete `src`), a diferencia de `python -m
scripts.calibrate_threshold`, que sí la agrega -- mismo motivo por el que
`src/cli.py`/`src/evaluate.py` se invocan como `-m src.cli`/`-m src.evaluate`:
    .venv/bin/python -m scripts.calibrate_threshold --run-dir runs/exp22_pretrain_ablation_imagenet_all
    .venv/bin/python -m scripts.calibrate_threshold --run-dir runs/exp05_fedmammobench_full_weighted --objective youden
    .venv/bin/python -m scripts.calibrate_threshold --run-dir runs/exp28_antioverfit_base --by-database
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.reporting import compute_confusion_matrix_metrics

_OBJECTIVE_KEYS: dict[str, str] = {
    "f1_macro": "cm_f1_macro",
    "youden": "cm_youden_j",
}


def _load_predictions(split_dir: Path) -> pd.DataFrame:
    """Lee `{split_dir}/predictions.csv`, con el error explícito de qué falta si no existe.

    Args:
        split_dir: `run_dir/val` o `run_dir/test`.

    Returns:
        pd.DataFrame: columnas `y_true`, `y_pred` (ignorada, se recalcula a
            partir de `y_prob`), `y_prob`.

    Raises:
        FileNotFoundError: si `predictions.csv` no existe en `split_dir` --
            típico de una corrida que nunca llegó a `evaluate_split()`.
    """
    path = split_dir / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path} -- ¿esta corrida llegó a evaluate_split()? "
            "(ver src/eval_pipeline.py, o corré src.evaluate sobre el checkpoint)."
        )
    return pd.read_csv(path)


def find_best_threshold(y_true: pd.Series, y_prob: pd.Series, objective: str) -> tuple[float, dict[str, float]]:
    """Barre los umbrales candidatos (los `y_prob` únicos de validación) y devuelve el que maximiza `objective`.

    Args:
        y_true: etiquetas reales de validación (0/1).
        y_prob: probabilidad de clase positiva de validación, mismo orden que `y_true`.
        objective: clave en `_OBJECTIVE_KEYS` -- `"f1_macro"` (default,
            balancea ambas clases) o `"youden"` (sensitivity + specificity - 1,
            el punto que maximiza la separación de la curva ROC).

    Returns:
        tuple[float, dict[str, float]]: `(mejor_umbral, métricas de
            compute_confusion_matrix_metrics() en ese umbral)`. Ante empate,
            se queda con el primer umbral (el más bajo) que lo alcanza --
            mismo criterio simple que el resto del repo usa para "mejor
            época" en `EarlyStopping` (estricto, sin desempate especial).

    Raises:
        ValueError: `objective` no reconocido.
    """
    if objective not in _OBJECTIVE_KEYS:
        raise ValueError(f"objective desconocido: {objective!r}. Opciones: {sorted(_OBJECTIVE_KEYS)}")
    metric_key = _OBJECTIVE_KEYS[objective]

    # Los únicos umbrales que pueden cambiar la matriz de confusión son los
    # valores de y_prob observados en val (cualquier punto intermedio entre
    # dos y_prob consecutivos clasifica exactamente igual) -- barrer más
    # finamente no encontraría un óptimo distinto, solo lo haría más lento.
    candidate_thresholds = sorted(y_prob.unique().tolist())

    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_score = float("-inf")
    for threshold in candidate_thresholds:
        y_pred_at_threshold = (y_prob >= threshold).astype(int).tolist()
        metrics = compute_confusion_matrix_metrics(y_true.tolist(), y_pred_at_threshold)
        score = metrics[metric_key]
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    assert best_metrics is not None  # candidate_thresholds nunca está vacío si val tiene filas
    return best_threshold, best_metrics


def _calibrate_pair(val_df: pd.DataFrame, test_df: pd.DataFrame, objective: str) -> dict[str, object]:
    """Núcleo de la calibración: elige umbral en `val_df` y lo compara contra 0.5 en `test_df`.

    Extraído para que `calibrate()` (global) y `calibrate_by_database()` (una
    llamada por base de datos) compartan exactamente la misma lógica -- la
    única diferencia entre ambos modos es QUÉ par (val, test) le pasan.

    Args:
        val_df: `predictions.csv` (o `predictions_{db}.csv`) de val -- columnas `y_true`, `y_prob`.
        test_df: su contraparte de test.
        objective: ver `find_best_threshold()`.

    Returns:
        dict[str, object]: `threshold`, `objective`, `val` (métricas en val a
            ese umbral), `test_baseline_0.5`, `test_calibrated`.
    """
    threshold, val_metrics = find_best_threshold(val_df["y_true"], val_df["y_prob"], objective)

    # Baseline: el umbral 0.5 fijo que evaluate_split()/evaluate_by_database()
    # ya usaron para escribir confusion_matrix_metrics.json/metrics_by_database.json
    # -- recalculado acá (no leído de disco) para que ambas filas de la
    # comparación vengan de la misma función en la misma corrida del script.
    test_pred_baseline = (test_df["y_prob"] >= 0.5).astype(int).tolist()
    test_metrics_baseline = compute_confusion_matrix_metrics(test_df["y_true"].tolist(), test_pred_baseline)

    test_pred_calibrated = (test_df["y_prob"] >= threshold).astype(int).tolist()
    test_metrics_calibrated = compute_confusion_matrix_metrics(test_df["y_true"].tolist(), test_pred_calibrated)

    return {
        "threshold": threshold,
        "objective": objective,
        "val": val_metrics,
        "test_baseline_0.5": test_metrics_baseline,
        "test_calibrated": test_metrics_calibrated,
    }


def calibrate(run_dir: Path, objective: str) -> dict[str, object]:
    """Corre la calibración completa (global) para un `run_dir` y persiste `test/metrics_calibrated.json`.

    Args:
        run_dir: carpeta de una corrida ya evaluada (`config.train.run_dir`),
            con `val/predictions.csv` y `test/predictions.csv` presentes.
        objective: ver `find_best_threshold()`.

    Returns:
        dict[str, object]: el mismo dict que se escribe a
            `test/metrics_calibrated.json` (`threshold`, `objective`, `val`,
            `test_baseline_0.5`, `test_calibrated`).
    """
    val_df = _load_predictions(run_dir / "val")
    test_df = _load_predictions(run_dir / "test")

    result = _calibrate_pair(val_df, test_df, objective)

    output_path = run_dir / "test" / "metrics_calibrated.json"
    output_path.write_text(json.dumps(result, indent=2))
    return result


def calibrate_by_database(run_dir: Path, objective: str) -> dict[str, dict[str, object]]:
    """Corre la calibración una vez POR BASE DE DATOS y persiste `test/metrics_calibrated_by_database.json`.

    Descubre las bases de datos disponibles listando
    `run_dir/val/predictions_*.csv` -- requiere haber corrido la corrida (o
    re-evaluado el checkpoint con `src.evaluate`) con la versión de
    `evaluate_by_database()` que también persiste val por base (ver su
    docstring en `src/eval_pipeline.py`); una corrida vieja, evaluada antes
    de ese cambio, no tiene esos archivos y esta función levanta
    `FileNotFoundError`.

    Args:
        run_dir: carpeta de una corrida evaluada con `by_database_manifests`
            (`DataConfig.by_database_manifests`, `src/config.py`).
        objective: ver `find_best_threshold()`.

    Returns:
        dict[str, dict[str, object]]: `{nombre_base_de_datos: resultado}`,
            mismo `resultado` que devuelve `_calibrate_pair()`. Una base
            cuyo `predictions_{db}.csv` de test no exista (aunque sí el de
            val) se omite con un aviso -- no debería ocurrir si ambos
            provienen de la misma corrida de `evaluate_by_database()`.

    Raises:
        FileNotFoundError: si `run_dir/val/` no tiene ningún
            `predictions_*.csv`.
    """
    val_dir = run_dir / "val"
    test_dir = run_dir / "test"

    db_names = sorted(p.stem.removeprefix("predictions_") for p in val_dir.glob("predictions_*.csv"))
    if not db_names:
        raise FileNotFoundError(
            f"No hay {val_dir}/predictions_<base_de_datos>.csv -- ¿esta corrida se evaluó con "
            "by_database_manifests fijado, y con la versión de evaluate_by_database() que también "
            "persiste val por base? (ver src/eval_pipeline.py). Una corrida vieja solo tendría "
            "estos archivos en test/, no en val/."
        )

    results: dict[str, dict[str, object]] = {}
    for db_name in db_names:
        test_path = test_dir / f"predictions_{db_name}.csv"
        if not test_path.is_file():
            print(f"Aviso: {test_path} no existe -- se omite '{db_name}'.")
            continue
        val_db_df = pd.read_csv(val_dir / f"predictions_{db_name}.csv")
        test_db_df = pd.read_csv(test_path)
        results[db_name] = _calibrate_pair(val_db_df, test_db_df, objective)

    output_path = test_dir / "metrics_calibrated_by_database.json"
    output_path.write_text(json.dumps(results, indent=2))
    return results


def _print_comparison(result: dict[str, object], label: str | None = None) -> None:
    """Imprime una tabla legible baseline-vs-calibrado para las métricas clínicas usuales.

    Args:
        result: salida de `_calibrate_pair()`/`calibrate()`.
        label: si se pasa (modo `--by-database`), se imprime como encabezado
            antes de la tabla -- el nombre de la base de datos.
    """
    if label is not None:
        print(f"--- {label} ---")
    baseline = result["test_baseline_0.5"]
    calibrated = result["test_calibrated"]
    print(f"Umbral calibrado en val (objective={result['objective']}): {result['threshold']:.4f}")
    print(f"{'métrica':<14}{'umbral 0.5':>12}{'calibrado':>12}{'delta':>10}")
    for key in ("cm_sensitivity", "cm_specificity", "cm_precision", "cm_f1", "cm_f1_macro", "cm_accuracy"):
        base_v, cal_v = baseline[key], calibrated[key]
        print(f"{key:<14}{base_v:>12.4f}{cal_v:>12.4f}{cal_v - base_v:>+10.4f}")


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Calibra el umbral de decisión de una corrida ya evaluada, sin reentrenar."
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Carpeta runs/<experiment_id>.")
    parser.add_argument(
        "--objective",
        choices=sorted(_OBJECTIVE_KEYS),
        default="f1_macro",
        help="Métrica de validación a maximizar al elegir el umbral (default: f1_macro).",
    )
    parser.add_argument(
        "--by-database",
        action="store_true",
        help=(
            "Calibra un umbral POR BASE DE DATOS en vez de uno global -- requiere "
            "val/predictions_<db>.csv (ver evaluate_by_database() en src/eval_pipeline.py). "
            "Escribe test/metrics_calibrated_by_database.json en vez de test/metrics_calibrated.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.by_database:
        results = calibrate_by_database(args.run_dir, args.objective)
        for db_name, result in results.items():
            _print_comparison(result, label=db_name)
        print(f"\nEscrito: {args.run_dir / 'test' / 'metrics_calibrated_by_database.json'}")
    else:
        result = calibrate(args.run_dir, args.objective)
        _print_comparison(result)
        print(f"\nEscrito: {args.run_dir / 'test' / 'metrics_calibrated.json'}")


if __name__ == "__main__":
    main()
