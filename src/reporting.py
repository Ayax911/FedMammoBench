"""Artefactos de evaluación post-entrenamiento: `metrics.json`, matriz de
confusión, curva ROC y `predictions.csv` -- para val y test por igual.

Ausentes del pipeline original — `src/cli.py` terminaba en `trainer.fit()`
y nunca tocaba `loaders["test"]` (ver `train/evaluation.py`). Estas
funciones son el equivalente al `test_model()` del proyecto INC
(`classification_images/training.py`), reescrito como funciones puras en
vez de un método de una clase de 426 líneas: cada una recibe exactamente
los datos que necesita y no conoce `Trainer`, `MetricsLogger`, ni argparse.

Usa `torchmetrics` (ya en `requirements.txt`, usado en `src/metrics.py`)
para la matriz de confusión y la curva ROC, en vez de scikit-learn como
hace el INC — para no añadir una dependencia nueva solo por dos funciones.
`matplotlib` sí es una dependencia nueva (ver `requirements.txt`); nada más
en `src/` necesita graficar.

Nada en este módulo sabe de "val" ni de "test" -- ambas etiquetas son
puramente de `cli.py`: `_evaluate_split()` (`src/cli.py`) llama exactamente
las mismas funciones de acá, una vez por split, con `run_dir/val/` o
`run_dir/test/` según toque. El `evaluate()` que corre por época dentro de
`Trainer.fit()` (`train/loop.py`) también evalúa val, pero solo el resumen
agregado -- `_evaluate_split()` vuelve a evaluar val (y test) una vez más,
al final, releyendo el mejor checkpoint, para poder generar matriz de
confusión/ROC/`predictions.csv`, que requieren las predicciones completas
en memoria (ver docstring de `predict_on_loader()`, `train/evaluation.py`).

Todo lo que sale de una de estas evaluaciones (metrics.json,
confusion_matrix_metrics.json, predictions.csv, confusion_matrix.png,
roc_curve.png) va a `run_dir/val/` o `run_dir/test/` -- una sola carpeta
por split para no repartir sus gráficas y predicciones entre `run_dir/` y
`run_dir/plots/`. `plots/` queda solo para lo que no es de un split en
particular (`loss_curve.png`, que compara ambos y se grafica antes de
evaluar ninguno).

`compute_confusion_matrix_metrics()` es la excepción a "cada una recibe
exactamente los datos que necesita para UN artefacto": no guarda ni grafica
nada por sí misma, solo deriva de `y_true`/`y_pred` la tabla completa de
métricas calculables desde una matriz de confusión 2x2 (más allá de las
siete que ya trackea `build_metric_collection()` en `src/metrics.py`)
-- pensada para combinarse con `test_metrics` y mandarse junto a
`MetricsLogger.log_summary()` (`src/tracking.py`), y así poder comparar
esas métricas entre corridas en W&B.

Ejemplo de uso:
    >>> from src.reporting import (
    ...     save_metrics_json, save_predictions_csv, plot_confusion_matrix,
    ...     plot_loss_curve, plot_metric_curve, plot_roc_curve, compute_confusion_matrix_metrics,
    ... )
    >>> plot_loss_curve(train_hist, val_hist, run_dir / "plots" / "loss_curve.png", best_epoch=7)
    >>> plot_metric_curve(train_auc_hist, val_auc_hist, run_dir / "plots" / "auc_curve.png", "AUC", best_epoch=7)
    >>> save_metrics_json(test_metrics, run_dir / "test" / "metrics.json")
    >>> save_predictions_csv(y_true, y_pred, y_prob, run_dir / "test" / "predictions.csv")
    >>> plot_confusion_matrix(y_true, y_pred, run_dir / "test" / "confusion_matrix.png")
    >>> plot_roc_curve(y_true, y_prob, run_dir / "test" / "roc_curve.png")
    >>> cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    >>> save_metrics_json(cm_metrics, run_dir / "test" / "confusion_matrix_metrics.json")
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display -- corre en servidores/CI sin X11
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import BinaryConfusionMatrix, BinaryROC


def save_metrics_json(metrics: dict[str, float], path: str | Path) -> None:
    """Vuelca un dict de métricas a JSON, creando el directorio padre si hace falta.

    Args:
        metrics: dict métrica -> valor, típicamente el retorno de
            `evaluate_checkpoint()` (`train/evaluation.py`).
        path: ruta destino del archivo `.json`.

    Example:
        >>> save_metrics_json({"loss": 0.31, "auc": 0.87}, "runs/exp01/metrics.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def save_predictions_csv(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
    path: str | Path,
) -> None:
    """Guarda etiquetas/predicciones/probabilidades alineadas por posición en un CSV.

    Args:
        y_true: etiquetas reales, mismo orden que `predict_on_loader()`.
        y_pred: clase predicha (umbral 0.5 sobre `y_prob`).
        y_prob: probabilidad de la clase positiva (malignant=1).
        path: ruta destino del archivo `.csv`.

    Example:
        >>> y_true, y_pred, y_prob = predict_on_loader(model, loaders["test"], loss_spec, device)
        >>> save_predictions_csv(y_true, y_pred, y_prob, "runs/exp01/test/predictions.csv")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "y_prob"])
        writer.writerows(zip(y_true, y_pred, y_prob))


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    path: str | Path,
    class_names: tuple[str, str] = ("Benign", "Malignant"),
) -> None:
    """Calcula y guarda una matriz de confusión binaria como imagen PNG.

    Args:
        y_true: etiquetas reales (0/1).
        y_pred: clase predicha (0/1).
        path: ruta destino de la imagen `.png`. El directorio padre se
            crea si no existe.
        class_names: nombres para los ejes, en orden `(clase 0, clase 1)`.

    Example:
        >>> plot_confusion_matrix(y_true, y_pred, "runs/exp01/test/confusion_matrix.png")
    """
    cm = BinaryConfusionMatrix()(torch.tensor(y_pred), torch.tensor(y_true)).numpy()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")  # pyright: ignore[reportUnknownMemberType]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xticks([0, 1], labels=class_names)  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticks([0, 1], labels=class_names)  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Predicho")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Real")  # pyright: ignore[reportUnknownMemberType]
    ax.set_title("Matriz de confusión")  # pyright: ignore[reportUnknownMemberType]
    fig.colorbar(im, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    fig.savefig(path, dpi=200)  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def _safe_div(numerator: float, denominator: float) -> float:
    """División con `zero_division=0.0` -- mismo criterio que `BinaryMacroF1Score` (`src/metrics.py`).

    Casi todo lo que deriva de una matriz de confusión es un cociente de
    conteos, y con un split pequeño o muy desbalanceado (ej. una clase sin
    ningún positivo predicho) el denominador puede ser exactamente 0. Un
    `ZeroDivisionError` a mitad de `cli.run()`, después de que el modelo ya
    entrenó, tiraría toda la corrida por una métrica secundaria -- en vez de
    eso, `0.0` dice "no hay evidencia suficiente para esta métrica", igual
    que scikit-learn.
    """
    return numerator / denominator if denominator != 0 else 0.0


def compute_confusion_matrix_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Deriva, a mano, todas las métricas binarias estándar calculables desde TP/TN/FP/FN.

    `build_metric_collection()` (`src/metrics.py`) ya cubre las siete
    métricas "clínicas" que se trackean por época (accuracy, auc,
    sensitivity, specificity, f1, f1_macro, precision) -- esta función no
    las reemplaza, las complementa: recorre la tabla completa de razones
    que se pueden armar con una matriz de confusión 2x2 (la misma que
    `plot_confusion_matrix` ya calcula y grafica), incluyendo varias que no
    se necesitan para entrenar/parar temprano pero sí son relevantes para
    diagnóstico clínico (NPV, likelihood ratios, odds ratio diagnóstico) o
    para comparar corridas en un dataset desbalanceado (balanced accuracy,
    MCC, kappa de Cohen). Pensada para llamarse una sola vez, sobre las
    predicciones completas de test (`predict_on_loader()`) -- igual que
    `plot_confusion_matrix()` y `plot_roc_curve()`, no dentro del loop de
    entrenamiento.

    Todas las claves llevan el prefijo `cm_` para que puedan combinarse sin
    colisión con las que devuelve `evaluate()` (`train/evaluation.py`) al
    pasarlas juntas a `MetricsLogger.log_summary()` o a `save_metrics_json()`
    -- aun cuando cubren el mismo concepto (ej. `cm_precision` vs
    `precision`), se recalculan acá desde cero para no depender de que
    `evaluate()` haya guardado los conteos crudos.

    Args:
        y_true: etiquetas reales (0=benigno, 1=maligno), mismo orden que
            devuelve `predict_on_loader()`.
        y_pred: clase predicha (umbral 0.5 sobre la probabilidad), mismo
            orden que `y_true`.

    Returns:
        dict[str, float]: conteos crudos (`cm_tp`, `cm_tn`, `cm_fp`, `cm_fn`)
        más, todas con el prefijo `cm_`:
            - `accuracy`, `balanced_accuracy`
            - `sensitivity` (TPR/recall), `specificity` (TNR)
            - `precision` (PPV), `npv`
            - `fpr` (fall-out), `fnr` (miss rate), `fdr`, `for` (false
              omission rate)
            - `f1`, `f1_macro`
            - `mcc` (coeficiente de correlación de Matthews)
            - `kappa` (kappa de Cohen)
            - `youden_j` (informedness), `markedness`
            - `prevalence`
            - `positive_likelihood_ratio`, `negative_likelihood_ratio`,
              `diagnostic_odds_ratio`
            - `g_mean`, `fowlkes_mallows`, `threat_score` (CSI/Jaccard)
        Cualquier cociente con denominador 0 vale `0.0` (`_safe_div`), nunca
        levanta -- ojo con `positive_likelihood_ratio` y
        `diagnostic_odds_ratio`: su denominador se anula justo cuando el
        modelo es PERFECTO en un eje (`specificity == 1` para el primero,
        `fp == 0` o `fn == 0` para el segundo), caso en el que el valor real
        es +infinito (mejor resultado posible), no 0. Un `0.0` ahí lee como
        "sin poder diagnóstico", el sentido contrario -- revisar
        `cm_specificity`/`cm_fp`/`cm_fn` antes de interpretar un 0.0 en
        cualquiera de las dos como "malo".

    Example:
        >>> y_true, y_pred, y_prob = predict_on_loader(model, loaders["test"], loss_spec, device)
        >>> cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
        >>> print(cm_metrics["cm_mcc"], cm_metrics["cm_npv"])
    """
    cm = BinaryConfusionMatrix()(torch.tensor(y_pred), torch.tensor(y_true)).numpy()
    # Mismo layout que plot_confusion_matrix: filas = real, columnas =
    # predicho -- [[TN, FP], [FN, TP]] (confirmado contra torchmetrics).
    tn, fp, fn, tp = (float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0]), float(cm[1, 1]))
    total = tp + tn + fp + fn

    sensitivity = _safe_div(tp, tp + fn)  # TPR / recall
    specificity = _safe_div(tn, tn + fp)  # TNR
    precision = _safe_div(tp, tp + fp)  # PPV
    npv = _safe_div(tn, tn + fn)

    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    fdr = _safe_div(fp, fp + tp)
    for_ = _safe_div(fn, fn + tn)  # false omission rate

    accuracy = _safe_div(tp + tn, total)
    balanced_accuracy = (sensitivity + specificity) / 2

    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    f1_negative = _safe_div(2 * npv * specificity, npv + specificity)
    f1_macro = (f1 + f1_negative) / 2

    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_div(tp * tn - fp * fn, mcc_denominator)

    prevalence = _safe_div(tp + fn, total)
    observed_agreement = accuracy
    expected_agreement = _safe_div((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), total * total)
    kappa = _safe_div(observed_agreement - expected_agreement, 1 - expected_agreement)

    youden_j = sensitivity + specificity - 1  # informedness / bookmaker informedness
    markedness = precision + npv - 1

    # Likelihood ratios y odds ratio diagnóstico: métricas estándar de
    # evaluación de pruebas diagnósticas (relevantes acá porque el problema
    # ES un diagnóstico -- benigno/maligno), sin equivalente en
    # `build_metric_collection()`.
    positive_likelihood_ratio = _safe_div(sensitivity, 1 - specificity)
    negative_likelihood_ratio = _safe_div(1 - sensitivity, specificity)
    diagnostic_odds_ratio = _safe_div(tp * tn, fp * fn)

    g_mean = (sensitivity * specificity) ** 0.5
    fowlkes_mallows = (precision * sensitivity) ** 0.5
    threat_score = _safe_div(tp, tp + fn + fp)  # critical success index / Jaccard de la clase positiva

    return {
        "cm_tp": tp,
        "cm_tn": tn,
        "cm_fp": fp,
        "cm_fn": fn,
        "cm_accuracy": accuracy,
        "cm_balanced_accuracy": balanced_accuracy,
        "cm_sensitivity": sensitivity,
        "cm_specificity": specificity,
        "cm_precision": precision,
        "cm_npv": npv,
        "cm_fpr": fpr,
        "cm_fnr": fnr,
        "cm_fdr": fdr,
        "cm_for": for_,
        "cm_f1": f1,
        "cm_f1_macro": f1_macro,
        "cm_mcc": mcc,
        "cm_kappa": kappa,
        "cm_youden_j": youden_j,
        "cm_markedness": markedness,
        "cm_prevalence": prevalence,
        "cm_positive_likelihood_ratio": positive_likelihood_ratio,
        "cm_negative_likelihood_ratio": negative_likelihood_ratio,
        "cm_diagnostic_odds_ratio": diagnostic_odds_ratio,
        "cm_g_mean": g_mean,
        "cm_fowlkes_mallows": fowlkes_mallows,
        "cm_threat_score": threat_score,
    }


def plot_loss_curve(
    train_loss: list[float],
    val_loss: list[float],
    path: str | Path,
    best_epoch: int | None = None,
) -> None:
    """Grafica pérdida de entrenamiento vs. validación por época.

    Equivalente al `plots/loss_curve.png` de la serie de notebooks — el
    gráfico de diagnóstico principal, y el único que faltaba por completo
    en este pipeline pese a que `metrics.csv` ya guardaba los datos.

    El eje Y arranca siempre en 0 y llega como mínimo a 1 — así las corridas
    se comparan de un vistazo sin importar en qué rango cayó la pérdida de
    esa corrida en particular. No es un `ax.set_ylim(0, 1)` fijo como en el
    notebook: si algún valor supera 1 (`CrossEntropyLoss` ponderada arranca
    por encima de 1 en las primeras épocas), el techo sube para no
    recortarlo — nunca al revés.

    Args:
        train_loss: pérdida media de entrenamiento por época, en orden.
        val_loss: pérdida media de validación por época, mismo largo que
            `train_loss`.
        path: ruta destino de la imagen `.png`. El directorio padre se
            crea si no existe.
        best_epoch: índice (base 0) de la época del mejor checkpoint. Si
            se pasa, se marca con una línea vertical. `None` no dibuja nada.

    Raises:
        ValueError: si `train_loss` y `val_loss` tienen distinto largo, o
            si están vacíos.

    Example:
        >>> plot_loss_curve(train_hist, val_hist, "runs/exp01/plots/loss_curve.png", best_epoch=7)
    """
    if len(train_loss) != len(val_loss):
        raise ValueError(
            f"train_loss y val_loss deben tener el mismo largo — "
            f"recibidos {len(train_loss)} y {len(val_loss)}"
        )
    if not train_loss:
        raise ValueError("No hay ninguna época que graficar (historial vacío).")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Épocas en base 1 para el eje, aunque best_epoch llegue en base 0 (como
    # lo numera Trainer.fit()) -- el +1 se aplica una sola vez, acá.
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Training loss", color="#1f77b4")  # pyright: ignore[reportUnknownMemberType]
    ax.plot(epochs, val_loss, label="Validation loss", color="#d62728")  # pyright: ignore[reportUnknownMemberType]
    if best_epoch is not None:
        ax.axvline(  # pyright: ignore[reportUnknownMemberType]
            best_epoch + 1,
            linestyle="--",
            color="gray",
            lw=1,
            label=f"Mejor época ({best_epoch + 1})",
        )
    # Techo mínimo 1 -- ver docstring. max(train_loss)/max(val_loss) ya
    # garantizan no-vacío (chequeado arriba), así que esto no puede lanzar.
    ax.set_ylim(0, max(1.0, max(train_loss), max(val_loss)))  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Época")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Loss")  # pyright: ignore[reportUnknownMemberType]
    ax.set_title("Pérdida de entrenamiento vs. validación")  # pyright: ignore[reportUnknownMemberType]
    ax.grid(True, linestyle="--", alpha=0.4)  # pyright: ignore[reportUnknownMemberType]
    ax.legend()  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    fig.savefig(path, dpi=200)  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def plot_metric_curve(
    train_values: list[float],
    val_values: list[float],
    path: str | Path,
    metric_name: str,
    best_epoch: int | None = None,
) -> None:
    """Grafica una métrica clínica de entrenamiento vs. validación por época.

    Generaliza `plot_loss_curve()` a cualquiera de las siete métricas de
    `build_metric_collection()` (`src/metrics.py`) -- accuracy, auc,
    sensitivity, specificity, f1, f1_macro, precision -- ahora que
    `train_one_epoch()` (`train/loop.py`) las calcula sobre entrenamiento
    igual que `evaluate()` ya hacía sobre validación. No reemplaza a
    `plot_loss_curve()`, que se mantiene aparte: loss no está acotada a
    `[0, 1]` (el techo del eje Y se recalcula por corrida), mientras que las
    siete métricas clínicas sí lo están siempre -- por eso acá el eje Y es
    fijo `[0, 1]`, lo que además hace comparables entre sí las gráficas de
    dos corridas distintas de un vistazo, sin mirar la escala primero.

    Args:
        train_values: valor de la métrica en entrenamiento por época, en
            orden (ej. `[e["train_auc"] for e in trainer.history]`).
        val_values: idem para validación, mismo largo que `train_values`
            (ej. `[e["val_auc"] for e in trainer.history]`).
        path: ruta destino de la imagen `.png`. El directorio padre se
            crea si no existe.
        metric_name: nombre a mostrar en título/eje Y/leyenda (ej. `"AUC"`,
            `"F1-macro"`, `"Sensibilidad"`) -- se usa tal cual, sin mapear
            ni traducir.
        best_epoch: índice (base 0) de la época del mejor checkpoint. Si
            se pasa, se marca con una línea vertical, igual que en
            `plot_loss_curve()`. `None` no dibuja nada.

    Raises:
        ValueError: si `train_values` y `val_values` tienen distinto largo,
            o si están vacíos.

    Example:
        >>> plot_metric_curve(
        ...     [e["train_auc"] for e in trainer.history],
        ...     [e["val_auc"] for e in trainer.history],
        ...     "runs/exp01/plots/auc_curve.png",
        ...     metric_name="AUC",
        ...     best_epoch=trainer.best_epoch,
        ... )
    """
    if len(train_values) != len(val_values):
        raise ValueError(
            f"train_values y val_values deben tener el mismo largo — "
            f"recibidos {len(train_values)} y {len(val_values)}"
        )
    if not train_values:
        raise ValueError("No hay ninguna época que graficar (historial vacío).")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Épocas en base 1 para el eje, aunque best_epoch llegue en base 0 (como
    # lo numera Trainer.fit()) -- mismo criterio que plot_loss_curve().
    epochs = range(1, len(train_values) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_values, label=f"Train {metric_name}", color="#1f77b4")  # pyright: ignore[reportUnknownMemberType]
    ax.plot(epochs, val_values, label=f"Val {metric_name}", color="#d62728")  # pyright: ignore[reportUnknownMemberType]
    if best_epoch is not None:
        ax.axvline(  # pyright: ignore[reportUnknownMemberType]
            best_epoch + 1,
            linestyle="--",
            color="gray",
            lw=1,
            label=f"Mejor época ({best_epoch + 1})",
        )
    ax.set_ylim(0, 1)  # pyright: ignore[reportUnknownMemberType] -- ver docstring: las 7 métricas viven en [0, 1]
    ax.set_xlabel("Época")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel(metric_name)  # pyright: ignore[reportUnknownMemberType]
    ax.set_title(f"{metric_name} — entrenamiento vs. validación")  # pyright: ignore[reportUnknownMemberType]
    ax.grid(True, linestyle="--", alpha=0.4)  # pyright: ignore[reportUnknownMemberType]
    ax.legend()  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    fig.savefig(path, dpi=200)  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def plot_roc_curve(y_true: list[int], y_prob: list[float], path: str | Path) -> None:
    """Calcula y guarda la curva ROC + AUC como imagen PNG.

    Args:
        y_true: etiquetas reales (0/1).
        y_prob: probabilidad de la clase positiva.
        path: ruta destino de la imagen `.png`. El directorio padre se
            crea si no existe.

    Example:
        >>> plot_roc_curve(y_true, y_prob, "runs/exp01/test/roc_curve.png")
    """
    fpr, tpr, _ = BinaryROC()(torch.tensor(y_prob), torch.tensor(y_true))
    roc_auc = torch.trapz(tpr, fpr).item()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr.numpy(), tpr.numpy(), label=f"ROC (AUC = {roc_auc:.3f})")  # pyright: ignore[reportUnknownMemberType]
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("False Positive Rate")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("True Positive Rate")  # pyright: ignore[reportUnknownMemberType]
    ax.set_title("Curva ROC")  # pyright: ignore[reportUnknownMemberType]
    ax.legend(loc="lower right")  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    fig.savefig(path, dpi=200)  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)
