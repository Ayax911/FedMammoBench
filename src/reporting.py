"""Artefactos de evaluación de test: `metrics.json`, matriz de confusión,
curva ROC y `predictions.csv`.

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

Todo lo que sale de la evaluación de test (metrics.json, predictions.csv,
confusion_matrix.png, roc_curve.png) va a `run_dir/test/` -- una sola
carpeta para no repartir gráficas y predicciones de test entre `run_dir/`
y `run_dir/plots/`. `plots/` queda solo para lo que no es de test
(`loss_curve.png`, que se grafica antes de tocar el set de test).

Ejemplo de uso:
    >>> from src.reporting import save_metrics_json, save_predictions_csv, plot_confusion_matrix, plot_loss_curve, plot_roc_curve
    >>> plot_loss_curve(train_hist, val_hist, run_dir / "plots" / "loss_curve.png", best_epoch=7)
    >>> save_metrics_json(test_metrics, run_dir / "test" / "metrics.json")
    >>> save_predictions_csv(y_true, y_pred, y_prob, run_dir / "test" / "predictions.csv")
    >>> plot_confusion_matrix(y_true, y_pred, run_dir / "test" / "confusion_matrix.png")
    >>> plot_roc_curve(y_true, y_prob, run_dir / "test" / "roc_curve.png")
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
