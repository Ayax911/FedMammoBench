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

Ejemplo de uso:
    >>> from src.reporting import save_metrics_json, save_predictions_csv, plot_confusion_matrix, plot_roc_curve
    >>> save_metrics_json(test_metrics, run_dir / "metrics.json")
    >>> save_predictions_csv(y_true, y_pred, y_prob, run_dir / "predictions.csv")
    >>> plot_confusion_matrix(y_true, y_pred, run_dir / "plots" / "confusion_matrix.png")
    >>> plot_roc_curve(y_true, y_prob, run_dir / "plots" / "roc_curve.png")
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
        >>> save_predictions_csv(y_true, y_pred, y_prob, "runs/exp01/predictions.csv")
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
        >>> plot_confusion_matrix(y_true, y_pred, "runs/exp01/plots/confusion_matrix.png")
    """
    cm = BinaryConfusionMatrix()(torch.tensor(y_pred), torch.tensor(y_true)).numpy()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: list[int], y_prob: list[float], path: str | Path) -> None:
    """Calcula y guarda la curva ROC + AUC como imagen PNG.

    Args:
        y_true: etiquetas reales (0/1).
        y_prob: probabilidad de la clase positiva.
        path: ruta destino de la imagen `.png`. El directorio padre se
            crea si no existe.

    Example:
        >>> plot_roc_curve(y_true, y_prob, "runs/exp01/plots/roc_curve.png")
    """
    fpr, tpr, _ = BinaryROC()(torch.tensor(y_prob), torch.tensor(y_true))
    roc_auc = torch.trapz(tpr, fpr).item()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr.numpy(), tpr.numpy(), label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
