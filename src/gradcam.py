"""Entrypoint CLI para generación e inspección de mapas de calor Grad-CAM.

Reconstruye el pipeline de datos y modelo de forma idéntica a `src/evaluate.py` a
partir de un archivo de configuración YAML y un checkpoint ya entrenado, y genera
mapas Grad-CAM superpuestos sobre las mamografías seleccionadas.

Uso:
    $ python -m src.gradcam --config configs/exp05_fedmammobench_full_weighted.yaml \\
        --checkpoint runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt \\
        --split test --select misclassified --n-per-class 5

    $ python -m src.gradcam --config configs/exp05_fedmammobench_full_weighted.yaml \\
        --checkpoint runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt \\
        --split test --select ids --image-ids CM000042,CM000107
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image

from .checkpoint import load_checkpoint
from .config import ExperimentConfig, load_config
from .datasets.dataset import MammoBenchDataset
from .datasets.manifest import Manifest
from .datasets.split import Split
from .datasets.transform import TransformBuilder
from .interpretability import compute_gradcam, overlay_heatmap
from .models.build import build_model
from .models.heads import get_head_strategy
from .seed import set_global_seed
from .train.build import build_loss


def run_gradcam(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    split_name: str,
    select: str,
    image_ids: list[str] | None = None,
    n_per_class: int = 5,
    output_dir: str | Path | None = None,
) -> None:
    """Genera mapas Grad-CAM para imágenes seleccionadas de un split.

    Args:
        config: `ExperimentConfig` correspondiente al entrenamiento del checkpoint.
        checkpoint_path: Ruta al archivo `.pt` con los pesos del modelo.
        split_name: Nombre del split a evaluar (`"val"` o `"test"`).
        select: Modo de selección (`"ids"`, `"misclassified"` o `"sample"`).
        image_ids: Lista de identificadores de imagen (requerido si `select == "ids"`).
        n_per_class: Número de imágenes por clase / tipo de error a seleccionar.
        output_dir: Directorio de salida. Si es `None`, se usa
            `config.train.run_dir / "gradcam" / split_name`.

    Raises:
        ValueError: Si `split_name` o `select` no son válidos, si `image_ids` falta
            o si los tamaños de split y predictions.csv no coinciden.
        FileNotFoundError: Si no se encuentra el checkpoint o `predictions.csv`.
    """
    checkpoint_path = Path(checkpoint_path)

    # Fijar semilla antes de cualquier reconstrucción
    set_global_seed(config.data.seed)

    manifest = Manifest(manifest_path=config.data.manifest_path, image_root=config.data.image_root)
    split = Split(manifest=manifest)

    # Solo eval_transform -- Grad-CAM no debe aplicar aumentación
    eval_transform_builder = TransformBuilder(
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    eval_transform = eval_transform_builder.build()

    # Reconstrucción del modelo y carga de pesos
    backbone, _ = build_model(
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

    load_checkpoint(model, checkpoint_path, device=config.train.device)
    loss_spec = build_loss(config.loss.name, device=config.train.device, **config.loss.hparams)

    # Selección del DataFrame de split
    if split_name == "val":
        df = split.val_df().reset_index(drop=True)
    elif split_name == "test":
        df = split.test_df().reset_index(drop=True)
    else:
        raise ValueError(f"split_name debe ser 'val' o 'test', recibido: {split_name!r}")

    # Definición de output_dir
    if output_dir is None:
        target_output_dir = Path(config.train.run_dir) / "gradcam" / split_name
    else:
        target_output_dir = Path(output_dir)
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Selección de filas según el modo
    if select == "ids":
        if not image_ids:
            raise ValueError("image_ids es requerido cuando select='ids'")
        selected_df = df[df["ID_image"].isin(image_ids)].reset_index(drop=True)
        if len(selected_df) == 0:
            raise ValueError(f"Ninguna imagen de {image_ids} fue encontrada en el split '{split_name}'.")

    elif select == "misclassified":
        predictions_path = Path(config.train.run_dir) / split_name / "predictions.csv"
        if not predictions_path.is_file():
            raise FileNotFoundError(
                f"Archivo de predicciones no encontrado en {predictions_path}. "
                f"Ejecute la evaluación del split '{split_name}' antes de usar select='misclassified'."
            )
        predictions_df = pd.read_csv(predictions_path)
        if len(df) != len(predictions_df):
            raise ValueError(
                f"El split tiene {len(df)} filas pero predictions.csv tiene {len(predictions_df)} filas."
            )
        combined_df = pd.concat([df, predictions_df], axis=1)
        errors_df = combined_df[combined_df["y_true"] != combined_df["y_pred"]].copy()
        errors_df["confidence_gap"] = (errors_df["y_prob"] - 0.5).abs()
        errors_df = errors_df.sort_values(by="confidence_gap", ascending=False)

        fp_df = errors_df[(errors_df["y_true"] == 0) & (errors_df["y_pred"] == 1)].head(n_per_class)
        fn_df = errors_df[(errors_df["y_true"] == 1) & (errors_df["y_pred"] == 0)].head(n_per_class)
        selected_df = pd.concat([fp_df, fn_df], axis=0).reset_index(drop=True)

    elif select == "sample":
        selected_rows: list[pd.DataFrame] = []
        for label in sorted(df["label_norm"].unique()):
            selected_rows.append(df[df["label_norm"] == label].head(n_per_class))
        selected_df = pd.concat(selected_rows, axis=0).reset_index(drop=True)

    else:
        raise ValueError(
            f"select debe ser 'ids', 'misclassified' o 'sample', recibido: {select!r}"
        )

    if len(selected_df) == 0:
        print(f"No se seleccionó ninguna imagen para Grad-CAM con select={select!r}.")
        return

    # Iterar sobre las imágenes seleccionadas
    dataset = MammoBenchDataset(df=selected_df, transform=eval_transform)
    print(f"Generando Grad-CAM para {len(selected_df)} imagen(es) en {target_output_dir}...")

    for idx in range(len(selected_df)):
        tensor, _ = dataset[idx]
        row = selected_df.iloc[idx]
        result = compute_gradcam(
            model,
            tensor.unsqueeze(0),
            loss_spec,
            device=config.train.device,
        )

        raw_img = Image.open(row["abs_image_path"])
        if raw_img.mode == "F":
            arr = np.array(raw_img, dtype=np.float32)
            p1, p99 = np.percentile(arr, [1, 99])
            if p99 > p1:
                arr_clipped = np.clip(arr, p1, p99)
                arr_norm = ((arr_clipped - p1) / (p99 - p1) * 255.0).astype(np.uint8)
            else:
                arr_norm = np.zeros_like(arr, dtype=np.uint8)
            rgb_arr = np.stack([arr_norm] * 3, axis=-1)
            display_img = Image.fromarray(rgb_arr, mode="RGB")
        else:
            display_img = raw_img.convert("RGB")

        if config.data.image_size is not None:
            h, w = config.data.image_size
            display_img = display_img.resize((w, h), Image.Resampling.BILINEAR)
        elif display_img.size != (result.heatmap.shape[1], result.heatmap.shape[0]):
            display_img = display_img.resize(
                (result.heatmap.shape[1], result.heatmap.shape[0]),
                Image.Resampling.BILINEAR,
            )

        composite = overlay_heatmap(display_img, result.heatmap)
        out_filename = (
            f"{row['ID_image']}_true{row['label_norm']}_score{result.target_score:.3f}.png"
        )
        out_path = target_output_dir / out_filename
        composite.save(out_path)
        print(f"  [{idx + 1}/{len(selected_df)}] Guardado {out_path.name} (score={result.target_score:.4f})")


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Objeto con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Genera mapas de calor Grad-CAM sobre mamografías evaluadas."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Ruta al YAML del experimento con el que se entrenó el checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Ruta al checkpoint .pt a evaluar.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Split sobre el cual calcular Grad-CAM ('val' o 'test'). Default: 'test'.",
    )
    parser.add_argument(
        "--select",
        default="misclassified",
        choices=["ids", "misclassified", "sample"],
        help="Criterio de selección: 'ids', 'misclassified' o 'sample'. Default: 'misclassified'.",
    )
    parser.add_argument(
        "--image-ids",
        default=None,
        type=str,
        help="Lista de IDs de imágenes separados por coma (ej. 'CM000042,CM000107'). Usado si select='ids'.",
    )
    parser.add_argument(
        "--n-per-class",
        default=5,
        type=int,
        help="Número de imágenes por clase/error a procesar. Default: 5.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Directorio destino para guardar los overlays (default: <run_dir>/gradcam/<split>).",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada invocable al ejecutar el script directamente."""
    args = parse_args()
    config = load_config(args.config)

    image_ids: list[str] | None = None
    if args.image_ids is not None:
        image_ids = [img_id.strip() for img_id in args.image_ids.split(",") if img_id.strip()]

    run_gradcam(
        config=config,
        checkpoint_path=args.checkpoint,
        split_name=args.split,
        select=args.select,
        image_ids=image_ids,
        n_per_class=args.n_per_class,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
