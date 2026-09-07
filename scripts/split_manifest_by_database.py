"""Deriva un manifest CSV por base de datos a partir de un manifest combinado (TIFF).

No toca ni un solo píxel: lee `manifests/fedmammobench_norm_{0_1,neg1_1}.csv`
(las variantes que apuntan a las imágenes TIFF de 224x224 ya normalizadas por
`Preproccesed/preprocess_images.py`), filtra filas por `source_dataset` y
escribe un CSV por base de datos con el mismo esquema de columnas -- mismos
`preprocessed_image_path` (TIFF), mismo `split` (train/val/test), sin
regenerar ni mover una sola imagen.

Se escriben en `manifests/by_database/`, una carpeta nueva y separada a
propósito de los 4 manifests "-grouped"/"-split-grouped" que ya viven en
`manifests/` (`cdd-cesm-split-grouped.csv`, `cmmd-tompei-grouped.csv`,
`inbreast-split-grouped.csv`, `kau-bcmd-split-grouped.csv`) -- esos apuntan a
`Preprocessed_Dataset/*.jpg` (la serie de notebooks, JPG), no a los TIFF de
este pipeline. Mezclarlos sería fácil de confundir porque comparten
`source_dataset`/`patient_id`/nombre de base de datos; viven en carpetas
distintas justamente para que un `manifest_path` mal copiado falle rápido
(ruta que no existe) en vez de cargar en silencio la imagen equivocada.

Cada archivo resultante es un `Manifest` válido por sí solo (conserva
`preprocessed_image_path`, `classification`, `split`, `patient_id` y todas
las demás columnas): `src.datasets.manifest.Manifest` + `src.datasets.split.Split`
lo procesan exactamente igual que al manifest combinado, incluida la
validación anti-leakage de `Split.verify_patient_consistency()` -- se cumple
sola porque cada fila de un `patient_id` dado ya pertenece a una sola
`source_dataset`.

`ExperimentConfig.data.by_database_manifests` (`src/config.py`) apunta a
estos archivos para que `cli.run()` evalúe el mejor checkpoint por base de
datos al final del entrenamiento y grafique `test/confusion_matrix_by_database.png`
y `test/metrics_by_database.png` -- ver `src/reporting.py` y `_evaluate_by_database()`
en `src/cli.py`.

Uso (re-ejecutable en cualquier momento; sobreescribe sin preguntar):
    .venv/bin/python scripts/split_manifest_by_database.py
"""

from pathlib import Path

import pandas as pd

# Manifests TIFF combinados de origen -- ver docs/DATA_PREPARATION.md y el
# header de CLAUDE.md sobre la pipeline pre-procesada de TIFFs flotantes.
SOURCE_MANIFESTS: tuple[Path, ...] = (
    Path("manifests/fedmammobench_norm_0_1.csv"),
    Path("manifests/fedmammobench_norm_neg1_1.csv"),
)
OUTPUT_DIR = Path("manifests/by_database")


def split_manifest_by_database(source_path: Path, output_dir: Path) -> list[Path]:
    """Filtra un manifest combinado por `source_dataset` y escribe un CSV por grupo.

    Args:
        source_path: manifest CSV de origen -- debe tener columna `source_dataset`.
        output_dir: carpeta destino; se crea si no existe.

    Returns:
        list[Path]: rutas de los CSVs escritos, uno por valor único de `source_dataset`.

    Raises:
        ValueError: si `source_path` no tiene columna `source_dataset`.

    Example:
        >>> split_manifest_by_database(
        ...     Path("manifests/fedmammobench_norm_neg1_1.csv"),
        ...     Path("manifests/by_database"),
        ... )
        [PosixPath('manifests/by_database/cdd-cesm_norm_neg1_1.csv'), ...]
    """
    df = pd.read_csv(source_path)
    if "source_dataset" not in df.columns:
        raise ValueError(f"{source_path} no tiene columna 'source_dataset' para agrupar.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # "fedmammobench_norm_neg1_1.csv" -> "norm_neg1_1" -- mismo sufijo que ya
    # usan los nombres de carpeta de imágenes (norm_0_1/, norm_neg1_1/), para
    # que el nombre del archivo diga a qué normalización apunta sin abrirlo.
    suffix = source_path.stem.replace("fedmammobench_", "")

    written: list[Path] = []
    for db_name, group in df.groupby("source_dataset"):
        out_path = output_dir / f"{db_name}_{suffix}.csv"
        group.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def main() -> None:
    """Corre `split_manifest_by_database()` sobre ambas variantes TIFF (norm_0_1 y norm_neg1_1)."""
    for source_path in SOURCE_MANIFESTS:
        written = split_manifest_by_database(source_path, OUTPUT_DIR)
        print(f"{source_path} -> {len(written)} manifest(s) por base de datos:")
        for path in written:
            print(f"  {path}")


if __name__ == "__main__":
    main()
