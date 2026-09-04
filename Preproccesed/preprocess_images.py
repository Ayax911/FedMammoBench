"""Preprocesamiento standalone de imagenes JPG organizadas en subcarpetas.
Uso:
    python preprocess_images.py --input_dir /ruta/a/BasesDeDatos --output_dir /ruta/salida
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)
TIFF_COMPRESSION = "tiff_deflate"  # sin perdida (lossless)


def preprocess_image(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga un .jpg, lo pasa a escala de grises, resize 224x224 y normaliza.

    Args:
        image_path: Ruta al archivo .jpg de entrada.

    Returns:
        arr_01: array float32 shape [224, 224], rango [0, 1].
        arr_pm1: array float32 shape [224, 224], rango [-1, 1].
    """
    image = Image.open(image_path).convert("L").resize(IMAGE_SIZE, Image.BILINEAR) # Se convierte a un canal con PIL, parametro "L"
    arr_01 = np.asarray(image, dtype=np.float32) / 255.0 # Convierte la imagen a float y la pasa de 0 - a 255
    arr_pm1 = (arr_01 - 0.5) / 0.5 # Pasa de 0-1 a [-0.5,0.5] y al dividir por dos pasa a [-1,1]
    return arr_01, arr_pm1


def save_as_tiff(array: np.ndarray, path: Path) -> None:
    """Guarda un array float32 2D como TIFF de 32 bits (modo 'F'), sin perdida."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="F").save(path, format="TIFF", compression=TIFF_COMPRESSION)


def process_directory(input_dir: Path, output_dir: Path) -> None:
    """Procesa recursivamente todos los .jpg/.jpeg bajo input_dir.

    Replica la ruta relativa completa (subcarpetas + nombre de archivo,
    con extension .tiff) bajo dos raices de salida separadas:
    output_dir/norm_0_1/... y output_dir/norm_neg1_1/...
    """
    out_01_root = output_dir / "norm_0_1"
    out_pm1_root = output_dir / "norm_neg1_1"

    jpg_paths = sorted(list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.jpeg")))
    if not jpg_paths:
        print(f"No se encontraron .jpg/.jpeg en {input_dir} (busqueda recursiva)")
        return

    for path in jpg_paths:
        rel_path = path.relative_to(input_dir).with_suffix(".tiff")
        arr_01, arr_pm1 = preprocess_image(path)
        save_as_tiff(arr_01, out_01_root / rel_path)
        save_as_tiff(arr_pm1, out_pm1_root / rel_path)

    print(f"Procesadas {len(jpg_paths)} imagenes ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, 1 canal).")
    print(f"  [0,1]  -> {out_01_root}")
    print(f"  [-1,1] -> {out_pm1_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directorio raiz que contiene las carpetas de bases de datos con los .jpg",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directorio raiz donde se crean norm_0_1/ y norm_neg1_1/ con la misma estructura",
    )
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()