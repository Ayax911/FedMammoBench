"""Preprocesa ISIC2019 como FLamby (resize_images.py + color_constancy.py), leyendo del .zip o de
una carpeta ya extraída.

Para cada imagen de los manifests de `manifests/fed_isic2019/` (23 247 de las 25 331 del
challenge; las demás no tienen centro): lado corto -> 224 px (aspect ratio preservado,
BILINEAR) + color constancy (shade-of-gray, power=6, método de Arora, igual que FLamby)
-> JPG en `<out>/ISIC_2019_Training_Input_preprocessed/<id>.jpg`.

Único cambio vs FLamby: el resultado de color constancy se recorta a [0, 255] antes de
castear a uint8 (FLamby castea float32 -> uint8 sin clip; el overflow es comportamiento
indefinido de numpy y envuelve píxeles saturados). No requiere cv2.

Dos modos de entrada, mutuamente excluyentes:
  --zip         lee del zip en streaming (no hace falta descomprimir los ~9 GB). Modo original.
  --image-dir   lee de una carpeta con los JPG ya extraídos (ej. `ISIC_2019_Training_Input/`
                que deja `flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py`
                tras su propio unzip+rm del zip -- en ese flujo el zip ya no existe cuando este
                script corre, así que --image-dir es la única opción).

Uso:
    .venv/bin/python scripts/fed_isic2019/preprocess_images.py \
        --zip /ruta/a/fed_isic2019/ISIC_2019_Training_Input.zip \
        --out /ruta/a/fed_isic2019

    .venv/bin/python scripts/fed_isic2019/preprocess_images.py \
        --image-dir /ruta/a/fed_isic2019/ISIC_2019_Training_Input \
        --out /ruta/a/fed_isic2019_fedmammobench
"""

import argparse
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

OUT_SUBDIR = "ISIC_2019_Training_Input_preprocessed"
SZ = 224


def color_constancy(img: np.ndarray, power: int = 6) -> np.ndarray:
    img = img.astype("float32")
    rgb_vec = np.power(np.mean(np.power(img, power), (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    return np.clip(img * rgb_vec, 0, 255).astype("uint8")


_zf = None


def _init_worker(zip_path: str | None) -> None:
    global _zf
    if zip_path is not None:
        _zf = zipfile.ZipFile(zip_path)


def process(args: tuple[str | None, str, str]) -> str:
    zip_path, member, out_path = args
    if Path(out_path).exists():
        return out_path
    if zip_path is not None:
        global _zf
        if _zf is None:
            _zf = zipfile.ZipFile(zip_path)
        img = Image.open(io.BytesIO(_zf.read(member))).convert("RGB")
    else:
        img = Image.open(member).convert("RGB")
    ratio = float(SZ) / min(img.size)
    img = img.resize(tuple(int(x * ratio) for x in img.size), resample=Image.BILINEAR)
    Image.fromarray(color_constancy(np.array(img))).save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--image-dir", default=None, help="carpeta con JPGs ya extraídos (alternativa a --zip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-dir", default="manifests/fed_isic2019")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="solo las primeras N imágenes (smoke)")
    args = ap.parse_args()

    if bool(args.zip) == bool(args.image_dir):
        raise SystemExit("pasa exactamente uno de --zip o --image-dir")

    ids = sorted(
        {i for p in Path(args.manifest_dir).glob("*.csv") for i in pd.read_csv(p)["image"]}
    )
    if args.limit:
        ids = ids[: args.limit]
    out_dir = Path(args.out) / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.zip:
        with zipfile.ZipFile(args.zip) as zf:
            names = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".jpg")}
        source = args.zip
    else:
        names = {p.stem: str(p) for p in Path(args.image_dir).glob("*.jpg")}
        source = None

    missing = [i for i in ids if i not in names]
    if missing:
        where = "zip" if args.zip else "carpeta"
        raise SystemExit(f"{len(missing)} imágenes del manifest no están en el {where}, ej. {missing[:3]}")
    jobs = [(source, names[i], str(out_dir / f"{i}.jpg")) for i in ids]
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(source,)) as ex:
        for n, _ in enumerate(ex.map(process, jobs, chunksize=32), 1):
            if n % 1000 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print(f"listo: {len(jobs)} imágenes en {out_dir}")


if __name__ == "__main__":
    main()
