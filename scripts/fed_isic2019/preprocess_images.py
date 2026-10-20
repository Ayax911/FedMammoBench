"""Preprocesa ISIC2019 como FLamby (resize_images.py + color_constancy.py), leyendo del .zip.

Para cada imagen de los manifests de `manifests/fed_isic2019/` (23 247 de las 25 331 del
challenge; las demás no tienen centro): lado corto -> 224 px (aspect ratio preservado,
BILINEAR) + color constancy (shade-of-gray, power=6, método de Arora, igual que FLamby)
-> JPG en `<out>/ISIC_2019_Training_Input_preprocessed/<id>.jpg`.

Único cambio vs FLamby: el resultado de color constancy se recorta a [0, 255] antes de
castear a uint8 (FLamby castea float32 -> uint8 sin clip; el overflow es comportamiento
indefinido de numpy y envuelve píxeles saturados). No requiere cv2.

Lee del zip en streaming: no hace falta descomprimir los ~9 GB.

Uso:
    .venv/bin/python scripts/fed_isic2019/preprocess_images.py \
        --zip /home/akira/data/fed_isic2019/ISIC_2019_Training_Input.zip \
        --out /home/akira/data/fed_isic2019
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


def _init_worker(zip_path: str) -> None:
    global _zf
    _zf = zipfile.ZipFile(zip_path)


def process(args: tuple[str, str, str]) -> str:
    zip_path, member, out_path = args
    if Path(out_path).exists():
        return out_path
    global _zf
    if _zf is None:
        _zf = zipfile.ZipFile(zip_path)
    img = Image.open(io.BytesIO(_zf.read(member))).convert("RGB")
    ratio = float(SZ) / min(img.size)
    img = img.resize(tuple(int(x * ratio) for x in img.size), resample=Image.BILINEAR)
    Image.fromarray(color_constancy(np.array(img))).save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-dir", default="manifests/fed_isic2019")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="solo las primeras N imágenes (smoke)")
    args = ap.parse_args()

    ids = sorted(
        {i for p in Path(args.manifest_dir).glob("*.csv") for i in pd.read_csv(p)["image"]}
    )
    if args.limit:
        ids = ids[: args.limit]
    out_dir = Path(args.out) / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        names = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".jpg")}
    missing = [i for i in ids if i not in names]
    if missing:
        raise SystemExit(f"{len(missing)} imágenes del manifest no están en el zip, ej. {missing[:3]}")
    jobs = [(args.zip, names[i], str(out_dir / f"{i}.jpg")) for i in ids]
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.zip,)) as ex:
        for n, _ in enumerate(ex.map(process, jobs, chunksize=32), 1):
            if n % 1000 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print(f"listo: {len(jobs)} imágenes en {out_dir}")


if __name__ == "__main__":
    main()
