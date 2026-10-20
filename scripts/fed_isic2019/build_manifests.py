"""Construye los 6 manifests por cliente de Fed-ISIC2019 + el vector `alpha` de la focal loss.

Fuente del split por cliente (H2): `train_test_split` del repo oficial de FLamby
(github.com/owkin/FLamby, flamby/datasets/fed_isic2019/dataset_creation_scripts/
train_test_split, commit edacf54d5211520583b0133d55ac39b6fda8324b). Trae, por imagen:
`image`, one-hot de las 9 columnas ISIC (MEL..UNK), `target` (0..7), `center` (0..5) y
`fold` (train/test). NO se re-deriva la heurística de dispositivos del paper.

Salida (por defecto bajo manifests/fed_isic2019/):
    <cliente>.csv   columnas: preprocessed_image_path, classification, split, patient_id
                    (+ image, center, target como contexto; Manifest las ignora)
    alpha.json      alpha = inverso de la proporción de clase sobre el TRAIN pooled
                    (los 6 clientes juntos, mismo vector en todos los nodos)
    summary.json    conteos por cliente/split

Criterios documentados (ver docs/FEDISIC2019_FLAMBY_REPRO_RESULTADOS.md):
- `split`: train/test = el split estático de FLamby. El paper no tiene val: este repo
  exige uno (Split.val_df), así que se recorta VAL_FRAC=10% del train de CADA cliente,
  estratificado por clase, seed 42. `test` queda intacto.
- `patient_id` = id de imagen (ISIC2019 no trae paciente real). Es un proxy que NO
  protege contra fuga de paciente/lesión entre splits: el script reporta cuántas
  lesiones (lesion_id de ISIC_2019_Training_Metadata.csv, si se pasa) cruzan splits.

Uso:
    .venv/bin/python scripts/fed_isic2019/build_manifests.py \
        [--split-csv RUTA|URL] [--metadata-csv ISIC_2019_Training_Metadata.csv]
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_URL = (
    "https://raw.githubusercontent.com/owkin/FLamby/edacf54d5211520583b0133d55ac39b6fda8324b/"
    "flamby/datasets/fed_isic2019/dataset_creation_scripts/train_test_split"
)
CLIENTS = {
    0: "bcn",
    1: "vidir_molemax",
    2: "vidir_modern",
    3: "rosendahl",
    4: "msk",
    5: "vienna_dias",
}
CLASS_NAMES = [
    "melanoma",
    "melanocytic_nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
    "squamous_cell_carcinoma",
]  # índice = `target` de FLamby = src.datasets.isic2019.ISIC2019_LABEL_MAP
# Tabla 8 del paper (train, test) por cliente.
PAPER_COUNTS = {0: (9930, 2483), 1: (3163, 791), 2: (2691, 672), 3: (1807, 452), 4: (655, 164), 5: (351, 88)}
VAL_FRAC = 0.10
SEED = 42
IMAGE_DIR = "ISIC_2019_Training_Input_preprocessed"


def load_split(src: str) -> pd.DataFrame:
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=60) as r:
            return pd.read_csv(io.BytesIO(r.read()))
    return pd.read_csv(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", default=SPLIT_URL)
    ap.add_argument("--metadata-csv", default=None, help="ISIC_2019_Training_Metadata.csv (opcional, para el reporte de fuga)")
    ap.add_argument("--out-dir", default="manifests/fed_isic2019")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_split(args.split_csv)
    assert len(df) == 23247, len(df)
    assert (df["UNK"] == 0).all() and df["target"].between(0, 7).all()
    df["classification"] = df["target"].map(dict(enumerate(CLASS_NAMES)))
    df["preprocessed_image_path"] = IMAGE_DIR + "/" + df["image"] + ".jpg"
    df["patient_id"] = df["image"]  # proxy, ver docstring
    df["split"] = df["fold"]

    # val = 10% del train de cada cliente, estratificado por clase.
    rng = np.random.default_rng(SEED)
    val_idx: list[int] = []
    for _, g in df[df["fold"] == "train"].groupby(["center", "target"]):
        n_val = int(round(len(g) * VAL_FRAC))
        if n_val:
            val_idx.extend(rng.choice(g.index.to_numpy(), size=n_val, replace=False).tolist())
    df.loc[val_idx, "split"] = "val"

    summary = {}
    cols = ["preprocessed_image_path", "classification", "split", "patient_id", "image", "center", "target"]
    for c, name in CLIENTS.items():
        sub = df[df["center"] == c][cols].sort_values("image")
        sub.to_csv(out / f"{name}.csv", index=False)
        n_tr_paper = int((df[(df.center == c)].fold == "train").sum())
        counts = sub["split"].value_counts().to_dict()
        summary[name] = {
            "center": c,
            "train": counts.get("train", 0),
            "val": counts.get("val", 0),
            "test": counts.get("test", 0),
            "train_plus_val": counts.get("train", 0) + counts.get("val", 0),
            "paper_train": PAPER_COUNTS[c][0],
            "paper_test": PAPER_COUNTS[c][1],
        }
        assert summary[name]["train_plus_val"] == PAPER_COUNTS[c][0] == n_tr_paper
        assert summary[name]["test"] == PAPER_COUNTS[c][1]

    # alpha: inverso de la proporción de clase sobre el train POOLED (train del paper = train+val
    # de este repo, los 6 clientes). Sin normalizar: alpha_c = N / n_c (== 1 / p_c).
    pooled_train = df[df["fold"] == "train"]
    n_c = pooled_train["target"].value_counts().sort_index()
    alpha = (len(pooled_train) / n_c).round(6).tolist()
    (out / "alpha.json").write_text(json.dumps({"classes": CLASS_NAMES, "class_counts": n_c.tolist(), "alpha": alpha}, indent=2))

    if args.metadata_csv:
        meta = pd.read_csv(args.metadata_csv)[["image", "lesion_id"]]
        m = df.merge(meta, on="image", how="left")
        m = m[m["lesion_id"].notna()]
        n_les = m.groupby("lesion_id")["split"].nunique()
        summary["_leakage_report"] = {
            "lesions_with_rows_in_more_than_one_split": int((n_les > 1).sum()),
            "lesions_total": int(len(n_les)),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("alpha:", alpha)


if __name__ == "__main__":
    main()
