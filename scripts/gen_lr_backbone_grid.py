#!/usr/bin/env python3
"""Genera los notebooks de los bloques 4 y 5 (LR diferencial de backbone).

Toma `configs/exp09/exp09.ipynb` como plantilla —los nueve notebooks de los
bloques 1-3 comparten estructura celda a celda— y sustituye solo las celdas que
cambian entre experimentos:

    [0]  imports            -> + os, random (para el sembrado)
    [2]  parámetros         -> EXP_ID/RUN_NAME/SEED/LR/LR_BACKBONE/...
    [3]  sembrado global    -> celda NUEVA, insertada tras el device
    [8]  DataLoaders        -> generator sembrado para el shuffle
    [9]  Classifier         -> variante lineal o MLP
    [10] instancia          -> re-sembrado antes de inicializar la cabeza
    [12] congelamiento      -> none / layer4 / layer4+layer3 / +layer2
    [15] optimizador        -> param groups con LR distinto por grupo
    [20] summary            -> registra seed, lr_head y lr_backbone

Los índices de arriba son los del notebook YA generado (con la celda de
sembrado insertada); la plantilla tiene una celda menos.

Uso:
    python scripts/gen_lr_backbone_grid.py            # escribe configs/expNN/
    python scripts/gen_lr_backbone_grid.py --dry-run  # solo lista lo que haría
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "exp09" / "exp09.ipynb"

SEED = 42

# --- índices de celda EN LA PLANTILLA (antes de insertar el sembrado) --------
C_IMPORTS = 0
C_DEVICE = 1
C_PARAMS = 2
C_LOADERS = 7
C_CLASSIFIER = 8
C_CLS_INSTANCE = 9
C_FREEZE = 11
C_OPTIM = 14
C_SUMMARY = 19

# --- definición de los bloques ----------------------------------------------
# unfreeze: none | layer4 | layer4_layer3 | layer4_layer3_layer2
# head:     linear | mlp (con dropout)
# lr_backbone: None cuando el backbone está congelado del todo.
EXPERIMENTS = [
    # Bloque 4 — barrido de LR_BACKBONE. Cabeza lineal y layer4+layer3 fijos,
    # de modo que la única variable sea el LR del backbone. exp10 y exp11 son
    # los anclajes: replican exp07 y exp09 pero ya sembrados.
    dict(exp="exp10", block=4, unfreeze="none", head="linear", dropout=None,
         lr_backbone=None, slug="frozen_linearhead_seed42",
         note="Control: linear probe con backbone congelado (ancla con exp07, ahora sembrado)."),
    dict(exp="exp11", block=4, unfreeze="layer4_layer3", head="linear", dropout=None,
         lr_backbone=1e-3, slug="l4l3_linearhead_lrbb1e-3",
         note="Control: LR único 1e-3, el comportamiento actual (ancla con exp09, ahora sembrado)."),
    dict(exp="exp12", block=4, unfreeze="layer4_layer3", head="linear", dropout=None,
         lr_backbone=3e-4, slug="l4l3_linearhead_lrbb3e-4",
         note="LR diferencial suave (ratio 3x)."),
    dict(exp="exp13", block=4, unfreeze="layer4_layer3", head="linear", dropout=None,
         lr_backbone=1e-4, slug="l4l3_linearhead_lrbb1e-4",
         note="LR diferencial estándar (ratio 10x)."),
    dict(exp="exp14", block=4, unfreeze="layer4_layer3", head="linear", dropout=None,
         lr_backbone=3e-5, slug="l4l3_linearhead_lrbb3e-5",
         note="LR diferencial conservador (ratio 33x)."),
    dict(exp="exp15", block=4, unfreeze="layer4_layer3", head="linear", dropout=None,
         lr_backbone=1e-5, slug="l4l3_linearhead_lrbb1e-5",
         note="LR diferencial muy conservador (ratio 100x)."),

    # Bloque 5 — profundidad x cabeza, ya con un LR de backbone sano (1e-4).
    # La cuarta celda de la rejilla 2x2 (l4+l3 x lineal) es exp13.
    dict(exp="exp16", block=5, unfreeze="layer4", head="linear", dropout=None,
         lr_backbone=1e-4, slug="l4_linearhead_lrbb1e-4",
         note="Profundidad: solo layer4, cabeza lineal."),
    dict(exp="exp17", block=5, unfreeze="layer4", head="mlp", dropout=0.3,
         lr_backbone=1e-4, slug="l4_mlp03_lrbb1e-4",
         note="Profundidad: solo layer4, cabeza MLP dropout 0.3."),
    dict(exp="exp18", block=5, unfreeze="layer4_layer3", head="mlp", dropout=0.3,
         lr_backbone=1e-4, slug="l4l3_mlp03_lrbb1e-4",
         note="Profundidad: layer4+layer3, cabeza MLP dropout 0.3."),
    dict(exp="exp19", block=5, unfreeze="layer4_layer3_layer2", head="linear", dropout=None,
         lr_backbone=1e-4, slug="l4l3l2_linearhead_lrbb1e-4",
         note="Extiende el eje de profundidad un bloque más: layer4+layer3+layer2."),
]

UNFREEZE_DESC = {
    "none": "backbone completamente congelado",
    "layer4": "solo la última capa (layer4) descongelada",
    "layer4_layer3": "última y penúltima capa (layer4 + layer3) descongeladas",
    "layer4_layer3_layer2": "últimas tres capas (layer4 + layer3 + layer2) descongeladas",
}

# backbone[5]=layer2, backbone[6]=layer3, backbone[7]=layer4 (ver el remap de
# load_radimagenet_backbone en la celda correspondiente del notebook).
UNFREEZE_IDX = {
    "none": [],
    "layer4": [7],
    "layer4_layer3": [7, 6],
    "layer4_layer3_layer2": [7, 6, 5],
}


def cell(source: str) -> dict:
    """Celda de código sin ejecutar, en el formato que usa la plantilla."""
    lines = source.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]],
    }


def imports_cell() -> str:
    return """
import json
import os
import random

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm.auto import tqdm
"""


def seeding_cell() -> str:
    return f"""
SEED = {SEED}


def seed_everything(seed=SEED):
    \"\"\"Fija todas las fuentes de aleatoriedad del notebook.

    Cubre la init de la cabeza, el shuffle del DataLoader y la
    RandomRotation del train_transform (que usa el RNG global de torch
    porque el DataLoader corre con num_workers=0).
    \"\"\"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo en cuDNN: reproducible a costa de algo de velocidad.
    # Con benchmark=True cuDNN elige algoritmos según el hardware y la
    # misma semilla deja de dar el mismo resultado entre máquinas.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
print(f"Semilla fijada: {{SEED}} (cudnn.deterministic=True, benchmark=False)")
"""


def _fmt_lr(value: float | None) -> str:
    """1e-4 en vez de 0.0001, para que la celda se lea igual que `LR = 1e-3`."""
    if value is None:
        return "None"
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e-{int(exponent[1:])}" if exponent[0] == "-" else f"{mantissa}e{int(exponent[1:])}"


def params_cell(spec: dict) -> str:
    run_name = f"{spec['exp']}_resnet50_radimagenet_mammobench_{spec['slug']}"
    lines = [
        f'# {spec["note"]}',
        f'EXP_ID = "{spec["exp"]}"',
        f'RUN_NAME = "{run_name}"',
        f'UNFROZEN_DESC = "{UNFREEZE_DESC[spec["unfreeze"]]}"',
        f'BLOCK = {spec["block"]}',
        f'HEAD = "{spec["head"]}"',
        "",
    ]
    if spec["head"] == "mlp":
        lines.append(f'DROPOUT = {spec["dropout"]}')
    lines += [
        "# LR de la cabeza (init aleatoria: necesita pasos grandes).",
        "LR = 1e-3",
        "# LR de los bloques descongelados del backbone. None = backbone",
        "# congelado, no hay grupo de parámetros que optimizar.",
        f"LR_BACKBONE = {_fmt_lr(spec['lr_backbone'])}",
        "WEIGHT_DECAY = 1e-4",
        "BATCH_SIZE = 32",
        "NUM_EPOCHS = 100",
        "PATIENCE = 20",
        "IMAGE_SIZE = 224",
        "ROTATION_DEGREES = 10",
        "",
        "PROJECT_ROOT = Path(",
        '    "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/"',
        '    "Federal Learning/infraestructura federada/Federal-Learning"',
        ")",
        "",
        "DATASET_ROOT = Path(",
        '    "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/"',
        '    "c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_"',
        '    "DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench"',
        ")",
        'MANIFEST_PATH = PROJECT_ROOT / "manifests" / "fedmammobench.csv"',
        'RADIMAGENET_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "RadImageNet-resnet50.pth"',
        "",
        'RUN_DIR = PROJECT_ROOT / "runs" / RUN_NAME',
        'PLOTS_DIR = RUN_DIR / "plots"',
        "RUN_DIR.mkdir(parents=True, exist_ok=True)",
        "PLOTS_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        'BEST_MODEL_PATH = RUN_DIR / "best_model.pth"',
        'LAST_MODEL_PATH = RUN_DIR / "last_model.pth"',
        "",
        'print(f"RUN_DIR: {RUN_DIR}")',
        'print(f"LR cabeza: {LR} | LR backbone: {LR_BACKBONE}")',
    ]
    return "\n".join(lines)


def loaders_cell() -> str:
    return """
# El shuffle del DataLoader consume el RNG global de torch; con un generator
# propio sembrado el orden de los batches es el mismo en cada ejecución
# aunque se re-ejecuten celdas de arriba en otro orden.
_loader_generator = torch.Generator()
_loader_generator.manual_seed(SEED)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train = DataLoader(
    train_processed, batch_size=BATCH_SIZE, shuffle=True,
    generator=_loader_generator, worker_init_fn=_seed_worker,
)
validation = DataLoader(validation_processed, batch_size=BATCH_SIZE, shuffle=False)
test = DataLoader(test_processed, batch_size=BATCH_SIZE, shuffle=False)
"""


LINEAR_CLASSIFIER = """
class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.linear = nn.Linear(2048, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=None)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)
"""

MLP_CLASSIFIER = """
class Classifier(nn.Module):
    def __init__(self, num_class, dropout=DROPOUT):
        super().__init__()

        self.linear1    = nn.Linear(2048, 1024)
        self.bn1        = nn.BatchNorm1d(1024)
        self.act1       = nn.LeakyReLU(0.2)
        self.dropout1   = nn.Dropout(dropout)

        self.linear2 = nn.Linear(1024, 512)
        self.bn2        = nn.BatchNorm1d(512)
        self.act2       = nn.LeakyReLU(0.2)
        self.dropout2   = nn.Dropout(dropout)

        self.linear3 = nn.Linear(512, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=None)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)
"""


def instance_cell(spec: dict) -> str:
    # Re-sembrar aquí hace que los pesos iniciales de la cabeza no dependan de
    # cuántas celdas se hayan ejecutado antes (Jupyter permite ejecutarlas en
    # cualquier orden, y el RNG global es estado compartido).
    head = "seed_everything()  # pesos iniciales de la cabeza reproducibles\n"
    if spec["head"] == "linear":
        return head + """
classifier = Classifier(num_class=2).to(device)
print("Clasificador lineal 2048->2")
print(f"Parámetros del clasificador: {sum(p.numel() for p in classifier.parameters()):,}")
"""
    return head + """
classifier = Classifier(num_class=2, dropout=DROPOUT).to(device)
print(f"Clasificador MLP 2048->1024->512->2, dropout={DROPOUT}")
print(f"Parámetros del clasificador: {sum(p.numel() for p in classifier.parameters()):,}")
"""


def freeze_cell(spec: dict) -> str:
    idx = UNFREEZE_IDX[spec["unfreeze"]]
    lines = []
    if idx:
        lines.append(
            "# backbone[7]=layer4, backbone[6]=layer3, backbone[5]=layer2"
            " (ver backbone_remap arriba)."
        )
    lines += [
        "for param in backbone.parameters():",
        "    param.requires_grad = False",
    ]
    for i in idx:
        lines += [
            f"for param in backbone[{i}].parameters():",
            "    param.requires_grad = True",
        ]
    lines += [
        "",
        "",
        "trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)",
        "total = sum(p.numel() for p in backbone.parameters())",
        "pct = 100 * trainable / total",
        'print(f"[{EXP_ID}] Backbone trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")',
    ]
    return "\n".join(lines)


def optim_cell() -> str:
    return """
criterion = nn.CrossEntropyLoss()

# LR diferencial: la cabeza parte de una init aleatoria y necesita 1e-3; el
# backbone trae pesos RadImageNet ya buenos y con ese mismo LR los destruye en
# 1-3 épocas (es lo que pasó en exp02/03/05/06/08/09). Grupos separados.
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
head_params = [p for p in model.classifier.parameters() if p.requires_grad]

param_groups = [{"params": head_params, "lr": LR, "name": "head"}]
if backbone_params:
    assert LR_BACKBONE is not None, (
        "Hay parámetros de backbone descongelados pero LR_BACKBONE es None."
    )
    param_groups.append({"params": backbone_params, "lr": LR_BACKBONE, "name": "backbone"})
elif LR_BACKBONE is not None:
    raise ValueError(
        "LR_BACKBONE está fijado pero el backbone está completamente congelado."
    )

optimizer = torch.optim.AdamW(param_groups, lr=LR, weight_decay=WEIGHT_DECAY)
# CosineAnnealingLR guarda un base_lr por grupo y decae los dos en paralelo,
# así que el ratio cabeza/backbone se mantiene casi todo el run. Solo se
# comprime en las últimas épocas, cuando el grupo del backbone toca el suelo
# absoluto eta_min=1e-7 antes que el de la cabeza; a esa altura los dos LR son
# ~1e-7 y ya no mueven los pesos, así que no afecta al resultado.
sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

print(f"Optimizer: AdamW(weight_decay={WEIGHT_DECAY})")
for g in optimizer.param_groups:
    n = sum(p.numel() for p in g["params"])
    print(f"  grupo {g['name']:9} lr={g['lr']:<8} params={n:,}")
"""


def summary_cell() -> str:
    return """
summary = {
    "experiment": EXP_ID,
    "block": BLOCK,
    "run_name": RUN_NAME,
    "unfrozen": UNFROZEN_DESC,
    "seed": SEED,
    "hyperparams": {
        "head": HEAD,
        "dropout": DROPOUT if HEAD == "mlp" else None,
        "lr": LR,
        "lr_head": LR,
        "lr_backbone": LR_BACKBONE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "image_size": IMAGE_SIZE,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "augmentation": f"RandomRotation({ROTATION_DEGREES})",
        "seed": SEED,
    },
    "epochs_run": len(train_loss_history),
    "best_epoch": best_epoch,
    "best_val_loss": float(min(val_loss_history)),
    "final_train_loss": float(train_loss_history[-1]),
    "final_val_loss": float(val_loss_history[-1]),
    "test": {
        "accuracy": float(test_acc),
        "auc": float(test_auc),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    },
    "checkpoints": {
        "best": str(BEST_MODEL_PATH),
        "last": str(LAST_MODEL_PATH),
    },
}

metrics_path = RUN_DIR / "metrics.json"
metrics_path.write_text(json.dumps(summary, indent=2))
print(f"Resumen guardado en: {metrics_path}")
print(json.dumps(summary["test"], indent=2))
"""


def build_notebook(template: dict, spec: dict) -> dict:
    cells = [copy.deepcopy(c) for c in template["cells"]]

    # Limpia salidas y contadores de la plantilla: estos notebooks no se han
    # ejecutado todavía.
    for c in cells:
        c["execution_count"] = None
        c["outputs"] = []
        c["metadata"] = {}

    cells[C_IMPORTS] = cell(imports_cell())
    cells[C_PARAMS] = cell(params_cell(spec))
    cells[C_LOADERS] = cell(loaders_cell())
    cells[C_CLASSIFIER] = cell(
        LINEAR_CLASSIFIER if spec["head"] == "linear" else MLP_CLASSIFIER
    )
    cells[C_CLS_INSTANCE] = cell(instance_cell(spec))
    cells[C_FREEZE] = cell(freeze_cell(spec))
    cells[C_OPTIM] = cell(optim_cell())
    cells[C_SUMMARY] = cell(summary_cell())

    # El sembrado va justo después de fijar el device y antes de cualquier
    # construcción de modelo o DataLoader.
    cells.insert(C_DEVICE + 1, cell(seeding_cell()))

    return {
        "cells": cells,
        # metadata mínima: la plantilla arrastra un blob enorme de estado de
        # widgets de tqdm que no tiene sentido copiar.
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": template.get("nbformat", 4),
        "nbformat_minor": template.get("nbformat_minor", 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="no escribe nada")
    ap.add_argument("--force", action="store_true", help="sobrescribe si ya existe")
    args = ap.parse_args()

    template = json.loads(TEMPLATE.read_text())

    for spec in EXPERIMENTS:
        out_dir = REPO / "configs" / spec["exp"]
        out_path = out_dir / f"{spec['exp']}.ipynb"
        rel = out_path.relative_to(REPO)

        if out_path.exists() and not args.force:
            print(f"[skip]  {rel} (ya existe; usa --force)")
            continue
        if args.dry_run:
            print(f"[dry]   {rel}  <- bloque {spec['block']}, {spec['slug']}")
            continue

        nb = build_notebook(template, spec)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"[write] {rel}  <- bloque {spec['block']}, {spec['slug']}")


if __name__ == "__main__":
    main()
