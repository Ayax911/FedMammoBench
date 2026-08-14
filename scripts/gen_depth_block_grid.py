#!/usr/bin/env python3
"""Genera el bloque 6 (profundidad de descongelamiento) a partir de exp23.

`configs/exp23/exp23.ipynb` es la base de este bloque: RadImageNet + layer4
descongelada, cabeza MLP (dropout 0.3) inicializada con normal(0, 0.01),
CrossEntropyLoss, best-checkpoint por F1-macro en validación, LR único 1e-4
(sin LR diferencial, intencional), semilla 42 y W&B. El bloque 6 mantiene TODO
eso fijo y solo varía qué tan profundo se descongela el backbone:

    exp23  UNFREEZE_IDX=[7]            layer4
    exp24  UNFREEZE_IDX=[]             backbone completamente congelado
    exp25  UNFREEZE_IDX=[7,6]          layer4+layer3
    exp26  UNFREEZE_IDX=[7,6,5]        layer4+layer3+layer2
    exp27  UNFREEZE_IDX=[7,6,5,4,1,0]  todo el backbone

exp23 no se regenera (ya está editado a mano y es el template); este script
solo escribe exp24-exp27. La única celda que cambia entre las 6 notebooks del
bloque es la de parámetros (EXP_ID/RUN_NAME/UNFROZEN_DESC/UNFREEZE_IDX) — la
celda de freeze y la del optimizador ya son dinámicas sobre UNFREEZE_IDX (ver
exp22-cell-11 / exp22-cell-14 de exp23), así que no hace falta tocarlas.

Índices de backbone (nn.Sequential, ver backbone_remap en
load_radimagenet_backbone dentro del notebook): 0=conv1, 1=bn1, 4=layer1,
5=layer2, 6=layer3, 7=layer4 (2=relu/3=maxpool/8=avgpool no tienen parámetros).

Uso:
    python scripts/gen_depth_block_grid.py            # escribe configs/exp2{4..7}/
    python scripts/gen_depth_block_grid.py --dry-run  # solo lista lo que haría
    python scripts/gen_depth_block_grid.py --force    # sobreescribe si ya existen
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "exp23" / "exp23.ipynb"
PARAMS_CELL_ID = "exp22-cell-02"

EXPERIMENTS = [
    dict(exp="exp24", unfreeze_idx=[], slug="frozen",
         desc="backbone completamente congelado",
         note="Profundidad 0/4: control con el backbone totalmente congelado."),
    dict(exp="exp25", unfreeze_idx=[7, 6], slug="layer4_layer3",
         desc="última y penúltima capa (layer4 + layer3) descongeladas",
         note="Profundidad 2/4: layer4 + layer3."),
    dict(exp="exp26", unfreeze_idx=[7, 6, 5], slug="layer4_layer3_layer2",
         desc="últimas tres capas (layer4 + layer3 + layer2) descongeladas",
         note="Profundidad 3/4: layer4 + layer3 + layer2."),
    dict(exp="exp27", unfreeze_idx=[7, 6, 5, 4, 1, 0], slug="all_unfrozen",
         desc="todo el backbone descongelado",
         note="Profundidad 4/4: todo el backbone (conv1/bn1/layer1-4) entrenable."),
]


def params_cell(spec: dict) -> str:
    run_name = (
        f"{spec['exp']}_resnet50_radimagenet_mammobench_{spec['slug']}"
        "_mlp03_ce_lr1e-4_img512_bs16_wandb"
    )
    return f"""EXP_ID = "{spec['exp']}"
RUN_NAME = "{run_name}"
UNFROZEN_DESC = "{spec['desc']}"
BLOCK = 6  # bloque 6: profundidad de descongelamiento (exp23-exp27), sobre la base con seed+wandb+CE+F1-macro
# Índices de backbone (nn.Sequential) a descongelar — ver freeze cell para el mapa completo.
UNFREEZE_IDX = {spec['unfreeze_idx']!r}
HEAD = "mlp"

LR = 1e-4            # cabeza (linear1/bn1/linear2)
LR_BACKBONE = 1e-4   # backbone descongelado — igual a LR, sin LR diferencial (intencional)
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
PATIENCE = 20
IMAGE_SIZE = 512
ROTATION_DEGREES = 7

PROJECT_ROOT = Path(
    "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/"
    "Federal Learning/infraestructura federada/Federal-Learning"
)

DATASET_ROOT = Path(
    "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/"
    "c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_"
    "DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench"
)
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "fedmammobench_tompei.csv"
RADIMAGENET_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "RadImageNet-resnet50.pth"

# W&B: la API key NUNCA se pega aquí. Se lee de la variable de entorno
# WANDB_API_KEY, o de las credenciales cacheadas en ~/.netrc tras correr
# `wandb login` una vez en este workstation (a diferencia de los
# contenedores Docker del paquete, aqui el notebook SI hereda ese
# ~/.netrc porque corre directo en el host, sin aislar el filesystem).
WANDB_PROJECT = "fedmammobench"
WANDB_ENTITY = None  # None -> entity por defecto de la API key
WANDB_MODE = "online"  # "online" | "offline" | "disabled"
WANDB_TAGS = ["notebook", "centralized", "radimagenet", "mammobench", "{spec['slug']}"]

RUN_DIR = PROJECT_ROOT / "runs" / RUN_NAME
PLOTS_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RUN_DIR / "best_model.pth"
LAST_MODEL_PATH = RUN_DIR / "last_model.pth"

print(f"RUN_DIR: {{RUN_DIR}}")"""


def build_notebook(template: dict, spec: dict) -> dict:
    nb = copy.deepcopy(template)
    found = False
    for cell in nb["cells"]:
        if cell.get("id") == PARAMS_CELL_ID:
            lines = params_cell(spec).split("\n")
            cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
            cell["outputs"] = []
            cell["execution_count"] = None
            found = True
        else:
            # exp23 no está ejecutado (sin runs/exp23...), pero por si acaso
            # se ejecutó una celda suelta mientras se editaba, cada notebook
            # del bloque arranca "limpio".
            cell["outputs"] = cell.get("outputs", [])
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    if not found:
        raise RuntimeError(f"No se encontró la celda de parámetros ({PARAMS_CELL_ID}) en el template.")
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Solo lista qué se generaría.")
    parser.add_argument("--force", action="store_true", help="Sobreescribe notebooks existentes.")
    args = parser.parse_args()

    template = json.loads(TEMPLATE.read_text())

    for spec in EXPERIMENTS:
        out_dir = REPO / "configs" / spec["exp"]
        out_path = out_dir / f"{spec['exp']}.ipynb"
        print(f"{spec['exp']}: {spec['note']}")
        print(f"  UNFREEZE_IDX={spec['unfreeze_idx']} -> {out_path.relative_to(REPO)}")

        if args.dry_run:
            continue
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} ya existe (usa --force para sobreescribir).")

        nb = build_notebook(template, spec)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

    if args.dry_run:
        print("\n(dry-run: no se escribió nada)")
    else:
        print(f"\n{len(EXPERIMENTS)} notebooks escritos.")


if __name__ == "__main__":
    main()
