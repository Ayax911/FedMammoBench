#!/usr/bin/env python3
"""Genera el bloque 7 (resolucion 256px) a partir de exp28.

`configs/exp28/exp28.ipynb` es la base de este bloque: una copia de exp23
(bloque 6 — RadImageNet + layer4 descongelada, cabeza MLP dropout 0.3, CE,
best-checkpoint por F1-macro, LR unico 1e-4, semilla 42, W&B) con tres cambios:

    - IMAGE_SIZE 512 -> 256
    - MANIFEST_PATH apunta a manifests/fedmammobench.csv en vez de
      fedmammobench_tompei.csv (borrado del repo en ee54a6b; exp23-exp27
      quedaron apuntando a un archivo que ya no existe, no propagar eso aqui)
    - agrega el desglose de test por source_dataset (cmmd/kau-bcmd/cdd-cesm/
      inbreast): funcion `evaluate_per_dataset` + metricas + graficas, escrito
      en RUN_DIR/per_dataset/ (separado de RUN_DIR/plots/, que sigue siendo
      solo el test global)

El bloque 7 mantiene TODO eso fijo y solo varia la profundidad de
descongelamiento, exactamente la misma grilla que el bloque 6:

    exp28  UNFREEZE_IDX=[7]            layer4              (== exp23, img256)
    exp29  UNFREEZE_IDX=[]             backbone congelado  (== exp24, img256)
    exp30  UNFREEZE_IDX=[7,6]          layer4+layer3       (== exp25, img256)
    exp31  UNFREEZE_IDX=[7,6,5]        layer4+layer3+layer2 (== exp26, img256)
    exp32  UNFREEZE_IDX=[7,6,5,4,1,0]  todo el backbone    (== exp27, img256)

exp28 no se regenera (ya esta editado a mano y es el template); este script
solo escribe exp29-exp32. La unica celda que cambia entre las 5 notebooks del
bloque es la de parametros (EXP_ID/RUN_NAME/UNFROZEN_DESC/UNFREEZE_IDX) — la
celda de freeze y la del optimizador ya son dinamicas sobre UNFREEZE_IDX
(heredado de exp23/gen_depth_block_grid.py), asi que no hace falta tocarlas;
tampoco las celdas de evaluacion per-dataset, que no dependen de la profundidad.

Indices de backbone (nn.Sequential, ver backbone_remap en
load_radimagenet_backbone dentro del notebook): 0=conv1, 1=bn1, 4=layer1,
5=layer2, 6=layer3, 7=layer4 (2=relu/3=maxpool/8=avgpool no tienen parametros).

Uso:
    python scripts/gen_img256_block_grid.py            # escribe configs/exp2{9}/, exp3{0,1,2}/
    python scripts/gen_img256_block_grid.py --dry-run  # solo lista lo que haria
    python scripts/gen_img256_block_grid.py --force    # sobreescribe si ya existen
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "exp28" / "exp28.ipynb"
PARAMS_CELL_ID = "exp22-cell-02"

EXPERIMENTS = [
    dict(exp="exp29", unfreeze_idx=[], slug="frozen",
         desc="backbone completamente congelado",
         note="Profundidad 0/4: control con el backbone totalmente congelado."),
    dict(exp="exp30", unfreeze_idx=[7, 6], slug="layer4_layer3",
         desc="última y penúltima capa (layer4 + layer3) descongeladas",
         note="Profundidad 2/4: layer4 + layer3."),
    dict(exp="exp31", unfreeze_idx=[7, 6, 5], slug="layer4_layer3_layer2",
         desc="últimas tres capas (layer4 + layer3 + layer2) descongeladas",
         note="Profundidad 3/4: layer4 + layer3 + layer2."),
    dict(exp="exp32", unfreeze_idx=[7, 6, 5, 4, 1, 0], slug="all_unfrozen",
         desc="todo el backbone descongelado",
         note="Profundidad 4/4: todo el backbone (conv1/bn1/layer1-4) entrenable."),
]


def params_cell(spec: dict) -> str:
    run_name = (
        f"{spec['exp']}_resnet50_radimagenet_mammobench_{spec['slug']}"
        "_mlp03_ce_lr1e-4_img256_bs16_wandb"
    )
    return f"""EXP_ID = "{spec['exp']}"
RUN_NAME = "{run_name}"
UNFROZEN_DESC = "{spec['desc']}"
BLOCK = 7  # bloque 7: resolucion de imagen (256px) sobre la base del bloque 6 (exp23-exp27, img512)
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
IMAGE_SIZE = 256
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
# fedmammobench_tompei.csv (usado por exp23-exp27) fue borrado del repo en
# ee54a6b y reemplazado por un fedmammobench.csv corregido — Block 7 apunta
# al manifest vigente, no al que exp23-exp27 dejaron hardcodeado.
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "fedmammobench.csv"
RADIMAGENET_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "RadImageNet-resnet50.pth"

# W&B: la API key NUNCA se pega aquí. Se lee de la variable de entorno
# WANDB_API_KEY, o de las credenciales cacheadas en ~/.netrc tras correr
# `wandb login` una vez en este workstation (a diferencia de los
# contenedores Docker del paquete, aqui el notebook SI hereda ese
# ~/.netrc porque corre directo en el host, sin aislar el filesystem).
WANDB_PROJECT = "fedmammobench"
WANDB_ENTITY = None  # None -> entity por defecto de la API key
WANDB_MODE = "online"  # "online" | "offline" | "disabled"
WANDB_TAGS = ["notebook", "centralized", "radimagenet", "mammobench", "{spec['slug']}", "img256"]

RUN_DIR = PROJECT_ROOT / "runs" / RUN_NAME
PLOTS_DIR = RUN_DIR / "plots"
# Métricas/gráficas globales (todo el test set) van en PLOTS_DIR y metrics.json,
# igual que en el bloque 6. El desglose por source_dataset va aparte, en su
# propia carpeta, para no mezclar los dos niveles de granularidad.
PER_DATASET_DIR = RUN_DIR / "per_dataset"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PER_DATASET_DIR.mkdir(parents=True, exist_ok=True)

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
            # exp28 no está ejecutado, pero por si acaso se ejecutó una celda
            # suelta mientras se editaba, cada notebook del bloque arranca "limpio".
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
