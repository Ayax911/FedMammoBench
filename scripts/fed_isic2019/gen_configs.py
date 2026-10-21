"""Genera los 7 experimentos del grid final de Fed-ISIC2019 (server + 6 nodos cada uno).

Valores de docs/FEDISIC2019_FLAMBY_REPRO.md §4.3 (Tablas 10-16 del paper de FLamby).
Salida: configs/federated/fedisic2019_<estrategia>_r<rounds>/{server.yaml,node_<cliente>.yaml}.

Los hparams finos de las augmentations (`random_scale_range`, `brightness`/`contrast`,
`affine_shear`, `coarse_dropout_scale`) y el `crop_size: [200, 200]` de abajo están
verificados contra el código real de FLamby (`flamby/datasets/fed_isic2019/dataset.py`,
clon oficial de github.com/owkin/FLamby), no inventados -- el paper (Apéndice H) solo
nombra las técnicas, nunca da los números.

Uso:
    .venv/bin/python scripts/fed_isic2019/gen_configs.py [--image-root /ruta/a/fed_isic2019]
"""

import argparse
import json
from pathlib import Path

CLIENTS = ["bcn", "vidir_molemax", "vidir_modern", "rosendahl", "msk", "vienna_dias"]
ALPHA = json.loads(Path("manifests/fed_isic2019/alpha.json").read_text())["alpha"]

# nombre -> (hparams de estrategia, lr local, rounds)
CELLS = {
    "fedavg": ({}, 0.01, 10),
    "fedprox": ({"proximal_mu": 0.001}, 0.01, 10),
    "fedadagrad": ({"eta": 0.0316, "eta_l": 0.01, "tau": 1e-8}, 0.01, 10),
    "fedadam": ({"eta": 0.0032, "eta_l": 0.01, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-8}, 0.01, 10),
    "fedyogi": ({"eta": 0.0032, "eta_l": 0.01, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-8}, 0.01, 10),
    # Cyclic: 1 cliente por ronda de flwr; una "ronda" del paper = vuelta de 6 clientes -> 10 vueltas = 60.
    "cyclic": ({"order": "fixed"}, 0.0032, 60),
    "scaffold": ({"server_lr": 1.0}, 0.01, 10),
}


def flow(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in d.items()) + "}" if d else "{}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image-root",
        default="/media/lenovo-mirp/TB2/DATA_TB2/04_PregradoJulian/data/fed_isic2019_fedmammobench",
    )
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    for i, (strategy, (hp, lr, rounds)) in enumerate(CELLS.items()):
        exp = f"fedisic2019_{strategy}_r{rounds}"
        d = Path("configs/federated") / exp
        d.mkdir(parents=True, exist_ok=True)
        port = 8090 + i
        arch = """architecture:
  name: efficientnet_b0_imagenet
  unfreeze_from: features        # fine-tune completo (paper FLamby)

head:
  name: configurable_mlp
  hparams:
    in_features: 1280
    hidden_layers: []
    input_dropout: 0.2           # el dropout(0.2) previo al _fc de efficientnet_pytorch
    num_classes: 8
"""
        header = f"""# {exp} -- Fed-ISIC2019, replica de FLamby (du Terrail et al., arXiv:2210.04620, Ap. H/J).
# Celda: strategy={strategy} {hp or ''}, lr local SGD={lr}, rounds={rounds}.
# Generado por scripts/fed_isic2019/gen_configs.py (NO editar a mano). Ver
# docs/FEDISIC2019_FLAMBY_REPRO_RESULTADOS.md para desviaciones y limitaciones:
#   - federation.local_updates: 100 (E = 100 BATCHES por ronda, no epocas).
#   - tracking.best_metric_name: loss (min) -- metricas multiclase fuera de alcance (§5.D4).
#   - rounds es una aproximacion de T_max (no el valor exacto de FLamby).
"""
        (d / "server.yaml").write_text(
            header
            + f"""
experiment_id: {exp}

federation:
  server_address: "127.0.0.1:{port}"
  rounds: {rounds}
  local_updates: 100
  num_nodes: 6
  aggregation_scope: full
  accept_failures: false

strategy:
  name: {strategy}
  hparams: {flow(hp)}

{arch}
tracking:
  best_metric_name: loss
  best_metric_mode: min
  save_every_rounds: null
  run_dir: runs/federado/{exp}/server
  checkpoint_dir: runs/federado/{exp}/server/weights
  device: cpu
  wandb_project: research
  wandb_group: {exp}
"""
        )
        for node in CLIENTS:
            (d / f"node_{node}.yaml").write_text(
                f"""# Nodo {node} de {exp} -- ver server.yaml. alpha = inverso de la proporcion de clase sobre el
# train POOLED de los 6 clientes (mismo vector en todos los nodos, como en el paper).

experiment_id: {exp}
node_name: {node}
server_address: "127.0.0.1:{port}"
aggregation_scope: full

{arch}
optimizer:
  name: sgd
  hparams:
    lr: {lr}

scheduler: null

loss:
  name: focal
  hparams:
    gamma: 2.0
    alpha: {ALPHA}

data:
  manifest_path: manifests/fed_isic2019/{node}.csv
  image_root: {args.image_root}
  batch_size: 64
  num_workers: 0
  seed: 42
  image_size: 224              # int = preserva aspect ratio con lado corto 224 (como FLamby)
  crop_size: [200, 200]        # RandomCrop(train)/CenterCrop(eval), igual que FLamby
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  augmentation:
    horizontal_flip: false       # ver joint_flip: reemplaza H+V independientes
    vertical_flip: false         # por el Flip(p=0.5) combinado real de FLamby
    joint_flip: true
    joint_flip_p: 0.5
    rotation_degrees: 50
    rotation_p: 0.5              # albumentations.Rotate(50) por default (verificado)
    random_scale: true
    random_brightness_contrast: true
    affine: true                 # affine_translate default None = solo shear, como FLamby
    random_crop: true
    coarse_dropout: true

run:
  device: {args.device}
  freeze_bn_stats: false
  run_dir: runs/federado/{exp}/nodes/{node}
  checkpoint_dir: runs/federado/{exp}/nodes/{node}/weights
  wandb_project: research
  wandb_group: {exp}
  # 6 nodos x EfficientNet-B0 x batch_size=64 (del paper) no entran juntos en
  # los 8GB de VRAM de esta GPU (OOM confirmado, ver
  # docs/FIXES_FEDISIC2019_FLAMBY_REPRO.md). Serializa el forward/backward
  # entre nodos sin tocar batch_size ni la semantica de la ronda federada.
  gpu_lock_path: /tmp/{exp}_gpu.lock
"""
            )
        print("escrito", d)


if __name__ == "__main__":
    main()
