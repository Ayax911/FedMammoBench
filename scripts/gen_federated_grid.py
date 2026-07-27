#!/usr/bin/env python3
"""Genera el grid federado exp32–49 (3 estrategias × 6 regímenes ronda/época).

Cada experimento produce 8 YAML: server.yaml, client.yaml y eval/{mammo_bench,
node1..5_partition}.yaml. Todo se mantiene constante salvo `rounds`, `local_epochs`
(= scheduler.t_max) y la estrategia de agregación, para aislar esos efectos.

Aprendizajes horneados (no repetir bugs de exp20–31):
  - FedAdam eta=0.001 (no el default 0.01 que colapsó FedYogi en exp29).
  - normalize_preset: radimagenet_rgb en TODOS los eval.
  - Manifests correctos: global mamo-bench-split-no-ddsm-rsna.csv; por-nodo <dataset>-split.csv.

Uso:  python scripts/gen_federated_grid.py            # genera los 144 archivos
      python scripts/gen_federated_grid.py --dry-run  # solo lista lo que crearía
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

# Bloque -> (rounds, epochs, {estrategia: num_exp})
BLOCKS: dict[str, dict] = {
    "A": {"rounds": 5,  "epochs": 20,  "nums": {"fedavg": 32, "fedprox": 33, "fedadam": 34}},
    "B": {"rounds": 10, "epochs": 20,  "nums": {"fedavg": 35, "fedprox": 36, "fedadam": 37}},
    "C": {"rounds": 30, "epochs": 5,   "nums": {"fedavg": 38, "fedprox": 39, "fedadam": 40}},
    "D": {"rounds": 1,  "epochs": 100, "nums": {"fedavg": 41, "fedprox": 42, "fedadam": 43}},
    "E": {"rounds": 3,  "epochs": 100, "nums": {"fedavg": 44, "fedprox": 45, "fedadam": 46}},
    "F": {"rounds": 5,  "epochs": 100, "nums": {"fedavg": 47, "fedprox": 48, "fedadam": 49}},
}

# Bloque federated.strategy.params (indentado bajo `strategy:` a 4/6 espacios)
STRATEGY_PARAMS: dict[str, str] = {
    "fedavg":  "    params: {}",
    "fedprox": "    params:\n      proximal_mu: 0.1",
    "fedadam": (
        "    params:\n"
        "      eta: 0.01          # recalibrado 2026-07-21: eta=0.001/tau=0.01 sub-entrenaba\n"
        "                         # (exp34 global congelado en AUC 0.46 aunque los clientes aprendían)\n"
        "      eta_l: 0.0316\n"
        "      beta_1: 0.9\n"
        "      beta_2: 0.99\n"
        "      tau: 0.001         # default flwr"
    ),
}

# node -> (dataset, manifest)
NODE_MAP = [
    (1, "cmmd",     "cmmd-split.csv"),
    (2, "inbreast", "inbreast-split.csv"),
    (3, "cdd-cesm", "cdd-cesm-split.csv"),
    (4, "kau-bcmd", "kau-bcmd-split.csv"),
    (5, "dmid",     "dmid-split.csv"),
]


def full_name(num: int, strat: str, rounds: int, epochs: int) -> str:
    return f"exp{num}_{strat}_r{rounds}e{epochs}_radimagenet"


def server_yaml(block: str, strat: str, rounds: int, epochs: int, full: str) -> str:
    params = STRATEGY_PARAMS[strat]
    return f"""# {full} — SERVIDOR CENTRAL (5 nodos, {strat}, RadImageNet init directo)
#
# Grid estrategia × régimen (exp32–49). Bloque {block}: {rounds} rondas × {epochs} épocas locales
# = {rounds * epochs} épocas-equivalentes. Trío comparable del bloque (fedavg/fedprox/fedadam,
# mismo régimen). Warm-start RadImageNet, manifest, LR y regularización idénticos en todo el grid;
# solo varían rounds, local_epochs (= scheduler.t_max) y la estrategia de agregación.
#
# WARM START: RadImageNet directo.  EVALUACIÓN: data.name=none → 100% federada.

name: {full}
mode: federated
seed: 42
output_dir: runs/{full}
device: auto

data:
  name: none              # servidor sin datos → evaluación federada
  image_size: 224
  grayscale: false
  num_classes: 2
  batch_size: 32
  num_workers: 0
  val_fraction: 0.1
  test_fraction: 0.1
  image_format: jpg
  normal_policy: benign

model:
  name: resnet50
  weight_source: radimagenet
  checkpoint_path: weights/RadImageNet-resnet50.pth
  in_channels: 3
  num_classes: 1
  dropout: 0.0
  freeze_backbone: false
  unfreeze_at_epoch: null
  local_unfreeze_at_epoch: null
  unfreeze_layers: null

training:
  local_epochs: {epochs}         # DEBE coincidir con el cliente y con scheduler.t_max
  grad_clip_norm: 1.0
  mixed_precision: true
  optimizer:
    name: adamw
    lr: 1.0e-4
    lr_head: 1.0e-3
    lr_backbone: 1.0e-4
    weight_decay: 1.0e-4
  scheduler:
    name: cosine
    t_max: {epochs}              # = local_epochs (el scheduler se reinicia cada ronda)
  augmentation:
    normalize_preset: radimagenet_rgb
  loss:
    name: bce
    auto_class_weights: true

evaluation:
  threshold: 0.5
  save_predictions: true

federated:
  num_clients: 5
  rounds: {rounds}
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  min_fit_clients: 5
  min_evaluate_clients: 5
  min_available_clients: 5
  accept_failures: false
  server_address: "0.0.0.0:8080"
  strategy:
    name: {strat}
{params}
"""


def client_yaml(block: str, strat: str, rounds: int, epochs: int, full: str) -> str:
    params = STRATEGY_PARAMS[strat]
    return f"""# {full} — NODOS CLIENTE (5 nodos, {strat}, RadImageNet init directo)
#
# Mapeo nodo → dataset (vía scripts/docker-deploy-federated.sh):
#   node1 → cmmd, node2 → inbreast, node3 → cdd-cesm, node4 → kau-bcmd, node5 → dmid
# Cada nodo usa ESTE YAML; el manifest se pasa por --manifest en el workflow.
#
# COHERENCIA servidor↔cliente: freeze_backbone, scheduler.t_max, rounds y strategy
# DEBEN coincidir con server.yaml. Bloque {block}: {rounds} rondas × {epochs} épocas.

name: {full}
mode: federated
seed: 42
output_dir: runs/{full}
device: auto

data:
  name: mammo_bench
  manifest_path: manifests/cmmd-split.csv   # sobreescrito por --manifest en el workflow
  image_root: data/
  image_size: 224
  grayscale: false
  num_classes: 2
  batch_size: 32
  num_workers: 0
  val_fraction: 0.15      # IGNORADO: el manifest ya trae split explícito
  test_fraction: 0.0      # IGNORADO: idem
  image_format: jpg
  normal_policy: benign
  balance_classes: true

model:
  name: resnet50
  weight_source: radimagenet
  checkpoint_path: weights/RadImageNet-resnet50.pth
  strict_load: false
  in_channels: 3
  num_classes: 1
  dropout: 0.0
  freeze_backbone: false
  unfreeze_at_epoch: null
  local_unfreeze_at_epoch: null
  unfreeze_layers: null

training:
  local_epochs: {epochs}         # DEBE coincidir con el servidor
  grad_clip_norm: 1.0
  mixed_precision: true
  optimizer:
    name: adamw
    lr: 1.0e-4
    lr_head: 1.0e-3
    lr_backbone: 1.0e-4
    weight_decay: 1.0e-4
  scheduler:
    name: cosine
    t_max: {epochs}              # = local_epochs (el scheduler se reinicia cada ronda)
  augmentation:
    horizontal_flip: true
    rotate_limit: 15
    brightness_contrast: true
    normalize_preset: radimagenet_rgb
  loss:
    name: bce
    auto_class_weights: true

evaluation:
  threshold: 0.5
  save_predictions: true

federated:
  rounds: {rounds}                 # DEBE coincidir con el servidor
  strategy:
    name: {strat}
{params}
"""


def eval_yaml(full: str, suffix: str, manifest: str) -> str:
    """Config de eval post-hoc (global o por-nodo). El checkpoint se pasa por --checkpoint."""
    return f"""name: {full}_{suffix}_eval
mode: centralized
output_dir: runs/{full}

data:
  name: mammo_bench
  manifest_path: manifests/{manifest}
  image_root: data/
  image_size: 224
  grayscale: false
  num_classes: 2
  batch_size: 32
  num_workers: 0
  image_format: jpg
  normal_policy: benign

model:
  name: resnet50
  pretrained: false
  in_channels: 3
  num_classes: 1
  dropout: 0.0
  weight_source: none

training:
  augmentation:
    normalize_preset: radimagenet_rgb
  loss:
    name: bce
    auto_class_weights: true

evaluation:
  threshold: 0.5
  save_predictions: true
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Solo listar, no escribir")
    args = ap.parse_args()

    written = 0
    for block, spec in BLOCKS.items():
        rounds, epochs = spec["rounds"], spec["epochs"]
        for strat, num in spec["nums"].items():
            full = full_name(num, strat, rounds, epochs)
            exp_dir = CONFIGS / f"exp{num}"
            eval_dir = exp_dir / "eval"

            files = {
                exp_dir / "server.yaml": server_yaml(block, strat, rounds, epochs, full),
                exp_dir / "client.yaml": client_yaml(block, strat, rounds, epochs, full),
                eval_dir / "mammo_bench.yaml": eval_yaml(
                    full, "mammo_bench", "mamo-bench-split-no-ddsm-rsna.csv"
                ),
            }
            for node, _dataset, manifest in NODE_MAP:
                files[eval_dir / f"node{node}_partition.yaml"] = eval_yaml(
                    full, f"node{node}", manifest
                )

            if args.dry_run:
                print(f"[{block}] exp{num} {strat:8s} {rounds}r×{epochs}e → {full} ({len(files)} files)")
                continue

            eval_dir.mkdir(parents=True, exist_ok=True)
            for path, content in files.items():
                path.write_text(content)
                written += 1
            print(f"[{block}] exp{num} {strat:8s} {rounds:2d}r×{epochs:3d}e → configs/exp{num}/ ({len(files)} files)")

    if not args.dry_run:
        print(f"\n✓ {written} archivos escritos ({len(BLOCKS) * 3} experimentos × 8).")


if __name__ == "__main__":
    main()
