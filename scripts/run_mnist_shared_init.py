#!/usr/bin/env python3
"""Corre configs/exp_mnist_smoketest_shared_init.yaml con la cabeza inicializada desde el
checkpoint compartido con INC (ver scripts/generate_shared_init_mnist.py para el porqué y
cómo se generan esos pesos).

Comparación de paridad puntual FMB<->INC -- no un modo de uso general de la CLI. El
backbone de este mismo experimento ya se resuelve solo desde el config
(`architecture.name: resnet18_mnist_shared_init` + `architecture.weights_path`, ver
`src/models/build.py`); solo la cabeza necesita este script, porque `HeadConfig` no tiene
un campo `weights_path` (ningún otro experimento necesita pesos iniciales congelados en la
cabeza) -- ver el docstring de `head_state_dict_path` en `src/cli.py:run()`.

Correr:
    .venv/bin/python -m scripts.run_mnist_shared_init
"""

from pathlib import Path

from src.cli import run
from src.config import load_config

CONFIG_PATH = "configs/exp_mnist_smoketest_shared_init.yaml"
HEAD_WEIGHTS_PATH = "runs/centralizado/shared_init_mnist/head_fmb.pt"


def main() -> None:
    if not Path(HEAD_WEIGHTS_PATH).exists():
        raise FileNotFoundError(
            f"{HEAD_WEIGHTS_PATH} no existe -- correr primero "
            "`.venv/bin/python -m scripts.generate_shared_init_mnist`."
        )
    config = load_config(CONFIG_PATH)
    run(config, head_state_dict_path=HEAD_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
