#!/usr/bin/env python3
"""Genera UNA vez los pesos iniciales aleatorios compartidos (backbone + cabeza) para la
comparación de paridad FMB<->INC del MNIST smoketest.

Por qué existe: ambos frameworks siembran `torch.manual_seed(42)`, pero cada codebase
construye su modelo con un orden distinto de llamadas a la RNG global (distinto orden de
instanciación de capas, dropout, etc.), así que "misma semilla" NO implica "mismos pesos
iniciales" entre dos implementaciones independientes -- eso por sí solo basta para que dos
corridas por lo demás idénticas terminen con accuracy/F1/sensibilidad/especificidad
ligeramente distintos (differencias de ~0.05 puntos porcentuales observadas, del orden de
un puñado de imágenes sobre 10.000). Este script congela los pesos iniciales UNA VEZ en
disco; ambos repos los cargan desde ahí en vez de que cada uno inicialice por su cuenta,
eliminando esa fuente de divergencia para esta comparación puntual.

No elimina el resto de la divergencia -- el orden de shuffle de cada DataLoader tampoco
está sincronizado entre los dos frameworks (FMB usa un `torch.Generator` explícito, INC usa
el estado global de RNG), así que ni con pesos iniciales idénticos las dos corridas son
bit-a-bit iguales, solo más cercanas. (Update: sincronizar ese shuffle vía
`--dataloader_seed`/`dataloader_images.py:Loader` movió el residuo de una clase a otra sin
reducir el total -- ver run_mnist_shared_init.sh. Dropout, que también lee del RNG global
sin generator propio, es la otra fuente conocida; por eso `HEAD_KWARGS["dropout"]` bajó a 0
más abajo.)

Genera tres archivos bajo runs/centralizado/shared_init_mnist/:
  - backbone.pt : resnet18(weights=None) truncado a sus primeros 9 hijos (conv1, bn1, relu,
    maxpool, layer1-4, avgpool -- sin fc), con cada clave prefijada "backbone." -- la misma
    convención de nombres que usa `ResNet18Model.backbone` en INC (ver
    inc-project-models-classification-detection-main/src/models/classification_images/
    models/image_models.py), así que ESTE MISMO archivo se carga sin transformación en:
      * INC, vía `--path_image_model backbone.pt --pretrained False --from_scratch False`
        (`base_model.load_state_dict(torch.load(weigths_file))` en `get_image_model()`).
      * FMB, vía `architecture.name: resnet18_mnist_shared_init` +
        `architecture.weights_path: backbone.pt` (`build.py` remapea "backbone.N." a los
        nombres nativos de torchvision antes de cargar, igual que ya hace para RadImageNet
        -- ver `_ARCHITECTURES["resnet18_mnist_shared_init"]`).
  - head_fmb.pt : cabeza `ConfigurableMLPHead(in_features=512, hidden_layers=[128],
    activation="leakyrelu", negative_slope=0.2, dropout=0, num_classes=2,
    use_batchnorm=False).build()` -- exactamente la arquitectura de
    `configs/exp_mnist_smoketest_shared_init.yaml`. Con dropout=0, `build()` NO instancia
    `nn.Dropout` (`if self.dropout > 0:`), así que la cabeza queda
    `Sequential(Flatten, Linear, LeakyReLU, Linear)` -- sin la capa Dropout que habría entre
    la activación y el Linear de salida si dropout > 0.
  - head_inc.pt : LOS MISMOS tensores de head_fmb.pt, solo renombrados a la convención de
    INC (`models/mlp_models.py`: `MLP.model = Sequential(Linear, Activation, Linear)` cuando
    dropout=0 -- mismo `if dropout > 0:` del lado de INC --, claves "model.0."/"model.2.").
    La cabeza de FMB antepone un `Flatten()` que la de INC no tiene, lo que desplaza en 1 el
    índice de cada `Linear`, y además FMB no envuelve la cabeza en un atributo "model." como
    sí hace INC -- sin este remapeo, un `load_state_dict()` directo fallaría por claves no
    encontradas en cualquiera de los dos lados.

Correr una sola vez (regenerar solo si HEAD_KWARGS o SEED cambian):
    .venv/bin/python -m scripts.generate_shared_init_mnist
"""

from pathlib import Path

import torch
from torchvision.models import resnet18

from src.models.mlp_configs.configurable_mlp import ConfigurableMLPHead

SEED = 42
OUT_DIR = Path("runs/centralizado/shared_init_mnist")

# Debe coincidir con architecture/head de configs/exp_mnist_smoketest_shared_init.yaml.
HEAD_KWARGS: dict[str, object] = {
    "in_features": 512,
    "hidden_layers": [128],
    "activation": "leakyrelu",
    "negative_slope": 0.2,
    "dropout": 0,  # 0 en ambos lados -- ver cabecera de configs/exp_mnist_smoketest_shared_init.yaml
    "num_classes": 2,
    "use_batchnorm": False,
}

# FMB antepone Flatten() (índice 0) a la cabeza; INC no. Con 1 sola capa oculta y
# dropout=0 (ninguno de los dos builders instancia la capa Dropout cuando dropout <= 0 --
# ver ConfigurableMLPHead.build() / INC's MLP.__init__, ambos con `if dropout > 0:`), los
# Linear de FMB quedan en los índices 1 (oculta) y 3 (salida); los de INC, envueltos en
# `self.model`, en los índices 0 y 2. (Con dropout>0 serían 1/4 y 0/3 -- un módulo Dropout
# de más en cada builder.) Si HEAD_KWARGS["hidden_layers"] cambia de longitud, o dropout
# vuelve a ser >0, este mapa hay que recalcularlo a mano (no vale la pena generalizarlo
# para un solo uso).
_FMB_TO_INC_LINEAR_INDEX = {"1": "0", "3": "2"}


def main() -> None:
    torch.manual_seed(SEED)

    # --- Backbone: mismo truncamiento que src/models/weights.py:truncate_backbone() y
    # que ResNet18Model.backbone en INC (list(model.children())[:9]). ---
    full_resnet = resnet18(weights=None)
    backbone = torch.nn.Sequential(*list(full_resnet.children())[:9])
    backbone_state = {f"backbone.{k}": v for k, v in backbone.state_dict().items()}

    # --- Cabeza ---
    head = ConfigurableMLPHead(**HEAD_KWARGS).build()  # pyright: ignore[reportArgumentType]
    head_fmb_state = head.state_dict()

    head_inc_state: dict[str, torch.Tensor] = {}
    for key, value in head_fmb_state.items():
        fmb_idx, _, suffix = key.partition(".")
        inc_idx = _FMB_TO_INC_LINEAR_INDEX[fmb_idx]
        head_inc_state[f"model.{inc_idx}.{suffix}"] = value

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(backbone_state, OUT_DIR / "backbone.pt")
    torch.save(head_fmb_state, OUT_DIR / "head_fmb.pt")
    torch.save(head_inc_state, OUT_DIR / "head_inc.pt")

    print(f"Backbone: {len(backbone_state)} tensores -> {OUT_DIR / 'backbone.pt'}")
    print(f"Cabeza FMB ({len(head_fmb_state)} tensores): {sorted(head_fmb_state.keys())} -> {OUT_DIR / 'head_fmb.pt'}")
    print(f"Cabeza INC ({len(head_inc_state)} tensores): {sorted(head_inc_state.keys())} -> {OUT_DIR / 'head_inc.pt'}")


if __name__ == "__main__":
    main()
