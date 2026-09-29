# `src/models/` — backbones, heads, congelamiento, carga de pesos

Contratos por método con ejemplos: **`src/models/DOCS.md`**. Esta hoja es el porqué y los invariantes.

Idioma: todo `models/` está en **inglés**.

`models/weights.py` **nunca importa `models/build.py`.**

---

## Dos ejes independientes: backbone y head

El backbone y el head se eligen por separado y se componen en `cli.py` con
`nn.Sequential(backbone, head.build())`. Eso es deliberado: el ensamblaje vive en el entrypoint, no
escondido en una factory.

### Backbone — `build_model(name, weights_path=None, unfreeze_from, device)`

Devuelve `(backbone, LoadReport)`. Instancia desde `_ARCHITECTURES`, **remapea las claves `backbone.N.`
del checkpoint a los nombres de torchvision**, y aplica una `FreezeStrategy`.

Arquitecturas registradas hoy:

| `name` | pesos | `weights_path` | `fc.in_features` |
|---|---|---|---|
| `resnet50_radimagenet` | checkpoint `.pth` externo | **obligatorio** | 2048 |
| `resnet50_scratch` | aleatorios desde cero (`weights_from_factory=True`) | prohibido | 2048 |
| `resnet18_scratch` | aleatorios desde cero (`weights_from_factory=True`) | prohibido | **512** |
| `resnet18_imagenet_v1` | torchvision (`weights_from_factory=True`) | prohibido | **512** |
| `resnet50_imagenet_v1` | torchvision (`weights_from_factory=True`) | prohibido | 2048 |
| `resnet50_imagenet_v2` | torchvision (`weights_from_factory=True`) | prohibido | 2048 |
| `custom_cnn_v1` | aleatorios desde cero (`weights_from_factory=True`) | prohibido | **256** |

> **Nota para ResNet18**: entrega 512 features (vs 2048 de ResNet50). Todo YAML con ResNet18 debe configurar `head.hparams.in_features: 512`.

> **Nota para `custom_cnn_v1`**: backbone custom (`models/custom_cnn.py`) -- 4 bloques
> `Conv2d(3x3, padding="same") -> BatchNorm2d -> ReLU` x2 con `MaxPool2d(2)` al final de
> los bloques 1-3 (canales 32/64/128/256, bloque 4 sin MaxPool) + `AdaptiveAvgPool2d(1)`.
> Entrega 256 features -- `head.hparams.in_features: 256`. Su `FreezeStrategy`
> (`CustomCNNFreezeStrategy`) usa `block_order: ["block1","block2","block3","block4"]`;
> `unfreeze_from: block1` = todo entrenable, igual semántica que `conv1` en los
> `*_scratch` de ResNet. Diseñado para 1 logit de salida (`head.hparams.num_classes: 1`
> + `loss.name: bce`) -- ver `configs/exp67_custom_cnn.yaml`.

Las de torchvision descargan y cachean en `~/.cache/torch/hub/checkpoints/` la primera vez que se
instancia el modelo — requiere internet **solo esa primera vez**. Para `*_scratch`, no se descarga nada.

### Head — `get_head_strategy(name)`

Devuelve una subclase **sin construir** de `HeadBuilder` (clave en `_HEAD_STRATEGIES`):

- **`standard_mlp`** — una capa oculta, **siempre con `BatchNorm1d`**.
- **`configurable_mlp`** — N capas ocultas, activación seleccionable, **sin BatchNorm por defecto**.
  Portado del INC (`PHASES.md`).

Las implementaciones concretas están en `models/mlp_configs/`.

---

## Invariantes

### `load_weights()` lanza excepción cuando `matched == 0`
Ese es exactamente el estado que produce un desajuste de prefijo `backbone.`, y **sin el raise la
corrida entrena desde inicialización aleatoria y solo parece mediocre** — no falla, que es peor. Es el
modo de falla legacy más caro de todos: se perdieron corridas enteras así. No lo degrades a warning.

`LoadReport` reporta cuántos tensores casaron; míralo si dudas de una carga.

### `BatchNorm1d` y el batch de tamaño 1
`StandardMLPHead` siempre lleva `BatchNorm1d`, que lanza excepción con un batch final de 1 elemento.
Por eso el loader de train usa `drop_last=True` (ver [DATASETS.md](DATASETS.md)).

### `unfreeze_from`
Nombre del bloque a partir del cual se descongelan gradientes; `"none"` (default) congela el backbone
entero. Los valores usados en los experimentos son `none`, `layer4` y descongelar todo.

**Cuánto se descongela pesa más que qué pesos trae el backbone**: en cada par comparable de la
ablación de pretraining, pasar de congelado a `layer4` a todo gana más AUC que cambiar
RadImageNet↔ImageNet. Ver [../experiments/CENTRALIZED.md](../experiments/CENTRALIZED.md).

### El split de checkpoint backbone/head
`checkpoint.py` escribe `best_epoch<N>_backbone.pt` y `_head.pt` además del `.pt` completo. Eso existe
para `aggregation_scope: backbone` del layer federado, donde solo agrega el backbone. Si cambias el
formato de checkpoint, [FEDERATED.md](FEDERATED.md) depende de él.
