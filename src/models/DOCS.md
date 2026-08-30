# Documentación del Módulo de Modelos (`src/models/`)

`src/models/` es el subsistema encargado de construir las arquitecturas de modelos de visión para aprendizaje federado / transferencia de conocimiento, cargar sus pesos preentrenados adaptando diferencias de nombres de tensores (`state_dict`), aplicar estrategias de congelamiento (`freeze`/`unfreeze`) por bloques de capas, construir las cabezas de clasificación y retornar reportes de carga.

---

## Flujo Típico de Uso

```python
import torch.nn as nn
from src.models.build import build_model
from src.models.heads import get_head_strategy

# 1. Construir backbone encoder y congelar/descongelar capas
backbone, report = build_model(
    name="resnet50_radimagenet",
    weights_path="checkpoints/RadImageNet-ResNet50_notop.pth",
    unfreeze_from="layer3",
    device="cuda"
)

# 2. Construir la cabeza de clasificación MLP (1 logit para BCE)
HeadClass = get_head_strategy("standard_mlp")
head_builder = HeadClass(in_features=2048, hidden_dim=512, num_classes=1)
head = head_builder.build()

# 3. Ensamblar modelo completo
model = nn.Sequential(backbone, head).to("cuda")
```

---

## Detalle por Archivo y Clase

### `build.py`

#### `ArchitectureSpec`

Dataclass que centraliza el mapeo de claves del checkpoint (`key_remap`), los prefijos válidos (`valid_prefixes`) y la estrategia de congelamiento (`freeze_strategy`).

#### `build_model(name, weights_path, *, unfreeze_from="none", device="cpu")`

Factory principal para instanciar el backbone truncado y con pesos cargados.

##### Cómo usar `build.py`:
```python
from src.models.build import build_model

# Construir backbone ResNet50 descongelando desde layer3 en adelante
backbone, report = build_model(
    name="resnet50_radimagenet",
    weights_path="checkpoints/RadImageNet-ResNet50_notop.pth",
    unfreeze_from="layer3",
    device="cpu"
)

print(f"Tensores coincidentes cargados: {report.matched}")
print(f"Tensores faltantes: {len(report.missing)}")
```

---

### `weights.py`

#### `load_weights(model_factory, weights_path, key_remap, valid_prefixes, device="cpu")`

Limpia el `state_dict` del checkpoint (remueve `"module."`), aplica `key_remap`, filtra por `valid_prefixes` y trunca el modelo a sus 9 bloques encoder.

##### Cómo usar `weights.py`:
```python
from torchvision.models import resnet50
from src.models.weights import load_weights

# Definir factory de modelo PyTorch base
model_factory = lambda: resnet50(weights=None)

# Mapeo de prefijos RadImageNet -> PyTorch estándar
key_remap = {"backbone.0.": "conv1.", "backbone.1.": "bn1."}
valid_prefixes = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")

backbone, report = load_weights(
    model_factory=model_factory,
    weights_path="checkpoints/RadImageNet-ResNet50_notop.pth",
    key_remap=key_remap,
    valid_prefixes=valid_prefixes,
    device="cpu"
)
```

---

### `freeze.py`

#### `FreezeStrategy` (ABC) & `ResNetFreezeStrategy`

Controla qué bloques posicionales del backbone permanecen congelados (`requires_grad=False`) o se descongelan (`requires_grad=True`).

* `block_order`: Lista ordenada de nombres de bloques (ej. `["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]`).
* `apply(backbone, unfreeze_from)`: Congela todo el backbone y descongela a partir del bloque indicado.

##### Cómo usar `freeze.py`:
```python
import torch.nn as nn
from src.models.freeze import ResNetFreezeStrategy

strategy = ResNetFreezeStrategy()
dummy_backbone = nn.Sequential() # Ejemplo conceptual

# Descongelar desde layer4 en adelante
param_counts = strategy.apply(dummy_backbone, unfreeze_from="layer4")
print(f"Parámetros entrenables: {param_counts['trainable']} / {param_counts['total']}")
```

---

### `reports.py`

#### `LoadReport`

Dataclass con el resultado de la carga de pesos (`matched`, `missing`, `unexpected`).

##### Cómo usar `reports.py`:
```python
from src.models.reports import LoadReport

report = LoadReport(matched=150, missing=["fc.weight"], unexpected=[])
print(f"Cargados con éxito: {report.matched} tensores")
```

---

### `head_builder.py` & `heads.py`

#### `HeadBuilder` (ABC)

Interfaz Strategy que define el método `.build() -> nn.Sequential` para cabezas de clasificación.

#### `get_head_strategy(name)`

Retorna la clase builder sin instanciar registrada en `_HEAD_STRATEGIES`.

##### Cómo usar `heads.py` & `head_builder.py`:
```python
from src.models.heads import get_head_strategy

# Obtener clase builder por nombre
HeadClass = get_head_strategy("standard_mlp")

# Instanciar builder con hiperparámetros
builder = HeadClass(in_features=2048, hidden_dim=256, dropout=0.3, num_classes=1)

# Construir módulo nn.Sequential
head_module = builder.build()
```

---

### `mlp_configs/standard_mlp.py`

#### `StandardMLPHead`

Cabeza de clasificación MLP estándar:
`Flatten -> Linear(in_features, hidden_dim) -> BatchNorm1d -> ReLU -> Dropout(p) -> Linear(hidden_dim, num_classes)`.

##### Cómo usar `StandardMLPHead`:
```python
from src.models.mlp_configs.standard_mlp import StandardMLPHead

head_builder = StandardMLPHead(in_features=2048, hidden_dim=512, dropout=0.5, num_classes=1)
head = head_builder.build()
```

---

## Exportaciones (`__init__.py`)

`src/models/__init__.py` reexporta las utilidades principales:
```python
from src.models import build_model, get_head_strategy, LoadReport
```
