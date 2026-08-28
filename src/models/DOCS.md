# Documentación de la carpeta

`src/models/` es el módulo encargado de construir las arquitecturas de modelos de visión para aprendizaje federado / transferencia de conocimiento, cargar sus pesos preentrenados adaptando diferencias de nombres de tensores (`state_dict`), aplicar estrategias de congelamiento (`freeze`/`unfreeze`) por bloques de capas y retornar reportes detallados del proceso de carga. Flujo típico:

```python
from src.models.build import build_model

# Construye el backbone ResNet50 con pesos de RadImageNet descongelando desde layer3 en adelante
backbone, report = build_model(
    name="resnet50_radimagenet",
    weights_path="checkpoints/RadImageNet-ResNet50_notop.pth",
    unfreeze_from="layer3",
    device="cuda",
)

print(f"Tensores coincidentes cargados: {report.matched}")
print(f"Tensores faltantes: {len(report.missing)}")
```

## build.py

### ArchitectureSpec

Dataclass que centraliza las especificaciones y variaciones de cada arquitectura de modelo soportada: el mapeo de claves de tensores del checkpoint, los prefijos válidos para filtrar capas pertenecientes al backbone y la estrategia de congelamiento correspondiente.

Atributos:
- `key_remap` (`dict[str, str]`): Diccionario con las sustituciones de prefijos en las claves del `state_dict` (ej. `"backbone.0."` -> `"conv1."`).
- `valid_prefixes` (`tuple[str, ...]`): Tupla de prefijos de nombres de capa permitidos en el backbone (ej. `("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")`).
- `freeze_strategy` (`FreezeStrategy`): Instancia de `FreezeStrategy` asociada a la arquitectura para controlar qué bloques congelar o entrenar.

#### Registro `_ARCHITECTURES`

Diccionario interno (`dict[str, ArchitectureSpec]`) que mapea un identificador único (ej. `"resnet50_radimagenet"`) con su `ArchitectureSpec` correspondiente.

---

### build_model(name, weights_path, *, unfreeze_from="none", device="cpu")

Función principal (factory) para la creación de modelos. Construye el backbone completo orquestando la instanciación de la arquitectura, la carga y remapeo de pesos preentrenados mediante `load_weights` y la aplicación de la estrategia de congelamiento seleccionada mediante `spec.freeze_strategy.apply`.

Inputs:
- `name` (`str`): Nombre/clave de la arquitectura registrada en `_ARCHITECTURES` (ej. `"resnet50_radimagenet"`).
- `weights_path` (`str`): Ruta al archivo checkpoint (`.pth` o `.pt`) con los pesos preentrenados.
- `unfreeze_from` (`str`, default `"none"`): Nombre del bloque a partir del cual se descongelarán las capas para fine-tuning. Se pasa a `FreezeStrategy.apply()`.
- `device` (`str`, default `"cpu"`): Dispositivo de cómputo donde se cargarán los tensores (ej. `"cpu"`, `"cuda"`).

Returns:
- `tuple[nn.Sequential, LoadReport]`: Una tupla conteniendo:
  1. El backbone `nn.Sequential` ya truncado y con los parámetros congelados/descongelados según la estrategia.
  2. El objeto `LoadReport` con el balance de tensores cargados.

Raises:
- `ValueError`: Si `name` no está registrado en `_ARCHITECTURES`.

## weights.py

### load_weights(model_factory, weights_path, key_remap, valid_prefixes, device="cpu")

Carga un checkpoint en la arquitectura instanciada por `model_factory`. Realiza la limpieza del `state_dict` (eliminación del prefijo `"module."` procedente de DataParallel), aplica el remapeo de prefijos especificado en `key_remap`, filtra únicamente los tensores pertenecientes a `valid_prefixes`, carga los pesos con `strict=False` y trunca la red para devolver solo las capas del backbone encoder (`nn.Sequential(*encoder_layers[:9])`).

Inputs:
- `model_factory` (`Callable[[], nn.Module]`): Función invocable sin argumentos que instancie la arquitectura base PyTorch (ej. `lambda: resnet50(weights=None)`).
- `weights_path` (`str | Path`): Ruta al archivo de pesos checkpoint.
- `key_remap` (`dict[str, str]`): Mapeo de prefijos de tensores `{prefijo_original: prefijo_destino}`.
- `valid_prefixes` (`tuple[str, ...]`): Tupla con los prefijos permitidos para filtrar el `state_dict`.
- `device` (`str`, default `"cpu"`): Dispositivo de destino (`"cpu"` o `"cuda"`).

Returns:
- `tuple[nn.Sequential, LoadReport]`: Tupla con el backbone truncado de 9 bloques y el reporte de carga.

Raises:
- `RuntimeError`: Si el `state_dict` remapeado y filtrado queda completamente vacío (0 tensores supervivientes), o si ninguna clave coincide durante `load_state_dict` (`matched == 0`).

## freeze.py

### FreezeStrategy(ABC)

Clase base abstracta que define la interfaz y lógica común para congelar (`requires_grad = False`) o descongelar (`requires_grad = True`) bloques de capas dentro de un backbone `nn.Sequential`.

#### block_order (propiedad abstracta)

`list[str]`: Retorna la lista de nombres descriptivos de los bloques de la arquitectura en el orden exacto en que están posicionados dentro del `nn.Sequential`.

#### apply(backbone, *, unfreeze_from)

Congela todos los parámetros del `backbone` y, si `unfreeze_from` es distinto de `"none"`, descongela los bloques a partir de la posición correspondiente a `unfreeze_from` hasta el final del modelo.

Inputs:
- `backbone` (`nn.Sequential`): El modelo cuyas capas se van a ajustar.
- `unfreeze_from` (`str`, *keyword-only*): Nombre del bloque a partir del cual habilitar gradientes. Si es `"none"`, todo el backbone permanece congelado.

Returns:
- `dict[str, int]`: Diccionario con la cantidad de parámetros entrenables y totales: `{"trainable": int, "total": int}`.

Raises:
- `ValueError`: Si `unfreeze_from` no es `"none"` ni pertenece a la lista devuelta por `block_order`.

---

### ResNetFreezeStrategy(FreezeStrategy)

Implementación concreta de `FreezeStrategy` adaptada para arquitecturas tipo ResNet50.

#### block_order

Devuelve la lista ordenada de bloques posicionales para ResNet50:
`["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]`.

## reports.py

### LoadReport

Dataclass que encapsula el resultado de la carga de pesos desde un checkpoint al modelo.

Inputs / Atributos:
- `matched` (`int`): Cantidad de tensores que coincidieron exitosamente entre el checkpoint y el modelo.
- `missing` (`list[str]`): Lista de nombres de parámetros esperados por el modelo pero ausentes en el checkpoint.
- `unexpected` (`list[str]`): Lista de nombres de tensores presentes en el checkpoint filtrado pero no esperados por el modelo.

#### __post_init__()

Validación ejecutada automáticamente al instanciar el dataclass.

Raises:
- `RuntimeError`: Si `matched == 0`, alertando que ningún tensor coincidió a pesar de que el `state_dict` no estaba vacío (indicativo de un fallo en el remapeo de claves).

## __init__.py

Módulo de inicialización del paquete `src/models/`. Reexporta los componentes públicos principales para facilitar su importación:

- `load_model` / `train_model` (interfaces de alto nivel).
- `LoadReport`, `TrainReport` (dataclasses de reporte).
