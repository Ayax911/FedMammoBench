# FEDISIC2019_FLAMBY_REPRO — replicar el benchmark Fed-ISIC2019 de FLamby (grid completo)

Brief para Antigravity (ciclo `/antigravity`, Fase 1). Redactado 2026-10-20. Informe esperado:
`docs/FEDISIC2019_FLAMBY_REPRO_RESULTADOS.md` (ver §7).

Fuente: du Terrail et al., *"FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in
Realistic Healthcare Settings"* (arXiv:2210.04620v3), Apéndice H (Fed-ISIC2019) + Apéndice J (Tablas
10-16, hiperparámetros finales del benchmark). Todas las tablas de este brief están transcritas
directamente del paper — no hace falta volver a leerlo.

---

## 1. Resumen ejecutivo

- Se quiere poder correr, con el código de este repo (`src/federated/`), el mismo grid de
  estrategias × hiperparámetros que FLamby corrió sobre Fed-ISIC2019: FedAvg, FedProx, FedAdagrad,
  FedAdam, FedYogi, Cyclic (orden fijo y shuffled) y **Scaffold**.
- Hoy el repo solo tiene registradas `fedavg`/`fedprox`/`fedadam`/`fedyogi` (clases stock de
  `flwr.server.strategy`, ver §3) y el optimizer local del paper (`SGD`) no existe (§3). Scaffold y
  Cyclic no existen en absoluto, en ningún lado.
- **Exclusión explícita y deliberada de esta vuelta: NO tocar `src/metrics.py` ni la fórmula de
  `balanced_accuracy` de `src/reporting.py`.** Esos dos archivos quedan tal cual están, asumiendo
  clasificación binaria. Esto implica que, para que el pipeline multiclase (8 clases de ISIC2019) no
  crashee contra esas métricas binarias, hace falta un bypass mínimo y explícito (§5.D) — las métricas
  "de verdad" para Fed-ISIC2019 (balanced accuracy multiclase correcta, por paper) quedan **fuera de
  alcance**, para una vuelta separada. El criterio de éxito de esta vuelta es que el pipeline
  **entrena** (loss desciende, no hay NaN/crash, las estrategias agregan bien, Scaffold/Cyclic/
  FedAdagrad corren de punta a punta) — no que las métricas reportadas sean clínicamente correctas.
- Baseline *pooled* (no-federado) del paper: **fuera de alcance** de esta vuelta (ver §6). Todo el
  trabajo pedido es para la rama federada.
- Entregable de código: las piezas de §5 (A-H) implementadas y verificadas con el harness sintético/
  MNIST smoketest que ya existe en el repo (sin necesitar aún datos reales de ISIC2019 para A-G), más
  la adquisición real del dataset y los YAML de una celda completa (servidor + 6 nodos) para H.

---

## 2. Restricción de entorno

**Esta vuelta es puramente de desarrollo — no de ejecución real.** Antigravity programa en la máquina
de desarrollo (sin GPU dedicada ni el dataset completo necesariamente disponible) y el despliegue con
hardware/datos reales pasa **después**, en otro PC, fuera del alcance de este brief. Por lo tanto:

- **No hace falta verificar ni esperar GPU, disco ni dataset completo antes de empezar a programar.**
  Si no hay GPU en la máquina de desarrollo, todo el código (A-G) se escribe y se verifica igual, por
  CPU, con tensores sintéticos (`torch.randn`, etc.) — ningún criterio de aceptación de esta vuelta
  exige GPU real (ver §5, §8).
- Internet y algo de disco sí hacen falta para lo mínimo de §5.H (bajar ISIC2019 y armar los
  manifests) — pero si tampoco hay esto, no es un bloqueante: documentarlo en el informe y entregar el
  código de construcción del manifest igual (sin haberlo corrido contra los datos reales todavía), para
  que se ejecute en el despliegue.
- No depender del mount `imagenesmedicas` para nada de lo pedido acá — Fed-ISIC2019 es un dataset
  público sin relación con esa data propietaria de mamografía, esté o no montado donde se programe.

En ningún caso parar el trabajo por falta de GPU/datos — avanzar con lo verificable por CPU/sintético
y dejar explícito en el informe qué quedó pendiente de validar con hardware/datos reales en el
despliegue.

Resto del entorno, igual que siempre: `.venv/bin/python` (Python 3.12, ver `CLAUDE.md` §Entorno). No
hay pytest — verificación con scripts desechables + asserts (convención de `PHASES.md`), **fuera del
repo**. `flwr==1.31.0` pinneado exacto — cualquier API nueva que se use (custom `Strategy`,
`ClientManager`, etc.) debe existir en esa versión exacta, verificarlo contra
`.venv/lib/python3.12/site-packages/flwr` antes de escribir código contra una API que cambió en
versiones más nuevas.

---

## 3. Lo que YA está resuelto / cerrado (no re-investigar)

- **Estrategias ya registradas** (`src/federated/strategies.py:30-35`): `fedavg`, `fedprox`,
  `fedadam`, `fedyogi` — clases stock de `flwr.server.strategy`, hparams (`proximal_mu`, `eta`,
  `eta_l`, `beta_1`, `beta_2`, `tau`) ya configurables por YAML vía `strategy.hparams`
  (`build_strategy()`, líneas 131-196). El comentario de línea 27-29 ya documenta por qué Scaffold no
  está — ahora sí hay que construirlo.
- **FedProx, el término proximal YA está implementado del lado del cliente**
  (`src/federated/client.py:166-185`): lee `proximal_mu` del config de ronda (que la clase stock
  `FedProx` inyecta sola) y arma el regularizer `(mu/2)·||w - w_global||²` que entra a
  `train_one_epoch(..., regularizer=...)`. **No tocar esto.**
- **`FocalLoss` (`src/train/focal_loss.py`) ya es genérico en número de clases** — verificado
  leyendo `forward()` (líneas 77-101): `log_softmax(dim=1)` + gather por índice + `alpha[targets]`
  funcionan para logits `[B, C]` con cualquier `C` y `alpha` de longitud `C`. **Cero cambios
  necesarios en ese archivo** — el único lugar binario es el wiring externo en `train/build.py`
  (§5.C). Su docstring de módulo/clase dice "binaria" porque hoy solo se usa así; actualizar esa
  frase cuando se generalice el wiring, no antes.
- **El patrón backbone+freeze es una ABC genérica y extensible**
  (`src/models/freeze.py:18-80`, `FreezeStrategy.apply()`): opera posicionalmente sobre
  `backbone.children()` usando `block_order` como una lista de NOMBRES arbitrarios (no hace falta que
  calcen con nombres reales de PyTorch) — agregar una arquitectura nueva es una subclase con su propio
  `block_order`, cero cambios a `apply()`. Mismo patrón para `ArchitectureSpec`/`_ARCHITECTURES`
  (`src/models/build.py:44-196`): agregar una entrada no toca `build_model()`.
- **`truncate_backbone(model, n=9)` (`src/models/weights.py:26-54`) ya acepta `n` como parámetro** —
  el `9` es el default para ResNet, no un hardcode ciego; para una arquitectura con otro número de
  hijos directos basta pasar el `n` correcto.
- **`ConfigurableMLPHead` ya es genérico en `num_classes`** (ya se usa con `num_classes: 2` en los
  YAML existentes vía `head.hparams.num_classes` — es un parámetro normal de construcción, no un
  hardcode).
- **`local_epochs` ya viaja servidor -> nodo por `on_fit_config_fn`, no por el YAML del nodo**
  (`src/federated/strategies.py:89-128` `make_on_config()`, consumido en
  `src/federated/client.py:152`) — el nodo nunca declara su propio `local_epochs`/`local_updates` en
  `node_*.yaml`; el mismo canal sirve para agregar el modo "E updates" de §5.F sin tocar el esquema de
  los YAML de nodo.
- **`FederationConfig.local_epochs: int = 1`** (`src/federated/config.py:107`) es el único campo hoy;
  agregar `local_updates` ahí es un campo hermano opcional, no un reemplazo.

---

## 4. El benchmark a replicar (self-contenido, ya extraído del paper)

### 4.1 Dataset y tarea

6 clientes por **dispositivo/sitio de adquisición** (no por hospital):

| # | Cliente | Train | Test |
|---|---|---|---|
| 0 | Hospital Clínic de Barcelona | 9930 | 2483 |
| 1 | ViDIR Viena (MoleMax HD) | 3163 | 791 |
| 2 | ViDIR Viena (DermLite FOTO) | 2691 | 672 |
| 3 | Cliff Rosendahl (Queensland) | 1807 | 452 |
| 4 | Memorial Sloan Kettering | 655 | 164 |
| 5 | ViDIR Viena (Heine Dermaphot) | 351 | 88 |

8 clases: Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis,
Dermatofibroma, Vascular lesion, Squamous cell carcinoma. Métrica del paper: balanced accuracy
(**fuera de alcance acá**, ver §1/§6).

### 4.2 Preprocesado y modelo

- Resize al lado corto = 224 px (aspect ratio preservado) + normalización de brillo/contraste
  ("color consistency", método de la solución Kaggle SIIM-ISIC de Arora).
- Modelo: **EfficientNet-B0** pre-entrenado ImageNet, *fine-tune completo* (no es un feature extractor
  congelado — el paper dice literalmente "we fine-tune an EfficientNet"), capa final reemplazada por
  lineal a 8 clases. `in_features = 1280`.
- Augmentations train: Random Scaling, Rotation, Random Brightness/Contrast, Flip, Affine deformation,
  Random crop, Coarse Dropout, Normalization. Test: Center cropping + Normalization.
- Loss: focal loss ponderada, `gamma=2`, `alpha` = inverso de la proporción de clase **calculado sobre
  el dataset pooled completo** (mismo vector en los 6 clientes — el paper asume que ese vector
  agregado está disponible a todos, una desviación deliberada de FL estricto que el propio paper
  reconoce).
- Optimizer local (TODAS las estrategias FL): **SGD** (`torch.optim.SGD`), nunca Adam.
- `E = 100` "local updates" = 100 **batches**, no épocas (convención explícita del paper, ver §5.F).
- 5 seeds por celda del grid.

### 4.3 Grid de búsqueda e hiperparámetros finales

Grids (Apéndice H.5, escala log de medio-década, `10^0.5 ≈ 3.162`):

| Estrategia | grid `lr` local | grid `server_lr` | otro |
|---|---|---|---|
| FedAvg, Cyclic | {0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316} | — | — |
| FedProx | {0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316} | — | μ ∈ {0.001, 0.01, 0.1, 1} |
| FedAdagrad, FedAdam, FedYogi, Scaffold | {0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316} | {0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1, 3.162, 10} | β1=0.9, β2=0.999, τ=1e-8 (fijos) |

Hiperparámetros finales elegidos (Tablas 10-16, fila Fed-ISIC2019):

| Estrategia | lr local | μ | server_lr |
|---|---|---|---|
| FedAvg | 0.01 | — | — |
| FedProx | 0.01 | 0.001 | — |
| FedAdagrad | 0.01 | — | 0.0316 |
| FedAdam | 0.01 | — | 0.0032 |
| FedYogi | 0.01 | — | 0.0032 |
| Cyclic | 0.0032 | — | — |
| Scaffold | 0.01 | — | 1.0 |

`T_max` (rondas): el paper define `T_max = n_epochs_pooled · ⌊n_T/(K·B·E)⌋` con `n_epochs_pooled=20`,
`n_T=18597` (train total), `K=6`, `B=64`, `E=100` → de ahí sale un número de una sola cifra (~9), pero
el paper no publica el entero exacto por dataset y la fórmula tal como está escrita (floor ANTES de
multiplicar) da 0, lo cual es absurdo — hay una ambigüedad de orden de operaciones que el texto del
paper no resuelve. **No perder tiempo re-derivando esto con precisión**: usar `rounds: 10` como valor
de trabajo razonable para todas las celdas de este grid y decirlo explícitamente en el informe como
una aproximación, no como el valor exacto de FLamby.

---

## 5. El trabajo pedido

### A — Backbone EfficientNet-B0 (`src/models/`)

**A1. `src/models/build.py`**: importar `efficientnet_b0, EfficientNet_B0_Weights` de
`torchvision.models`. Nueva entrada en `_ARCHITECTURES`:

```python
"efficientnet_b0_imagenet": ArchitectureSpec(
    model_factory=lambda: efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
    key_remap={},
    valid_prefixes=(),       # no se usa con weights_from_factory=True, ver docstring
    freeze_strategy=EfficientNetFreezeStrategy(),
    weights_from_factory=True,
),
```

`torchvision.models.efficientnet_b0(...).children()` tiene **3** hijos directos: `features`
(Sequential, el backbone real), `avgpool` (AdaptiveAvgPool2d), `classifier` (Sequential con Dropout +
Linear a 1000). Para extraer el backbone hace falta truncar a **`n=2`** (`features` + `avgpool`),
NO el default `n=9` (ese es específico de ResNet) — ver si conviene exponer `n` como parte de
`ArchitectureSpec` (un campo nuevo `truncate_n: int = 9`, con default 9 para no tocar las entradas
existentes) en vez de hardcodear un `if` en `build_model()` para esta arquitectura puntual. Verificar
con `torch.randn(2,3,224,224)` antes de seguir: shape de salida tras `nn.Flatten()` debe ser
`(2, 1280)`.

**A2. `src/models/freeze.py`**: nueva clase:

```python
class EfficientNetFreezeStrategy(FreezeStrategy):
    """FreezeStrategy para el backbone truncado de EfficientNet-B0 (`features`, `avgpool`)."""

    @property
    def block_order(self) -> list[str]:
        return ["features", "avgpool"]
```

`unfreeze_from: "features"` (índice 0) = todo entrenable = el default del paper ("fine-tune
completo"). `unfreeze_from: "none"` = todo congelado (feature extractor puro, útil como ablación pero
no es lo que usa el paper). No hace falta granularidad más fina (por bloque MBConv) — el paper no la
usa para este dataset.

**Criterio de aceptación A**: forward `(2,3,224,224) -> (2,1280)` tras el backbone, `(2,1280) ->
(2,8)` con la cabeza (`ConfigurableMLPHead(in_features=1280, hidden_layers=[], num_classes=8)`).
`unfreeze_from="features"` dejar `requires_grad=True` en el 100% del backbone; `"none"` en 0%.

### B — Optimizer SGD (`src/train/build.py`)

Una línea: `"sgd": torch.optim.SGD` en `_OPTIMIZERS` (línea 25-28). Nada más — `build_param_groups()`
y el resto de `build_optimizer()` ya son agnósticos al optimizer concreto.

**Criterio de aceptación B**: `build_optimizer(model.parameters(), "sgd", lr=0.01)` devuelve un
`torch.optim.SGD` y un `.step()` real mueve los pesos.

### C — Loss focal multiclase (`src/train/build.py`)

`_make_focal` (líneas 211-227) hoy fija `probs=lambda outputs: torch.softmax(outputs, dim=1)[:, 1]`
sin importar cuántas clases tenga la cabeza. Generalizar con un parámetro `num_classes` que, con
default `2`, reproduce EXACTAMENTE el comportamiento actual (ningún YAML existente cambia de
comportamiento):

```python
def _make_focal(*, device: str = "cpu", num_classes: int = 2, **hparams: Any) -> LossSpec:
    ...
    loss_fn = FocalLoss(**hparams).to(device)
    if num_classes == 2:
        probs_fn = lambda outputs: torch.softmax(outputs, dim=1)[:, 1]
    else:
        probs_fn = lambda outputs: torch.softmax(outputs, dim=1)
    return LossSpec(compute=lambda outputs, labels: loss_fn(outputs, labels), probs=probs_fn)
```

`build_loss()` (línea 237) debe aceptar y pasar `num_classes` solo a los fabricantes que lo acepten
(no es un hparam de `FocalLoss` — no debe terminar en su constructor). Mismo tratamiento opcional en
`_make_cross_entropy` si da tiempo, no es bloqueante para Fed-ISIC2019 (usa focal, no cross_entropy
plano, según el paper).

**Criterio de aceptación C**: `build_loss("focal", num_classes=8, alpha=[1]*8, gamma=2.0)` sobre
`logits=torch.randn(4,8)` da `probs.shape == (4,8)` y `probs.sum(dim=1)` ≈ 1. Con `num_classes=2`
(o sin pasarlo), comportamiento byte-idéntico al actual.

### D — Mínimo indispensable para que el pipeline multiclase no crashee

**Esto es la ÚNICA zona donde el trabajo roza "métricas" — y está acotado a propósito a lo mínimo
para no crashear, NO a calcular bien balanced accuracy multiclase. No tocar `src/metrics.py` ni
`src/reporting.py` en ningún punto de este ítem.**

**D1. `src/datasets/manifest.py::normalize_labels()`** (líneas 80-96): hoy el mapa `{"benign": 0,
"malignant": 1}` está hardcodeado. Generalizar para aceptar un mapa de etiquetas arbitrario (ej. las
8 clases de ISIC2019) — el mecanismo más simple y menos invasivo: agregar un parámetro opcional
`label_map: dict[str, int] | None = None` a `normalize_labels()` (default `None` = el mapa binario
actual, cero cambio de comportamiento para mamografía); cuando se pasa, se usa ese mapa en vez del
binario. Decidir junto con quien llama a `Manifest` cómo ese `label_map` llega desde el config (puede
ser un campo nuevo en el config de datos, o simplemente construirse en el loader del manifest de
ISIC2019 antes de llamar `normalize_labels()` — lo segundo es menos invasivo y evita tocar
`src/config.py`).

**D2. `src/train/loop.py` (`train_one_epoch`, `evaluate`)**: agregar un parámetro `num_classes: int =
2` a ambas funciones. Con el default, comportamiento idéntico a hoy. Con `num_classes > 2`: **saltar
por completo** la construcción de `build_metric_collection(device)` y la llamada a `metrics.update(...)`
— el `return` de ambas funciones queda como `{"loss": total_loss / n_batches}` (más `"prox_loss"` si
hay `regularizer`, igual que hoy). Comentario corto explicando que esto es un bypass deliberado,
scopeado fuera en `docs/FEDISIC2019_FLAMBY_REPRO.md §5.D`, no un diseño definitivo.

**D3. `src/federated/client.py`**: dos puntos que asumen el keyset binario y van a `KeyError`/
`format` fallido si `evaluate()`/`train_one_epoch()` devuelven solo `{"loss": ...}`:
- Líneas 230 y 282 (`_log_line` f-strings con `val_auc`): usar `.get("auc")` con un valor por defecto
  o condicionar el mensaje de log a si la clave existe.
- `_ROUNDS_CSV_FIELDS` (líneas 67-70): hoy una lista fija de 8 columnas. Con `num_classes > 2`,
  `rounds.csv` solo va a tener `round`, `num_examples`, `loss` disponibles — construir el fieldnames
  dinámicamente a partir de las claves que realmente trae el primer `metrics` reportado, en vez de la
  lista fija (o mantener la lista fija para el caso binario y una lista reducida para el multiclase —
  lo que sea menos invasivo, decidirlo y documentarlo).
- `fit()`/`evaluate()` de `FedMammoBenchClient` deben pasar `num_classes` a `train_one_epoch`/
  `evaluate_loader` — viene de `self.config.head.hparams["num_classes"]` o similar, ya disponible en
  el config del nodo.

**D4. `src/federated/round_tracking.py`**: con las métricas limitadas a `loss`, el YAML de este
experimento DEBE fijar `tracking.best_metric_name: loss` y `tracking.best_metric_mode: min` (ver
`§5.H`, el ejemplo de YAML) — `round_tracking.py:165` arma la clave como `f"val_{best_metric_name}"`,
así que `"val_loss"` existe siempre y no requiere ningún cambio de código ahí. Documentarlo en el
informe como la limitación conocida de esta vuelta, no intentar arreglarlo con un metric_name falso.

**D5. `src/federated/evaluate_node.py`**: hacer un grep rápido antes de tocarlo — probablemente asume
binario para su reporting final (curvas, matriz de confusión). Para esta vuelta, con `num_classes >
2` debe **omitirse sin crashear** (loggear que se omite esa evaluación final, no reimplementar su
reporting multiclase).

**Criterio de aceptación D**: un run sintético de 2-3 rondas con un DataLoader de tensores aleatorios
(`torch.randn(B,3,224,224)`, labels `randint(0,8,(B,))`) corre de punta a punta por
`train_one_epoch`/`evaluate`/`FedMammoBenchClient.fit()`/`.evaluate()` sin excepciones, con
`metrics.csv`/`rounds.csv` escribiéndose con el keyset reducido.

### E — Estrategias nuevas (`src/federated/strategies.py` + archivos nuevos)

**E1. FedAdagrad** (trivial): ya existe en `flwr.server.strategy` — agregar el import y la entrada
`"fedadagrad": FedAdagrad` en `_STRATEGIES` (línea 20 y 30-35). Mismo contrato hparams que
FedAdam/FedYogi (`eta`, `eta_l`, `beta_1` no aplica a Adagrad puro — verificar la firma real de
`flwr.server.strategy.FedAdagrad` antes de splatear hparams, puede no aceptar `beta_2`/`tau`).

**E2. Cyclic** (nuevo, pero más simple que Scaffold): NO encaja como una estrategia de agregación
simultánea — el paper la define como entrenar un cliente, pasar el modelo entrenado al siguiente, en
secuencia, dando una vuelta completa a todos los clientes por "ronda". Diseño recomendado sobre la
API de `flwr.server.strategy.Strategy`:
- `min_fit_clients=1`, `fraction_fit` efectivamente irrelevante — `configure_fit()` sobreescrito para
  seleccionar **un solo** cliente por ronda (de `client_manager.all()`, en un orden estable —
  ordenar por `cid` para la variante "fija"; con una permutación de semilla fija que se reconsulta
  cada vez que se completa una vuelta completa a todos los clientes, para la variante "shuffled").
- `aggregate_fit()` no promedia nada: los parámetros devueltos por ese único cliente SON los nuevos
  parámetros globales (N=1, sin ponderación).
- `aggregate_evaluate()`/evaluación: puede seguir siendo la evaluación normal sobre todos los nodos
  (eso no cambia por Cyclic — Cyclic solo afecta el FIT).
- Implementar en `src/federated/cyclic.py` (`CyclicStrategy`, hparam `order: Literal["fixed",
  "shuffled"] = "fixed"`), registrar `"cyclic": CyclicStrategy` en `_STRATEGIES`. El cliente
  (`client.py`) **no necesita ningún cambio** — desde su perspectiva es un fit normal.

**E3. Scaffold** (el más caro, bespoke — no hay clase stock de flwr para esto en 1.31.0, verificarlo
antes igual). Fórmula (Karimireddy et al. 2020, *SCAFFOLD*, opción II):

En la ronda `t`, el servidor tiene el modelo global `x_t` y el control-variate global `c` (vector del
mismo tamaño que los parámetros entrenables). Cada cliente `i` mantiene su propio `c_i` **persistente
entre rondas** (estado de instancia de `FedMammoBenchClient`, inicializado a ceros la primera vez que
ese cliente hace `fit()`). En su ronda de entrenamiento local (E=100 updates, ver §5.F), en vez de
`w <- w - eta_l * grad`, cada paso de SGD es:

```
w <- w - eta_l * (grad(w) - c_i + c)
```

Al terminar los E updates (`y_i` = pesos finales locales), el cliente calcula:

```
c_i_nuevo = c_i - c + (1 / (E * eta_l)) * (x_t - y_i)
delta_y_i = y_i - x_t
delta_c_i = c_i_nuevo - c_i
c_i <- c_i_nuevo          # se guarda para la próxima ronda de ESTE cliente
```

y devuelve `(delta_y_i, delta_c_i)` al servidor (ver mecanismo de transporte abajo). El servidor
agrega (participación completa, `|S|=N=num_nodes`):

```
x_{t+1} = x_t + server_lr * (1/N) * sum_i delta_y_i
c_{t+1} = c + (1/N) * sum_i delta_c_i
```

**Transporte sobre Flower**: el canal `config` (dict) de flwr solo acepta `Scalar` (str/int/float/
bool/bytes) — no tensores — así que `c`/`c_i` NO viajan por ahí. Usar el mismo truco que las
implementaciones de referencia de Scaffold sobre Flower: extender el `Parameters`/`NDArrays` que el
servidor manda en `configure_fit()` concatenando, al final de los arrays del modelo, los arrays de
`c` (mismas formas que los tensores entrenables, en el mismo orden que
`get_model_ndarrays(model, scope)`); el cliente sabe dónde cortar porque conoce `len(model_arrays)` de
antemano (es el mismo split que ya usa `set_model_ndarrays`/`get_model_ndarrays`,
`src/federated/param_utils.py`). El cliente devuelve, en su respuesta de `fit()`, `[*delta_y_i_arrays,
*delta_c_i_arrays]` con la misma convención. Señalizar que Scaffold está activo vía el `config` de
ronda (`"scaffold_active": True` en el dict que arma `make_on_config()` o una variante propia) para
que `client.py` sepa que debe aplicar la corrección de gradiente y partir los arrays recibidos/
devueltos en dos mitades — en vez de intentar inferirlo de la longitud, que es más frágil.

Vive en `src/federated/scaffold.py` (`ScaffoldStrategy(Strategy)`, hparam `server_lr`). El estado
`c` global es un atributo de instancia de `ScaffoldStrategy`, inicializado a ceros en
`initialize_parameters()` o en el primer `aggregate_fit()`. El estado `c_i` de cada cliente vive en
`FedMammoBenchClient.__init__` (un dict `self._scaffold_c: dict[str, Tensor] | None = None`,
poblado perezosamente la primera vez que `fit()` ve `scaffold_active=True`).

`train_one_epoch`/la nueva función de §5.F necesitan aceptar este término de corrección exactamente
igual que ya aceptan `regularizer` para FedProx (mismo patrón: un `Callable[[], Tensor] | None`, pero
acá el término se suma al GRADIENTE, no a la loss — revisar si conviene un segundo parámetro
`grad_correction: Callable[[nn.Module], None] | None` que se aplica a los `.grad` de los parámetros
justo después de `loss.backward()` y antes de `optimizer.step()`, en vez de forzar que viva dentro de
la loss como el proximal de FedProx).

**Criterio de aceptación E**: Cyclic — un toy de 3 "clientes" en memoria (sin red real), cada uno con
datos deterministas distintos, 3 rondas: el modelo final es literalmente el resultado de entrenar
secuencialmente sobre los 3 (verificable comparando pesos contra una implementación de referencia
hecha a mano en el script de verificación). Scaffold — un toy de heterogeneidad fuerte (2-3 "clientes"
sintéticos con distribuciones de datos muy distintas, ej. regresión lineal con covariables
desplazadas), corriendo FedAvg puro vs Scaffold el mismo número de rondas: Scaffold debe converger a
una pérdida pooled final igual o menor que FedAvg en ese toy heterogéneo (la propiedad que Scaffold
promete). Además, integración real sobre gRPC: 2 nodos del MNIST smoketest (`manifests/
mnist_smoketest/`, ya existente, ver `CLAUDE.md` raíz) con `strategy.name: scaffold`/`cyclic`/
`fedadagrad`, 3-5 rondas, verificando que el proceso termina sin excepción y que `rounds.csv` muestra
pérdida descendiendo.

### F — Modo "E=100 local updates" por ronda

Hoy el nodo cuenta en épocas locales completas (`local_epochs`, `src/federated/config.py:107`,
consumido en `client.py:152,189`). El paper cuenta en **batches**, con wraparound si `E` excede los
batches de una época.

**F1. `src/federated/config.py`**: agregar `local_updates: int | None = None` como campo hermano de
`local_epochs` en `FederationConfig` (línea 106-107). Ambos presentes a la vez es un error de
validación (`model_validator` que lo chequee, mismo patrón `extra="forbid"` del resto del repo).

**F2. `src/federated/strategies.py::make_on_config()`** (líneas 89-128): el dict que arma debe incluir
`"local_updates"` cuando el server.yaml lo trae (en vez de, o junto a, `"local_epochs"` — decidir si
mandar ambos con uno de los dos en `None`/ausente, o mandar siempre los dos y que el cliente mire cuál
de los dos no es `None`).

**F3. Nueva función en `src/train/loop.py`**: `train_n_updates(model, loader, optimizer, loss_spec,
device, n_updates, freeze_bn_stats=True, regularizer=None, grad_correction=None, num_classes=2)` —
mismo cuerpo que `train_one_epoch` pero iterando con un iterador que da la vuelta sobre `loader`
cuantas veces haga falta hasta completar exactamente `n_updates` batches (`itertools.cycle(loader)` +
`itertools.islice(..., n_updates)`, o un loop manual con `try/except StopIteration` + reconstruir el
iterador — cualquiera de los dos, documentar cuál y por qué). Las métricas devueltas se acumulan sobre
esos `n_updates` batches (pueden pisar el mismo dato más de una vez si `n_updates` > batches por
época — es el comportamiento que el paper documenta y asume).

**F4. `src/federated/client.py::fit()`**: branch — si `config` trae `"local_updates"` (no `None`),
usar `train_n_updates` UNA vez por ronda (no en un loop de épocas); si trae `"local_epochs"`, el
camino actual sin cambios. El log por época/update y el `metrics.csv` por fila pueden simplificarse a
una sola fila por ronda en el modo `local_updates` (no hay "épocas locales" que loguear una por una) —
decidirlo y documentarlo, no es necesario que `metrics.csv` tenga el mismo significado de fila en
ambos modos.

**Criterio de aceptación F**: con un loader de 10 batches y `n_updates=25`, `train_n_updates` procesa
exactamente 25 batches (verificable contando llamadas a `optimizer.step()` con un mock/spy), dando 2
vueltas completas al loader + 5 batches de una tercera.

### G — Augmentations faltantes (`src/datasets/transform.py`)

Hoy `TransformBuilder` solo tiene flip horizontal/vertical, rotación y blur gaussiano (líneas 93-114).
Agregar, cada una como flag booleano + hparams propios (mismo patrón que los existentes, ej.
`horizontal_flip`/`horizontal_flip_p`):
- `random_scale` (`transforms.RandomResizedCrop` con rango de escala, o `transforms.RandomAffine`
  con `scale=`).
- `random_brightness_contrast` (`transforms.ColorJitter(brightness=..., contrast=...)` — **ojo**:
  `CLAUDE.md` ya documenta que `ColorJitter` "destruye TIFFs float" para el pipeline de mamografía
  (ver memoria `diagnostico-sobreajuste-2026-09`); para ISIC2019 las imágenes son RGB uint8 normales
  vía JPG/PNG, no TIFF float — confirmarlo antes de reusar `ColorJitter` sin preocupación, pero NO
  cambiar el comportamiento default para los YAML de mamografía existentes (el flag debe ser opt-in,
  default `False`).
- `affine` (`transforms.RandomAffine` con `degrees`/`translate`/`shear` según lo que falte además de
  `random_scale`).
- `random_crop` (`transforms.RandomCrop` o `RandomResizedCrop` sobre el tensor ya reseteado a 224).
- `coarse_dropout` — no existe nativo en `torchvision.transforms`; puede portarse del INC si ese
  proyecto ya lo tiene (ver si `src/models/classification_images/` del repo hermano INC lo usa,
  CLAUDE.md raíz documenta el cross-reference entre proyectos) o implementarse como una clase simple
  (`RandomErasing` de torchvision es funcionalmente equivalente — evaluarlo como sustituto antes de
  escribir uno propio).

**Criterio de aceptación G**: cada augmentation nueva, aplicada sobre un tensor sintético `(3,224,224)`
con `p=1.0`, produce una salida de la MISMA forma y dtype, pero distinta del input (`torch.equal(...)`
es `False`). Con todos los flags en `False` (default), la salida es byte-idéntica a la de hoy.

### H — Dataset Fed-ISIC2019: adquisición + manifest + YAMLs

**H1. Descarga**: ISIC2019 challenge dataset, público — buscar la fuente oficial (ISIC archive /
challenge page citada en el paper, referencias [TRK18]/[CGC+18]/[CCR+19] de FLamby) y documentar en
el informe exactamente de dónde se bajó y qué licencia tiene (`CC BY-NC 4.0`, ya lo dice el paper). Si
la máquina de desarrollo tiene disco/internet para los ~9 GB, bajarlo; si no, **no es bloqueante**
(ver §2) — documentar el código/instrucciones de descarga igual, para correrlo en el despliegue, y
seguir con H2-H4 usando una muestra chica (unas pocas imágenes por clase/cliente, sintéticas o reales)
para verificar que el manifest y los YAML cargan bien.

**H2. Split por cliente**: el paper reconstruye la metadata de dispositivo de adquisición a partir del
dataset público (Apéndice H.1) — si esa metadata no viene en el CSV público estándar de ISIC2019,
revisar si el repo oficial de FLamby (`github.com/owkin/FLamby`, citado en el paper) publica el split
ya hecho (un CSV o lista de IDs por cliente) en vez de reconstruirlo desde cero; usar eso si existe, es
muchísimo más confiable que re-derivar la heurística del paper. Documentar la fuente exacta.

**H3. Manifest**: construir un CSV por cliente (6 manifests) bajo
`manifests/fed_isic2019/<cliente>.csv`, con las columnas del esquema de este repo
(`preprocessed_image_path`, `classification`, `split`, `patient_id`) — `classification` lleva el
nombre de clase (ej. `"melanoma"`, `"melanocytic_nevus"`, ...), no `benign`/`malignant`; `split` es
`train`/`test` según el split estático del paper (no hay val separado en el paper — ver si este repo
necesita un val propio recortado del train, documentando el criterio). `patient_id`: si ISIC2019 no
trae ID de paciente real, usar el ID de imagen como proxy y decirlo explícitamente (compromete el
invariante de "no fuga de paciente entre splits" documentado en `check_patients_id()`
(`manifest.py:63-78`) — reportarlo, no silenciarlo).

**H4. YAMLs**: servidor + 6 nodos para la celda **FedProx** (la más fácil de verificar, el término
proximal ya existe del lado del cliente), siguiendo el esquema de
`configs/federated/exp44_fedgrid_fedprox_r10/` como plantilla de forma (no de contenido — los valores
cambian). Ejemplo ya preparado en la conversación previa de este ciclo (adjuntar/citar si quedó
guardado en el historial de Antigravity, si no, reconstruir con los valores de §4.3): `strategy.name:
fedprox`, `proximal_mu: 0.001`, `architecture.name: efficientnet_b0_imagenet`, `unfreeze_from:
features`, `head.num_classes: 8`, `optimizer.name: sgd`, `optimizer.hparams.lr: 0.01`, `loss.name:
focal`, `loss.hparams.gamma: 2.0`, `loss.hparams.alpha` = vector de 8 pesos (inverso de proporción de
clase sobre el manifest pooled — calcularlo de verdad sobre los 6 manifests juntos, no inventarlo),
`tracking.best_metric_name: loss`, `tracking.best_metric_mode: min` (ver §5.D4), `rounds: 10` (ver
nota de T_max en §4.3), `federation.local_updates: 100` (nunca `local_epochs`).

Generar también las 6 celdas restantes (FedAvg, FedAdagrad, FedAdam, FedYogi, Cyclic, Scaffold) con los
valores de la tabla de §4.3 — mecánicamente, una vez el YAML de FedProx está bien formado el resto es
cambiar `strategy.name`/`hparams`; **no hace falta haber corrido FedProx con datos/GPU reales primero**
para escribir y dejar listas las otras 6 (ver §2 — esta vuelta es de desarrollo, la corrida real de las
7 celdas es parte del despliegue, no de este entregable).

**Criterio de aceptación H**: los 6 manifests cargan con `Manifest` (una vez generalizado
`normalize_labels`, §5.D1) sin excepción; si se pudo bajar el dataset completo, los conteos de fila por
split calzan con la Tabla 8 del paper (§4.1) dentro de un margen razonable (el paper excluye algunas
imágenes por falta de metadata — no tiene que ser exacto al dígito, pero si difiere mucho, investigar
por qué antes de seguir). Los 7 YAML (uno por estrategia) cargan sin error de Pydantic. Correr al menos
1-2 rondas reales con datos/GPU **si están disponibles en la máquina de desarrollo**; si no, dejarlo
documentado como pendiente de validar en el despliegue — no es un criterio bloqueante de esta vuelta.

---

## 6. Qué NO hacer

- **No tocar `src/metrics.py` ni la fórmula de `balanced_accuracy`/`BinaryConfusionMatrix` de
  `src/reporting.py`.** Es la exclusión central de este brief (ver §1, §5.D). Si en el camino parece
  "casi gratis" arreglarlo bien, igual no — queda para una vuelta separada del usuario.
- No implementar ni correr el baseline *pooled* (no-federado) del paper — ni como script separado ni
  como `configs/exp*.yaml` centralizado. Esta vuelta es 100% sobre `src/federated/`.
- No renombrar ni tocar las estrategias ya registradas (`fedavg`/`fedprox`/`fedadam`/`fedyogi`) más
  allá de agregar `fedadagrad` al dict.
- No tocar el pipeline de mamografía existente: ningún YAML de `configs/exp*.yaml` ni
  `configs/federated/exp4*/` se edita. Todos los defaults nuevos (`num_classes`, `label_map`,
  `local_updates`, los flags de augmentation nuevos) deben ser opcionales con un default que reproduce
  el comportamiento actual byte-idéntico para esos configs.
- No intentar que `T_max` calce exacto con el valor interno de `flamby` — usar la aproximación de
  §4.3 y decirlo.
- No commitear ni pushear — dejar los cambios en el árbol de trabajo.

---

## 7. Entregable

`docs/FEDISIC2019_FLAMBY_REPRO_RESULTADOS.md` con:
1. Lista de archivos tocados/creados, con diff o resumen fiel de cada uno (A-H).
2. Salida literal de cada verificación de §5 (criterios de aceptación A-H).
3. Fuente exacta de descarga de ISIC2019 y del split por cliente (H1/H2), con el conteo de filas real
   por cliente comparado contra la Tabla 8 del paper.
4. Cualquier desviación de este brief, con su motivo — en particular, cualquier decisión de diseño
   tomada en Scaffold/Cyclic que este brief dejó abierta (transporte exacto de arrays, formato de
   `rounds.csv` reducido, etc.).
5. Explícito: confirmar que `src/metrics.py` y `src/reporting.py` quedaron sin tocar (`git diff
   --stat` de esos dos archivos debe estar vacío).

---

## 8. Verificación

Scripts desechables fuera del repo, salida pegada en el informe.

1. **Smoke de arquitectura** (§A): forward EfficientNet-B0 truncado + cabeza, conteo de
   `requires_grad` para `unfreeze_from="features"`/`"none"`.
2. **Smoke de optimizer/loss** (§B/§C): un `.step()` de SGD mueve los pesos; `build_loss("focal",
   num_classes=8, ...)` da `probs.shape==(4,8)` que suma ~1 por fila.
3. **Harness multiclase sintético** (§D): 2-3 rondas con tensores aleatorios de punta a punta por
   `train_one_epoch`/`evaluate`/`FedMammoBenchClient`, sin excepción, `rounds.csv` con el keyset
   reducido.
4. **Cyclic** (§E): toy de 3 clientes en memoria, comparar pesos finales contra entrenamiento
   secuencial de referencia hecho a mano.
5. **Scaffold** (§E): toy heterogéneo, Scaffold converge a pérdida pooled igual o mejor que FedAvg en
   el mismo número de rondas.
6. **Integración real sobre gRPC** (§E, Scaffold/Cyclic/FedAdagrad): 2 nodos del MNIST smoketest ya
   existente (`manifests/mnist_smoketest/`), 3-5 rondas por estrategia nueva, termina sin excepción,
   pérdida desciende.
7. **Local updates** (§F): contador de `optimizer.step()` da exactamente `n_updates` para varios
   valores de `n_updates` mayores y menores que el tamaño del loader.
8. **Augmentations** (§G): cada flag nuevo en `p=1.0` cambia el tensor; todos en `False` reproduce el
   tensor de hoy exacto.
9. **Manifest + YAMLs** (§H): los 7 YAML (uno por estrategia) cargan sin error de Pydantic; si hubo
   datos reales disponibles, conteos de fila por cliente/split vs Tabla 8 del paper, y 1-2 rondas de
   FedProx corriendo sin excepción. Si no hubo datos/GPU en la máquina de desarrollo, dejarlo anotado
   como pendiente de validar en el despliegue — no bloquea el resto de la verificación (ver §2).
10. `.venv/bin/python -c "import src.federated.server; import src.federated.client;
    import src.federated.strategies; import src.federated.scaffold; import src.federated.cyclic"`
    (ajustar nombres de módulo a lo que realmente se haya creado).
11. `git diff --stat src/metrics.py src/reporting.py` vacío (confirma §6).
