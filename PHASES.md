# Integración INC → FedMammoBench: bitácora por fase

Este documento registra, fase por fase, qué se portó del proyecto INC
(`inc-project-models-classification-detection-main/src/models/classification_images/`)
hacia `src/` de FedMammoBench, qué se modificó y qué se añadió. Análisis previo
completo (componente → impacto → propuesta) en el historial de la sesión que
originó esta rama; este archivo documenta la *implementación*, no la
justificación completa de cada decisión.

Convención de ramas: `develop` es la base de integración; cada fase vive en
`phaseN-<nombre>`, se implementa ahí y se mergea a `develop` con
`--no-ff` para dejar el límite de cada fase visible en `git log --graph`.
`main` no se toca hasta que `develop` esté validado.

---

## Fase 1 — Quick fixes: métricas, `drop_last`, determinismo cuDNN

Rama: `phase1-quick-fixes`. Sin dependencias de las fases siguientes — todo
lo demás se apoya en estos tres cambios.

### Modificado

- **[`src/metrics.py`](src/metrics.py)** — `build_metric_collection()` gana
  `"f1": BinaryF1Score()` y `"precision": BinaryPrecision()`. El proyecto INC
  selecciona su mejor checkpoint por F1 de validación
  (`classification_images/early_stopping.py`, ver Fase 3); sin F1 en la
  colección, `metric_name: f1` en el YAML no era ni siquiera expresable.
  `precision` es el equivalente de torchmetrics al `vpp()` que el INC calcula
  a mano (`TP/(TP+FP)`) — se reutiliza la métrica ya provista por la
  librería en vez de portar la función.
- **[`src/datasets/build.py`](src/datasets/build.py)** — `builder_dataloader()`
  añade `drop_last=True` al `DataLoader` de `"train"` (val/test sin cambios).
  `REFACTOR.md` §6 ya marcaba esto como obligatorio y nunca se implementó:
  la cabeza (`StandardMLPHead`) usa `BatchNorm1d`, que lanza excepción con un
  batch de una sola muestra.
- **[`src/seed.py`](src/seed.py)** — `set_global_seed()` gana un parámetro
  `cudnn_deterministic: bool = True` que fija
  `torch.backends.cudnn.deterministic = True` y `benchmark = False`. Sin
  esto, dos corridas con la misma semilla podían dar resultados distintos en
  GPU — necesario para poder atribuir cualquier diferencia numérica frente
  al baseline del INC a un cambio de código real, no a ruido de cuDNN.

### Añadido

Nada nuevo en esta fase — son extensiones puntuales de módulos existentes.

### Compatibilidad

Todos los cambios son aditivos con default que preserva el comportamiento
previo excepto `drop_last` (que corrige un bug de omisión, no introduce uno).
Ningún YAML existente deja de validar.


---

## Fase 2 — Balanceo de clases: fix del bug de pesos + Focal Loss

Rama: `phase2-loss-balancing`. Depende de fase 1 solo por orden de merge, no
por contenido.

### Modificado

- **[`src/train/build.py`](src/train/build.py)** — dos cambios:
  1. **Bug real corregido, confirmado en runtime.** `_make_bce()` y
     `_make_cross_entropy()` ahora convierten `pos_weight`/`weight` de lista
     (lo único que YAML puede expresar) a `torch.Tensor` antes de construir
     la loss. Antes de este fix, `build_loss("cross_entropy",
     weight=[2.0, 1.0])` fallaba con:
     ```
     TypeError: cannot assign 'list' object to buffer 'weight' (torch
     Tensor or None required)
     ```
     (reproducido directamente contra `nn.CrossEntropyLoss` antes de
     aplicar el fix — ver verificación abajo). Es decir: el balanceo de
     clases parecía soportado por `build_loss(name, **hparams)`, pero era
     **inalcanzable desde config** en la práctica. Con el dataset
     ~66/34 (benigno/maligno) de `fedmammobench.csv`, esto bloqueaba
     justo el mecanismo que compensa el desbalance.
  2. **`build_loss()` gana un parámetro `device: str = "cpu"`**, mismo
     patrón que `build_metric_collection(device=...)` en `src/metrics.py`.
     Necesario porque el tensor de pesos debe vivir en el mismo device que
     los logits — antes no había forma de moverlo, ni de mover la loss en
     sí (`loss_fn.to(device)`, ahora aplicado en las tres fábricas).
  3. Añadido `_make_focal()` y registrado `"focal"` en `_LOSSES`.
- **[`src/cli.py`](src/cli.py)** — el único call site de `build_loss()`
  ahora pasa `device=config.train.device`.
- **[`src/train/__init__.py`](src/train/__init__.py)** — reexporta `FocalLoss`.

### Añadido

- **[`src/train/focal_loss.py`](src/train/focal_loss.py)** — `FocalLoss`,
  portada de `classification_images/focal_loss.py` del proyecto INC.
  Diferencia deliberada respecto al original: `alpha` se registra como
  buffer de `nn.Module` (`register_buffer`) en vez de como atributo plano,
  así `loss_fn.to(device)` lo mueve junto con el resto del módulo — el INC
  asumía `torch.device('cuda')` hardcodeado dentro de `losses.py`, lo que
  rompería en CPU.

### Verificación (runtime real, no solo `py_compile`)

Sin dependencias instaladas en el repo (ver `CLAUDE.md`), se instalaron en
un venv de scratch (`torch==2.13+cpu`, `torchvision`, `torchmetrics`,
`pandas`, `pydantic`, `tensorboard`, `matplotlib` — el set de
`requirements.txt` más `matplotlib` de la fase 4) para probar el código real
en vez de solo verificar sintaxis:

```python
build_loss("cross_entropy", weight=[2.0, 1.0], device="cpu")   # antes: TypeError: cannot assign 'list' object to buffer 'weight'
build_loss("bce", pos_weight=[2.0], device="cpu")               # antes: mismo error
build_loss("focal", alpha=[2.0, 1.0], gamma=2.0, device="cpu")  # nuevo esquema
build_loss("focal", device="cpu")                                # alpha=None también funciona
build_loss("bogus")                                              # ValueError, como antes
```
Las cinco llamadas pasan; `FocalLoss(alpha=...).to("cpu")` confirma que
`alpha` se mueve correctamente como buffer.

### Compatibilidad

`build_loss(name, **hparams)` sin `device` sigue funcionando (default
`"cpu"`) para cualquier código que aún no pase el parámetro. Losses sin
`weight`/`pos_weight`/`alpha` (el caso común hasta ahora) no cambian de
comportamiento.

---

## Fase 3 — EarlyStopping y el fix del "siempre maximiza"

Rama: `phase3-early-stopping`. Depende de fase 1 (necesita `f1`/`precision`
en `src/metrics.py` para que `metric_name: f1` tenga sentido).

### Añadido

- **[`src/train/early_stopping.py`](src/train/early_stopping.py)** —
  `EarlyStopping(patience, min_delta, mode)`, `dataclass` sin I/O:
  `step(value) -> bool` (¿esta época mejoró?) + atributos `best_value`,
  `counter`, `should_stop`. A diferencia del `EarlyStopping` del proyecto
  INC (`classification_images/early_stopping.py`), no guarda checkpoints
  dentro de `__call__` — esa responsabilidad se queda enteramente en
  `Trainer.fit()`, que ya la tenía.
  **Decisión de diseño no listada explícitamente en el análisis previo:**
  el mismo objeto resuelve dos problemas a la vez —"¿hay que guardar
  checkpoint?" y "¿hay que parar?"— porque ambos dependen exactamente del
  mismo `mode` (`max`/`min`). Tenerlos separados habría duplicado la
  lógica de comparación en dos sitios que podrían desincronizarse.

### Modificado

- **[`src/train/trainer.py`](src/train/trainer.py)** — `Trainer` gana
  `metric_mode`, `patience`, `min_delta`, y construye internamente
  `self.tracker = EarlyStopping(...)`. `fit()` reemplaza el
  `if current_metric > self.best_metric` hardcodeado (que **siempre
  maximizaba**, sin importar la métrica — bug ya señalado en el análisis
  previo) por `self.tracker.step(current_metric)`, y añade el `break` de
  early stopping. `self.best_metric` queda como propiedad de solo lectura
  que delega a `self.tracker.best_value`, para no romper código que ya lo
  leyera.
- **[`src/config.py`](src/config.py)** — `TrainConfig` gana `metric_mode:
  str = "max"`, `patience: int | None = None`, `min_delta: float = 0.0`.
  Todos con default que reproduce el comportamiento previo exacto
  (siempre maximizaba, nunca paraba antes de tiempo) para YAML que no los
  mencione.
- **[`src/cli.py`](src/cli.py)** — pasa los tres campos nuevos de
  `config.train` al constructor de `Trainer`.
- **[`src/train/__init__.py`](src/train/__init__.py)** — reexporta
  `EarlyStopping`.

### Verificación (runtime real)

1. **Unit tests de `EarlyStopping`** (modo `max`, modo `min`, `patience=None`
   nunca dispara, `mode` inválido lanza `ValueError`) — los cuatro casos
   pasan.
2. **`Trainer.fit()` end-to-end** con un MLP sintético de 2 capas sobre
   datos aleatorios (`torch.manual_seed(0)`), tres casos:
   - `patience=1, min_delta=1.0` (imposible de satisfacer) → confirma
     `tracker.should_stop=True` y que el loop corta antes de `epochs=20`
     (se detiene en la época 1), guardando igual un checkpoint válido.
   - `metric_name="loss", metric_mode="min"` → confirma que `best_metric`
     sigue bajando época a época. **Esto es la prueba directa del fix**:
     con el código anterior (`current_metric > self.best_metric`), una loss
     que *baja* nunca hubiera vuelto a superar el mejor valor tras la
     época 0, y el checkpoint se habría congelado ahí — con el fix,
     seguido baja limpiamente por 5 épocas.
   - Sin pasar `patience`/`metric_mode` (defaults) → corre las `epochs`
     completas sin activar `should_stop`, igual que el comportamiento
     previo a esta fase.
3. **`TrainConfig` validado desde YAML real**, con y sin los campos nuevos
   — ambos casos validan, y el caso sin campos nuevos recupera exactamente
   los defaults de compatibilidad (`max`, `None`, `0.0`).

### Compatibilidad

Un `ExperimentConfig` que no mencione `metric_mode`/`patience`/`min_delta`
se comporta exactamente igual que antes de esta fase — la única diferencia
observable es que, si `metric_name` fuera alguna vez `"loss"`, ahora
selecciona la *mejor* época en vez de congelarse en la primera.

---

## Fase 4 — Evaluación en test + artefactos (el gap más grave del análisis)

Rama: `phase4-test-evaluation`. Depende de fase 1 (F1/precision) y fase 3
(el `best_checkpoint` que devuelve `Trainer.fit()`).

Este era el hueco más serio identificado en el análisis previo:
`src/cli.py:run()` construía `loaders["test"]` (`datasets/build.py`) y
**nunca lo usaba** — la función terminaba en `trainer.fit()`. FedMammoBench
no podía producir ni una sola métrica de test, que es literalmente el
número que va al paper.

### Añadido

- **[`src/train/evaluation.py`](src/train/evaluation.py)** — dos funciones
  puras:
  - `evaluate_checkpoint(model, checkpoint_path, loader, loss_spec, device)`
    — carga el checkpoint in-place (reutiliza `checkpoint.load_checkpoint()`)
    y delega en `train/loop.py:evaluate()` para el resumen agregado. No
    reimplementa el forward pass.
  - `predict_on_loader(model, loader, loss_spec, device)` — recorre todo el
    loader acumulando `(y_true, y_pred, y_prob)` en memoria. Deliberadamente
    **separada** de `evaluate()`: esa función corre una vez por época
    durante entrenamiento y solo necesita el resumen; acumular todo el
    split en memoria solo vale la pena una vez, al final, sobre test.
- **[`src/reporting.py`](src/reporting.py)** — el equivalente al
  `test_model()` de 130+ líneas del proyecto INC, partido en cuatro
  funciones puras: `save_metrics_json()`, `save_predictions_csv()`,
  `plot_confusion_matrix()`, `plot_roc_curve()`. Usa `BinaryConfusionMatrix`
  y `BinaryROC` de **torchmetrics** (ya en `requirements.txt`, ya usado en
  `src/metrics.py`) en vez de scikit-learn como el INC — para no añadir una
  dependencia nueva solo por dos cálculos que la librería ya instalada
  provee. `matplotlib` sí es dependencia nueva (backend `"Agg"`, sin
  necesidad de X11/display).

### Modificado

- **[`src/cli.py`](src/cli.py)** — `run()` ahora, después de
  `trainer.fit()`: llama `evaluate_checkpoint()` sobre `loaders["test"]`
  con el `best_checkpoint` que `fit()` acaba de devolver (nunca el estado
  final del modelo, nunca una bandera de config aparte — la invariante que
  el análisis previo señalaba como obligatoria), imprime el resumen, y
  escribe `metrics.json`, `predictions.csv`, `plots/confusion_matrix.png`
  y `plots/roc_curve.png` en `config.train.run_dir`.
- **[`requirements.txt`](requirements.txt)** — añade `matplotlib>=3.11,<4.0`
  con comentario explicando por qué (antes deliberadamente excluido, "nada
  en `src/` lo importa" — ahora `reporting.py` sí).

### Verificación (runtime real, end-to-end completo)

Corrida completa contra un MLP sintético (no ResNet, para no depender de
pesos preentrenados que no existen en este entorno): `Trainer.fit()` de 5
épocas → `evaluate_checkpoint()` sobre un `test_loader` separado →
`predict_on_loader()` → los cuatro artefactos.

- `test_metrics` trae las 7 claves esperadas (`loss`, `accuracy`, `auc`,
  `sensitivity`, `specificity`, `f1`, `precision`).
- `len(y_true) == len(y_pred) == len(y_prob) == len(test_loader.dataset)`
  (16/16) — confirma que `predict_on_loader()` no pierde ni duplica
  muestras.
- `y_pred` todo en `{0, 1}`, `y_prob` todo en `[0, 1]`.
- Los cuatro archivos se crean, no están vacíos, y:
  - `metrics.json` recarga con `json.load()` y coincide byte a byte
    (como dict) con `test_metrics`.
  - `predictions.csv` tiene el header correcto y exactamente una fila por
    muestra de test (16 filas + header).
  - `plots/confusion_matrix.png` y `plots/roc_curve.png` se abren con
    `PIL.Image.open(...).verify()` sin error — son PNG válidos, no
    archivos truncados o corruptos.

### Compatibilidad

Cambio de comportamiento observable, intencional: correr `python -m
src.cli` ahora escribe archivos nuevos en `run_dir` (`metrics.json`,
`predictions.csv`, `plots/`) que antes no existían. Nada de lo que ya se
escribía (`config.yaml`, `metrics.csv` de `MetricsLogger`, checkpoints)
cambia de formato o ubicación.

---

## Fase 5 — Cabeza MLP configurable + augmentación (flip vertical, blur)

Rama: `phase5-head-augmentation`. Independiente de las fases 2-4 en
contenido, solo depende de fase 1 por orden de merge.

### Añadido

- **[`src/models/mlp_configs/configurable_mlp.py`](src/models/mlp_configs/configurable_mlp.py)**
  — `ConfigurableMLPHead(HeadBuilder)`: `hidden_layers: list[int]`
  arbitraria, `activation` elegible por nombre (`relu`, `leakyrelu`,
  `sigmoid`, `tanh`, `gelu`, `linear`), `use_batchnorm: bool = False`.
  Complementa a `StandardMLPHead` (no la reemplaza — ambas quedan
  registradas). Reproduce la cabeza `2048 → 1024 → 256` con GELU y
  dropout 0.5 del proyecto INC (`classification_images/models/mlp_models.py`),
  que `StandardMLPHead` (una sola capa oculta fija, `BatchNorm1d` siempre)
  no podía expresar.

### Modificado

- **[`src/models/heads.py`](src/models/heads.py)** — registra
  `"configurable_mlp"` en `_HEAD_STRATEGIES` junto a `"standard_mlp"`.
- **[`src/datasets/transform.py`](src/datasets/transform.py)** —
  `TransformBuilder` gana `use_vertical_flip`/`vertical_flip_p` y
  `use_blur`/`blur_p`/`blur_kernel_size`/`blur_sigma`, portados de
  `RandomVerticalFlip` y el `RandomGaussianBlur` casero del proyecto INC
  (`classification_images/dataloaders/dataloader_images.py`). El blur se
  implementa con `transforms.RandomApply([GaussianBlur(...)], p=...)` de
  torchvision en vez de portar la clase `nn.Module` propia del INC — el
  building block ya existe en la librería. Orden de augmentación
  preservado como PIL-first (Resize → flips/rotación/blur → ToTensor →
  Normalize), a diferencia del INC que hace `ToTensor` primero — ambos
  órdenes son válidos con torchvision, se mantiene la convención existente
  del repo en vez de adoptar la del INC.
- **[`src/config.py`](src/config.py)** — nuevo modelo `AugmentationConfig`
  (`horizontal_flip`, `horizontal_flip_p`, `rotation_degrees`,
  `vertical_flip`, `vertical_flip_p`, `blur`, `blur_p`), anidado en
  `DataConfig.augmentation`. Todos los defaults reproducen exactamente lo
  que `src/cli.py` tenía hardcodeado (`use_horizontal_flip=True,
  use_rotation=True`, sin flip vertical ni blur, `rotation_degrees` en el
  default de `TransformBuilder`) — antes de esta fase, ninguno de esos
  parámetros era alcanzable desde YAML.
- **[`src/cli.py`](src/cli.py)** — el `TransformBuilder` de train ahora se
  construye a partir de `config.data.augmentation` en vez de literales
  hardcodeados; `use_rotation` se deriva de `rotation_degrees > 0` (una
  sola fuente de verdad, no dos flags que puedan desincronizarse).

### Verificación (runtime real)

- **`ConfigurableMLPHead` con la arquitectura exacta del INC**
  (`in_features=2048, hidden_layers=[2048, 1024, 256], activation="gelu",
  dropout=0.5, num_classes=2, use_batchnorm=False`): `build()` produce la
  secuencia de 11 capas esperada (`Flatten → [Linear→GELU→Dropout]×3 →
  Linear`), un forward real con `torch.randn(4, 2048, 1, 1)` da salida
  `[4, 2]`, y se confirma que no hay ningún `BatchNorm1d` en el módulo.
- Casos borde: `hidden_layers=[]` produce un único `Linear` (sale `[2, 2]`
  con entrada `[2, 64]`); `use_batchnorm=True` sí inserta `BatchNorm1d`;
  `dropout=0.0` omite la capa `Dropout` por completo (no un `Dropout(p=0)`
  inerte); `activation="bogus"` lanza `ValueError`.
- Ambas cabezas (`"standard_mlp"`, `"configurable_mlp"`) resuelven
  correctamente vía `get_head_strategy()`.
- **`TransformBuilder`** con las cuatro augmentaciones activas
  (`use_horizontal_flip, use_rotation, use_vertical_flip, use_blur`) sobre
  una imagen PIL sintética real: el pipeline resultante trae
  `RandomVerticalFlip` y `RandomApply` (el blur) en el orden esperado, y
  produce un tensor `[3, 64, 64]` sin excepciones. El pipeline por defecto
  (sin ningún flag) sigue siendo exactamente `[Resize, ToTensor,
  Normalize]`, igual que antes de esta fase.
- **`AugmentationConfig()` sin argumentos** reproduce los defaults exactos
  que `cli.py` tenía hardcodeados antes de esta fase.
- **`ExperimentConfig` completo, dos casos**: un YAML "old-style" sin
  bloque `augmentation` ni campos nuevos de `train` valida igual que antes;
  un YAML equivalente al experimento del proyecto INC
  (`configurable_mlp` con `hidden_layers: [2048, 1024, 256]`,
  `cross_entropy` con `weight: [2.0, 1.0]`, augmentación completa,
  `metric_name: f1`, `patience: 50`) valida de punta a punta.

### Compatibilidad

Un `ExperimentConfig` que no mencione `head.name: configurable_mlp` ni
`data.augmentation` se comporta exactamente igual que antes de esta fase —
verificado explícitamente arriba, no solo argumentado.

---

## Fase 6 — Checkpoints periódicos, desglose backbone/cabeza, duración por época

Rama: `phase6-checkpoint-tracking`. Última fase de código de esta
integración — mejoras de observabilidad, no bloqueantes.

### Modificado

- **[`src/checkpoint.py`](src/checkpoint.py)** — `save_checkpoint()` gana
  `extra_state_dicts: dict[str, nn.Module] | None = None`. Cuando se pasa,
  guarda además cada submódulo por separado
  (`<path.stem>_<nombre><path.suffix>`), junto al checkpoint principal.
  Generaliza `Best_Model_image.pth`/`Best_Model_classifier.pth` del
  proyecto INC — pensado para cuando llegue la fase federada, donde solo
  el backbone se agrega entre nodos: tener el checkpoint ya partido evita
  reconstruir esa separación después.
- **[`src/train/trainer.py`](src/train/trainer.py)** — `Trainer` gana
  `save_every: int | None = None` (checkpoints periódicos
  `epoch{N}.pt`, independientes del mejor). Además:
  - `_split_state_dicts()`: si `self.model` es `nn.Sequential(backbone,
    head)` (la forma exacta en la que `src/cli.py` ensambla el modelo),
    devuelve `{"backbone": ..., "head": ...}` para pasarlo a
    `save_checkpoint(extra_state_dicts=...)` en cada mejor checkpoint.
    Cualquier otra forma de modelo devuelve `None` — el desglose es una
    conveniencia, no un requisito.
  - Cada época registra `duration_seconds` (via `time.time()`) en el dict
    que se pasa a `MetricsLogger.log()` — sin cambios en `MetricsLogger`
    en sí, que ya persiste lo que reciba.
- **[`src/config.py`](src/config.py)** — `TrainConfig` gana `save_every:
  int | None = None`, portado del guardado periódico implícito del
  proyecto INC (`training.py`, cada 10 épocas).
- **[`src/cli.py`](src/cli.py)** — pasa `config.train.save_every` al
  constructor de `Trainer`.

### Añadido

Nada nuevo en esta fase — son extensiones de módulos ya existentes.

### Verificación (runtime real)

Entrenamiento de 6 épocas con `save_every=2` sobre un modelo
`nn.Sequential(nn.Sequential(Linear, ReLU), Linear)` (la forma exacta
`backbone + head` que ensambla `cli.py`):

- Checkpoints periódicos `epoch0.pt`, `epoch2.pt`, `epoch4.pt` existen
  (cada 2 épocas, como se pidió).
- Para el mejor checkpoint, existen `best_epoch3_backbone.pt` y
  `best_epoch3_head.pt` junto al `.pt` principal.
- Esos dos archivos **cargan de verdad** con `load_state_dict()` en
  submódulos frescos idénticos en forma — no solo "el archivo existe", sino
  que el `state_dict` que contienen es válido para esa arquitectura.
- `metrics.csv` de la corrida trae `duration_seconds` en cada fila, todas
  no negativas.

### Compatibilidad

`save_every=None` (default) no guarda ningún checkpoint periódico, igual
que antes de esta fase. `extra_state_dicts=None` (default) en
`save_checkpoint()` no guarda ningún archivo adicional. Un modelo que no
sea `nn.Sequential` de longitud 2 nunca activa el desglose backbone/cabeza
— no falla, `_split_state_dicts()` simplemente devuelve `None`.

---

## Fuera de alcance de esta integración (deliberado)

Del análisis original, dos puntos quedan fuera de estas seis fases:

- **B4 — equivalencia `num_freeze` (INC) ↔ `unfreeze_from` (FedMammoBench).**
  Es un análisis, no código: el proyecto INC congela por índice de
  parámetro (`freeze_layers(model, num_freeze=80)`,
  `classification_images/models/image_models.py`) mientras FedMammoBench
  congela por nombre de bloque (`ResNetFreezeStrategy.block_order`). Para
  saber a qué `unfreeze_from` equivale `num_freeze=80` hace falta
  instanciar un `resnet50` real y contar `len(list(m.parameters()))`
  acumulado por bloque — esta sesión no tiene pesos de RadImageNet
  disponibles para reproducir el experimento exacto del INC, solo un venv
  de scratch para verificar lógica con tensores sintéticos. Queda como
  tarea abierta, no resuelta por omisión.
- **C4 — inspección visual del batch** (`show_batch_images()` del INC,
  guarda un PNG de muestras del train loader con sus etiquetas al
  arrancar). Prioridad baja en el análisis original; no implementado.

### Cierre de fases

Con esta fase termina la implementación planeada. `develop` tiene las seis
fases mergeadas con `--no-ff`; `main` no fue tocado. Antes de fusionar
`develop` a `main`, validar contra un experimento real (pesos RadImageNet +
`manifests/fedmammobench.csv`) — todo lo de arriba se verificó con modelos
y datos sintéticos en un venv de scratch, nunca contra el pipeline completo
con datos reales, porque ese entorno no está disponible en esta sesión (ver
`CLAUDE.md`: sin venv, sin dependencias instaladas en el repo).
