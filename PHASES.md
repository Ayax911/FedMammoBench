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
