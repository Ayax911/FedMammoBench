# Documentación de la carpeta

`src/train/` construye optimizer/scheduler/loss por nombre, define el loop puro de
entrenamiento/evaluación de una época, y orquesta el `fit()` de varias épocas guardando
solo el mejor checkpoint. Flujo típico:

```python
optimizer = build_optimizer(model.parameters(), "adamw", lr=1e-4)
loss_spec = build_loss("cross_entropy")
scheduler = build_scheduler(optimizer, "cosine", T_max=20)

trainer = Trainer(model, optimizer, loss_spec, checkpoint_dir="runs/exp/weights",
                   device="cuda", scheduler=scheduler, metric_name="auc")
best_ckpt_path = trainer.fit(train_loader, val_loader, epochs=20)
```

## build.py

### build_optimizer(params, name, **hparams) / build_scheduler(optimizer, name, **hparams)

Dict dispatch simple: `name` es una clave en `_OPTIMIZERS`/`_SCHEDULERS`
(`"adam"`/`"adamw"`; `"cosine"`/`"reduceonplateu"`), `**hparams` son los kwargs propios de
esa clase de PyTorch (`lr`, `weight_decay`, `T_max`, `patience`, ...).

Raises: `ValueError` si `name` no está registrado.

### LossSpec

Empareja, para un esquema de salida dado, la función de pérdida con la forma correcta
de convertir logits crudos en la probabilidad de la clase positiva. Existe porque esa
conversión depende de **cuántos logits emite la cabeza** — 1 con BCE, 2 con
CrossEntropy — nunca del nombre de la loss ni de un flag aparte que se pueda
desincronizar del modelo real construido en `src/models/`.

`train_one_epoch()`/`evaluate()` (`train/loop.py`) reciben un `LossSpec` en vez de una
`nn.Module` suelta y llaman `spec.compute(...)` / `spec.probs(...)` sin ningún `if`
propio. El único `if` de todo este mecanismo vive en `build_loss()` (ver abajo) y se
evalúa **una vez, al construir el `LossSpec`** — no una vez por batch.

Atributos:
- `compute` (`Callable[[Tensor, Tensor], Tensor]`): `(outputs, labels) -> loss escalar`.
  Ya sabe si tiene que castear/hacer `squeeze` de `outputs`/`labels` para el esquema
  elegido — el caller nunca lo decide.
- `probs` (`Callable[[Tensor], Tensor]`): `outputs -> probabilidad de la clase positiva`,
  shape `[B]`. Solo lo usa `evaluate()`, para alimentar las métricas (AUC, etc.) —
  nunca se usa para entrenar.

### _make_bce(**hparams) -> LossSpec

Esquema de **1 logit**: `BCEWithLogitsLoss` + sigmoid. La cabeza debe emitir `[B, 1]`.

- `compute`: `loss_fn(outputs.squeeze(1), labels.float())` — `squeeze(1)` deja
  `outputs` en `[B]` para que calce con `labels` (que llega `[B]`, `Long`, desde el
  `DataLoader` por defecto); `BCEWithLogitsLoss` exige además `labels` en `float`.
- `probs`: `sigmoid(outputs.squeeze(1))`.

### _make_cross_entropy(**hparams) -> LossSpec

Esquema de **2 logits**: `CrossEntropyLoss` + softmax. La cabeza debe emitir `[B, 2]`.

- `compute`: `loss_fn(outputs, labels)` sin casteo — `CrossEntropyLoss` ya espera
  `labels` como índice de clase `Long` `[B]`, que es justo lo que entrega el
  `DataLoader`.
- `probs`: `softmax(outputs, dim=1)[:, 1]` — la columna 1 es `malignant`
  (ver `Manifest.normalize_labels()` en `src/datasets/manifest.py`: `benign=0`,
  `malignant=1`).

### build_loss(name, **hparams) -> LossSpec

Factory: `name` es `"bce"` o `"cross_entropy"`, clave en `_LOSSES` (dict de nombre →
factory de `LossSpec`, mismo patrón que `_OPTIMIZERS`/`_SCHEDULERS`).

**El nombre elegido debe ser consistente con `num_classes` de la cabeza del modelo**
(`src/models/mlp_configs/standard_mlp.py`): `"bce"` espera una cabeza de 1 logit,
`"cross_entropy"` una de 2. Esta función no puede verificar eso — no ve el modelo — así
que un mismatch no falla acá: se manifiesta como un error de shape/dtype de PyTorch
dentro de `LossSpec.compute()`, la primera vez que se llama con un batch real.

Raises: `ValueError` si `name` no está en `_LOSSES`.

## loop.py

Funciones puras — sin estado entre llamadas, sin memoria de la mejor época (eso vive en
`Trainer`).

### train_one_epoch(model, loader, optimizer, loss_spec, device) -> dict[str, float]

Una época completa de entrenamiento: `model.train()`, re-congela BN (ver
`_set_frozen_bn_eval` abajo), y por cada batch hace forward → `loss_spec.compute()` →
`backward()` → `optimizer.step()`.

Returns: `{"loss": promedio de la pérdida sobre todos los batches}`.

### _set_frozen_bn_eval(model)

Mantiene en modo `eval()` las capas `BatchNorm{1,2,3}d` cuyos parámetros están
congelados (`requires_grad=False`), incluso después de `model.train()`. Sin esto, BN
congelado sigue actualizando `running_mean`/`running_var` con datos de entrenamiento
aunque γ/β no se actualicen — el bug legacy documentado del proyecto (ver `REFACTOR.md`
§7 y `CLAUDE.md`).

### evaluate(model, loader, loss_spec, device) -> dict[str, float]

Decorada con `@torch.no_grad()`. Evalúa sobre cualquier loader (val o test,
indistintamente): `model.eval()`, y por batch, forward → `loss_spec.compute()` (para el
loss reportado) → `loss_spec.probs()` (para las métricas) → `metrics.update(probs,
labels)`.

Returns: dict con `"loss"` + lo que devuelva `build_metric_collection()` (accuracy, auc,
sensitivity, specificity), cada valor ya como `float` (`.item()`).

## trainer.py

### Trainer

Única pieza de `train/` con memoria entre épocas: compara la métrica de validación
contra la mejor vista hasta ahora, y guarda el checkpoint **solo cuando mejora**.

Inputs (`__init__`):
- `model` (`nn.Module`): backbone + cabeza ya unidos.
- `optimizer` (`Optimizer`): construido vía `build_optimizer`.
- `loss_spec` (`LossSpec`): construido vía `build_loss`. Encapsula tanto el cálculo de
  la pérdida como la conversión de logits a probabilidad — ni `Trainer` ni
  `train/loop.py` necesitan saber si el esquema activo es BCE o CrossEntropy.
- `checkpoint_dir` (`str | Path`): carpeta donde se guardan los checkpoints.
- `device` (`str`, default `"cpu"`).
- `scheduler` (`LRScheduler | None`, default `None`): si se pasa, `scheduler.step()` se
  llama al final de cada época.
- `metric_name` (`str`, default `"auc"`): clave del dict que devuelve `evaluate()` a
  **maximizar** para decidir el mejor checkpoint.

#### fit(train_loader, val_loader, epochs) -> Path

Corre el loop completo de épocas. Por cada una: `train_one_epoch()`, `evaluate()`,
`scheduler.step()` si hay scheduler, y si `val_metrics[metric_name]` mejora sobre
`self.best_metric`, guarda `checkpoint_dir/best_epoch<N>.pt` vía `save_checkpoint()`.

Returns: ruta al **mejor** checkpoint según `metric_name` — nunca el de la última
época. Es el único valor que debe usarse para la evaluación final en test (ver bug
legacy "evaluar con el checkpoint final en vez del mejor" en `REFACTOR.md` §7).

Raises: `RuntimeError` si ninguna época produjo un checkpoint válido (p. ej.
`epochs == 0`, o `val_loader` vacío).

## Sin `__init__.py`

A diferencia de `src/datasets/` y `src/models/`, esta carpeta todavía no tiene
`__init__.py` — funciona igual como *namespace package* (PEP 420), pero nada se
reexporta. Importar directo: `from src.train.build import build_loss, LossSpec` /
`from src.train.trainer import Trainer`.
