# Documentación del Módulo de Entrenamiento (`src/train/`)

`src/train/` es el subsistema encargado de construir optimizadores, schedulers y especificadores de funciones de pérdida (`LossSpec`), ejecutar las épocas puras de entrenamiento y evaluación manteniendo BatchNorm congelado cuando corresponde, y orquestar la corrida multi-época con `Trainer` guardando el mejor checkpoint según la métrica objetivo.

---

## Flujo Típico de Uso

```python
from src.train.build import build_optimizer, build_scheduler, build_loss
from src.train.trainer import Trainer

# 1. Construir componentes de optimización y loss (BCE = 1 logit)
optimizer = build_optimizer(model.parameters(), name="adamw", lr=1e-4, weight_decay=1e-2)
loss_spec = build_loss(name="bce")
scheduler = build_scheduler(optimizer, name="cosine", T_max=20)

# 2. Instanciar Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_spec=loss_spec,
    checkpoint_dir="runs/exp01/weights",
    run_dir="runs/exp01",
    device="cuda",
    scheduler=scheduler,
    metric_name="auc"
)

# 3. Correr el bucle de entrenamiento
best_ckpt_path = trainer.fit(train_loader, val_loader, epochs=20)
print(f"Entrenamiento finalizado. Mejor checkpoint en: {best_ckpt_path}")
```

---

## Detalle por Archivo y Clase

### `build.py`

#### `build_optimizer(params, name, **hparams)`

Factory de optimizadores PyTorch (`"adam"`, `"adamw"`).

#### `build_scheduler(optimizer, name, **hparams)`

Factory de schedulers de tasa de aprendizaje (`"cosine"`, `"reduceonplateau"`).

#### `LossSpec` & `build_loss(name, **hparams)`

Empareja la función de pérdida con la conversión correcta de logits a probabilidades según el esquema de salida:
* `"bce"`: 1 logit -> `BCEWithLogitsLoss` + `sigmoid(outputs.squeeze(1))`.
* `"cross_entropy"`: 2 logits -> `CrossEntropyLoss` + `softmax(outputs, dim=1)[:, 1]`.

##### Cómo usar `build.py`:
```python
import torch.nn as nn
from src.train.build import build_optimizer, build_scheduler, build_loss

model = nn.Linear(10, 1)

# Crear optimizador AdamW con lr=1e-4
optimizer = build_optimizer(model.parameters(), "adamw", lr=1e-4, weight_decay=1e-4)

# Crear scheduler CosineAnnealingLR
scheduler = build_scheduler(optimizer, "cosine", T_max=10)

# Crear LossSpec para 1 logit (BCE)
loss_spec = build_loss("bce")
```

---

### `loop.py`

#### `train_one_epoch(model, loader, optimizer, loss_spec, device)`

Ejecuta una época de entrenamiento: pone el modelo en `.train()`, congela el comportamiento estadístico de las capas BatchNorm congeladas (`_set_frozen_bn_eval`), realiza forward, calcula loss con `loss_spec.compute`, backward y `optimizer.step()`.

#### `evaluate(model, loader, loss_spec, device)`

Ejecuta evaluación bajo `@torch.no_grad()`: pone el modelo en `.eval()`, calcula loss y métricas clínicas (`accuracy`, `auc`, `sensitivity`, `specificity`) transformando logits con `loss_spec.probs`.

##### Cómo usar `loop.py`:
```python
from src.train.loop import train_one_epoch, evaluate
from src.train.build import build_loss, build_optimizer

loss_spec = build_loss("bce")
optimizer = build_optimizer(model.parameters(), "adamw", lr=1e-4)

# Entrenar una época
train_metrics = train_one_epoch(model, train_loader, optimizer, loss_spec, device="cuda")
print(f"Pérdida en train: {train_metrics['loss']:.4f}")

# Evaluar en conjunto de validación
val_metrics = evaluate(model, val_loader, loss_spec, device="cuda")
print(f"AUC en val: {val_metrics['auc']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
```

---

### `trainer.py`

#### `Trainer`

Orquestador principal con memoria entre épocas. Mantiene seguimiento de la mejor época según `metric_name`, guarda el checkpoint con `save_checkpoint()` únicamente cuando la métrica mejora y registra los logs mediante `MetricsLogger`.

##### Cómo usar `Trainer`:
```python
from src.train.trainer import Trainer
from src.train.build import build_loss, build_optimizer

optimizer = build_optimizer(model.parameters(), "adamw", lr=1e-4)
loss_spec = build_loss("bce")

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_spec=loss_spec,
    checkpoint_dir="runs/exp01/weights",
    run_dir="runs/exp01",
    device="cuda",
    metric_name="auc"
)

best_path = trainer.fit(train_loader, val_loader, epochs=10)
```

---

## Exportaciones (`__init__.py`)

`src/train/__init__.py` reexporta las utilidades principales:
```python
from src.train import Trainer, LossSpec, build_loss, build_optimizer, build_scheduler, train_one_epoch, evaluate
```
