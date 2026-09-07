# Documentación del Paquete Raíz (`src/`)

`src/` contiene la infraestructura compartida, configuraciones, utilidades de reproducibilidad, persistencia de métricas y la interfaz de línea de comandos (CLI) del proyecto `Federal-Learning`.

---

## Estructura de Paquetes

```
src/
├── config.py         Validación y serialización de experimentos con Pydantic v2 y YAML.
├── cli.py            Entrypoint de entrenamiento: ensambla datos, modelo, optimizador, Trainer.fit() y evaluación final.
├── evaluate.py        Entrypoint de re-evaluación: mismo config, un checkpoint ya entrenado, CERO reentrenamiento.
├── eval_pipeline.py   Evaluación de val/test/por-base-de-datos, compartida por cli.py y evaluate.py.
├── checkpoint.py     Guardado y carga determinista de state_dict con metadata.
├── metrics.py        Colección de métricas clínicas binarias (Accuracy, AUC, Sensibilidad, Especificidad).
├── seed.py           Control de reproducibilidad global y por worker en PyTorch, NumPy y Random.
├── tracking.py       Logger unificado para CSV, TensorBoard y Weights & Biases (W&B).
├── datasets/         Carga de datos, manifests, splits anti-fuga y transformaciones (ver src/datasets/DOCS.md).
├── models/           Factory de backbones, pesos preentrenados, estrategias de congelamiento y cabezas (ver src/models/DOCS.md).
└── train/            Loss specs, bucle por época y orquestador Trainer (ver src/train/DOCS.md).
```

---

## Descripción por Archivo y Clase

### `config.py`

Maneja la validación de configuraciones mediante Pydantic v2. Cada sub-configuracion utiliza `extra="forbid"` para evitar que typos en archivos YAML sean ignorados silenciosamente.

* **`ArchitectureConfig`**: Dataclass de configuración para construir el backbone encoder (`name`, `weights_path`, `unfreeze_from`). `weights_path` es `None` por defecto -- obligatorio en la práctica para `resnet50_radimagenet` (checkpoint externo), innecesario para `resnet50_imagenet_v1`/`resnet50_imagenet_v2` (pesos embebidos en torchvision, ver `models/DOCS.md`).
* **`NamedComponentConfig`**: Configuración genérica por nombre e hiperparámetros (`name`, `hparams`) para componentes dinámicos (optimizadores, schedulers, funciones de pérdida y cabezas de clasificación).
* **`DataConfig`**: Parámetros de dataset y DataLoader (`manifest_path`, `image_root`, `batch_size`, `num_workers`, `seed`, `image_size`). `by_database_manifests` (opcional, `None` desactiva) mapea `nombre_base_de_datos -> manifest_path` para el desglose de test por base de datos que `cli.run()` hace al final del entrenamiento -- ver `_evaluate_by_database()` más abajo y `scripts/split_manifest_by_database.py`.
* **`TrainConfig`**: Parámetros del bucle de entrenamiento (`epochs`, `metric_name`, `checkpoint_dir`, `run_dir`, `device`, `wandb_project`, `wandb_group`).
* **`ExperimentConfig`**: Modelo principal que integra todas las secciones de un experimento.
* **`load_config(path)`**: Carga y valida un archivo YAML contra `ExperimentConfig`.
* **`save_config(config, path)`**: Serializa un `ExperimentConfig` a formato YAML.

#### Cómo usar `config.py`:
```python
from pathlib import Path
from src.config import ExperimentConfig, load_config, save_config

# 1. Cargar y validar una configuración desde YAML
config = load_config("configs/experiment_example.yaml")
print(f"ID del experimento: {config.experiment_id}")
print(f"Arquitectura: {config.architecture.name}")
print(f"Learning rate: {config.optimizer.hparams.get('lr')}")

# 2. Guardar la configuración procesada junto a los artefactos de la corrida
save_config(config, Path("runs/exp_01/config.yaml"))
```

---

### `checkpoint.py`

Proporciona funciones para guardar y cargar el estado de modelos PyTorch junto con metadata relevante (época y valor de métrica).

* **`save_checkpoint(model, path, epoch, metric_value)`**: Persiste `model.state_dict()` y diccionario de metadata en formato `.pt`.
* **`load_checkpoint(model, path, device="cpu")`**: Carga los pesos en `model` in-place y retorna la metadata guardada.

#### Cómo usar `checkpoint.py`:
```python
import torch
import torch.nn as nn
from src.checkpoint import save_checkpoint, load_checkpoint

# Crear modelo de prueba
model = nn.Linear(10, 1)

# 1. Guardar un checkpoint cuando la métrica mejora
save_checkpoint(
    model=model,
    path="runs/exp_01/weights/best_epoch_10.pt",
    epoch=10,
    metric_value=0.895
)

# 2. Cargar los pesos en un modelo recién instanciado
new_model = nn.Linear(10, 1)
metadata = load_checkpoint(
    model=new_model,
    path="runs/exp_01/weights/best_epoch_10.pt",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Cargado checkpoint de la época {metadata['epoch']} (AUC: {metadata['metric_value']:.3f})")
```

---

### `metrics.py`

Centraliza las métricas de evaluación para clasificación binaria de mamografías usando `torchmetrics`.

* **`build_metric_collection(device="cpu")`**: Retorna una `MetricCollection` con `accuracy`, `auc`, `sensitivity` (recall positivo), `specificity`, `f1`, `f1_macro` y `precision`. Todas se actualizan con la misma llamada `update(probs, labels)`, donde `probs` es la probabilidad de la clase positiva con shape `[B]`.
* **`BinaryMacroF1Score(threshold=0.5)`**: F1-macro binario (promedio del F1 de ambas clases), equivalente a `sklearn.metrics.f1_score(..., average="macro")` — el criterio de mejor checkpoint de la serie de notebooks. torchmetrics no trae un equivalente binario: `MulticlassF1Score` espera `[B, C]` o etiquetas enteras, no la probabilidad `[B]` con la que se alimenta esta colección.

> **`"f1"` vs `"f1_macro"` en `train.metric_name`.** `f1` (`BinaryF1Score`) mide **solo la clase positiva** y vale `0.0` exacto mientras el modelo no prediga ningún maligno — el estado normal de las primeras épocas con el backbone congelado y el desbalance ~66/34 del manifest. Como `EarlyStopping` exige mejora estricta, esa racha de ceros no resetea el contador de paciencia: con `patience: 10` la corrida se detiene en la época 10 y `fit()` devuelve el checkpoint de la **época 0**, o sea un modelo sin entrenar. Usar `f1_macro` para reproducir los notebooks.

#### Cómo usar `metrics.py`:
```python
import torch
from src.metrics import build_metric_collection

# Instanciar colección de métricas en el dispositivo correspondiente
metrics = build_metric_collection(device="cpu")

# Simular batches de predicciones (probabilidades) y etiquetas reales
probs = torch.tensor([0.1, 0.8, 0.3, 0.9])
labels = torch.tensor([0, 1, 0, 1])

# Acumular estado batch a batch
metrics.update(probs, labels)

# Calcular métricas consolidadas
results = metrics.compute()
for name, value in results.items():
    print(f"{name}: {value.item():.4f}")

# Reiniciar para la siguiente época
metrics.reset()
```

---

### `seed.py`

Asegura la reproducibilidad determinista en todas las fuentes de aleatoriedad del pipeline.

* **`set_global_seed(seed)`**: Fija la semilla en `random`, `numpy`, `torch` y `torch.cuda`.
* **`seed_worker(worker_id)`**: Función de inicialización de workers para `torch.utils.data.DataLoader`.
* **`make_generator(seed)`**: Crea un `torch.Generator` determinista para el shuffle de DataLoaders.

#### Cómo usar `seed.py`:
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.seed import set_global_seed, seed_worker, make_generator

# 1. Establecer la semilla global al inicio del script
SEED = 42
set_global_seed(SEED)

# 2. Configurar DataLoader determinista
dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=make_generator(SEED)
)
```

---

### `tracking.py`

Logger unificado que registra métricas en archivo CSV (`metrics.csv`), eventos de TensorBoard y opcionalmente integra con Weights & Biases (W&B) sin bloquear si no hay conectividad.

* **`MetricsLogger`**: Maneja los streams de salida para persistencia de métricas por época y de test.
  - `log(epoch, metrics)`: Escribe una fila en el CSV y scalars en TensorBoard / W&B (una por época). `metrics.csv` y el `SummaryWriter` de TensorBoard se abren PEREZOSAMENTE, recién en la primera llamada a `log()` -- no en `__init__` -- así que instanciar un `MetricsLogger` sobre un `run_dir` que ya tiene un `metrics.csv` real (ej. `evaluate.py:run_evaluation()`, que solo usa `log_summary`/`log_image`/`log_table`) no lo trunca si nunca se llama a `log()`.
  - `log_image(name, path)`: Sube un PNG ya guardado en disco (plots de `reporting.py`) al summary de W&B. No-op sin W&B.
  - `log_summary(metrics)`: Registra métricas de una sola medición (test) en el summary de la corrida, no en la serie por época. No-op sin W&B.
  - `log_table(name, csv_path)`: Sube un CSV (ej. `predictions.csv`) como tabla explorable en W&B. No-op sin W&B.
  - `close()`: Libera descriptores de archivo (si se llegaron a abrir) y cierra sesiones (`wandb.finish()` incluido).

`cli.run()` es dueño de la corrida de W&B: la abre antes de construir `Trainer` y la cierra después de la evaluación de test, para que entrenamiento y test queden en la misma corrida (`Trainer.fit()` ya no abre/cierra su propia corrida si se le inyecta un `logger` — ver `train/DOCS.md`).

#### Cómo usar `tracking.py`:
```python
from src.tracking import MetricsLogger

# Usar como gestor de contexto (Context Manager) para garantizar el cierre correcto.
# config= adjunta los hiperparámetros de la corrida (ExperimentConfig.model_dump())
# para poder filtrar/agrupar corridas por config en la UI de W&B.
with MetricsLogger(
    run_dir="runs/exp_01",
    wandb_project="Federal-Learning",
    wandb_run_name="exp_01",
    wandb_group="bce_freeze_posweight_sweep",  # junta esta corrida con las del mismo bloque en la UI
    config={"lr": 1e-4, "unfreeze_idx": [7]},
) as logger:
    for epoch in range(1, 5):
        epoch_metrics = {
            "train_loss": 0.45 / epoch,
            "val_loss": 0.50 / epoch,
            "val_auc": 0.75 + (epoch * 0.04),
            "val_accuracy": 0.80 + (epoch * 0.02)
        }
        logger.log(epoch, epoch_metrics)

    # Después de entrenar: plots, métricas de test y predictions.csv, todo
    # en la misma corrida de W&B (ver src/cli.py y src/reporting.py).
    logger.log_image("plots/loss_curve", "runs/exp_01/plots/loss_curve.png")
    logger.log_summary({"test_auc": 0.91})
    logger.log_table("test/predictions", "runs/exp_01/test/predictions.csv")
```

---

### `cli.py`

Punto de entrada ejecutable para experimentos centralizados. Ensambla la configuración, semilla, datasets, modelo, optimizador, loss y trainer.

* **`run(config)`**: Ejecuta un experimento completo dado un `ExperimentConfig` -- entrena con `Trainer.fit()` y evalúa el mejor checkpoint vía `eval_pipeline.py`.
* **`parse_args()`**: Lee `--config` desde argumentos CLI.
* **`main()`**: Función de entrada invocada al ejecutar el archivo como módulo.

#### Cómo usar `cli.py`:
```bash
# Ejecutar desde la línea de comandos
python -m src.cli --config configs/exp01.yaml
```

```python
# O ejecutar programáticamente desde Python
from src.config import load_config
from src.cli import run

config = load_config("configs/exp01.yaml")
run(config)
```

---

### `eval_pipeline.py`

Las dos funciones de evaluación que `cli.py` necesita DESPUÉS de entrenar, extraídas a su propio módulo para que `evaluate.py` (re-evaluación sin reentrenar) también pueda usarlas sin importar `cli.py` -- que, por diseño, ningún otro módulo debe importar.

* **`evaluate_split(split_name, model, best_checkpoint, loader, loss_spec, device, run_dir, logger)`**: evalúa un checkpoint sobre un split (`"val"`/`"test"`) y persiste/loguea `metrics.json`, `confusion_matrix_metrics.json`, `predictions.csv`, matriz de confusión y ROC -- misma función para ambos splits.
* **`evaluate_by_database(by_database_manifests, image_root, eval_transform_builder, model, best_checkpoint, loss_spec, batch_size, num_workers, device, run_dir, logger)`**: opt-in vía `DataConfig.by_database_manifests`. Reconstruye un `Manifest`+`Split`+`DataLoader` independiente por cada base de datos del dict, evalúa el mismo checkpoint sobre el `.test_df()` de cada uno, y escribe en `run_dir/test/`: `metrics_by_database.json`, `confusion_matrix_by_database.png` (panel 1xN) y `metrics_by_database.png` (barras agrupadas por base de datos y por métrica, eje Y `[0, 1]`) -- ver `reporting.py`.
* **`METRIC_DISPLAY_NAMES`**: nombres a mostrar de las 7 métricas clínicas, compartidos entre las curvas train-vs-val de `cli.run()` y las barras de `evaluate_by_database()`.

---

### `evaluate.py`

Entrypoint para re-evaluar un checkpoint YA entrenado (val/test + desglose por base de datos) sin volver a entrenar -- lo que `cli.run()` no ofrece a propósito (ver CLAUDE.md: "no hay resume, ni forma de puntuar un checkpoint existente sin reentrenar"). Útil para producir `run_dir/test/*_by_database.*` de una corrida entrenada ANTES de que `DataConfig.by_database_manifests` existiera, sin pagar el costo de reentrenar.

* **`run_evaluation(config, checkpoint_path)`**: reconstruye manifest/split/transforms/dataloaders/backbone+cabeza igual que `cli.run()` (con el MISMO YAML que se usó para entrenar), pero en vez de `Trainer.fit()` carga `checkpoint_path` y llama a `eval_pipeline.evaluate_split()`/`evaluate_by_database()`. Nunca toca `config.yaml`/`metrics.csv`/`plots/` del `run_dir` (artefactos del entrenamiento original) -- solo `run_dir/val/` y `run_dir/test/`.
* **`parse_args()`** / **`main()`**: `--config` + `--checkpoint` desde línea de comandos.

#### Cómo usar `evaluate.py`:
```bash
python -m src.evaluate \
    --config configs/exp05_fedmammobench_full_weighted.yaml \
    --checkpoint runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt
```

```python
from src.config import load_config
from src.evaluate import run_evaluation

config = load_config("configs/exp05_fedmammobench_full_weighted.yaml")
run_evaluation(config, "runs/exp05_fedmammobench_full_weighted/weights/best_epoch123.pt")
```
