# Estructura de `src/fedmammobench/`

## `cli/` — puntos de entrada del CLI
- `centralized.py` — entrypoint `fedmammobench-centralized`
- `federated.py` — entrypoint `fedmammobench-federated`
- `evaluate.py` — entrypoint `fedmammobench-evaluate` (carga checkpoint, corre `Evaluator`)

## `configs/` — sistema de configuración (dataclasses tipadas)
- `loader.py` — resuelve YAML con herencia `defaults:` desde `base.yaml`
- `schema.py` — re-exporta todo de los submódulos (clave para que `loader.py` encuentre los tipos)
- `data_config.py`, `model_config.py`, `training_config.py`, `federated_config.py`, `wandb_config.py` — una sección cada uno
- `experiment.py` — `ExperimentConfig`, validación cruzada entre secciones

## `datasets/` — carga y partición de datos
- `base.py` — clase abstracta `MammographyDataset`, IO de imágenes (PNG/JPG/DICOM)
- `cbis_ddsm.py`, `vindr_mammo.py`, `mammo_bench.py` — datasets concretos
- `factory.py` / `registry.py` — patrón `@register_dataset`
- `loaders.py` — construcción de `DataLoader` con balanceo de clases opcional
- `partitioning.py` — partición federada (iid / dirichlet / quantity_skew)
- `transforms.py` — pipelines Albumentations (train con augmentación, eval sin)

## `models/` — arquitecturas y pesos preentrenados
- `factory.py` — patrón `@register_model`
- `resnet.py`, `densenet.py`, `efficientnet.py`, `inception.py` — arquitecturas concretas
- `_adapt.py` — adapta el primer conv para distinto número de canales de entrada
- `_head.py` — constructor de cabeza de clasificación compartido (logits crudos, sin activación)
- `weight_loaders/` — inyección de pesos preentrenados:
  - `base.py` — interfaz
  - `imagenet.py`, `radimagenet.py`, `custom.py`, `none.py` — un loader por `weight_source`
  - `_keymaps.py` — mapeos de claves entre checkpoints
  - `custom.py` contiene el fix de normalización de prefijo `backbone.` (ver bug de warm-start federado en memoria del proyecto)

## `federated/` — lógica de aprendizaje federado (Flower)
- `server.py` — `run_simulation` (Ray) y `run_grpc_server` (multi-dispositivo real)
- `client.py` — `FedMammoBenchClient`, lógica de `fit` (freeze policy, unfreeze cíclico, param groups nuevos)
- `param_utils.py` — conversión `state_dict` ↔ formato `list[ndarray]` de Flower
- `node_logging.py` — `NodeMetricsRecorder`, envuelve la strategy para loggear métricas por nodo, timing y guardar checkpoints (`global_model.pt`/`global_best.pt`)
- `server_training.py` — entrenamiento híbrido server-side (`attach_server_training`)
- `strategies/` — patrón `@register_strategy`: `fedavg.py`, `fedprox.py`, `fedadam.py`, `fedyogi.py`, `fedbn.py`, `scaffold.py`, más `registry.py`

## `training/` — bucle de entrenamiento
- `trainer.py` — `Trainer`, usado tanto por centralizado como por el cliente federado; incluye early stopping y fix de BN bajo freeze
- `losses.py`, `optim.py` — pérdidas y optimizadores/schedulers

## `evaluation/`
- `evaluator.py` — `Evaluator`, evaluación post-hoc sobre checkpoint
- `metrics.py` — cálculo de métricas (accuracy, AUC, etc.)

## `utils/` — utilidades transversales
- `checkpoint.py` — save/load de checkpoints
- `csv_logger.py` — logger CSV append-only, robusto a crashes
- `metrics_sink.py` — fan-out de métricas (TensorBoard, W&B, CSV) vía interfaz duck-typed común
- `tensorboard_utils.py`, `wandb_utils.py` — writers específicos
- `device.py` — selección de dispositivo (CPU/GPU)
- `seeding.py` — reproducibilidad
- `logging_utils.py` — configuración de logging estándar

## `plotting.py`
Genera plots de un run completado (`autoplot()`), detecta automáticamente modo centralizado/federado según los CSV presentes; nunca falla el run.
