"""Federal-Learning (`src`): Framework de Aprendizaje Federado para Clasificación de Mamografías.

Este paquete proporciona utilidades centralizadas y federadas para:
    - Validación y serialización de configuraciones de experimentos (src.config).
    - Carga de manifests, divisiones anti-fuga por paciente y transformaciones de imágenes (src.datasets).
    - Construcción de backbones encoders, carga de pesos preentrenados y cabezas de clasificación (src.models).
    - Bucles de entrenamiento, loss specs y orquestación multi-época (src.train).
    - reproducibilidad determinista, persistencia de checkpoints y registro de métricas (src.seed, src.checkpoint, src.tracking, src.metrics).
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, load_config, save_config
from .metrics import build_metric_collection
from .seed import make_generator, seed_worker, set_global_seed
from .tracking import MetricsLogger

__all__ = [
    "ExperimentConfig",
    "load_config",
    "save_config",
    "save_checkpoint",
    "load_checkpoint",
    "build_metric_collection",
    "set_global_seed",
    "seed_worker",
    "make_generator",
    "MetricsLogger",
]
