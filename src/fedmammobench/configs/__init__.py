"""Configuration system for fedmammo.

Per-section modules (each with a ``validate()`` method):

- :mod:`~fedmammo.configs.data_config`      — DataConfig, PartitioningConfig
- :mod:`~fedmammo.configs.model_config`     — ModelConfig, NORMALIZE_PRESETS
- :mod:`~fedmammo.configs.training_config`  — TrainingConfig and sub-configs
- :mod:`~fedmammo.configs.federated_config` — FederatedConfig, StrategyConfig
- :mod:`~fedmammo.configs.experiment`       — ExperimentConfig (top-level)

Use :func:`load_config` to load a YAML file into an :class:`ExperimentConfig`.
Call :meth:`ExperimentConfig.validate` after loading to catch configuration
errors early (before any dataset or model is constructed).
"""

from fedmammo.configs.data_config import (
    DataColumnMapping,
    DataConfig,
    PartitioningConfig,
    check_patient_ids_for_nan,
)
from fedmammo.configs.experiment import EvaluationConfig, ExperimentConfig
from fedmammo.configs.federated_config import FederatedConfig, StrategyConfig
from fedmammo.configs.loader import load_config, save_config
from fedmammo.configs.model_config import NORMALIZE_PRESETS, ModelConfig
from fedmammo.configs.training_config import (
    AugmentationConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

__all__ = [
    "AugmentationConfig",
    "DataColumnMapping",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FederatedConfig",
    "LossConfig",
    "ModelConfig",
    "NORMALIZE_PRESETS",
    "OptimizerConfig",
    "PartitioningConfig",
    "SchedulerConfig",
    "StrategyConfig",
    "TrainingConfig",
    "check_patient_ids_for_nan",
    "load_config",
    "save_config",
]
