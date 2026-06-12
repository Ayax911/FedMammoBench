"""Configuration system for fedmammobench.

Per-section modules (each with a ``validate()`` method):

- :mod:`~fedmammobench.configs.data_config`      — DataConfig, PartitioningConfig
- :mod:`~fedmammobench.configs.model_config`     — ModelConfig, NORMALIZE_PRESETS
- :mod:`~fedmammobench.configs.training_config`  — TrainingConfig and sub-configs
- :mod:`~fedmammobench.configs.federated_config` — FederatedConfig, StrategyConfig
- :mod:`~fedmammobench.configs.experiment`       — ExperimentConfig (top-level)

Use :func:`load_config` to load a YAML file into an :class:`ExperimentConfig`.
Call :meth:`ExperimentConfig.validate` after loading to catch configuration
errors early (before any dataset or model is constructed).
"""

from fedmammobench.configs.data_config import (
    DataColumnMapping,
    DataConfig,
    PartitioningConfig,
    check_patient_ids_for_nan,
)
from fedmammobench.configs.experiment import EvaluationConfig, ExperimentConfig
from fedmammobench.configs.federated_config import FederatedConfig, StrategyConfig
from fedmammobench.configs.loader import load_config, save_config
from fedmammobench.configs.model_config import NORMALIZE_PRESETS, ModelConfig
from fedmammobench.configs.training_config import (
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
