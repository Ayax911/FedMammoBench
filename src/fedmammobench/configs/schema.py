"""Typed configuration schema — re-exports from per-section modules.

All dataclasses now live in dedicated modules:

- :mod:`fedmammo.configs.data_config`       — DataConfig, DataColumnMapping, PartitioningConfig
- :mod:`fedmammo.configs.model_config`      — ModelConfig, NORMALIZE_PRESETS
- :mod:`fedmammo.configs.training_config`   — TrainingConfig, OptimizerConfig, SchedulerConfig,
                                               AugmentationConfig, LossConfig
- :mod:`fedmammo.configs.federated_config`  — FederatedConfig, StrategyConfig
- :mod:`fedmammo.configs.experiment`        — ExperimentConfig, EvaluationConfig

This module re-exports every public name so that existing imports such as
``from fedmammo.configs.schema import ExperimentConfig`` continue to work
without modification.
"""

from fedmammo.configs.data_config import (
    DataColumnMapping,
    DataConfig,
    PartitioningConfig,
    check_patient_ids_for_nan,
)
from fedmammo.configs.experiment import EvaluationConfig, ExperimentConfig
from fedmammo.configs.federated_config import FederatedConfig, StrategyConfig
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
]
