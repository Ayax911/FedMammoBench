"""Typed configuration schema — re-exports from per-section modules.

All dataclasses now live in dedicated modules:

- :mod:`fedmammobench.configs.data_config`       — DataConfig, DataColumnMapping, PartitioningConfig
- :mod:`fedmammobench.configs.model_config`      — ModelConfig, NORMALIZE_PRESETS
- :mod:`fedmammobench.configs.training_config`   — TrainingConfig, OptimizerConfig, SchedulerConfig,
                                               AugmentationConfig, LossConfig
- :mod:`fedmammobench.configs.federated_config`  — FederatedConfig, StrategyConfig
- :mod:`fedmammobench.configs.experiment`        — ExperimentConfig, EvaluationConfig

This module re-exports every public name so that existing imports such as
``from fedmammobench.configs.schema import ExperimentConfig`` continue to work
without modification.
"""

from fedmammobench.configs.data_config import (
    DataColumnMapping,
    DataConfig,
    PartitioningConfig,
    check_patient_ids_for_nan,
)
from fedmammobench.configs.experiment import EvaluationConfig, ExperimentConfig
from fedmammobench.configs.federated_config import (
    FederatedConfig,
    ServerTrainingConfig,
    StrategyConfig,
)
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
    "ServerTrainingConfig",
    "StrategyConfig",
    "TrainingConfig",
    "check_patient_ids_for_nan",
]
