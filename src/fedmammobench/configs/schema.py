"""Typed configuration schema — re-exports from per-section modules.

All dataclasses now live in dedicated modules:

- :mod:`fedmammobench.configs.data_config`       — DataConfig, DataColumnMapping, PartitioningConfig
- :mod:`fedmammobench.configs.model_config`      — ModelConfig, NORMALIZE_PRESETS
- :mod:`fedmammobench.configs.training_config`   — TrainingConfig, OptimizerConfig, SchedulerConfig,
                                               AugmentationConfig, LossConfig
- :mod:`fedmammobench.configs.federated_config`  — FederatedConfig, StrategyConfig
- :mod:`fedmammobench.configs.experiment`        — ExperimentConfig, EvaluationConfig
- :mod:`fedmammobench.configs.wandb_config`       — WandbConfig

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
    FEDERATED_BEST_CHECKPOINT_METRICS,
    FederatedConfig,
    ServerTrainingConfig,
    StrategyConfig,
)
from fedmammobench.configs.model_config import NORMALIZE_PRESETS, HeadConfig, ModelConfig
from fedmammobench.configs.training_config import (
    BEST_CHECKPOINT_METRICS,
    AugmentationConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from fedmammobench.configs.wandb_config import WandbConfig

__all__ = [
    "BEST_CHECKPOINT_METRICS",
    "FEDERATED_BEST_CHECKPOINT_METRICS",
    "NORMALIZE_PRESETS",
    "AugmentationConfig",
    "DataColumnMapping",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FederatedConfig",
    "HeadConfig",
    "LossConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PartitioningConfig",
    "SchedulerConfig",
    "ServerTrainingConfig",
    "StrategyConfig",
    "TrainingConfig",
    "WandbConfig",
    "check_patient_ids_for_nan",
]
