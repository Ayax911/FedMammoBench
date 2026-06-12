"""Top-level experiment configuration assembling all section modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fedmammobench.configs.data_config import DataConfig, PartitioningConfig
from fedmammobench.configs.federated_config import FederatedConfig
from fedmammobench.configs.model_config import ModelConfig
from fedmammobench.configs.training_config import TrainingConfig


@dataclass
class EvaluationConfig:
    """Evaluation-time settings.

    Attributes:
        threshold: Probability cutoff for the positive class.
        save_predictions: If True, dump per-sample predictions to CSV.
    """

    threshold: float = 0.5
    save_predictions: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Every YAML file maps onto this. ``mode`` selects which entrypoint should
    consume it; CLI scripts will assert a compatible mode.
    """

    name: str = "experiment"
    mode: Literal["centralized", "federated"] = "federated"
    seed: int = 42
    output_dir: str = "runs"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    data: DataConfig = field(default_factory=DataConfig)
    partitioning: PartitioningConfig = field(default_factory=PartitioningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)

    def validate(self) -> None:
        """Run all section validators plus cross-section consistency checks.

        Raises:
            ValueError: if any configuration is invalid or inconsistent.
        """
        self.data.validate()
        self.partitioning.validate()

        # Cross-section: normalize_preset ↔ in_channels
        preset = self.training.augmentation.normalize_preset
        self.model.validate(normalize_preset=preset)

        # Cross-section: FedProx strategy params
        proximal_mu = float(self.federated.strategy.params.get("mu", 0.0))
        self.training.validate(
            strategy_name=self.federated.strategy.name,
            proximal_mu=proximal_mu,
        )

        self.federated.validate()

        # Cross-section: unfreeze_at_epoch should be reachable within training rounds
        if self.mode == "federated" and self.model.unfreeze_at_epoch is not None:
            if self.model.unfreeze_at_epoch >= self.federated.rounds:
                import warnings
                warnings.warn(
                    f"model.unfreeze_at_epoch={self.model.unfreeze_at_epoch} is >= "
                    f"federated.rounds={self.federated.rounds}. "
                    "The backbone will never be unfrozen during training.",
                    UserWarning,
                    stacklevel=2,
                )
        elif self.mode == "centralized" and self.model.unfreeze_at_epoch is not None:
            if self.model.unfreeze_at_epoch >= self.training.epochs:
                import warnings
                warnings.warn(
                    f"model.unfreeze_at_epoch={self.model.unfreeze_at_epoch} is >= "
                    f"training.epochs={self.training.epochs}. "
                    "The backbone will never be unfrozen during training.",
                    UserWarning,
                    stacklevel=2,
                )


__all__ = [
    "EvaluationConfig",
    "ExperimentConfig",
]
