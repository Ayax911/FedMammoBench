"""Typed configuration schema.

Configurations are plain dataclasses; YAML is mapped onto them by
:mod:`fedmammo.configs.loader`. Using dataclasses (instead of pydantic) keeps
the dependency surface small and the data model trivially serializable.

Top-level: :class:`ExperimentConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Normalization presets
# ---------------------------------------------------------------------------

NORMALIZE_PRESETS: dict[str, dict[str, tuple[float, ...]]] = {
    # Standard ImageNet RGB stats
    "imagenet_rgb": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    # Luminance-weighted single-channel equivalent of ImageNet RGB
    "imagenet_gray": {"mean": (0.449,), "std": (0.226,)},
    # RadImageNet publishes (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5) for RGB
    "radimagenet_rgb": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    # Single-channel RadImageNet (grayscale mammography)
    "radimagenet_gray": {"mean": (0.5,), "std": (0.5,)},
    # Legacy default used in earlier fedmammo configs
    "mammo_default": {"mean": (0.5,), "std": (0.25,)},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataColumnMapping:
    """Column-name mapping for tabular manifests.

    Override these in YAML when your CSV uses different column names.
    """

    image_path: str = "image_path"
    label: str = "pathology"
    patient_id: str = "patient_id"
    split: str = "split"


@dataclass
class DataConfig:
    """Dataset selection and IO settings.

    Attributes:
        name: Dataset identifier registered in :mod:`fedmammo.datasets.factory`.
            One of ``cbis_ddsm``, ``vindr_mammo``, ``synthetic``, ``mammo_bench``.
        manifest_path: Optional path to a CSV manifest (CBIS-DDSM, Mammo-Bench).
        annotations_path: Optional path to breast-level annotations (VinDr-Mammo).
        image_root: Root directory containing the image files.
        image_format: ``png``, ``jpg``, or ``dicom``. Only consulted by loaders
            that accept multiple formats.
        image_size: Square resize target (height == width).
        grayscale: If True, load images as 1-channel; otherwise broadcast to 3.
        num_classes: Should remain 2 for binary classification.
        batch_size: Mini-batch size for both training and evaluation by default.
        num_workers: PyTorch DataLoader workers per client.
        val_fraction: Fraction of the train set reserved for validation when
            the manifest has no explicit ``split`` column.
        test_fraction: Fraction of the manifest reserved for test under the
            same condition.
        birads_3_policy: VinDr-Mammo only. How to map BI-RADS 3:
            ``drop`` (default), ``benign``, or ``malignant``.
        normal_policy: Mammo-Bench only. How to treat rows whose
            ``classification`` column is ``Normal``: map to benign (``benign``,
            default) or drop them (``drop``).
        synthetic_num_samples: For the synthetic loader, how many samples to
            generate per split.
        columns: Column-name mapping for tabular manifests.
        balance_classes: If True, build a WeightedRandomSampler at train time.
    """

    name: Literal["cbis_ddsm", "vindr_mammo", "synthetic", "mammo_bench", "none"] = "synthetic"
    manifest_path: str | None = None
    annotations_path: str | None = None
    image_root: str | None = None
    image_format: Literal["png", "jpg", "dicom"] = "png"
    image_size: int = 224
    grayscale: bool = True
    num_classes: int = 2
    batch_size: int = 32
    num_workers: int = 2
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    birads_3_policy: Literal["drop", "benign", "malignant"] = "drop"
    normal_policy: Literal["benign", "drop"] = "benign"
    synthetic_num_samples: int = 256
    columns: DataColumnMapping = field(default_factory=DataColumnMapping)
    balance_classes: bool = True


# ---------------------------------------------------------------------------
# Federated partitioning
# ---------------------------------------------------------------------------

@dataclass
class PartitioningConfig:
    """How to split the training set across federated clients.

    Attributes:
        scheme: ``iid`` (uniform random), ``dirichlet`` (label-skewed
            Dirichlet), or ``quantity_skew`` (different amounts per client).
        alpha: Dirichlet concentration parameter. Lower = more non-IID.
            Only used by ``dirichlet``.
        min_per_client: Minimum samples per client; partitioning is retried
            up to ``max_retries`` times if any client falls below this.
        max_retries: How many times to redraw a Dirichlet partition.
        quantity_skew_sigma: Std-dev of log-normal scaling for
            ``quantity_skew``. 0 reduces to IID amounts.
    """

    scheme: Literal["iid", "dirichlet", "quantity_skew"] = "iid"
    alpha: float = 0.5
    min_per_client: int = 16
    max_retries: int = 20
    quantity_skew_sigma: float = 0.5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model architecture and head settings.

    Attributes:
        name: Model identifier registered in :mod:`fedmammo.models.factory`.
        pretrained: Legacy flag — kept for backward compatibility. When
            ``weight_source`` is ``"auto"`` (default), this flag is consulted:
            ``True`` → ``"imagenet"``, ``False`` → ``"none"``. Explicit
            ``weight_source`` values take precedence.
        weight_source: Where to load pretrained weights from.
            ``"auto"`` infers from the legacy ``pretrained`` flag.
            ``"imagenet"`` uses torchvision ImageNet defaults.
            ``"radimagenet"`` loads a RadImageNet PyTorch checkpoint.
            ``"custom"`` loads an arbitrary local checkpoint (requires
            ``checkpoint_path``).
            ``"none"`` keeps random initialization.
        checkpoint_path: Absolute (or ``~``-expanded) path to a ``.pth``
            checkpoint file. Required when ``weight_source="custom"``.
            For ``"radimagenet"`` this overrides the ``FEDMAMMO_RADIMAGENET_DIR``
            environment variable lookup.
        pretrained_num_classes: Number of output classes in the source
            checkpoint's head. Used for shape validation; the loaded head is
            always discarded in favor of a fresh head sized to ``num_classes``.
        strict_load: If ``False`` (default), missing and unexpected keys during
            checkpoint loading are logged as warnings rather than errors.
            ``True`` makes ``load_state_dict`` strict.
        dropout: Dropout probability applied at the classification head.
        num_classes: Should match :attr:`DataConfig.num_classes`.
        in_channels: 1 if grayscale; 3 otherwise. The model factory adapts
            the first conv layer accordingly.
        freeze_backbone: Freeze all backbone parameters (requires_grad=False)
            and set BatchNorm layers to eval mode to prevent running-stat drift.
        freeze_head: Freeze the classification head parameters.
        unfreeze_at_epoch: If set, backbone freezing is lifted when the
            federated round (or centralized epoch) reaches this value,
            enabling progressive unfreezing.
    """

    name: Literal[
        "resnet18", "resnet50", "efficientnet_b0", "densenet121", "inception_v3"
    ] = "resnet18"
    pretrained: bool = True
    weight_source: Literal["imagenet", "radimagenet", "custom", "none", "auto"] = "auto"
    checkpoint_path: str | None = None
    pretrained_num_classes: int | None = None
    strict_load: bool = False

    dropout: float = 0.2
    num_classes: int = 2
    in_channels: int = 1

    freeze_backbone: bool = False
    freeze_head: bool = False
    unfreeze_at_epoch: int | None = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    name: Literal["sgd", "adam", "adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # only used by SGD


@dataclass
class SchedulerConfig:
    name: Literal["none", "cosine", "step", "plateau"] = "none"
    step_size: int = 10
    gamma: float = 0.1
    t_max: int = 50  # cosine


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotate_limit: int = 15
    brightness_contrast: bool = True
    elastic: bool = False
    normalize_preset: str | None = None
    # Accept scalar (replicated across channels) or per-channel list.
    # YAML sequences are loaded as list[float]; scalars as float.
    normalize_mean: Any = 0.5
    normalize_std: Any = 0.25


@dataclass
class LossConfig:
    """Loss function selection.

    ``ce`` is class-weighted cross-entropy (weights derived from train counts
    if :attr:`auto_class_weights` is True). ``focal`` uses focal loss with
    :attr:`focal_gamma`.
    """

    name: Literal["ce", "focal"] = "ce"
    auto_class_weights: bool = True
    focal_gamma: float = 2.0


@dataclass
class TrainingConfig:
    """Centralized / per-client local training hyperparameters.

    Attributes:
        epochs: Number of epochs for centralized training (run_centralized.py).
        local_epochs: Number of local epochs per Flower round per client.
        grad_clip_norm: Optional gradient-norm clipping value (0 disables).
        mixed_precision: Use ``torch.cuda.amp`` autocast when CUDA is available.
    """

    epochs: int = 20
    local_epochs: int = 1
    grad_clip_norm: float = 0.0
    mixed_precision: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Evaluation-time settings.

    Attributes:
        threshold: Probability cutoff for the positive class.
        save_predictions: If True, dump per-sample predictions to CSV.
    """

    threshold: float = 0.5
    save_predictions: bool = False


# ---------------------------------------------------------------------------
# Federated strategy + runtime
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Federated strategy selection.

    ``params`` is a free-form dict consumed by the strategy implementation
    (e.g. ``{"mu": 0.01}`` for FedProx).
    """

    name: Literal["fedavg", "fedprox", "scaffold", "fedbn"] = "fedavg"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedConfig:
    """Flower simulation / gRPC parameters.

    Attributes:
        num_clients: Number of simulated clients (i.e. virtual hospitals).
            In real gRPC mode this is informational; the actual count is
            determined by ``min_*_clients`` and which devices connect.
        rounds: Number of FL rounds.
        fraction_fit: Fraction of clients sampled for training per round.
        fraction_evaluate: Fraction sampled for federated evaluation per round.
        min_fit_clients: Hard minimum for training selection.
        min_evaluate_clients: Hard minimum for evaluation selection.
        min_available_clients: Minimum clients before a round starts.
        accept_failures: Whether to tolerate individual client failures.
        client_resources: Per-client Ray quotas (simulation only).
        ray_init_args: Forwarded to Ray (simulation only).
        server_address: ``HOST:PORT`` the gRPC server binds to / clients
            connect to. Used only by ``run_grpc_server`` and the standalone
            client script; ignored by ``run_simulation``.
        strategy: Strategy selection (see :class:`StrategyConfig`).
    """

    num_clients: int = 4
    rounds: int = 10
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    accept_failures: bool = True
    client_resources: dict[str, float] = field(
        default_factory=lambda: {"num_cpus": 1.0, "num_gpus": 0.0}
    )
    ray_init_args: dict[str, Any] = field(default_factory=dict)
    server_address: str = "0.0.0.0:8080"
    strategy: StrategyConfig = field(default_factory=StrategyConfig)


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

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
]
