"""Data and partitioning configuration with built-in validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataColumnMapping:
    """Column-name mapping for tabular manifests.

    Override in YAML when your CSV uses different column names.
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

    def validate(self) -> None:
        """Raise ValueError for any invalid combination of data settings."""
        total = self.val_fraction + self.test_fraction
        if total > 1.0:
            raise ValueError(
                f"val_fraction ({self.val_fraction}) + test_fraction ({self.test_fraction}) "
                "must be <= 1.0."
            )
        if total == 1.0 and self.name not in ("none", "synthetic"):
            # Allowed for gRPC server configs where the full local dataset is
            # used as a centralized evaluation holdout (no training on server).
            pass
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {self.image_size}")


def check_patient_ids_for_nan(patient_ids: list) -> bool:
    """Return True if any patient_id is None or NaN (pandas float NaN).

    Use this at runtime (after loading a dataset manifest) to detect missing
    patient identifiers before partitioning.  A patient_id that is None or NaN
    silently disables patient-level partitioning, which can cause data leakage.
    """
    for pid in patient_ids:
        if pid is None:
            return True
        if isinstance(pid, float) and math.isnan(pid):
            return True
    return False


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

    def validate(self) -> None:
        """Raise ValueError for invalid partitioning settings."""
        if self.alpha <= 0.0:
            raise ValueError(f"Dirichlet alpha must be > 0, got {self.alpha}")
        if self.min_per_client < 1:
            raise ValueError(f"min_per_client must be >= 1, got {self.min_per_client}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


__all__ = [
    "DataColumnMapping",
    "DataConfig",
    "PartitioningConfig",
    "check_patient_ids_for_nan",
]
