"""Dataset factory.

:func:`build_dataset` consumes an :class:`ExperimentConfig` and returns a
``{"train", "val", "test"}`` mapping of :class:`MammographyDataset` instances
ready to be wrapped in DataLoaders.
"""

from __future__ import annotations

from fedmammo.configs.schema import ExperimentConfig
from fedmammo.datasets.base import MammographyDataset
from fedmammo.datasets.cbis_ddsm import CBISDDSMDataset
from fedmammo.datasets.mammo_bench import MammoBenchDataset
from fedmammo.datasets.synthetic import SyntheticMammographyDataset
from fedmammo.datasets.transforms import build_transforms
from fedmammo.datasets.vindr_mammo import VinDrMammoDataset
from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def build_dataset(cfg: ExperimentConfig) -> dict[str, MammographyDataset]:
    """Construct {"train", "val", "test"} datasets from a config.

    Args:
        cfg: A fully populated :class:`ExperimentConfig`. Only ``cfg.data``
            and ``cfg.training.augmentation`` are consumed.

    Raises:
        ValueError: if required paths for non-synthetic datasets are missing.
        FileNotFoundError: if the manifest / annotations file is absent.
    """
    train_tx, eval_tx = build_transforms(
        image_size=cfg.data.image_size,
        augmentation=cfg.training.augmentation,
        in_channels=cfg.model.in_channels,
    )

    name = cfg.data.name
    if name == "none":
        # Sentinel: the caller (typically a gRPC server with no holdout) does
        # not want any local dataset constructed. Returning an empty mapping
        # signals downstream code to disable any dataset-dependent path
        # (centralized evaluation, loss-from-labels heuristics, etc.).
        _logger.info("data.name='none' — skipping dataset construction.")
        return {}
    if name == "synthetic":
        return _build_synthetic(cfg, train_tx, eval_tx)
    if name == "cbis_ddsm":
        if not cfg.data.manifest_path or not cfg.data.image_root:
            raise ValueError(
                "data.name=cbis_ddsm requires both `manifest_path` and `image_root`."
            )
        return CBISDDSMDataset.from_manifest(
            manifest_path=cfg.data.manifest_path,
            image_root=cfg.data.image_root,
            columns=cfg.data.columns,
            image_format=cfg.data.image_format,
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            seed=cfg.seed,
            grayscale=cfg.data.grayscale,
            transform_train=train_tx,
            transform_eval=eval_tx,
        )
    if name == "vindr_mammo":
        if not cfg.data.annotations_path or not cfg.data.image_root:
            raise ValueError(
                "data.name=vindr_mammo requires both `annotations_path` and `image_root`."
            )
        return VinDrMammoDataset.from_annotations(
            annotations_path=cfg.data.annotations_path,
            image_root=cfg.data.image_root,
            val_fraction=cfg.data.val_fraction,
            birads_3_policy=cfg.data.birads_3_policy,
            seed=cfg.seed,
            grayscale=cfg.data.grayscale,
            transform_train=train_tx,
            transform_eval=eval_tx,
        )
    if name == "mammo_bench":
        if not cfg.data.manifest_path or not cfg.data.image_root:
            raise ValueError(
                "data.name=mammo_bench requires both `manifest_path` and `image_root`."
            )
        return MammoBenchDataset.from_manifest(
            manifest_path=cfg.data.manifest_path,
            image_root=cfg.data.image_root,
            normal_policy=cfg.data.normal_policy,
            image_format=cfg.data.image_format,
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            seed=cfg.seed,
            grayscale=cfg.data.grayscale,
            transform_train=train_tx,
            transform_eval=eval_tx,
        )
    raise ValueError(f"Unknown dataset name: {name!r}")


def _build_synthetic(
    cfg: ExperimentConfig,
    train_tx,  # noqa: ANN001
    eval_tx,  # noqa: ANN001
) -> dict[str, MammographyDataset]:
    n = cfg.data.synthetic_num_samples
    n_val = max(8, int(n * cfg.data.val_fraction))
    n_test = max(8, int(n * cfg.data.test_fraction))
    n_train = max(16, n - n_val - n_test)
    _logger.info(
        "Building synthetic datasets: train=%d, val=%d, test=%d, image_size=%d",
        n_train,
        n_val,
        n_test,
        cfg.data.image_size,
    )
    return {
        "train": SyntheticMammographyDataset(
            num_samples=n_train,
            image_size=cfg.data.image_size,
            grayscale=cfg.data.grayscale,
            seed=cfg.seed,
            transform=train_tx,
        ),
        "val": SyntheticMammographyDataset(
            num_samples=n_val,
            image_size=cfg.data.image_size,
            grayscale=cfg.data.grayscale,
            seed=cfg.seed + 1,
            transform=eval_tx,
        ),
        "test": SyntheticMammographyDataset(
            num_samples=n_test,
            image_size=cfg.data.image_size,
            grayscale=cfg.data.grayscale,
            seed=cfg.seed + 2,
            transform=eval_tx,
        ),
    }


__all__ = ["build_dataset"]
