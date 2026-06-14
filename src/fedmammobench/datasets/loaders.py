"""DataLoader construction with optional class balancing."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from fedmammobench.datasets.base import MammographyDataset
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def build_dataloader(
    dataset: MammographyDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    balance_classes: bool = False,
    drop_last: bool = False,
    pin_memory: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Return a DataLoader, optionally with a WeightedRandomSampler.

    When ``balance_classes`` is True, ``shuffle`` is ignored (the sampler
    handles iteration order) and each sample's weight is the inverse of its
    class frequency in ``dataset``. This is appropriate when the positive
    class is rare, which is typical for mammography.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot build a DataLoader over an empty dataset.")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    if balance_classes:
        labels = dataset.labels
        n_classes = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        absent = [c for c in range(n_classes) if counts[c] == 0]
        if absent:
            _logger.warning(
                "balance_classes=True but class(es) %s are absent from this dataset — "
                "WeightedRandomSampler cannot oversample what does not exist. "
                "Check the data partition.",
                absent,
            )
        # Avoid divide-by-zero for absent classes.
        counts = np.where(counts == 0, 1.0, counts)
        inv = 1.0 / counts
        sample_weights = inv[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
    )


__all__ = ["build_dataloader"]
