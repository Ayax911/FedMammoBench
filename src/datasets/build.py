"""PyTorch DataLoader builder factory instantiating train, validation, and test splits."""

import torch
from torch.utils.data import DataLoader

from .dataset import MammoBenchDataset
from .split import Split
from .transform import TransformBuilder


def builder_dataloader(
    split: Split,
    train_transform_builder: TransformBuilder,
    eval_transform_builder: TransformBuilder,
    batch_size: int = 16,
    num_workers: int = 1,
) -> dict[str, DataLoader[tuple[torch.Tensor, int]]]:
    """Constructs PyTorch DataLoaders for train, validation, and test dataset splits.

    Args:
        split: `Split` instance containing patient-disjoint DataFrames for train, val, and test.
        train_transform_builder: `TransformBuilder` configuring training augmentations.
        eval_transform_builder: `TransformBuilder` configuring deterministic evaluation transforms.
        batch_size: Number of image samples per mini-batch. Default is 16.
        num_workers: Number of subprocesses used for data loading. Default is 1.

    Returns:
        dict[str, DataLoader[tuple[torch.Tensor, int]]]: Dictionary mapping split names
            (`"train"`, `"val"`, `"test"`) to their corresponding PyTorch DataLoaders.
    """
    # Compile torchvision transform pipelines
    train_transform = train_transform_builder.build()
    eval_transform = eval_transform_builder.build()

    # Instantiate PyTorch Datasets over split DataFrames
    train_ds = MammoBenchDataset(df=split.train_df(), transform=train_transform)
    val_ds = MammoBenchDataset(df=split.val_df(), transform=eval_transform)
    test_ds = MammoBenchDataset(df=split.test_df(), transform=eval_transform)

    # Wrap in PyTorch DataLoaders (shuffling enabled strictly for training split)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }