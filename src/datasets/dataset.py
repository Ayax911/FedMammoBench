"""PyTorch Dataset implementation for loading mammography images from validated manifests."""

from typing import cast

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]


class MammoBenchDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset loading mammography image samples and normalized binary labels.

    Expects a DataFrame processed by `Manifest` containing `abs_image_path` and `label_norm` columns.

    Args:
        df: Input pandas DataFrame containing sample metadata and absolute image file paths.
        grayscale: If True, opens PIL images in single-channel luminance mode (`"L"`).
            If False, opens images in 3-channel RGB mode (`"RGB"`). Default is False.
        transform: torchvision transformation pipeline to apply on loaded PIL Images.
            If None, applies default resize (224x224) and tensor conversion.

    Example:
        >>> from src.datasets.manifest import Manifest
        >>> from src.datasets.dataset import MammoBenchDataset
        >>> manifest = Manifest("manifests/fedmammobench.csv", "data/images")
        >>> dataset = MammoBenchDataset(df=manifest.df)
        >>> img_tensor, label = dataset[0]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        grayscale: bool = False,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = df
        self.grayscale = grayscale
        self.transform = transform or self._default_transform()

    def _default_transform(self) -> transforms.Compose:
        """Constructs fallback default image transform (Resize 224x224 + ToTensor).

        Returns:
            transforms.Compose: Minimal transformation pipeline without normalization.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        """Returns total sample count in the dataset split.

        Returns:
            int: Number of rows in `self.df`.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Loads, converts, transforms, and returns sample image tensor and label.

        Args:
            idx: Sample index in dataset split DataFrame.

        Returns:
            tuple[torch.Tensor, int]: A tuple `(image_tensor, label)` where `image_tensor`
                has shape `[C, H, W]` and `label` is 0 (benign) or 1 (malignant).

        Example:
            >>> img, label = dataset[0]
            >>> print(img.shape, label)
        """
        row = self.df.iloc[idx]

        # Determine PIL color space conversion ("L" = 1 channel grayscale, "RGB" = 3 channel)
        mode = "L" if self.grayscale else "RGB"
        image = Image.open(row["abs_image_path"]).convert(mode)

        # Apply spatial augmentations, tensor conversion, and normalization
        image_tensor = cast(torch.Tensor, self.transform(image))
        label = int(row["label_norm"])

        return image_tensor, label