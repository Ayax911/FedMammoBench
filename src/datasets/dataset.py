"""PyTorch Dataset implementation for loading mammography images from validated manifests."""

from typing import cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]
import torchvision.transforms.functional as TF


class MammoBenchDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset loading mammography image samples and normalized binary labels.

    Expects a DataFrame processed by `Manifest` containing `abs_image_path` and `label_norm` columns.

    Supports two image encodings, dispatched on the opened PIL image's mode:
      - Standard 8-bit images (JPG/PNG/uint8 TIFF, PIL modes other than `"F"`): converted
        to `"RGB"`, converted to tensor via `TF.to_tensor()` (`[0, 255] -> [0.0, 1.0]`),
        and then handed to `transform`.
      - 32-bit float TIFFs (PIL mode `"F"`, e.g. `Preproccesed/preprocess_images.py`'s
        `norm_neg1_1` or `norm_0_1` output): single-channel, pixel values already normalized
        on disk (e.g. to `[-1, 1]` for RadImageNet). Following the INC scheme
        (`classification_images/dataloaders/dataloader_images.py`), these are converted to a
        native PyTorch tensor without PIL `.convert()` (which would clip/truncate floats),
        replicated to 3 channels via `torch.cat`, and then passed to `transform` (which
        operates on tensors).

    Args:
        df: Input pandas DataFrame containing sample metadata and absolute image file paths.
        transform: torchvision transformation pipeline to apply on input tensors.
            If None, applies default resize (224x224).

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
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = df
        self.transform = transform or self._default_transform()

    def _default_transform(self) -> transforms.Compose:
        """Constructs fallback default image transform (Resize 224x224).

        Returns:
            transforms.Compose: Minimal transformation pipeline without normalization.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
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
        image = Image.open(row["abs_image_path"])

        if image.mode == "F":
            # 32-bit float TIFF (e.g. Preproccesed/preprocess_images.py's
            # norm_neg1_1 output): single-channel, pixel values already
            # rescaled/normalized on disk (e.g. to [-1, 1] for RadImageNet).
            # PIL's .convert("L"/"RGB") on mode "F" does NOT rescale -- it
            # clips-and-casts the floats directly into 0-255, which collapses
            # a [-1, 1] image to near-all-zero. We follow the INC scheme
            # (classification_images/dataloaders/dataloader_images.py):
            # array nativo -> tensor -> réplica a 3 canales ANTES del transform,
            # que opera sobre tensores.
            array = np.array(image)[np.newaxis, ...]
            image_tensor = torch.from_numpy(array)
            image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=0)
        else:
            image_tensor = TF.to_tensor(image.convert("RGB"))  # [0, 255] -> [0, 1], como el ToTensor de antes

        image_tensor = cast(torch.Tensor, self.transform(image_tensor))
        label = int(row["label_norm"])

        return image_tensor, label