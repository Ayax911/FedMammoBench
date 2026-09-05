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

    Supports two image encodings, dispatched on the opened PIL image's mode:
      - Standard 8-bit images (JPG/PNG/uint8 TIFF, PIL modes other than `"F"`): converted
        to `"L"` or `"RGB"` as usual, then handed to `transform` (which typically includes
        `ToTensor()` to rescale `[0, 255] -> [0, 1]` and a `Normalize`).
      - 32-bit float TIFFs (PIL mode `"F"`, e.g. `Preproccesed/preprocess_images.py`'s
        `norm_neg1_1` output): single-channel, pixel values already normalized on disk
        (e.g. to `[-1, 1]` for RadImageNet). These are passed to `transform` *without*
        `.convert()` -- PIL clips/truncates floats rather than rescaling them when
        converting mode `"F"` to `"L"`/`"RGB"`, which would collapse an already-normalized
        image to near-all-zero. `transform`'s `Normalize` (if any) should therefore use
        1-element `mean`/`std` tuples for this encoding, or be identity if the values are
        already in the desired range. Channel replication to 3-channel RGB (when
        `grayscale=False`) happens after `transform` runs, on the resulting tensor.

    Args:
        df: Input pandas DataFrame containing sample metadata and absolute image file paths.
        grayscale: If True, keeps images single-channel (PIL luminance mode `"L"` for
            standard images; untouched for mode `"F"`). If False, expands to 3 channels
            (PIL `"RGB"` for standard images; tensor-level channel repeat for mode `"F"`).
            Default is False.
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
        image = Image.open(row["abs_image_path"])

        if image.mode == "F":
            # 32-bit float TIFF (e.g. Preproccesed/preprocess_images.py's
            # norm_neg1_1 output): single-channel, pixel values already
            # rescaled/normalized on disk (e.g. to [-1, 1] for RadImageNet).
            # PIL's .convert("L"/"RGB") on mode "F" does NOT rescale -- it
            # clips-and-casts the floats directly into 0-255, which collapses
            # a [-1, 1] image to near-all-zero. So this path skips .convert()
            # entirely and lets `self.transform` operate on the raw "F" image
            # (torchvision's Resize/ToTensor both handle mode "F" correctly:
            # Resize interpolates the floats as-is, and ToTensor recognizes
            # "F" and copies values through without the usual /255 rescale).
            # Channel replication to 3-channel RGB happens after the
            # transform, on the tensor, instead of via PIL conversion.
            image_tensor = cast(torch.Tensor, self.transform(image))
            if not self.grayscale and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.expand(3, -1, -1).contiguous()
        else:
            # Determine PIL color space conversion ("L" = 1 channel grayscale, "RGB" = 3 channel)
            mode = "L" if self.grayscale else "RGB"
            image = image.convert(mode)
            image_tensor = cast(torch.Tensor, self.transform(image))

        label = int(row["label_norm"])

        return image_tensor, label