from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MammographyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable | None = None) -> None:
        """Wrap a manifest slice as a PyTorch dataset.

        Args:
            df: A DataFrame produced by Split (e.g. train_df()/val_df()/test_df()).
                Must carry 'abs_image_path' and 'label_norm' columns — both added
                by Manifest, never computed here.
            transform: Callable applied to the PIL image before returning it
                (e.g. a torchvision.transforms.Compose). None returns the raw
                PIL image untouched.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (image, label) for row idx.

        idx is positional (.iloc), not a pandas index label — required because
        the DataFrame Split hands back can carry a non-contiguous index even
        after the reset_index() above resets it, since any further slicing by
        the caller (e.g. filtering by source_dataset) reintroduces gaps.
        """
        row = self.df.iloc[idx]

        image = Image.open(row["abs_image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["label_norm"])
