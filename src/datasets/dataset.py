from typing import cast

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]
from PIL import Image


class MammoBenchDataset(Dataset[tuple[torch.Tensor, int]]):
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
        """Transform mínimo: redimensiona y convierte a tensor.
        SUPUESTO: 224x224 — ajústalo si tu modelo espera otro tamaño."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        mode = "L" if self.grayscale else "RGB" # "L" is parameter to Pillow "luminance" to one canal
        image = Image.open(row["abs_image_path"]).convert(mode)

        image_tensor = cast(torch.Tensor, self.transform(image))
        label = int(row["label_norm"])

        return image_tensor, label