"""Tests for `datasets.transforms.build_transforms` resize handling (v0.9.0).

Run::

    pytest tests/test_transforms.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running ``pytest tests/`` without ``pip install -e .``.
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _non_square_image(fill: int = 200) -> np.ndarray:
    # Tall rectangle, like a real mammography source image, filled with a
    # distinguishable non-zero value so padding (black) is easy to detect.
    return np.full((400, 150), fill, dtype=np.uint8)


class TestResizeModes:
    def test_squash_output_shape(self):
        from fedmammobench.configs.training_config import AugmentationConfig
        from fedmammobench.datasets.transforms import build_transforms

        aug = AugmentationConfig(resize_mode="squash")
        _, eval_tx = build_transforms(image_size=224, augmentation=aug, in_channels=1)
        out = eval_tx(image=_non_square_image())["image"]
        assert tuple(out.shape) == (1, 224, 224)

    def test_letterbox_output_shape(self):
        from fedmammobench.configs.training_config import AugmentationConfig
        from fedmammobench.datasets.transforms import build_transforms

        aug = AugmentationConfig(resize_mode="letterbox")
        _, eval_tx = build_transforms(image_size=224, augmentation=aug, in_channels=1)
        out = eval_tx(image=_non_square_image())["image"]
        assert tuple(out.shape) == (1, 224, 224)

    def test_letterbox_pads_with_black(self):
        from fedmammobench.configs.training_config import AugmentationConfig
        from fedmammobench.datasets.transforms import build_transforms

        # normalize_mean/std=0 -> raw pixel value 0 maps to 0.0 in the tensor,
        # so a black-padded corner is exactly 0.0 and easy to assert on.
        aug = AugmentationConfig(
            resize_mode="letterbox", normalize_mean=0.0, normalize_std=1.0
        )
        _, eval_tx = build_transforms(image_size=224, augmentation=aug, in_channels=1)
        out = eval_tx(image=_non_square_image(fill=200))["image"]
        # A 400x150 image padded to a 224x224 square leaves black bars on the
        # left/right; the top-left corner falls in the padded region.
        assert out[0, 0, 0].item() == pytest.approx(0.0)

    def test_squash_does_not_pad(self):
        from fedmammobench.configs.training_config import AugmentationConfig
        from fedmammobench.datasets.transforms import build_transforms

        aug = AugmentationConfig(
            resize_mode="squash", normalize_mean=0.0, normalize_std=1.0
        )
        _, eval_tx = build_transforms(image_size=224, augmentation=aug, in_channels=1)
        out = eval_tx(image=_non_square_image(fill=200))["image"]
        # Squash stretches the fill value across the whole square -> no
        # black corners.
        assert out[0, 0, 0].item() > 0.0

    def test_upscale_does_not_raise(self):
        # image_size larger than the source is a degenerate case in this
        # benchmark (all real sources are >=224px) but must not crash.
        from fedmammobench.configs.training_config import AugmentationConfig
        from fedmammobench.datasets.transforms import build_transforms

        aug = AugmentationConfig()
        _, eval_tx = build_transforms(image_size=512, augmentation=aug, in_channels=1)
        out = eval_tx(image=_non_square_image())["image"]
        assert tuple(out.shape) == (1, 512, 512)
