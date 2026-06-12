"""Albumentations-based transform pipelines for mammography.

Two pipelines are produced by :func:`build_transforms`: a training transform
with stochastic augmentation, and an evaluation transform that only resizes
and normalizes. The augmentation set is conservative and clinically defensible
(no aggressive vertical flips on standard CC/MLO views unless explicitly
enabled).

Normalization
-------------
Mean and standard deviation are resolved in this priority order:

1. ``augmentation.normalize_preset`` — a named preset from
   :data:`fedmammobench.configs.schema.NORMALIZE_PRESETS` (e.g.
   ``"radimagenet_gray"``).  Length must match ``in_channels``.
2. ``augmentation.normalize_mean`` / ``normalize_std`` as ``list[float]``
   (one value per channel).  Length validated against ``in_channels``.
3. ``augmentation.normalize_mean`` / ``normalize_std`` as a scalar ``float``
   — replicated to ``in_channels`` channels (legacy behaviour).
"""

from __future__ import annotations

import warnings
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from fedmammobench.configs.schema import NORMALIZE_PRESETS, AugmentationConfig


# ---------------------------------------------------------------------------
# Norm resolution helper
# ---------------------------------------------------------------------------

def _resolve_norm(
    aug: AugmentationConfig,
    in_channels: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(mean, std)`` tuples of length ``in_channels``.

    Raises:
        ValueError: if a preset is unknown, or if mean/std lengths differ
            from ``in_channels``.
    """
    if aug.normalize_preset is not None:
        preset = NORMALIZE_PRESETS.get(aug.normalize_preset)
        if preset is None:
            raise ValueError(
                f"Unknown normalize_preset: {aug.normalize_preset!r}. "
                f"Available presets: {sorted(NORMALIZE_PRESETS)}"
            )
        mean: tuple[float, ...] = preset["mean"]
        std: tuple[float, ...] = preset["std"]
        if len(mean) != in_channels:
            raise ValueError(
                f"normalize_preset={aug.normalize_preset!r} has {len(mean)} "
                f"channel(s) but in_channels={in_channels}. Choose a preset "
                f"that matches your channel count or set normalize_preset=None."
            )
        return mean, std

    # --- scalar or per-channel list ---
    raw_mean = aug.normalize_mean
    raw_std = aug.normalize_std

    if isinstance(raw_mean, (int, float)):
        mean = (float(raw_mean),) * in_channels
    else:
        mean = tuple(float(v) for v in raw_mean)

    if isinstance(raw_std, (int, float)):
        std = (float(raw_std),) * in_channels
    else:
        std = tuple(float(v) for v in raw_std)

    if len(mean) != in_channels:
        raise ValueError(
            f"normalize_mean has {len(mean)} value(s) but in_channels="
            f"{in_channels}. Provide a scalar or a list of exactly "
            f"{in_channels} value(s)."
        )
    if len(std) != in_channels:
        raise ValueError(
            f"normalize_std has {len(std)} value(s) but in_channels="
            f"{in_channels}. Provide a scalar or a list of exactly "
            f"{in_channels} value(s)."
        )
    return mean, std


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_transforms(
    image_size: int,
    augmentation: AugmentationConfig,
    *,
    in_channels: int = 1,
    grayscale: bool | None = None,
) -> tuple[A.Compose, A.Compose]:
    """Build (train_transform, eval_transform).

    Args:
        image_size: Square resize target (height == width).
        augmentation: :class:`~fedmammobench.configs.schema.AugmentationConfig`
            instance with knobs for the training pipeline and normalization.
        in_channels: Number of image channels fed to the model (1 for
            grayscale, 3 for RGB).  Drives mean/std tuple length.
        grayscale: *Deprecated.* If provided, overrides ``in_channels``
            (``True`` → 1, ``False`` → 3).  Use ``in_channels`` instead.

    Returns:
        Tuple ``(train_transform, eval_transform)`` of Albumentations
        ``Compose`` objects. Both yield ``torch.Tensor`` via
        :class:`~albumentations.pytorch.ToTensorV2`.

    Raises:
        ValueError: if mean/std length does not match ``in_channels``, or if
            the requested preset is unknown.
    """
    if grayscale is not None:
        warnings.warn(
            "build_transforms(..., grayscale=...) is deprecated. "
            "Use in_channels=1 (grayscale) or in_channels=3 (RGB) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        in_channels = 1 if grayscale else 3

    mean, std = _resolve_norm(augmentation, in_channels)

    train_ops: list[Any] = [A.Resize(image_size, image_size)]
    if augmentation.horizontal_flip:
        train_ops.append(A.HorizontalFlip(p=0.5))
    if augmentation.vertical_flip:
        train_ops.append(A.VerticalFlip(p=0.5))
    if augmentation.rotate_limit > 0:
        train_ops.append(
            A.Rotate(limit=augmentation.rotate_limit, border_mode=0, p=0.5)
        )
    if augmentation.brightness_contrast:
        train_ops.append(
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        )
    if augmentation.elastic:
        train_ops.append(A.ElasticTransform(alpha=20, sigma=5, p=0.3))
    train_ops.extend(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    eval_ops: list[Any] = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(train_ops), A.Compose(eval_ops)


__all__ = ["build_transforms", "_resolve_norm"]
