"""Image transformation and data augmentation pipeline builder module."""

from typing import Any, Callable

from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]


class TransformBuilder:
    """Configurable factory builder for torchvision image processing pipelines.

    Constructs a `transforms.Compose` pipeline incorporating spatial resizing, optional
    data augmentations (horizontal flip, random rotation, vertical flip, gaussian blur),
    tensor conversion, and normalization.

    Args:
        image_size: Spatial target resolution tuple `(height, width)` after resizing. Default is `(224, 224)`.
        use_horizontal_flip: If True, adds random horizontal flip augmentation. Default is False.
        use_rotation: If True, adds random rotation augmentation. Default is False.
        rotation_degrees: Maximum degree range for random rotation (used if `use_rotation=True`). Default is 15.
        horizontal_flip_p: Probability of applying horizontal flip (used if `use_horizontal_flip=True`). Default is 0.5.
        use_vertical_flip: If True, adds random vertical flip augmentation. Default is False.
            Ported from the INC project (classification_images/dataloaders/dataloader_images.py),
            which applies it in addition to horizontal flip and rotation.
        vertical_flip_p: Probability of applying vertical flip (used if `use_vertical_flip=True`). Default is 0.5.
        use_blur: If True, randomly applies gaussian blur with probability `blur_p`. Default is False.
            Ported from the INC project's `RandomGaussianBlur` — implemented here via
            `transforms.RandomApply([GaussianBlur(...)], p=...)` instead of a bespoke
            `nn.Module`, since torchvision already provides the building block.
        blur_p: Probability of applying the blur (used if `use_blur=True`). Default is 0.3.
        blur_kernel_size: Kernel size for `GaussianBlur`. Default is 3.
        blur_sigma: `(min, max)` range for `GaussianBlur`'s sigma. Default is `(0.1, 0.6)`.
        normalize_mean: Per-channel normalization mean tuple `(R, G, B)`. Default is `(0.5, 0.5, 0.5)`.
        normalize_std: Per-channel normalization standard deviation tuple `(R, G, B)`. Default is `(0.5, 0.5, 0.5)`.

    Example:
        >>> from src.datasets.transform import TransformBuilder
        >>> builder = TransformBuilder(image_size=(224, 224), use_horizontal_flip=True)
        >>> pipeline = builder.build()
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        use_horizontal_flip: bool = False,
        use_rotation: bool = False,
        rotation_degrees: int = 15,
        horizontal_flip_p: float = 0.5,
        use_vertical_flip: bool = False,
        vertical_flip_p: float = 0.5,
        use_blur: bool = False,
        blur_p: float = 0.3,
        blur_kernel_size: int = 3,
        blur_sigma: tuple[float, float] = (0.1, 0.6),
        normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.image_size = image_size
        self.use_horizontal_flip = use_horizontal_flip
        self.use_rotation = use_rotation
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_p = horizontal_flip_p
        self.use_vertical_flip = use_vertical_flip
        self.vertical_flip_p = vertical_flip_p
        self.use_blur = use_blur
        self.blur_p = blur_p
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def build(self) -> transforms.Compose:
        """Assembles and returns the configured torchvision transformation pipeline.

        Returns:
            transforms.Compose: Ordered torchvision transformation pipeline.

        Example:
            >>> tx = TransformBuilder(image_size=(512, 512)).build()
        """
        # Step 1: Spatial resolution resizing
        steps: list[Callable[[Any], Any]] = [transforms.Resize(self.image_size)]

        # Step 2: Conditional training data augmentations (order matches the
        # INC project's dataloader_images.py, minus the ToTensor-first
        # ordering it uses -- these all support PIL input equally, so this
        # project keeps its existing PIL-first convention).
        if self.use_horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip(p=self.horizontal_flip_p))

        if self.use_rotation:
            steps.append(transforms.RandomRotation(degrees=self.rotation_degrees, fill=0))

        if self.use_vertical_flip:
            steps.append(transforms.RandomVerticalFlip(p=self.vertical_flip_p))

        if self.use_blur:
            steps.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=self.blur_sigma)],
                    p=self.blur_p,
                )
            )

        # Step 3: PyTorch tensor conversion [0, 255] -> [0.0, 1.0]
        steps.append(transforms.ToTensor())
        # Step 4: Channel normalization (x - mean) / std
        steps.append(transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std))

        return transforms.Compose(steps)
