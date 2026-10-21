"""Image transformation and data augmentation pipeline builder module."""

import random
from typing import Any, Callable

import torch
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]
from torchvision.transforms import functional as TF  # pyright: ignore[reportMissingTypeStubs]


class _JointFlip:
    """Replica `albumentations.Flip(p=...)` de FLamby: UNA sola tirada de probabilidad
    `p`; si sale, elige horizontal/vertical/ambos con igual probabilidad (1/3 cada
    uno). Deliberadamente distinto de componer `RandomHorizontalFlip`+
    `RandomVerticalFlip` independientes: eso da 75% de probabilidad de algún flip
    (25% cada combinación), mientras que `Flip(p=0.5)` da 50% (verificado contra la
    firma real de `albumentations` instalada -- ver docs/FEDISIC2019_FLAMBY_REPRO.md).
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img
        direction = random.choice(("horizontal", "vertical", "both"))
        if direction == "horizontal":
            return TF.hflip(img)
        if direction == "vertical":
            return TF.vflip(img)
        return TF.vflip(TF.hflip(img))


class TransformBuilder:
    """Configurable factory builder for torchvision image processing pipelines.

    Constructs a `transforms.Compose` pipeline incorporating spatial resizing, optional
    data augmentations (horizontal flip, random rotation, vertical flip, gaussian blur),
    and normalization on 3-channel PyTorch tensors.

    Args:
        image_size: Spatial target resolution tuple `(height, width)` or integer `size`
            for the shorter edge (preserving aspect ratio) after resizing. Default is `(224, 224)`.
        use_horizontal_flip: If True, adds random horizontal flip augmentation. Default is False.
        use_rotation: If True, adds random rotation augmentation. Default is False.
        rotation_degrees: Maximum degree range for random rotation (used if `use_rotation=True`). Default is 15.
        rotation_p: Probability of applying the rotation step at all (used if `use_rotation=True`).
            Default `1.0` reproduces the previous byte-identical behavior (rotation always
            applied, since `transforms.RandomRotation` itself has no `p=`). FLamby's own
            `albumentations.Rotate(50)` for Fed-ISIC2019 defaults to `p=0.5` — pass that
            explicitly when replicating FLamby, don't rely on this default.
        horizontal_flip_p: Probability of applying horizontal flip (used if `use_horizontal_flip=True`). Default is 0.5.
        use_vertical_flip: If True, adds random vertical flip augmentation. Default is False.
            Ported from the INC project (classification_images/dataloaders/dataloader_images.py),
            which applies it in addition to horizontal flip and rotation.
        vertical_flip_p: Probability of applying vertical flip (used if `use_vertical_flip=True`). Default is 0.5.
        use_joint_flip: If True, replaces independent horizontal/vertical flips with a single
            FLamby-style `albumentations.Flip(p=...)` draw (see `_JointFlip`): one coin flip at
            probability `joint_flip_p`, and if it lands, horizontal/vertical/both are chosen with
            equal probability. Mutually exclusive with `use_horizontal_flip`/`use_vertical_flip` —
            enable this ALONE, don't combine with those (composing independent H+V flips gives a
            75% chance of some flip; FLamby's single draw gives `joint_flip_p`, default 50%).
            Default `False` (no existing config is affected).
        joint_flip_p: Probability used by `use_joint_flip`. Default `0.5`, matching
            `albumentations.Flip`'s own default.
        use_blur: If True, randomly applies gaussian blur with probability `blur_p`. Default is False.
            Ported from the INC project's `RandomGaussianBlur` — implemented here via
            `transforms.RandomApply([GaussianBlur(...)], p=...)` instead of a bespoke
            `nn.Module`, since torchvision already provides the building block.
        blur_p: Probability of applying the blur (used if `use_blur=True`). Default is 0.3.
        blur_kernel_size: Kernel size for `GaussianBlur`. Default is 3.
        blur_sigma: `(min, max)` range for `GaussianBlur`'s sigma. Default is `(0.1, 0.6)`.
        normalize_mean: Per-channel normalization mean tuple/sequence. Default is `(0.5, 0.5, 0.5)`.
            Because channel replication to 3 channels occurs in `MammoBenchDataset` BEFORE
            `transform` runs, `Normalize` always receives a 3-channel tensor `(3, H, W)`.
            Both 1-element tuples/lists (broadcast across all 3 channels) and 3-element
            tuples/lists are supported. Existing configs (`[0.449]`, `null`, `(0.5, 0.5, 0.5)`)
            remain valid without changes.
        use_random_scale / random_scale_p / random_scale_range: random zoom via
            `RandomAffine(scale=...)` (opt-in, default off; Fed-ISIC2019). Default
            range `(0.93, 1.07)` matches FLamby's own
            `albumentations.RandomScale(0.07)` (±7%) — verified against
            `flamby/datasets/fed_isic2019/dataset.py` in the official repo.
        use_random_brightness_contrast / random_brightness_contrast_p / brightness / contrast:
            `ColorJitter` (opt-in; OJO: destroys float TIFFs, only for uint8-origin RGB).
            Defaults `brightness=0.15, contrast=0.1` match FLamby's
            `RandomBrightnessContrast(0.15, 0.1)`.
        use_affine / affine_p / affine_degrees / affine_translate / affine_shear:
            `RandomAffine` translate + shear (opt-in). Default `affine_shear=0.1`
            matches FLamby's `albumentations.Affine(shear=0.1)` — a barely
            perceptible shear range, not a typo: that's what the paper's reference
            implementation actually uses. `affine_translate` defaults to `None`
            (no translation) — FLamby's own `Affine(shear=0.1)` call leaves
            `translate_percent` unset (`None` in the installed `albumentations` API,
            verified against the real signature), so it never translates; pass an
            explicit `(tx, ty)` only if you actually want translation on top of shear.
        crop_size / use_random_crop / random_crop_p: deterministic square crop down
            from `image_size` to `crop_size` — `RandomCrop` when `use_random_crop`
            is set (train), `CenterCrop` otherwise (eval), matching FLamby's
            `RandomCrop(200, 200)` (train) / `CenterCrop(200, 200)` (test). Unlike
            the other entries here, this one applies whenever `crop_size` is set,
            train AND eval alike — it's a sizing step, not a probabilistic
            augmentation (FLamby's own crop has no `p=`, it always runs).
        use_coarse_dropout / coarse_dropout_p / coarse_dropout_scale: `RandomErasing`
            as a substitute for Coarse Dropout (opt-in). `coarse_dropout_scale`
            approximates FLamby's `CoarseDropout(1-8 holes, 16, 16)` (1 to 8 holes
            of 16x16px on a 200x200 image, i.e. ~0.6%-5.1% of the image area) as a
            single erased rectangle of comparable aggregate area — not identical
            (one hole vs several), documented approximation.
        normalize_std: Per-channel normalization standard deviation tuple/sequence.
            Default is `(0.5, 0.5, 0.5)`. Follows the same broadcast behavior as `normalize_mean`.

    Example:
        >>> from src.datasets.transform import TransformBuilder
        >>> builder = TransformBuilder(image_size=(224, 224), use_horizontal_flip=True)
        >>> pipeline = builder.build()
    """

    def __init__(
        self,
        image_size: tuple[int, int] | int | None = (224, 224),
        crop_size: tuple[int, int] | None = None,
        use_horizontal_flip: bool = False,
        use_rotation: bool = False,
        rotation_degrees: int = 15,
        rotation_p: float = 1.0,
        horizontal_flip_p: float = 0.5,
        use_vertical_flip: bool = False,
        vertical_flip_p: float = 0.5,
        use_joint_flip: bool = False,
        joint_flip_p: float = 0.5,
        use_blur: bool = False,
        blur_p: float = 0.3,
        blur_kernel_size: int = 3,
        blur_sigma: tuple[float, float] = (0.1, 0.6),
        normalize_mean: tuple[float, ...] | None = (0.5, 0.5, 0.5),
        normalize_std: tuple[float, ...] | None = (0.5, 0.5, 0.5),
        use_random_scale: bool = False,
        random_scale_p: float = 0.5,
        random_scale_range: tuple[float, float] = (0.93, 1.07),
        use_random_brightness_contrast: bool = False,
        random_brightness_contrast_p: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.1,
        use_affine: bool = False,
        affine_p: float = 0.5,
        affine_degrees: float = 0.0,
        affine_translate: tuple[float, float] | None = None,
        affine_shear: float = 0.1,
        use_random_crop: bool = False,
        use_coarse_dropout: bool = False,
        coarse_dropout_p: float = 0.5,
        coarse_dropout_scale: tuple[float, float] = (0.006, 0.05),
    ) -> None:
        self.image_size = image_size
        self.crop_size = crop_size
        self.use_horizontal_flip = use_horizontal_flip
        self.use_rotation = use_rotation
        self.rotation_degrees = rotation_degrees
        self.rotation_p = rotation_p
        self.horizontal_flip_p = horizontal_flip_p
        self.use_vertical_flip = use_vertical_flip
        self.vertical_flip_p = vertical_flip_p
        self.use_joint_flip = use_joint_flip
        self.joint_flip_p = joint_flip_p
        self.use_blur = use_blur
        self.blur_p = blur_p
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.use_random_scale = use_random_scale
        self.random_scale_p = random_scale_p
        self.random_scale_range = random_scale_range
        self.use_random_brightness_contrast = use_random_brightness_contrast
        self.random_brightness_contrast_p = random_brightness_contrast_p
        self.brightness = brightness
        self.contrast = contrast
        self.use_affine = use_affine
        self.affine_p = affine_p
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_shear = affine_shear
        self.use_random_crop = use_random_crop
        self.use_coarse_dropout = use_coarse_dropout
        self.coarse_dropout_p = coarse_dropout_p
        self.coarse_dropout_scale = coarse_dropout_scale

    def build(self) -> transforms.Compose:
        """Assembles and returns the configured torchvision transformation pipeline.

        Returns:
            transforms.Compose: Ordered torchvision transformation pipeline.

        Example:
            >>> tx = TransformBuilder(image_size=(512, 512)).build()
        """
        steps: list[Callable[[Any], Any]] = []

        # Step 1: Spatial resolution resizing (skipped if image_size is None)
        if self.image_size is not None:
            steps.append(transforms.Resize(self.image_size))

        # Step 2: Conditional training data augmentations (order matches the
        # INC project's dataloader_images.py: HorizontalFlip, RandomRotation,
        # VerticalFlip, GaussianBlur; all operate directly on tensors).
        if self.use_horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip(p=self.horizontal_flip_p))

        if self.use_rotation:
            # OJO: transforms.RandomApply.forward() llama torch.rand(1) SIEMPRE,
            # incluso con p=1.0 -- envolverlo incondicionalmente correría un
            # draw extra del RNG global en cada muestra y rompería la
            # reproducibilidad bit a bit de cualquier config existente (todas
            # usan rotación por default, rotation_degrees=15). Por eso el
            # wrapper solo se usa cuando rotation_p<1.0 (el único caso en que
            # cambia el comportamiento); con el default 1.0 el camino es
            # exactamente el de siempre, sin draw adicional.
            if self.rotation_p < 1.0:
                steps.append(
                    transforms.RandomApply(
                        [transforms.RandomRotation(degrees=self.rotation_degrees, fill=0)],
                        p=self.rotation_p,
                    )
                )
            else:
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

        # Step 2b: augmentations añadidas para Fed-ISIC2019 (FLamby; ver
        # docs/FEDISIC2019_FLAMBY_REPRO.md §5.G). Todas opt-in, default False:
        # con todos los flags apagados el pipeline es el de siempre. Operan
        # sobre tensores float; ColorJitter es seguro para las imágenes RGB
        # uint8->float [0,1] de ISIC pero destruye TIFFs float de mamografía
        # (ver CLAUDE.md) -- por eso no se activa por default.
        if self.use_random_scale:
            steps.append(
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, scale=self.random_scale_range)],
                    p=self.random_scale_p,
                )
            )

        if self.use_random_brightness_contrast:
            steps.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast)],
                    p=self.random_brightness_contrast_p,
                )
            )

        if self.use_joint_flip:
            # Posición exacta del `Flip(p=0.5)` de FLamby: después de
            # brightness/contrast, antes del affine (ver dataset.py oficial).
            # Mutuamente excluyente con use_horizontal_flip/use_vertical_flip --
            # AugmentationConfig lo valida (model_validator en src/config.py).
            steps.append(_JointFlip(p=self.joint_flip_p))

        if self.use_affine:
            steps.append(
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=self.affine_degrees,
                            translate=self.affine_translate,
                            shear=self.affine_shear,
                        )
                    ],
                    p=self.affine_p,
                )
            )

        if self.crop_size is not None:
            # Recorte determinista (no envuelto en RandomApply: FLamby's propio
            # RandomCrop/CenterCrop no tiene `p=`, siempre corre) -- train usa
            # una ubicación aleatoria, eval siempre el centro. Requiere que
            # `image_size` ya sea >= `crop_size` en ambas dimensiones (Step 1
            # ya corrió arriba).
            if self.use_random_crop:
                steps.append(transforms.RandomCrop(self.crop_size))
            else:
                steps.append(transforms.CenterCrop(self.crop_size))

        if self.use_coarse_dropout:
            # RandomErasing (torchvision) como sustituto de Coarse Dropout: borra
            # UN rectángulo con ceros (el de albumentations borra varios).
            steps.append(
                transforms.RandomErasing(
                    p=self.coarse_dropout_p, scale=self.coarse_dropout_scale, value=0.0
                )
            )

        # Step 3: Channel normalization (x - mean) / std (skipped if None)
        if self.normalize_mean is not None and self.normalize_std is not None:
            steps.append(transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std))

        return transforms.Compose(steps)
