from typing import Any, Callable

from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]


class TransformBuilder:
    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        use_horizontal_flip: bool = False,
        use_rotation: bool = False,
        rotation_degrees: int = 15,
        horizontal_flip_p: float = 0.5,
        normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        """
        Args:
            image_size: Tamaño final (alto, ancho) tras el resize.
            use_horizontal_flip: Si True, incluye flip horizontal aleatorio.
            use_rotation: Si True, incluye rotación aleatoria.
            rotation_degrees: Rango máximo de rotación, solo si use_rotation=True.
            horizontal_flip_p: Probabilidad de flip, solo si use_horizontal_flip=True.
        """
        self.image_size = image_size
        self.use_horizontal_flip = use_horizontal_flip
        self.use_rotation = use_rotation
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_p = horizontal_flip_p
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def build(self) -> transforms.Compose:
        """Construye el pipeline según qué flags estén activos en self."""
        steps: list[Callable[[Any], Any]] = [transforms.Resize(self.image_size)]

        if self.use_horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip(p=self.horizontal_flip_p))

        if self.use_rotation:
            steps.append(transforms.RandomRotation(degrees=self.rotation_degrees, fill=0))

        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std))

        return transforms.Compose(steps)