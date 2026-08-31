"""Configuración de un experimento: modelos Pydantic v2 + carga/guardado en YAML.

Sin herencia entre configs (sin defaults:/base.yaml) — decisión del proyecto:
cada experimento se lee de punta a punta en un solo archivo, nada se hereda
ni se sobreescribe desde otro lado. extra="forbid" en cada modelo atrapa
typos en el YAML como error de validación, no como un campo ignorado en
silencio.

Ejemplo de uso:
    >>> from src.config import load_config, save_config
    >>> config = load_config("configs/experiment_example.yaml")
    >>> save_config(config, "runs/exp_01/config.yaml")
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ArchitectureConfig(BaseModel):
    """Configuración para la instanciación del backbone del modelo (`models/build.py`).

    Attributes:
        name: Identificador registrado en `_ARCHITECTURES` (ej. `"resnet50_radimagenet"`).
        weights_path: Ruta al archivo checkpoint `.pth` o `.pt` con pesos preentrenados.
        unfreeze_from: Nombre del bloque a partir del cual descongelar gradientes (default: `"none"`).

    Example:
        >>> arch_cfg = ArchitectureConfig(
        ...     name="resnet50_radimagenet",
        ...     weights_path=Path("checkpoints/RadImageNet-ResNet50_notop.pth"),
        ...     unfreeze_from="layer3"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    name: str  # clave en _ARCHITECTURES, ej. "resnet50_radimagenet"
    weights_path: Path
    unfreeze_from: str = "none"


class NamedComponentConfig(BaseModel):
    """Configuración genérica para componentes seleccionados por nombre e hiperparámetros.

    Aplica para optimizadores, schedulers, funciones de pérdida (`train/build.py`) y
    cabezas de clasificación (`models/heads.py`).

    Attributes:
        name: Clave del algoritmo o estrategia registrada (ej. `"adamw"`, `"bce"`, `"standard_mlp"`).
        hparams: Diccionario arbitrario con los hiperparámetros pasados al constructor.

    Example:
        >>> opt_cfg = NamedComponentConfig(name="adamw", hparams={"lr": 0.0001, "weight_decay": 0.01})
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    hparams: dict[str, Any] = Field(default_factory=dict)


class AugmentationConfig(BaseModel):
    """Configuración de augmentación de imágenes del train split (`datasets/transform.py`).

    Todos los defaults reproducen exactamente el comportamiento previo a
    esta opción (`src/cli.py` construía el `TransformBuilder` de train con
    `use_horizontal_flip=True, use_rotation=True` hardcodeados y
    `rotation_degrees` en su default de `TransformBuilder`, sin flip
    vertical ni blur) — un `DataConfig` que no mencione `augmentation` en
    YAML se comporta igual que antes de que este modelo existiera.

    `vertical_flip` y `blur` están portados del proyecto INC
    (`classification_images/dataloaders/dataloader_images.py`), que además
    de flip horizontal y rotación aplica `RandomVerticalFlip(p=0.2)` y un
    blur gaussiano aleatorio — ver PHASES.md fase 5.

    Attributes:
        horizontal_flip: si aplicar flip horizontal aleatorio.
        horizontal_flip_p: probabilidad del flip horizontal.
        rotation_degrees: grados máximos de rotación aleatoria. `0`
            desactiva la rotación.
        vertical_flip: si aplicar flip vertical aleatorio.
        vertical_flip_p: probabilidad del flip vertical.
        blur: si aplicar blur gaussiano aleatorio.
        blur_p: probabilidad de aplicar el blur cuando `blur=True`.

    Example:
        >>> aug_cfg = AugmentationConfig(rotation_degrees=7, vertical_flip=True)
    """

    model_config = ConfigDict(extra="forbid")

    horizontal_flip: bool = True
    horizontal_flip_p: float = 0.5
    rotation_degrees: int = 15
    vertical_flip: bool = False
    vertical_flip_p: float = 0.5
    blur: bool = False
    blur_p: float = 0.3


class DataConfig(BaseModel):
    """Configuración para carga de datos, splits y DataLoaders (`datasets/`).

    Attributes:
        manifest_path: Ruta al archivo CSV manifest con imágenes y metadatos.
        image_root: Directorio raíz para resolver rutas relativas del manifest.
        batch_size: Tamaño de lote para los DataLoaders (default: 16).
        num_workers: Número de subprocesos paralelos para carga de datos (default: 1).
        seed: Semilla aleatoria para reproducibilidad de shuffles y augmentations (default: 42).
        image_size: Dimensiones (alto, ancho) para resize de imágenes (default: (224, 224)).
        augmentation: Configuración de augmentación del train split — ver
            `AugmentationConfig`. val/test nunca se augmentan
            (`src/cli.py` construye su `TransformBuilder` sin pasarle esto).

    Example:
        >>> data_cfg = DataConfig(
        ...     manifest_path=Path("manifests/fedmammobench.csv"),
        ...     image_root=Path("data/images"),
        ...     batch_size=32
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    manifest_path: Path
    image_root: Path
    batch_size: int = 16
    num_workers: int = 1
    seed: int = 42
    image_size: tuple[int, int] = (224, 224)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


class TrainConfig(BaseModel):
    """Configuración del bucle de entrenamiento y persistencia (`train/trainer.py`).

    Attributes:
        epochs: Número máximo de épocas a entrenar (puede terminar antes por
            early stopping si `patience` está fijado).
        metric_name: Nombre de la métrica de validación a trackear para
            guardar el mejor checkpoint (default: `"auc"`).
        metric_mode: `"max"` si más `metric_name` es mejor (auc, f1,
            accuracy, sensitivity, specificity, precision), `"min"` si menos
            es mejor (loss). Default `"max"`.
        patience: épocas sin mejora antes de activar early stopping. `None`
            (default) desactiva la parada temprana. Portado del
            `--patience_early` del proyecto INC.
        min_delta: mejora mínima para contar como mejora real, tanto para
            guardar checkpoint como para el contador de `patience`. Default
            `0.0`. El proyecto INC usa `0.005` sobre F1 de validación.
        save_every: si se fija, guarda un checkpoint periódico cada
            `save_every` épocas (independiente del mejor checkpoint), como
            `--save_every 10` implícito en el proyecto INC. `None`
            (default) no guarda checkpoints periódicos.
        checkpoint_dir: Directorio destino para archivos de peso `.pt`.
        run_dir: Directorio destino para archivos de logs (`metrics.csv`, TensorBoard).
        device: Dispositivo de cómputo (ej. `"cpu"`, `"cuda"`).
        wandb_project: Nombre opcional del proyecto en Weights & Biases (None desactiva W&B).

    Example:
        >>> train_cfg = TrainConfig(
        ...     epochs=200,
        ...     metric_name="f1",
        ...     patience=50,
        ...     min_delta=0.005,
        ...     save_every=10,
        ...     checkpoint_dir=Path("runs/exp01/weights"),
        ...     run_dir=Path("runs/exp01"),
        ...     device="cuda"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    epochs: int
    metric_name: str = "auc"
    metric_mode: str = "max"
    patience: int | None = None
    min_delta: float = 0.0
    save_every: int | None = None
    checkpoint_dir: Path
    run_dir: Path
    device: str = "cpu"
    wandb_project: str | None = None  # None -> W&B desactivado, ver tracking.py


class ExperimentConfig(BaseModel):
    """Modelo contenedor principal que valida el experimento completo desde YAML.

    Attributes:
        experiment_id: Identificador único del experimento.
        architecture: Configuración del backbone encoder.
        head: Configuración de la cabeza de clasificación.
        optimizer: Configuración del optimizador.
        scheduler: Configuración opcional del scheduler de learning rate.
        loss: Configuración del esquema de pérdida (BCE vs CrossEntropy).
        data: Configuración de datos y DataLoaders.
        train: Configuración de entrenamiento y tracking.

    Example:
        >>> config = load_config("configs/exp01.yaml")
        >>> print(config.experiment_id)
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    architecture: ArchitectureConfig
    head: NamedComponentConfig
    optimizer: NamedComponentConfig
    scheduler: NamedComponentConfig | None = None
    loss: NamedComponentConfig
    data: DataConfig
    train: TrainConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Carga y valida un ExperimentConfig desde un archivo YAML.

    Args:
        path: Ruta al archivo YAML de configuración del experimento.

    Returns:
        ExperimentConfig: Objeto Pydantic validado con todos los campos del experimento.

    Raises:
        FileNotFoundError: Si el archivo en `path` no existe.
        pydantic.ValidationError: Si el YAML tiene campos faltantes, tipos incorrectos
            o claves no reconocidas (`extra="forbid"`).

    Example:
        >>> cfg = load_config("configs/exp01.yaml")
        >>> print(cfg.architecture.name)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config no encontrado: {path}")
    raw = yaml.safe_load(path.read_text())
    return ExperimentConfig.model_validate(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Guarda un ExperimentConfig en formato YAML para registrar exactamente la corrida.

    Args:
        config: Objeto ExperimentConfig a serializar.
        path: Ruta destino del archivo YAML. La carpeta contenedora se crea si no existe.

    Example:
        >>> save_config(config, Path("runs/exp01/config.snapshot.yaml"))
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
