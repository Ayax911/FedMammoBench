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
        name: Identificador registrado en `_ARCHITECTURES` (ej.
            `"resnet50_radimagenet"`, `"resnet50_imagenet_v1"`,
            `"resnet50_imagenet_v2"`).
        weights_path: Ruta al archivo checkpoint `.pth` o `.pt` con pesos
            preentrenados. `None` (default) para arquitecturas que ya traen
            sus pesos incluidos en el `model_factory` -- ej.
            `resnet50_imagenet_v1`/`_v2`, que usan los pesos de ImageNet
            embebidos en torchvision (`ArchitectureSpec.weights_from_factory
            = True`, ver `models/build.py`) y no leen ningún archivo local.
            Para el resto (`resnet50_radimagenet`) sigue siendo obligatorio
            EN LA PRÁCTICA: `build_model()` levanta `ValueError` si falta,
            porque esas arquitecturas no tienen otra forma de conseguir sus
            pesos. No se valida acá (en `config.py`) para no hacer que este
            módulo dependa de `_ARCHITECTURES` -- ver la dirección de
            dependencias única de CLAUDE.md (`config.py` va ANTES que
            `models/` en la cadena, nunca al revés).
        unfreeze_from: Nombre del bloque a partir del cual descongelar gradientes (default: `"none"`).

    Example:
        >>> arch_cfg = ArchitectureConfig(
        ...     name="resnet50_radimagenet",
        ...     weights_path=Path("checkpoints/RadImageNet-ResNet50_notop.pth"),
        ...     unfreeze_from="layer3"
        ... )
        >>> arch_cfg = ArchitectureConfig(name="resnet50_imagenet_v2", unfreeze_from="layer4")
    """

    model_config = ConfigDict(extra="forbid")

    name: str  # clave en _ARCHITECTURES, ej. "resnet50_radimagenet"
    weights_path: Path | None = None
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
        normalize_mean: media por canal `(R, G, B)` de `transforms.Normalize`,
            aplicada tanto en train como en val/test. Default `(0.5, 0.5, 0.5)`,
            el mismo que traía hardcodeado `TransformBuilder` antes de que
            este campo existiera — un YAML que no lo mencione se comporta
            igual que antes.
        normalize_std: desviación estándar por canal `(R, G, B)` de
            `transforms.Normalize`. Default `(0.5, 0.5, 0.5)`.

            `mean=(0,0,0)` + `std=(1,1,1)` deja los píxeles en `[0, 1]`, es
            decir desactiva la normalización — que es exactamente lo que
            hace el proyecto INC, cuyo `dataloader_images.py` aplica solo
            `ToTensor()` sin ningún `Normalize`. Úsalo para reproducirlo;
            el default `0.5/0.5` (rango `[-1, 1]`) NO es equivalente.
        by_database_manifests: mapa opcional `nombre_base_de_datos ->
            manifest_path` para el desglose de test por base de datos (ver
            `_evaluate_by_database()` en `cli.py`). `None` (default)
            desactiva el desglose por completo — un YAML que no lo mencione
            se comporta exactamente igual que antes de que este campo
            existiera. Cuando se fija, `cli.run()` construye, al final del
            entrenamiento, un `Manifest`+`Split` independiente por cada
            entrada (mismo `image_root`, mismo `eval_transform_builder` que
            el split de test principal) y evalúa el mismo mejor checkpoint
            sobre el `.test_df()` de cada uno — no filtra en memoria el test
            split principal por una columna, porque construir un
            `Manifest`/`Split` real por base de datos reutiliza tal cual la
            validación anti-leakage de `Split` y deja un artefacto en disco
            (el CSV) auditable por separado. Los archivos que
            `scripts/split_manifest_by_database.py` genera en
            `manifests/by_database/` (uno por `source_dataset` de
            `fedmammobench_norm_{0_1,neg1_1}.csv`, sin tocar ninguna imagen)
            son el caso de uso pensado para este campo, pero cualquier CSV
            con el esquema de `Manifest` sirve. Las claves son solo
            etiquetas para las gráficas/summary de W&B (`test_by_database_
            {clave}_...`) — no necesitan coincidir con ningún valor de
            `source_dataset`, aunque en la práctica sí coinciden.

    Example:
        >>> data_cfg = DataConfig(
        ...     manifest_path=Path("manifests/fedmammobench.csv"),
        ...     image_root=Path("data/images"),
        ...     batch_size=32
        ... )
        >>> data_cfg = DataConfig(
        ...     manifest_path=Path("manifests/fedmammobench_norm_neg1_1.csv"),
        ...     image_root=Path("data/images"),
        ...     by_database_manifests={
        ...         "cmmd": Path("manifests/by_database/cmmd_norm_neg1_1.csv"),
        ...         "inbreast": Path("manifests/by_database/inbreast_norm_neg1_1.csv"),
        ...     },
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    manifest_path: Path
    image_root: Path
    batch_size: int = 16
    num_workers: int = 1
    seed: int = 42
    image_size: tuple[int, int] | None = (224, 224)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    normalize_mean: tuple[float, ...] | None = (0.5, 0.5, 0.5)
    normalize_std: tuple[float, ...] | None = (0.5, 0.5, 0.5)
    by_database_manifests: dict[str, Path] | None = None


class TrainConfig(BaseModel):
    """Configuración del bucle de entrenamiento y persistencia (`train/trainer.py`).

    Attributes:
        epochs: Número máximo de épocas a entrenar (puede terminar antes por
            early stopping si `patience` está fijado).
        metric_name: Nombre de la métrica de validación a trackear para
            guardar el mejor checkpoint (default: `"auc"`). Debe ser una
            clave de `build_metric_collection()` (`src/metrics.py`) o
            `"loss"`. Para reproducir la serie de notebooks el criterio es
            `"f1_macro"`, no `"f1"` — este último mide solo la clase
            positiva y se clava en 0.0 cuando el modelo no predice ningún
            maligno, lo que dispara el early stopping antes de tiempo (ver
            la nota del módulo `src/metrics.py`).
        metric_mode: `"max"` si más `metric_name` es mejor (auc, f1,
            f1_macro, accuracy, sensitivity, specificity, precision),
            `"min"` si menos es mejor (loss). Default `"max"`.
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
        freeze_bn_stats: si `True` (default), las capas BatchNorm cuyos
            parámetros están congelados se mantienen en `eval()` durante el
            entrenamiento, así sus `running_mean`/`running_var` NO se
            actualizan (`train/loop.py:_set_frozen_bn_eval`).

            `False` reproduce el comportamiento del proyecto INC, que nunca
            las re-evalúa: con `model.train()` las estadísticas del backbone
            siguen adaptándose a los datos de entrenamiento aunque los pesos
            estén congelados. Con el backbone 100% congelado esa es la
            diferencia entre un backbone que efectivamente cambia (INC) y
            uno que se queda con las estadísticas de RadImageNet (default).
        wandb_project: Nombre opcional del proyecto en Weights & Biases (None desactiva W&B).
        wandb_group: Nombre opcional de grupo en Weights & Biases (`wandb.init(group=...)`).
            Junta en la UI las corridas de un mismo bloque de experimentos (ej. el barrido
            fullfreeze/layer4 x posweight, o el barrido de profundidad de cabeza) bajo una
            sola fila expandible, sin tocar `experiment_id` ni `run_dir`. Ignorado si
            `wandb_project` es `None`. `None` (default) dejaría la corrida sin grupo — visible
            suelta en la lista de la UI en vez de agrupada.

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
    freeze_bn_stats: bool = True
    wandb_project: str | None = None  # None -> W&B desactivado, ver tracking.py
    wandb_group: str | None = None  # agrupa corridas en la UI de W&B, ver tracking.py


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
