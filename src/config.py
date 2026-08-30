"""Config de un experimento: modelos Pydantic v2 + carga/guardado en YAML.

Sin herencia entre configs (sin defaults:/base.yaml) — decisión del proyecto:
cada experimento se lee de punta a punta en un solo archivo, nada se hereda
ni se sobreescribe desde otro lado. extra="forbid" en cada modelo atrapa
typos en el YAML como error de validación, no como un campo ignorado en
silencio.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ArchitectureConfig(BaseModel):
    """Config para build_model() (models/build.py)."""

    model_config = ConfigDict(extra="forbid")

    name: str  # clave en _ARCHITECTURES, ej. "resnet50_radimagenet"
    weights_path: Path
    unfreeze_from: str = "none"


class NamedComponentConfig(BaseModel):
    """Config genérico para cualquier componente elegido por nombre +
    hiperparámetros propios: optimizer/scheduler/loss (dict-dispatch en
    train/build.py — el algoritmo ya existe en PyTorch, solo cambian los
    hiperparámetros) y también head (Strategy, models/heads.py).

    head usaba antes un HeadConfig con campos fijos (hidden_dim, dropout,
    num_classes) — se corrigió a este mismo shape genérico: head_cls
    (Type[HeadBuilder]) solo expone el constructor de la ABC base, nunca el
    de la subclase concreta que get_head_strategy() devuelva en runtime, así
    que ningún set fijo de kwargs puede tipar correctamente contra cualquier
    HeadBuilder futuro — mismo problema, mismo shape de solución que
    optimizer/scheduler/loss.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    hparams: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Config para Manifest/Split/builder_dataloader (datasets/)."""

    model_config = ConfigDict(extra="forbid")

    manifest_path: Path
    image_root: Path
    batch_size: int = 16
    num_workers: int = 1
    seed: int = 42
    image_size: tuple[int, int] = (224, 224)


class TrainConfig(BaseModel):
    """Config para Trainer (train/trainer.py)."""

    model_config = ConfigDict(extra="forbid")

    epochs: int
    metric_name: str = "auc"
    checkpoint_dir: Path
    run_dir: Path
    device: str = "cpu"
    wandb_project: str | None = None  # None -> W&B desactivado, ver tracking.py


class ExperimentConfig(BaseModel):
    """Config completo de un experimento — un YAML, de punta a punta."""

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
        path: ruta al YAML del experimento.

    Returns:
        ExperimentConfig ya validado (tipos, extra="forbid" en cada nivel).

    Raises:
        FileNotFoundError: path no existe.
        pydantic.ValidationError: el YAML tiene un campo faltante,
            de tipo incorrecto, o un campo desconocido (typo).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config no encontrado: {path}")
    raw = yaml.safe_load(path.read_text())
    return ExperimentConfig.model_validate(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Guarda un ExperimentConfig a YAML — para dejar registrado, junto al
    resto de runs/<RUN_NAME>/, exactamente qué config produjo esa corrida.

    Args:
        config: config a serializar.
        path: ruta destino — la carpeta contenedora se crea si no existe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
