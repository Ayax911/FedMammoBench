"""Ensamblado del lado del nodo: manifest -> split -> loaders -> modelo -> loss.

Es el bloque de reconstrucción de `evaluate.py:run_evaluation()` (que a su
vez replica el de `cli.run()`), factorizado UNA vez para los dos
consumidores federados que lo necesitan idéntico: `client.py` (entrena
sobre él ronda a ronda) y `evaluate_node.py` (lo reconstruye para la
evaluación final del mejor modelo global). Si viviera copiado en ambos,
cualquier corrección al pipeline de datos (el gotcha de los TIFF modo "F",
el orden semilla-antes-que-datasets, el índice posicional del backbone)
tendría tres copias que mantener en vez de una.

NO llama a `set_global_seed()`: eso es responsabilidad del entrypoint (una
sola vez por proceso, ANTES de construir nada — mismo orden que
`cli.run()`), no de una función que un proceso puede invocar más de una
vez.
"""

from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import DataLoader

from ..datasets.build import builder_dataloader
from ..datasets.manifest import Manifest
from ..datasets.split import Split
from ..datasets.transform import TransformBuilder
from ..models.build import build_model
from ..models.heads import get_head_strategy
from ..train.build import LossSpec, build_loss
from .config import FederatedNodeConfig


@dataclass(frozen=True)
class NodeAssembly:
    """Todo lo que un proceso de nodo necesita tener construido una sola vez.

    Attributes:
        model: `nn.Sequential(backbone, head)` ya en `config.run.device`.
            Mismo ensamblado posicional que `cli.run()` — `model[0]` es el
            backbone, `model[1]` la cabeza (el contrato que
            `param_utils.scope_module` y `Trainer._split_state_dicts`
            comparten).
        loaders: `{"train", "val", "test"}` de `builder_dataloader()` —
            train con shuffle+drop_last+generator seedeado, val/test sin
            shuffle.
        loss_spec: `LossSpec` del YAML del nodo, en el device del nodo.
        eval_transform_builder: el transform SIN augmentación — lo reusa
            `evaluate_node.py` si hiciera falta reconstruir loaders de
            evaluación.
        n_train / n_val / n_test: tamaños de los splits locales. `n_train`
            y `n_val` son los `num_examples` que el nodo reporta a Flower
            (el peso de este nodo en la agregación de pesos y de métricas).
    """

    model: nn.Sequential
    loaders: dict[str, DataLoader]
    loss_spec: LossSpec
    eval_transform_builder: TransformBuilder
    n_train: int
    n_val: int
    n_test: int


def build_node_assembly(config: FederatedNodeConfig) -> NodeAssembly:
    """Construye el ensamblado completo de un nodo desde su YAML.

    Mismo orden y mismas decisiones que `cli.run()`/`evaluate.py` (ver el
    docstring del módulo). El backbone se construye CON sus pesos
    preentrenados (`weights_path` local del nodo) aunque la ronda 1 los
    vaya a pisar con los parámetros iniciales del servidor: mantiene la
    validación de `LoadReport` (matched == 0 revienta acá, no en silencio)
    y deja al nodo utilizable para depuración sin servidor. El costo es una
    carga de checkpoint que en la workstation toma segundos.

    Args:
        config: el YAML del nodo ya validado.

    Returns:
        NodeAssembly: ver la dataclass.

    Raises:
        Todo lo que levantan las piezas subyacentes: `FileNotFoundError`/
        `ValueError` del manifest, `ValueError` de split con fuga de
        pacientes, `RuntimeError` de `LoadReport` con matched == 0, etc.

    Example:
        >>> assembly = build_node_assembly(load_node_config("node_cmmd.yaml"))
        >>> assembly.n_train
        3776
    """
    manifest = Manifest(manifest_path=config.data.manifest_path, image_root=config.data.image_root)
    split = Split(manifest=manifest)

    aug = config.data.augmentation
    train_transform_builder = TransformBuilder(
        image_size=config.data.image_size,
        use_horizontal_flip=aug.horizontal_flip,
        horizontal_flip_p=aug.horizontal_flip_p,
        use_rotation=aug.rotation_degrees > 0,
        rotation_degrees=aug.rotation_degrees,
        use_vertical_flip=aug.vertical_flip,
        vertical_flip_p=aug.vertical_flip_p,
        use_blur=aug.blur,
        blur_p=aug.blur_p,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    eval_transform_builder = TransformBuilder(
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
    )
    loaders = builder_dataloader(
        split,
        train_transform_builder,
        eval_transform_builder,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        seed=config.data.seed,
    )

    backbone, load_report = build_model(
        config.architecture.name,
        weights_path=(
            str(config.architecture.weights_path)
            if config.architecture.weights_path is not None
            else None
        ),
        unfreeze_from=config.architecture.unfreeze_from,
        device=config.run.device,
    )
    print(
        f"[{config.node_name}] backbone {config.architecture.name}: "
        f"matched={load_report.matched} missing={len(load_report.missing)} "
        f"unexpected={len(load_report.unexpected)}"
    )
    head_cls = get_head_strategy(config.head.name)
    head = head_cls(**config.head.hparams)
    model = nn.Sequential(backbone, head.build()).to(config.run.device)

    loss_spec = build_loss(config.loss.name, device=config.run.device, **config.loss.hparams)

    return NodeAssembly(
        model=model,
        loaders=loaders,
        loss_spec=loss_spec,
        eval_transform_builder=eval_transform_builder,
        n_train=len(split.train_df()),
        n_val=len(split.val_df()),
        n_test=len(split.test_df()),
    )
