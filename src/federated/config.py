"""Configuración federada: modelos Pydantic v2 + carga/guardado en YAML.

Dos formas de YAML, una por rol:

- `server.yaml` -> `FederatedServerConfig`: rondas, estrategia, alcance de
  agregación y tracking del mejor modelo global. SIN sección `data`, SIN
  `loss`/`optimizer`/`scheduler` — el servidor nunca ve imágenes ni computa
  una pérdida; solo necesita `architecture` + `head` para construir el
  modelo plantilla del que salen los parámetros iniciales y en el que se
  materializan los checkpoints globales.
- `node_<nombre>.yaml` -> `FederatedNodeConfig`: el equivalente por-nodo de
  `ExperimentConfig`, con el manifest LOCAL del nodo (uno de
  `manifests/by_database/`) y una variante recortada de `TrainConfig`
  (`NodeRunConfig`): sin `epochs`/`patience`/`metric_name` porque las
  rondas, la parada y la elección del mejor modelo son decisiones del
  SERVIDOR, no del nodo.

Mismas reglas que `src/config.py` (ver su docstring): sin herencia entre
YAMLs y `extra="forbid"` en todo. Consecuencia deliberada: las secciones
`architecture`/`head` van DUPLICADAS textualmente en el server.yaml y en
cada node_*.yaml — cada archivo se lee de punta a punta. La deriva entre
copias no se vigila con comentarios (el mecanismo del legacy, que falló:
"DEBE coincidir con server.yaml" no detiene nada) sino con el handshake por
hash de `handshake.model_config_hash()`, verificado en runtime en cada
ronda.

Ejemplo de uso:
    >>> from src.federated.config import load_server_config, load_node_config
    >>> server_cfg = load_server_config("configs/federated/exp40/server.yaml")
    >>> node_cfg = load_node_config("configs/federated/exp40/node_cmmd.yaml")
"""

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from ..config import (
    ArchitectureConfig,
    DataConfig,
    NamedComponentConfig,
)

# Alcance de agregación (federation.aggregation_scope / aggregation_scope):
#   "full"     -> viaja y se agrega el state_dict COMPLETO de
#                 nn.Sequential(backbone, head) — FedAvg estándar, comparable
#                 con las corridas centralizadas y con el exp70 legacy.
#   "backbone" -> solo model[0] (el backbone) viaja y se agrega; cada nodo
#                 conserva una cabeza personalizada local. Es la intención
#                 detrás del split best_epochN_backbone.pt/_head.pt que los
#                 checkpoints centralizados ya guardan (ver
#                 Trainer._split_state_dicts()).
AggregationScope = Literal["full", "backbone"]


class FederationConfig(BaseModel):
    """Parámetros de la federación en sí: transporte, rondas y participación.

    Attributes:
        server_address: `host:puerto` donde el servidor escucha gRPC. Los
            nodos apuntan acá desde su propio YAML (`server_address` del
            nodo). Default `"127.0.0.1:8080"` — todo corre en la
            workstation; con `network_mode: host` en Docker la MISMA
            dirección sirve dentro y fuera de contenedores.
        rounds: número FIJO de rondas de agregación. No hay early stopping
            federado a propósito: el legacy lo implementaba levantando una
            excepción desde dentro de la estrategia y des-envolviéndola
            distinto por transporte (una verruga documentada en su
            auditoría). Acá el invariante del repo "evaluar el MEJOR
            checkpoint, nunca el último" se cumple con el tracker de mejor
            ronda del servidor (`round_tracking.py`) + rondas fijas.
        local_epochs: épocas locales que cada nodo entrena por ronda. Lo
            fija el SERVIDOR y viaja a los nodos en el config de cada
            ronda (`on_fit_config_fn`) — el YAML del nodo no tiene campo
            `epochs` que pueda contradecirlo.
        num_nodes: cuántos nodos participan. Se usa como
            `min_fit_clients = min_evaluate_clients = min_available_clients`
            en la estrategia: con 4 bases de datos como 4 nodos, una ronda
            sin TODOS los nodos no es el experimento diseñado.
        aggregation_scope: ver `AggregationScope` arriba. Duplicado en cada
            node_*.yaml y cubierto por el hash del handshake — un nodo con
            el alcance equivocado enviaría una lista de ndarrays de largo
            distinto y corrompería la agregación si no se detectara antes.
        accept_failures: si `False` (default), un nodo caído o con config
            incoherente aborta la corrida en vez de seguir agregando en
            silencio con menos nodos. `True` solo tiene sentido para
            depurar.
        round_timeout_seconds: timeout por ronda que Flower pasa a
            `ServerConfig`. `None` (default) = sin timeout — una ronda de
            fit sobre cmmd (4722 imágenes) a varias épocas locales puede
            tardar varios minutos legítimamente.
        grpc_max_message_length: tamaño máximo de mensaje gRPC en bytes.
            Default 512 MiB, heredado del legacy: un ResNet50 completo en
            FP32 son ~100 MB por dirección y por ronda, y el default de
            gRPC (4 MB) lo rechaza.

    Example:
        >>> fed = FederationConfig(rounds=30, local_epochs=2, num_nodes=4)
    """

    model_config = ConfigDict(extra="forbid")

    server_address: str = "127.0.0.1:8080"
    rounds: int
    local_epochs: int = 1
    num_nodes: int
    aggregation_scope: AggregationScope = "full"
    accept_failures: bool = False
    round_timeout_seconds: float | None = None
    grpc_max_message_length: int = 536_870_912  # 512 MiB


class ServerTrackingConfig(BaseModel):
    """Tracking del mejor modelo global y artefactos del servidor.

    Attributes:
        best_metric_name: métrica (clave de `build_metric_collection()` o
            `"loss"`) sobre el PROMEDIO PONDERADO de la evaluación federada
            de cada ronda, usada para decidir si la ronda produjo un nuevo
            mejor modelo global. Default `"auc"`. OJO: el promedio ponderado
            de AUCs por nodo NO es el AUC del pool combinado de
            predicciones — ver docs/FEDERATED_DESIGN.md.
        best_metric_mode: `"max"` o `"min"`, igual que
            `TrainConfig.metric_mode`.
        save_every_rounds: si se fija, guarda un checkpoint global periódico
            cada N rondas además del mejor. `None` (default) no guarda
            periódicos — a ~100 MB por checkpoint de alcance `full`, uno
            por ronda serían gigabytes sin valor (y `*.pt` está gitignoreado
            de todos modos).
        run_dir: carpeta de artefactos del servidor
            (`runs/<exp>/server/`): `config.yaml`, `metrics.csv` (una fila
            por RONDA — la columna `epoch` de `MetricsLogger` lleva el
            número de ronda), `server.log`, `best.json`, TensorBoard,
            `plots/`.
        checkpoint_dir: carpeta de pesos globales
            (`runs/<exp>/server/weights/`).
        device: dispositivo donde vive el modelo plantilla del servidor.
            `"cpu"` (default) alcanza: el servidor nunca hace forward, el
            modelo existe solo para generar parámetros iniciales y para
            serializar checkpoints.
        wandb_project: proyecto W&B opcional (`None` desactiva W&B), igual
            que `TrainConfig.wandb_project`.
        wandb_group: grupo W&B. La convención federada es
            `wandb_group = experiment_id` en el servidor Y en los nodos,
            para que las 1+N corridas del experimento queden bajo una sola
            fila expandible en la UI.

    Example:
        >>> tr = ServerTrackingConfig(
        ...     run_dir=Path("runs/exp40/server"),
        ...     checkpoint_dir=Path("runs/exp40/server/weights"),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    best_metric_name: str = "auc"
    best_metric_mode: str = "max"
    save_every_rounds: int | None = None
    run_dir: Path
    checkpoint_dir: Path
    device: str = "cpu"
    wandb_project: str | None = None
    wandb_group: str | None = None


class FederatedServerConfig(BaseModel):
    """Modelo contenedor del `server.yaml` completo.

    Deliberadamente SIN `data`, SIN `loss`, SIN `optimizer`/`scheduler`: el
    servidor no ve imágenes (requisito de diseño) y no computa ninguna
    pérdida — los nodos reportan las suyas y el servidor solo promedia.
    `architecture` + `head` están porque el servidor construye un modelo
    plantilla para (1) los parámetros iniciales de la ronda 1 y (2)
    materializar los checkpoints globales en el mismo formato que los
    centralizados (`checkpoint.save_checkpoint`).

    Attributes:
        experiment_id: identificador del experimento. Debe coincidir con el
            de cada node_*.yaml — entra al hash del handshake.
        federation: transporte, rondas y participación (`FederationConfig`).
        strategy: estrategia de agregación por nombre + hparams
            (`NamedComponentConfig`, mismo patrón que optimizer/loss).
            Nombres registrados en `strategies._STRATEGIES`: `fedavg`,
            `fedprox`, `fedadam`, `fedyogi`. Los `hparams` se splatean al
            constructor stock de flwr (`proximal_mu`, `eta`, `beta_1`, ...).
        architecture: backbone — DUPLICADO en cada node_*.yaml, cubierto por
            el handshake. `weights_path` puede diferir entre archivos (rutas
            locales); NO entra al hash porque los pesos que importan llegan
            del servidor en la ronda 1.
        head: cabeza — DUPLICADA en cada node_*.yaml, cubierta por el
            handshake (sus `hparams` definen la forma del modelo:
            `num_classes` decide 1 vs 2 logits).
        tracking: mejor-modelo y artefactos (`ServerTrackingConfig`).

    Example:
        >>> cfg = load_server_config("configs/federated/exp40/server.yaml")
        >>> cfg.federation.rounds
        30
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    federation: FederationConfig
    strategy: NamedComponentConfig
    architecture: ArchitectureConfig
    head: NamedComponentConfig
    tracking: ServerTrackingConfig


class NodeRunConfig(BaseModel):
    """Variante recortada de `TrainConfig` para un nodo federado.

    Lo que se quitó, y por qué (comparar con `TrainConfig`):
    - `epochs`: las épocas locales por ronda las fija el servidor
      (`FederationConfig.local_epochs`) y llegan por `on_fit_config_fn`.
    - `metric_name`/`metric_mode`/`patience`/`min_delta`: la elección del
      mejor modelo y la duración de la corrida son del servidor
      (rondas fijas + tracker de mejor ronda). Un "mejor checkpoint local"
      por nodo sería el invariante equivocado.
    - `save_every`: los checkpoints globales periódicos son del servidor
      (`ServerTrackingConfig.save_every_rounds`). El nodo solo persiste su
      cabeza por ronda cuando `aggregation_scope: backbone` (ver
      `client.py`).

    Attributes:
        device: dispositivo de cómputo del nodo (`"cpu"`, `"cuda"`).
        freeze_bn_stats: idéntico a `TrainConfig.freeze_bn_stats` — se pasa
            tal cual a `train_one_epoch()`.
        run_dir: carpeta de artefactos del nodo
            (`runs/<exp>/nodes/<node_name>/`), con la MISMA forma que un
            `run_dir` centralizado (requisito de diseño): `config.yaml`,
            `metrics.csv` (una fila por época local), `rounds.csv`,
            `train.log`, `plots/`, `val/`, `test/`.
        checkpoint_dir: carpeta de pesos del nodo
            (`runs/<exp>/nodes/<node_name>/weights/`).
        wandb_project: proyecto W&B opcional (`None` desactiva).
        wandb_group: grupo W&B — convención: `experiment_id`, ver
            `ServerTrackingConfig.wandb_group`.

    Example:
        >>> run = NodeRunConfig(
        ...     device="cuda",
        ...     run_dir=Path("runs/exp40/nodes/cmmd"),
        ...     checkpoint_dir=Path("runs/exp40/nodes/cmmd/weights"),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    device: str = "cpu"
    freeze_bn_stats: bool = True
    run_dir: Path
    checkpoint_dir: Path
    wandb_project: str | None = None
    wandb_group: str | None = None


class FederatedNodeConfig(BaseModel):
    """Modelo contenedor de un `node_<nombre>.yaml` completo.

    Attributes:
        experiment_id: debe coincidir con el del server.yaml — entra al hash
            del handshake, así un nodo lanzado con el YAML de OTRO
            experimento falla en la ronda 1 en vez de contaminar la
            agregación.
        node_name: nombre del nodo (`"cmmd"`, `"kau-bcmd"`, ...). Etiqueta
            para métricas, W&B (`<experiment_id>-<node_name>`) y la columna
            `node_name` que el nodo reporta al servidor.
        server_address: `host:puerto` del servidor gRPC. Copia del
            `federation.server_address` del server.yaml.
        aggregation_scope: duplicado del server.yaml, cubierto por el hash —
            ver `AggregationScope`.
        architecture: backbone — duplicado del server.yaml (hash). El
            `weights_path` local puede diferir; los pesos efectivos llegan
            del servidor.
        head: cabeza — duplicada del server.yaml (hash).
        optimizer: optimizador LOCAL del nodo (se reconstruye fresco cada
            ronda, ver `client.py`). `backbone_lr` en `hparams` se respeta
            vía `train.build.build_param_groups()` — el mismo mecanismo de
            LR discriminativo de `cli.py`.
        scheduler: scheduler local opcional. OJO: se reconstruiría CADA
            ronda, así que un schedule tipo `cosine` con `T_max` pensado
            para toda la corrida solo abarcaría `local_epochs` épocas. Los
            configs de ejemplo usan `null`; si se fija, que sea consciente
            de esa semántica.
        loss: pérdida local del nodo. A propósito NO entra al hash del
            handshake: `pos_weight`/`weight` se calculan sobre el train
            split de CADA nodo (los desbalances difieren radicalmente:
            kau-bcmd ~4% maligno vs cmmd ~49%) y son legítimamente
            distintos entre nodos. Lo que sí define la forma del modelo
            (1 vs 2 logits) ya está cubierto por `head.hparams.num_classes`.
        data: `DataConfig` reusado tal cual, apuntando al manifest LOCAL del
            nodo (uno de `manifests/by_database/`). Restricciones extra
            validadas acá abajo.
        run: `NodeRunConfig` (variante recortada de `TrainConfig`).

    Example:
        >>> cfg = load_node_config("configs/federated/exp40/node_cmmd.yaml")
        >>> cfg.node_name
        'cmmd'
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    node_name: str
    server_address: str
    aggregation_scope: AggregationScope
    architecture: ArchitectureConfig
    head: NamedComponentConfig
    optimizer: NamedComponentConfig
    scheduler: NamedComponentConfig | None = None
    loss: NamedComponentConfig
    data: DataConfig
    run: NodeRunConfig

    @model_validator(mode="after")
    def _check_data_constraints(self) -> "FederatedNodeConfig":
        """Restricciones de `data` que solo aplican en federado.

        - `by_database_manifests` es una funcionalidad centralizada (el
          desglose de test por base de datos de `eval_pipeline`). En
          federado el desglose ES la federación misma — cada nodo ya es una
          base de datos — así que fijarlo en un nodo es señal de un YAML
          centralizado pegado a medias: error, no warning.
        - `num_workers > 0` es un warning (no error): los workers del
          DataLoader hacen fork() con el canal gRPC ya abierto, y
          grpc-python no lo soporta de forma confiable (dolor documentado
          del legacy, que forzaba 0). Con los TIFF 224x224 preprocesados la
          carga es barata y 0 cuesta poco. Se deja correr porque puede
          haber contextos (depuración sin servidor) donde sí funcione.
        """
        if self.data.by_database_manifests is not None:
            raise ValueError(
                "data.by_database_manifests no aplica en un nodo federado: "
                "el desglose por base de datos es la federación misma "
                "(un manifest local por nodo). Quita esa clave del YAML "
                f"del nodo '{self.node_name}'."
            )
        if self.data.num_workers > 0:
            warnings.warn(
                f"Nodo '{self.node_name}': data.num_workers="
                f"{self.data.num_workers} > 0 con gRPC abierto puede colgar "
                "el DataLoader (fork tras crear el canal, limitación de "
                "grpc-python documentada en el legacy). Recomendado: 0.",
                stacklevel=2,
            )
        return self


def load_server_config(path: str | Path) -> FederatedServerConfig:
    """Carga y valida un `FederatedServerConfig` desde YAML.

    Args:
        path: ruta al `server.yaml`.

    Returns:
        FederatedServerConfig validado.

    Raises:
        FileNotFoundError: si el archivo no existe.
        pydantic.ValidationError: campos faltantes, tipos incorrectos o
            claves no reconocidas (`extra="forbid"`).

    Example:
        >>> cfg = load_server_config("configs/federated/exp40/server.yaml")
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config de servidor no encontrado: {path}")
    raw = yaml.safe_load(path.read_text())
    return FederatedServerConfig.model_validate(raw)


def load_node_config(path: str | Path) -> FederatedNodeConfig:
    """Carga y valida un `FederatedNodeConfig` desde YAML.

    Args:
        path: ruta al `node_<nombre>.yaml`.

    Returns:
        FederatedNodeConfig validado.

    Raises:
        FileNotFoundError: si el archivo no existe.
        pydantic.ValidationError: campos faltantes, tipos incorrectos,
            claves no reconocidas, o `by_database_manifests` fijado (ver
            `FederatedNodeConfig._check_data_constraints`).

    Example:
        >>> cfg = load_node_config("configs/federated/exp40/node_cmmd.yaml")
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config de nodo no encontrado: {path}")
    raw = yaml.safe_load(path.read_text())
    return FederatedNodeConfig.model_validate(raw)


def save_federated_config(
    config: FederatedServerConfig | FederatedNodeConfig, path: str | Path
) -> None:
    """Guarda un config federado (servidor o nodo) como snapshot YAML.

    Mismo propósito que `src.config.save_config`: dejar en el `run_dir` una
    copia exacta de lo que corrió. Una sola función para ambos roles porque
    la serialización es idéntica (`model_dump(mode="json")`).

    Args:
        config: `FederatedServerConfig` o `FederatedNodeConfig`.
        path: destino del YAML. La carpeta contenedora se crea si no existe.

    Example:
        >>> save_federated_config(cfg, Path("runs/exp40/server/config.yaml"))
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
