"""Servidor Flower: `python -m src.federated.server --config <server.yaml>`.

Un solo proceso, sin datos: construye un modelo PLANTILLA (arquitectura +
cabeza del YAML, ver `FederatedServerConfig`) únicamente para (1) generar
los parámetros iniciales de la ronda 1 y (2) materializar los checkpoints
globales en el mismo formato que los centralizados
(`checkpoint.save_checkpoint`). Nunca hace forward, nunca ve una imagen.

Todos los puntos de contacto con la API de flwr del lado servidor viven en
este archivo (ver el docstring del paquete).
"""

import argparse
from pathlib import Path

import torch.nn as nn
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, start_server

from ..eval_pipeline import METRIC_DISPLAY_NAMES
from ..models.build import build_model
from ..models.heads import get_head_strategy
from ..reporting import plot_loss_curve, plot_metric_curve
from ..seed import set_global_seed
from ..tracking import MetricsLogger
from .config import FederatedServerConfig, load_server_config, save_federated_config
from .handshake import model_config_hash
from .param_utils import get_model_ndarrays
from .round_tracking import TrackedStrategy
from .strategies import build_strategy, make_on_config

# Semilla de la inicialización de la cabeza del modelo plantilla -- el
# servidor no tiene sección `data` (no hay data.seed que reusar), así que
# se fija un valor fijo, documentado acá, en vez de agregar un campo de
# config sin ningún otro uso. Solo importa en aggregation_scope: full (ahí
# la cabeza inicial SÍ viaja a los nodos como parte de los parámetros
# iniciales); en backbone es irrelevante -- la cabeza nunca se agrega.
_TEMPLATE_SEED = 42



def _build_template_model(config: FederatedServerConfig) -> nn.Sequential:
    """Arquitectura + cabeza del server.yaml, ensamblados igual que en `cli.py`.

    Sin pesos de ningún checkpoint de ronda -- solo `weights_path` (si la
    arquitectura lo requiere, ej. RadImageNet) para que `build_model` no
    falle su chequeo `LoadReport.matched == 0`. El resultado es la forma
    correcta del modelo; los VALORES los define la ronda 1 (parámetros
    iniciales) y luego cada mejora (`best_round<N>.pt`).
    """
    set_global_seed(_TEMPLATE_SEED)
    backbone, load_report = build_model(
        config.architecture.name,
        weights_path=(
            str(config.architecture.weights_path)
            if config.architecture.weights_path is not None
            else None
        ),
        unfreeze_from=config.architecture.unfreeze_from,
        device=config.tracking.device,
    )
    print(
        f"[server] backbone plantilla {config.architecture.name}: "
        f"matched={load_report.matched} missing={len(load_report.missing)} "
        f"unexpected={len(load_report.unexpected)}"
    )
    head_cls = get_head_strategy(config.head.name)
    head = head_cls(**config.head.hparams)
    return nn.Sequential(backbone, head.build()).to(config.tracking.device)


def _plot_round_history(tracked: TrackedStrategy, run_dir: Path) -> None:
    """Curvas train-vs-val por ronda, mismo estilo que las de época del centralizado.

    Reusa `reporting.plot_loss_curve`/`plot_metric_curve` tal cual --
    `TrackedStrategy.round_history` ya trae una fila por ronda con las
    mismas claves `train_*`/`val_*` que `Trainer.history` trae por época,
    solo que el eje es "ronda" en vez de "época".
    """
    history = tracked.round_history
    if not history:
        print("[server] sin rondas completadas -- se omiten los plots.")
        return
    plots_dir = run_dir / "plots"
    if "train_loss" in history[0] and "val_loss" in history[0]:
        plot_loss_curve(
            [r["train_loss"] for r in history], [r["val_loss"] for r in history],
            plots_dir / "loss_curve.png", best_epoch=tracked.best_round,
        )
        tracked.logger.log_image("plots/loss_curve", plots_dir / "loss_curve.png")
    for key, display_name in METRIC_DISPLAY_NAMES.items():
        train_key, val_key = f"train_{key}", f"val_{key}"
        if train_key not in history[0] or val_key not in history[0]:
            continue
        path = plots_dir / f"{key}_curve.png"
        plot_metric_curve(
            [r[train_key] for r in history], [r[val_key] for r in history],
            path, metric_name=display_name, best_epoch=tracked.best_round,
        )
        tracked.logger.log_image(f"plots/{key}_curve", path)


def run_server(config: FederatedServerConfig) -> TrackedStrategy:
    """Arma la estrategia envuelta y corre `start_server()` hasta la última ronda.

    Args:
        config: `FederatedServerConfig` ya validado.

    Returns:
        TrackedStrategy: la estrategia (con `round_history`/`best_round`
        ya poblados) -- devuelta para que `main()` genere plots/resumen y
        para que un script de verificación pueda inspeccionarla sin pasar
        por `main()`.

    Example:
        >>> tracked = run_server(load_server_config("configs/federated/exp40_fedavg_full/server.yaml"))
        >>> tracked.best_round
        7
    """
    run_dir = Path(config.tracking.run_dir)
    checkpoint_dir = Path(config.tracking.checkpoint_dir)
    save_federated_config(config, run_dir / "config.yaml")

    template_model = _build_template_model(config)
    scope = config.federation.aggregation_scope
    model_hash = model_config_hash(config.experiment_id, config.architecture, config.head, scope)
    initial_parameters = ndarrays_to_parameters(get_model_ndarrays(template_model, scope))
    on_config = make_on_config(model_hash, config.federation.local_epochs, scope)

    inner_strategy = build_strategy(
        config.strategy.name,
        initial_parameters=initial_parameters,
        num_nodes=config.federation.num_nodes,
        on_config=on_config,
        accept_failures=config.federation.accept_failures,
        **config.strategy.hparams,
    )

    logger = MetricsLogger(
        run_dir,
        wandb_project=config.tracking.wandb_project,
        wandb_run_name=f"{config.experiment_id}-server",
        wandb_group=config.tracking.wandb_group,
        config=config.model_dump(mode="json"),
    )
    tracked = TrackedStrategy(
        inner_strategy,
        logger=logger,
        template_model=template_model,
        scope=scope,
        model_hash=model_hash,
        best_metric_name=config.tracking.best_metric_name,
        best_metric_mode=config.tracking.best_metric_mode,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        save_every_rounds=config.tracking.save_every_rounds,
    )

    try:
        start_server(
            server_address=config.federation.server_address,
            config=ServerConfig(
                num_rounds=config.federation.rounds,
                round_timeout=config.federation.round_timeout_seconds,
            ),
            strategy=tracked,
            grpc_max_message_length=config.federation.grpc_max_message_length,
        )
    finally:
        _plot_round_history(tracked, run_dir)
        if tracked.best_round is not None:
            print(
                f"[server] mejor ronda global: {tracked.best_round} "
                f"({config.tracking.best_metric_name}="
                f"{tracked.tracker.best_value:.4f}) -> {tracked.best_checkpoint_path}"
            )
            logger.log_summary({
                "best_round": tracked.best_round,
                f"best_{config.tracking.best_metric_name}": tracked.tracker.best_value,
            })
        else:
            print("[server] AVISO: ninguna ronda produjo un mejor modelo (0 rondas completadas?).")
        logger.close()

    return tracked


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Servidor federado FedMammoBench (agregación Flower sobre gRPC real)."
    )
    parser.add_argument("--config", required=True, type=str, help="Ruta al server.yaml.")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del proceso servidor."""
    args = parse_args()
    config = load_server_config(args.config)
    run_server(config)


if __name__ == "__main__":
    main()
