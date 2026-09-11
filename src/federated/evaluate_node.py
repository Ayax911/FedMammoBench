"""Evaluación final de un nodo contra el MEJOR modelo global de la corrida.

Produce, dentro de `runs/<exp>/nodes/<node_name>/`, exactamente los
artefactos finales de un `run_dir` centralizado: `plots/` (curvas de
entrenamiento reconstruidas desde `metrics.csv`) y `val/` + `test/`
(`metrics.json`, `confusion_matrix_metrics.json`, `predictions.csv`,
`confusion_matrix.png`, `roc_curve.png`) — vía el MISMO
`eval_pipeline.evaluate_split()` que usan `cli.py` y `evaluate.py`, sin
tocarlo.

Qué checkpoint se evalúa — según `aggregation_scope`:
- `full`: el `best_round<N>.pt` que apunta `best.json` del servidor, tal
  cual (formato idéntico al centralizado, `evaluate_split` lo carga
  directo).
- `backbone`: el mejor backbone global (`best_round<N>_backbone.pt`) MÁS
  la cabeza local que este nodo entrenó EN ESA MISMA ronda
  (`weights/round<N>_head.pt`, guardada por `client.fit()`): la pareja que
  realmente se midió cuando la ronda N resultó ser la mejor. La
  composición se persiste como `weights/final_round<N>.pt` — un checkpoint
  por-nodo auditable, en formato centralizado.

Dos formas de invocarlo:
- En proceso, desde `client.main()` justo después de que `start_client()`
  retorna (el servidor terminó -> `best.json` está final, sin carrera: se
  reescribe DURANTE las rondas en cada mejora). Reusa el ensamblado y el
  logger ya vivos del cliente.
- Standalone, el fallback de recuperación (nodo que murió a mitad de
  corrida, o re-generar `val/`/`test/` después):

      python -m src.federated.evaluate_node \\
          --config configs/federated/exp40_fedavg_full/node_cmmd.yaml \\
          --server-run-dir runs/exp40_fedavg_full/server

Nada acá reentrena, y nada toca `metrics.csv`/`rounds.csv`/`config.yaml`
del nodo: `MetricsLogger` abre sus writers perezosamente en el primer
`log()` — que nunca ocurre acá — exactamente la invariante por la que esa
pereza existe (ver CLAUDE.md).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..checkpoint import save_checkpoint
from ..eval_pipeline import METRIC_DISPLAY_NAMES, evaluate_split
from ..reporting import plot_loss_curve, plot_metric_curve
from ..seed import set_global_seed
from ..tracking import MetricsLogger
from .assembly import NodeAssembly, build_node_assembly
from .config import FederatedNodeConfig, load_node_config
from .handshake import model_config_hash

# Espera acotada por best.json: cubre el caso patológico de un servidor que
# murió antes de la primera mejora (best.json nunca escrito) sin colgar al
# nodo para siempre -- pasado el plazo, FileNotFoundError con instrucciones.
_BEST_JSON_TIMEOUT_SECONDS = 60.0
_BEST_JSON_POLL_SECONDS = 2.0


def _wait_for_best_json(server_run_dir: Path) -> dict[str, Any]:
    """Lee `best.json` del servidor, esperando hasta el timeout si no está."""
    best_path = server_run_dir / "best.json"
    deadline = time.monotonic() + _BEST_JSON_TIMEOUT_SECONDS
    while not best_path.is_file():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                f"best.json no apareció en {best_path} tras "
                f"{_BEST_JSON_TIMEOUT_SECONDS:.0f}s. ¿El servidor murió antes "
                "de la primera mejora? Revisa runs/<exp>/server/server.log; "
                "cuando exista un best.json, re-lanza la evaluación con "
                "`python -m src.federated.evaluate_node --config <node.yaml> "
                "--server-run-dir <server_run_dir>`."
            )
        time.sleep(_BEST_JSON_POLL_SECONDS)
    return json.loads(best_path.read_text())


def _plots_from_metrics_csv(run_dir: Path, logger: MetricsLogger) -> None:
    """Reconstruye `plots/` del nodo desde su `metrics.csv`.

    Desde el CSV y no desde un `history` en memoria a propósito: así el
    fallback standalone (proceso nuevo, sin historia) produce EXACTAMENTE
    los mismos plots que la ruta en-proceso. `best_epoch=None` en todas las
    curvas: la "mejor época" es una decisión de RONDA del servidor, marcar
    una época local acá sería inventar una señal que no existe.
    """
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.is_file():
        print(f"AVISO: {metrics_csv} no existe -- se omiten los plots del nodo.")
        return
    df = pd.read_csv(metrics_csv)

    loss_path = run_dir / "plots" / "loss_curve.png"
    plot_loss_curve(list(df["train_loss"]), list(df["val_loss"]), loss_path)
    logger.log_image("plots/loss_curve", loss_path)

    for metric_key, display_name in METRIC_DISPLAY_NAMES.items():
        train_col, val_col = f"train_{metric_key}", f"val_{metric_key}"
        if train_col not in df.columns or val_col not in df.columns:
            continue
        path = run_dir / "plots" / f"{metric_key}_curve.png"
        plot_metric_curve(list(df[train_col]), list(df[val_col]), path, metric_name=display_name)
        logger.log_image(f"plots/{metric_key}_curve", path)


def run_final_evaluation(
    config: FederatedNodeConfig,
    server_run_dir: Path,
    logger: MetricsLogger | None = None,
    assembly: NodeAssembly | None = None,
) -> None:
    """Evalúa el mejor modelo global sobre val y test locales del nodo.

    Args:
        config: YAML del nodo ya validado.
        server_run_dir: carpeta de artefactos del servidor (contiene
            `best.json` y `weights/`).
        logger: `MetricsLogger` ya abierto para reusar (la ruta en-proceso
            desde `client.main()` — misma corrida de W&B que el
            entrenamiento). `None` (standalone) abre uno propio con
            run-name `<exp>-<nodo>-final-eval` y lo cierra al terminar.
        assembly: ensamblado ya construido para reusar (ruta en-proceso —
            evita un SEGUNDO ResNet50 en la GPU mientras el del cliente
            sigue vivo: con 4 nodos contra una sola GPU eso puede ser la
            diferencia entre caber y OOM). `None` (standalone) lo
            construye.

    Raises:
        FileNotFoundError: `best.json` ausente tras el timeout, o el
            checkpoint/cabeza que referencia no existe.
        RuntimeError: `model_hash` de `best.json` distinto del de este
            nodo — el checkpoint global no corresponde a este YAML.

    Example:
        >>> run_final_evaluation(cfg, Path("runs/exp40_fedavg_full/server"))
    """
    best = _wait_for_best_json(Path(server_run_dir))

    node_hash = model_config_hash(
        config.experiment_id, config.architecture, config.head, config.aggregation_scope
    )
    if best.get("model_hash") != node_hash:
        raise RuntimeError(
            f"[{config.node_name}] best.json trae model_hash="
            f"{best.get('model_hash')!r} pero este nodo calcula {node_hash!r}: "
            "el checkpoint global no corresponde a este YAML (¿experimento "
            "equivocado, o YAML editado después de la corrida?)."
        )

    if assembly is None:
        assembly = build_node_assembly(config)
    model = assembly.model
    best_round = int(best["best_round"])
    run_dir = Path(config.run.run_dir)

    if config.aggregation_scope == "full":
        # Formato centralizado completo: evaluate_split lo carga tal cual.
        checkpoint_path = Path(best["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint global no encontrado: {checkpoint_path}")
    else:
        # backbone: componer mejor-backbone-global + cabeza local de ESA ronda.
        backbone_path = Path(best["checkpoint"])
        head_path = Path(config.run.checkpoint_dir) / f"round{best_round}_head.pt"
        for p, what in ((backbone_path, "backbone global"), (head_path, "cabeza local")):
            if not p.is_file():
                raise FileNotFoundError(f"{what} de la ronda {best_round} no encontrado: {p}")
        backbone_ckpt = torch.load(backbone_path, map_location=config.run.device)
        model[0].load_state_dict(backbone_ckpt["model_state_dict"])
        model[1].load_state_dict(torch.load(head_path, map_location=config.run.device))
        checkpoint_path = Path(config.run.checkpoint_dir) / f"final_round{best_round}.pt"
        save_checkpoint(
            model,
            checkpoint_path,
            epoch=best_round,
            metric_value=float(best["metric_value"]),
        )

    own_logger = logger is None
    if logger is None:
        logger = MetricsLogger(
            run_dir,
            wandb_project=config.run.wandb_project,
            wandb_run_name=f"{config.experiment_id}-{config.node_name}-final-eval",
            wandb_group=config.run.wandb_group,
            config=config.model_dump(mode="json"),
        )
    try:
        _plots_from_metrics_csv(run_dir, logger)
        for split_name, loader in (("val", assembly.loaders["val"]), ("test", assembly.loaders["test"])):
            split_metrics = evaluate_split(
                split_name,
                model,
                checkpoint_path,
                loader,
                assembly.loss_spec,
                config.run.device,
                run_dir,
                logger,
            )
            print(
                f"[{config.node_name}] {split_name} vs mejor modelo global "
                f"(ronda {best_round}): auc={split_metrics['auc']:.4f} "
                f"f1_macro={split_metrics['f1_macro']:.4f}"
            )
    finally:
        if own_logger:
            logger.close()


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación final de un nodo federado contra el mejor modelo "
            "global (fallback standalone de la que client.py corre solo)."
        )
    )
    parser.add_argument("--config", required=True, type=str, help="node_<nombre>.yaml del nodo.")
    parser.add_argument(
        "--server-run-dir",
        type=str,
        default=None,
        help="Carpeta con best.json. Default: <run_dir del nodo>/../../server.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada standalone (mismo patrón que src.evaluate)."""
    args = parse_args()
    config = load_node_config(args.config)
    set_global_seed(config.data.seed)
    server_run_dir = (
        Path(args.server_run_dir)
        if args.server_run_dir is not None
        else Path(config.run.run_dir).parent.parent / "server"
    )
    run_final_evaluation(config, server_run_dir)


if __name__ == "__main__":
    main()
