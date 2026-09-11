"""Cliente Flower de un nodo: `python -m src.federated.client --config <node.yaml>`.

Un proceso por nodo, cada uno con su YAML (`FederatedNodeConfig`). El nodo
entrena `local_epochs` épocas locales por ronda con las funciones puras de
`train/loop.py` — NO con `Trainer.fit()`, a propósito: `Trainer` empaqueta
checkpointing por época, early stopping y "mejor época local", todas
decisiones de granularidad época que acá pertenecen al SERVIDOR a
granularidad ronda (el invariante "evaluar el mejor, nunca el último" se
cumple con el tracker de mejor ronda global de `round_tracking.py`; un
"mejor checkpoint local" por nodo sería el invariante equivocado).

Artefactos del nodo — cada proceso escribe SOLO su carpeta
(`runs/<exp>/nodes/<node_name>/`), con la misma forma que un `run_dir`
centralizado:
- `config.yaml`: snapshot del YAML del nodo.
- `metrics.csv` (+ TensorBoard): UNA fila por época local, claves
  `round` + `train_*` + `val_*` + `duration_seconds` — keyset idéntico en
  todas las filas (contrato de `MetricsLogger.log()`); `train_prox_loss`
  existe en TODA la corrida o en ninguna (la estrategia es fija por
  experimento).
- `rounds.csv`: UNA fila por ronda con las métricas del MODELO AGREGADO
  sobre el val local (lo que `evaluate()` le reporta al servidor). Aparte
  de `metrics.csv` porque mezclar filas por-época y por-ronda rompería el
  keyset estable de `MetricsLogger`.
- `train.log`: espejo en texto, append (misma convención que `Trainer`).
- `weights/round<r>_head.pt`: SOLO con `aggregation_scope: backbone` — la
  cabeza local tras el fit de la ronda `r`, el registro de personalización
  que `evaluate_node.py` recompone con el mejor backbone global.
- `plots/`, `val/`, `test/`: los escribe `evaluate_node.run_final_evaluation()`
  al terminar la corrida (ver ese módulo).

Todos los puntos de contacto con la API de flwr del lado nodo viven en
este archivo (ver el docstring del paquete).
"""

import argparse
import csv
import time
from pathlib import Path

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..seed import set_global_seed
from ..tracking import MetricsLogger
from ..train.build import build_optimizer, build_param_groups, build_scheduler
from ..train.loop import evaluate as evaluate_loader
from ..train.loop import train_one_epoch
from .assembly import NodeAssembly, build_node_assembly
from .config import FederatedNodeConfig, load_node_config, save_federated_config
from .evaluate_node import run_final_evaluation
from .handshake import model_config_hash
from .param_utils import get_model_ndarrays, set_model_ndarrays

# Debe cubrir el modelo serializado por mensaje (~100 MB un ResNet50 FP32
# en alcance full). Mismo valor que el default de
# FederationConfig.grpc_max_message_length -- si se sube en el server.yaml
# más allá de esto, subirlo acá también (no está en el YAML del nodo a
# propósito: no hay caso de uso conocido para que difiera por nodo).
_GRPC_MAX_MESSAGE_LENGTH = 536_870_912

# Esquema fijo de rounds.csv -- csv.DictWriter con fieldnames estables,
# mismas 8 métricas que evaluate() de train/loop.py.
_ROUNDS_CSV_FIELDS = [
    "round", "num_examples", "loss", "accuracy", "auc",
    "sensitivity", "specificity", "f1", "f1_macro", "precision",
]


class FedMammoBenchClient(fl.client.NumPyClient):
    """NumPyClient de un nodo FedMammoBench.

    Construye el ensamblado UNA vez (`assembly.build_node_assembly`) y lo
    reusa todas las rondas; lo que se reconstruye fresco CADA ronda es el
    optimizador (y scheduler, si hay): tras el reemplazo server-side de los
    pesos, los momentos de Adam de la ronda anterior apuntan a un paisaje
    que ya no existe — arrastrarlos optimizaría contra estado obsoleto
    (práctica estándar en Flower; también la del legacy).

    El logger de métricas queda abierto toda la sesión; lo cierra `main()`
    en su `finally`, no esta clase — mismo contrato de propiedad que
    `Trainer` con logger inyectado.
    """

    def __init__(self, config: FederatedNodeConfig) -> None:
        self.config = config
        self.assembly: NodeAssembly = build_node_assembly(config)
        self.model_hash = model_config_hash(
            config.experiment_id, config.architecture, config.head, config.aggregation_scope
        )

        run_dir = Path(config.run.run_dir)
        save_federated_config(config, run_dir / "config.yaml")
        self.logger = MetricsLogger(
            run_dir,
            wandb_project=config.run.wandb_project,
            wandb_run_name=f"{config.experiment_id}-{config.node_name}",
            wandb_group=config.run.wandb_group,
            config=config.model_dump(mode="json"),
        )
        self.train_log_path = run_dir / "train.log"
        self.rounds_csv_path = run_dir / "rounds.csv"
        # Perezoso como los writers de MetricsLogger: se trunca ("w") en la
        # primera ronda de ESTA corrida, no en __init__, y nunca si el
        # proceso muere antes de evaluar.
        self._rounds_csv_started = False
        # Contador de época GLOBAL (continuo entre rondas): el eje X de
        # metrics.csv/TensorBoard. round r con local_epochs=e produce las
        # épocas (r-1)*e+1 .. r*e.
        self.global_epoch = 0

    # ------------------------------------------------------------------ #

    def _check_hash(self, config: dict[str, Scalar], phase: str) -> None:
        """Handshake: aborta si el hash del servidor no es el de este YAML."""
        server_hash = config.get("model_hash")
        if server_hash != self.model_hash:
            raise RuntimeError(
                f"[{self.config.node_name}] Handshake fallido en {phase}: "
                f"hash del servidor={server_hash!r}, hash de este nodo="
                f"{self.model_hash!r}. El server.yaml y este node_*.yaml "
                "difieren en experiment_id/architecture/head/"
                "aggregation_scope -- corrige los YAML antes de relanzar."
            )

    def _log_line(self, text: str) -> None:
        """Espejo en texto de una línea de progreso (train.log, append)."""
        self.train_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.train_log_path, "a") as f:
            f.write(text + "\n")
        print(f"[{self.config.node_name}] {text}", flush=True)

    # ------------------------------------------------------------------ #

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Una ronda de entrenamiento local sobre los parámetros agregados.

        Orden: handshake -> cargar parámetros (scope-aware) -> optimizer/
        scheduler frescos -> regularizador FedProx si el servidor mandó
        `proximal_mu` -> `local_epochs` x (train_one_epoch + evaluate) con
        una fila de metrics.csv por época local -> (scope backbone) guardar
        la cabeza de la ronda -> devolver parámetros + num_examples +
        métricas de la última época local (prefijo `train_`).
        """
        self._check_hash(config, "fit")
        current_round = int(config["current_round"])
        local_epochs = int(config.get("local_epochs", 1))
        scope = self.config.aggregation_scope
        model = self.assembly.model

        set_model_ndarrays(model, parameters, scope)

        params, optimizer_hparams = build_param_groups(model, self.config.optimizer.hparams)
        optimizer = build_optimizer(params, self.config.optimizer.name, **optimizer_hparams)
        scheduler = (
            build_scheduler(optimizer, self.config.scheduler.name, **self.config.scheduler.hparams)
            if self.config.scheduler is not None
            else None
        )

        # FedProx: la clase stock del servidor inyecta proximal_mu en el
        # config de fit. El término penal (mu/2)·||w - w_global||² entra a
        # train_one_epoch como regularizer (LossSpec no ve el modelo, ver
        # su docstring). Solo parámetros entrenables: los congelados no se
        # mueven y aportarían exactamente 0 (los buffers de BN quedan fuera
        # -- asimetría heredada de la definición de FedProx sobre
        # parámetros optimizados, documentada en FEDERATED_DESIGN.md). Y
        # nada de autocast/FP16 en este pipeline: el underflow del término
        # penal que invalidó las comparaciones del legacy (auditoría N7) no
        # puede reproducirse acá.
        proximal_mu = float(config.get("proximal_mu", 0.0))
        regularizer = None
        if proximal_mu > 0.0:
            trainable = [p for p in model.parameters() if p.requires_grad]
            global_vec = parameters_to_vector(trainable).detach().clone()

            def regularizer() -> torch.Tensor:
                return (proximal_mu / 2.0) * torch.sum(
                    (parameters_to_vector(trainable) - global_vec) ** 2
                )

        fit_start = time.perf_counter()
        train_metrics: dict[str, float] = {}
        for local_epoch in range(1, local_epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                self.assembly.loaders["train"],
                optimizer,
                self.assembly.loss_spec,
                self.config.run.device,
                freeze_bn_stats=self.config.run.freeze_bn_stats,
                regularizer=regularizer,
            )
            val_metrics = evaluate_loader(
                model,
                self.assembly.loaders["val"],
                self.assembly.loss_spec,
                self.config.run.device,
            )
            if scheduler is not None:
                # Mismo contrato que Trainer.fit(): ReduceLROnPlateau
                # necesita un valor -- acá la val loss local (el nodo no
                # tiene metric_name propio; la métrica "de verdad" es del
                # servidor).
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            self.global_epoch += 1
            duration = time.perf_counter() - epoch_start
            row = {
                "round": float(current_round),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "duration_seconds": duration,
            }
            self.logger.log(self.global_epoch, row)
            self._log_line(
                f"ronda {current_round} época local {local_epoch}/{local_epochs} "
                f"(global {self.global_epoch}): "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} ({duration:.1f}s)"
            )

        if scope == "backbone":
            # El registro de personalización de la ronda: la cabeza que
            # entrenó CONTRA el backbone agregado de esta ronda.
            # evaluate_node.py recompone mejor-backbone-global + esta
            # cabeza (la de la mejor ronda) para la evaluación final.
            head_path = Path(self.config.run.checkpoint_dir) / f"round{current_round}_head.pt"
            head_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model[1].state_dict(), head_path)

        fit_seconds = time.perf_counter() - fit_start
        report: dict[str, Scalar] = {
            "node_name": self.config.node_name,
            "fit_seconds": fit_seconds,
            **{f"train_{k}": float(v) for k, v in train_metrics.items()},
        }
        return get_model_ndarrays(model, scope), self.assembly.n_train, report

    # ------------------------------------------------------------------ #

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evalúa el modelo AGREGADO de esta ronda sobre el val local.

        Es la mitad nodo de la "evaluación federada pura": el servidor
        promedia estos reportes ponderando por `num_examples` (el tamaño
        del val local). Además deja una fila en `rounds.csv` — el registro
        local de la calidad del modelo global ronda a ronda.
        """
        self._check_hash(config, "evaluate")
        current_round = int(config["current_round"])
        model = self.assembly.model

        set_model_ndarrays(model, parameters, self.config.aggregation_scope)
        metrics = evaluate_loader(
            model, self.assembly.loaders["val"], self.assembly.loss_spec, self.config.run.device
        )

        mode = "a" if self._rounds_csv_started else "w"
        with open(self.rounds_csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_ROUNDS_CSV_FIELDS)
            if not self._rounds_csv_started:
                writer.writeheader()
                self._rounds_csv_started = True
            writer.writerow(
                {"round": current_round, "num_examples": self.assembly.n_val, **metrics}
            )
        self._log_line(
            f"ronda {current_round} modelo agregado: "
            f"val_loss={metrics['loss']:.4f} val_auc={metrics['auc']:.4f}"
        )

        report: dict[str, Scalar] = {
            "node_name": self.config.node_name,
            **{f"val_{k}": float(v) for k, v in metrics.items()},
        }
        return float(metrics["loss"]), self.assembly.n_val, report


# ---------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Nodo federado FedMammoBench (cliente Flower sobre gRPC real)."
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Ruta al node_<nombre>.yaml de este nodo."
    )
    parser.add_argument(
        "--server-run-dir",
        type=str,
        default=None,
        help=(
            "Carpeta de artefactos del servidor (donde vive best.json). "
            "Default: <run_dir del nodo>/../../server -- la forma estándar "
            "runs/<exp>/{server,nodes/<n>}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del proceso nodo.

    Orden: config -> semilla (ANTES de construir nada, mismo orden que
    `cli.run()`) -> cliente (construye el ensamblado) -> `start_client()`
    (bloquea hasta que el servidor termina las rondas; con flwr 1.31
    RETORNA limpio en la desconexión final, verificado en la spike de
    Fase 0) -> evaluación final del mejor modelo global
    (`evaluate_node.run_final_evaluation`, escribe `val/`, `test/` y
    `plots/` del nodo) -> cerrar logger.
    """
    args = parse_args()
    config = load_node_config(args.config)
    set_global_seed(config.data.seed)

    server_run_dir = (
        Path(args.server_run_dir)
        if args.server_run_dir is not None
        else Path(config.run.run_dir).parent.parent / "server"
    )

    client = FedMammoBenchClient(config)
    try:
        fl.client.start_client(
            server_address=config.server_address,
            client=client.to_client(),
            grpc_max_message_length=_GRPC_MAX_MESSAGE_LENGTH,
            insecure=True,
            # Tolera que el servidor tarde en levantar (arranque de
            # contenedores): reintenta la conexión inicial en vez de morir.
            max_retries=10,
            max_wait_time=120.0,
        )
        # start_client retornó == el servidor terminó todas las rondas ==
        # best.json está final (se reescribe DURANTE las rondas, en cada
        # mejora). La evaluación final corre dentro del try: si falla,
        # queda el fallback manual `python -m src.federated.evaluate_node`.
        run_final_evaluation(config, server_run_dir, logger=client.logger)
    finally:
        client.logger.close()


if __name__ == "__main__":
    main()
