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
import contextlib
import csv
import time
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from filelock import FileLock
from flwr.common import NDArrays, Scalar
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..seed import set_global_seed
from ..tracking import MetricsLogger
from ..train.build import build_optimizer, build_param_groups, build_scheduler
from ..train.loop import evaluate as evaluate_loader
from ..train.loop import train_n_updates, train_one_epoch
from .assembly import NodeAssembly, build_node_assembly
from .config import (
    FederatedNodeConfig,
    load_node_config,
    node_num_classes,
    save_federated_config,
)
from .evaluate_node import run_final_evaluation
from .handshake import model_config_hash
from .param_utils import get_model_ndarrays, scope_module, set_model_ndarrays

# Debe cubrir el modelo serializado por mensaje (~100 MB un ResNet50 FP32
# en alcance full). Mismo valor que el default de
# FederationConfig.grpc_max_message_length -- si se sube en el server.yaml
# más allá de esto, subirlo acá también (no está en el YAML del nodo a
# propósito: no hay caso de uso conocido para que difiera por nodo).
_GRPC_MAX_MESSAGE_LENGTH = 536_870_912

# Esquema fijo de rounds.csv para el caso binario -- csv.DictWriter con
# fieldnames estables, mismas métricas que evaluate() de train/loop.py. Con
# num_classes > 2 evaluate() solo devuelve "loss" (bypass, ver
# docs/FEDISIC2019_FLAMBY_REPRO.md §5.D) y se usa la lista reducida de abajo.
_ROUNDS_CSV_FIELDS = [
    "round", "num_examples", "loss", "accuracy", "auc",
    "sensitivity", "specificity", "f1", "f1_macro", "precision",
]
_ROUNDS_CSV_FIELDS_MULTICLASS = ["round", "num_examples", "loss"]


def _fmt_auc(metrics: dict[str, float]) -> str:
    """` val_auc=...` si la métrica existe (binario); cadena vacía si no (multiclase)."""
    auc = metrics.get("auc")
    return f" val_auc={auc:.4f}" if auc is not None else ""


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
        # Scaffold: control variate local c_i, PERSISTENTE entre rondas.
        # Se puebla (ceros) la primera vez que fit() ve scaffold_active=True.
        # Clave = nombre en el state_dict del alcance, solo parámetros entrenables.
        self._scaffold_c: dict[str, torch.Tensor] | None = None

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

    @contextlib.contextmanager
    def _gpu_section(self):
        """Serializa el cómputo GPU pesado entre nodos cuando
        `run.gpu_lock_path` está fijado (ver `NodeRunConfig.gpu_lock_path`):
        adquiere el lock, corre el bloque, y libera caché de CUDA al salir
        para no dejar memoria fragmentada retenida mientras este nodo espera
        su próximo turno (el allocator de PyTorch no la devuelve sola --
        confirmado necesario, ver docs/FIXES_FEDISIC2019_FLAMBY_REPRO.md).
        `nullcontext()` + sin `empty_cache()` si `gpu_lock_path` es `None` --
        cero cambio de comportamiento por default.
        """
        if self.config.run.gpu_lock_path is None:
            yield
            return
        with FileLock(str(self.config.run.gpu_lock_path)):
            try:
                yield
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        local_updates_cfg = config.get("local_updates")
        local_updates = int(local_updates_cfg) if local_updates_cfg is not None else None
        local_epochs = int(config.get("local_epochs", 1))
        scope = self.config.aggregation_scope
        model = self.assembly.model
        num_classes = node_num_classes(self.config)

        # Scaffold: el servidor concatenó [*x_t, *c] -- se parte por posición
        # (la señal explícita es config["scaffold_active"], no el largo).
        scaffold_active = bool(config.get("scaffold_active", False))
        c_global: dict[str, torch.Tensor] | None = None
        if scaffold_active:
            n_model = len(scope_module(model, scope).state_dict())
            if len(parameters) != 2 * n_model:
                raise ValueError(
                    f"[{self.config.node_name}] scaffold_active pero llegaron "
                    f"{len(parameters)} arrays (se esperaban {2 * n_model})."
                )
            c_arrays = parameters[n_model:]
            parameters = parameters[:n_model]

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

        # Scaffold: corrección de gradiente g <- g - c_i + c, aplicada a los
        # .grad tras backward() y antes de optimizer.step() (ver
        # train/loop.py `grad_correction`). Cuenta los pasos para E.
        grad_correction = None
        step_counter = [0]
        param_lr: dict[str, float] = {}
        x_global: dict[str, np.ndarray] = {}
        if scaffold_active:
            module = scope_module(model, scope)
            state_keys = list(module.state_dict().keys())
            trainable_names = [n for n, p in module.named_parameters() if p.requires_grad]
            device = self.config.run.device
            c_global = {
                k: torch.as_tensor(c_arrays[state_keys.index(k)], dtype=torch.float32, device=device)
                for k in trainable_names
            }
            if self._scaffold_c is None:
                self._scaffold_c = {
                    k: torch.zeros_like(c_global[k]) for k in trainable_names
                }
            c_local = self._scaffold_c
            x_global = {k: np.array(a, copy=True) for k, a in zip(state_keys, parameters)}
            named = dict(module.named_parameters())
            for group in optimizer.param_groups:
                for p in group["params"]:
                    for k in trainable_names:
                        if named[k] is p:
                            param_lr[k] = float(group["lr"])
            trainable_params = [(k, named[k]) for k in trainable_names]
            assert c_global is not None

            def grad_correction(_model: torch.nn.Module) -> None:
                for k, p in trainable_params:
                    if p.grad is not None:
                        p.grad.add_(c_global[k] - c_local[k])
                step_counter[0] += 1

        fit_start = time.perf_counter()
        train_metrics: dict[str, float] = {}
        if local_updates is not None:
            # Modo FLamby: UNA llamada de local_updates batches por ronda; una
            # sola fila de metrics.csv por ronda (no hay "épocas locales"
            # que loguear una por una; el eje X de metrics.csv es la ronda).
            epoch_start = time.perf_counter()
            with self._gpu_section():
                train_metrics = train_n_updates(
                    model,
                    self.assembly.loaders["train"],
                    optimizer,
                    self.assembly.loss_spec,
                    self.config.run.device,
                    n_updates=local_updates,
                    freeze_bn_stats=self.config.run.freeze_bn_stats,
                    regularizer=regularizer,
                    grad_correction=grad_correction,
                    num_classes=num_classes,
                )
                val_metrics = evaluate_loader(
                    model,
                    self.assembly.loaders["val"],
                    self.assembly.loss_spec,
                    self.config.run.device,
                    num_classes=num_classes,
                )
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()
            self.global_epoch += 1
            duration = time.perf_counter() - epoch_start
            self.logger.log(
                self.global_epoch,
                {
                    "round": float(current_round),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "duration_seconds": duration,
                },
            )
            self._log_line(
                f"ronda {current_round} {local_updates} updates locales: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}{_fmt_auc(val_metrics)} ({duration:.1f}s)"
            )
        for local_epoch in range(1, (local_epochs + 1) if local_updates is None else 1):
            epoch_start = time.perf_counter()
            with self._gpu_section():
                train_metrics = train_one_epoch(
                    model,
                    self.assembly.loaders["train"],
                    optimizer,
                    self.assembly.loss_spec,
                    self.config.run.device,
                    freeze_bn_stats=self.config.run.freeze_bn_stats,
                    regularizer=regularizer,
                    grad_correction=grad_correction,
                    num_classes=num_classes,
                )
                val_metrics = evaluate_loader(
                    model,
                    self.assembly.loaders["val"],
                    self.assembly.loss_spec,
                    self.config.run.device,
                    num_classes=num_classes,
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
                f"val_loss={val_metrics['loss']:.4f}"
                f"{_fmt_auc(val_metrics)} ({duration:.1f}s)"
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
        new_arrays = get_model_ndarrays(model, scope)
        if scaffold_active:
            assert c_global is not None and self._scaffold_c is not None
            new_arrays = self._scaffold_finish(
                model, new_arrays, x_global, c_global, param_lr, step_counter[0]
            )
        return new_arrays, self.assembly.n_train, report

    def _scaffold_finish(
        self,
        model: torch.nn.Sequential,
        y_arrays: list[np.ndarray],
        x_global: dict[str, np.ndarray],
        c_global: dict[str, torch.Tensor],
        param_lr: dict[str, float],
        n_steps: int,
    ) -> list[np.ndarray]:
        """Cierra la ronda Scaffold (opción II) y arma `[*delta_y, *delta_c]`.

            c_i_nuevo = c_i - c + (x_t - y_i) / (E * eta_l)
            delta_y   = y_i - x_t        delta_c = c_i_nuevo - c_i

        `E` = pasos de optimizador realmente dados; `eta_l` = lr del grupo del
        parámetro (respeta el LR discriminativo backbone/cabeza). `c_i` se
        actualiza in-place para la próxima ronda de ESTE nodo. Las entradas
        no entrenables (buffers de BN) llevan `delta_c = 0`.
        """
        assert self._scaffold_c is not None
        module = scope_module(model, self.config.aggregation_scope)
        state_keys = list(module.state_dict().keys())
        delta_y: list[np.ndarray] = []
        delta_c: list[np.ndarray] = []
        for key, y in zip(state_keys, y_arrays):
            x = x_global[key]
            delta_y.append((y.astype(np.float64) - x.astype(np.float64)).astype(y.dtype))
            if key in self._scaffold_c:
                c_i = self._scaffold_c[key]
                scale = 1.0 / (max(n_steps, 1) * param_lr[key])
                x_t = torch.as_tensor(x, dtype=torch.float32, device=c_i.device)
                y_t = torch.as_tensor(y, dtype=torch.float32, device=c_i.device)
                c_new = c_i - c_global[key] + scale * (x_t - y_t)
                delta_c.append((c_new - c_i).cpu().numpy())
                self._scaffold_c[key] = c_new
            else:
                delta_c.append(np.zeros(y.shape, dtype=np.float32))
        return [*delta_y, *delta_c]

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
        num_classes = node_num_classes(self.config)
        with self._gpu_section():
            metrics = evaluate_loader(
                model,
                self.assembly.loaders["val"],
                self.assembly.loss_spec,
                self.config.run.device,
                num_classes=num_classes,
            )

        # Binario: esquema fijo de siempre. Multiclase: lista reducida
        # (round, num_examples, loss) -- ver docs/FEDISIC2019_FLAMBY_REPRO.md §5.D3.
        fieldnames = _ROUNDS_CSV_FIELDS if num_classes == 2 else _ROUNDS_CSV_FIELDS_MULTICLASS
        mode = "a" if self._rounds_csv_started else "w"
        with open(self.rounds_csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._rounds_csv_started:
                writer.writeheader()
                self._rounds_csv_started = True
            writer.writerow(
                {"round": current_round, "num_examples": self.assembly.n_val, **metrics}
            )
        self._log_line(
            f"ronda {current_round} modelo agregado: "
            f"val_loss={metrics['loss']:.4f}{_fmt_auc(metrics)}"
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
