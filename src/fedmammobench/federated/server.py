"""Server entrypoint: configure FedAvg (or any registered strategy) and run
``flwr.simulation.start_simulation``.

This server prefers **federated evaluation**: after each round, every selected
client evaluates the freshly aggregated parameters on its *own* local
validation split and reports back ``(loss, num_examples, metrics)``. The
strategy's ``evaluate_metrics_aggregation_fn`` then weighted-averages those
metrics; Flower itself weighted-averages the losses. The result is logged
to ``server_federated_metrics.csv`` (alongside the legacy
``server_metrics.csv`` which is reserved for the centralized phase) and to
TensorBoard under ``server/federated``.

The legacy server-side **centralized evaluation** path remains available but
is now opt-in: only when ``cfg.data`` produces a non-empty ``test`` split
(typically via an independent dataset the server operator owns, e.g. a
public benchmark) does the server build an ``evaluate_fn`` and log rows
with ``phase="centralized"``. Set ``data.name: none`` (or ``test_fraction:
0.0`` with no manifest) to keep the server image-free.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import flwr as fl
from flwr.common import EvaluateRes, NDArrays, Scalar
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy

from fedmammobench.configs.schema import ExperimentConfig
from fedmammobench.datasets import MammographyDataset, build_dataloader, build_dataset
from fedmammobench.evaluation import Evaluator
from fedmammobench.federated.client import client_fn_factory
from fedmammobench.federated.node_logging import EarlyStoppingTriggered, NodeMetricsRecorder
from fedmammobench.federated.param_utils import (
    load_ndarrays_to_state_dict,
    state_dict_to_ndarrays,
)
from fedmammobench.federated.strategies import build_strategy
from fedmammobench.models import build_model
from fedmammobench.training import build_loss
from fedmammobench.utils.checkpoint import load_checkpoint
from fedmammobench.utils.csv_logger import CSVLogger
from fedmammobench.utils.device import resolve_device
from fedmammobench.utils.logging_utils import get_logger
from fedmammobench.utils.metrics_sink import MetricWriter, build_metric_sink

_logger = get_logger(__name__)


def _autoplot(out_root: Path) -> None:
    """Render plots for the just-finished run. Imported lazily so this module's
    import graph stays matplotlib-free (matplotlib is in requirements.txt but
    not a pyproject.toml dependency, so CI doesn't have it). ``autoplot``
    itself never raises — see fedmammobench.plotting.
    """
    from fedmammobench.plotting import autoplot

    autoplot(out_root)


def _build_evaluate_fn(
    cfg: ExperimentConfig,
    test_dataset: MammographyDataset | None,
    train_labels,  # noqa: ANN001 - numpy.ndarray
    tb_writer: MetricWriter | None,
    csv_logger: CSVLogger | None,
    round_offset: int = 0,
) -> Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None:
    """Build a centralized evaluate_fn that runs after each aggregation.

    ``round_offset`` shifts only the round number used for logging (CSV/TB) —
    never the raw ``server_round`` Flower itself passes in — so a resumed run
    (Flower's internal counter always restarts at 1, see run_simulation) still
    logs absolute round numbers, matching every other file in the same run.
    """
    if test_dataset is None or len(test_dataset) == 0:
        return None

    device = resolve_device(cfg.device)
    template_model = build_model(cfg.model).to(device)
    criterion = build_loss(
        cfg.training.loss, train_labels=train_labels, num_classes=cfg.model.num_classes
    ).to(device)
    loader = build_dataloader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        balance_classes=False,
    )
    evaluator = Evaluator(template_model, device=device, threshold=cfg.evaluation.threshold)

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],  # noqa: ARG001
    ) -> tuple[float, dict[str, Scalar]]:
        abs_round = server_round + round_offset
        load_ndarrays_to_state_dict(template_model, parameters, strict=True)
        result = evaluator.evaluate(loader, criterion=criterion)
        loss = float(result.get("loss", 0.0))
        numeric = {
            k: float(v)
            for k, v in result.items()
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))
        }
        _logger.info(
            "[server] round %d centralized: loss=%.4f auc=%.4f f1=%.4f sens=%.4f spec=%.4f",
            abs_round,
            loss,
            numeric.get("roc_auc", float("nan")),
            numeric.get("f1", float("nan")),
            numeric.get("sensitivity", float("nan")),
            numeric.get("specificity", float("nan")),
        )
        if tb_writer is not None:
            tb_writer.log_scalars("server/centralized", numeric, abs_round)
        if csv_logger is not None:
            csv_logger.append({"round": abs_round, "phase": "centralized", **numeric})
        # Flower expects (loss, metrics_dict). Drop non-numeric keys.
        return loss, {k: v for k, v in numeric.items() if k != "loss"}

    return evaluate_fn


_METRIC_KEYS: tuple[str, ...] = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "sensitivity",
    "specificity",
)


def _attach_federated_logging(
    strategy: Any,
    tb_writer: MetricWriter | None,
    csv_logger: CSVLogger | None,
    round_offset: int = 0,
) -> None:
    """Wrap ``strategy.aggregate_evaluate`` to log per-round federated metrics.

    Flower's :meth:`Strategy.aggregate_evaluate` returns ``(loss, metrics)``
    after weighted-averaging the per-client evaluate responses. This wrapper
    preserves that return value and, as a side effect, writes a CSV row and
    a TensorBoard scalar group under ``server/federated``. The original
    callable is left in place if no sinks are provided.

    ``round_offset`` shifts only the logged round number (CSV/TB) — the call
    to ``original`` (the real strategy) always gets Flower's own raw round
    number, unmodified, same principle as NodeMetricsRecorder.wrap().
    """
    if tb_writer is None and csv_logger is None:
        return

    original = strategy.aggregate_evaluate

    def wrapped(
        server_round: int,
        results: Sequence[tuple[ClientProxy, EvaluateRes]],
        failures: Sequence[Any],
    ) -> tuple[float | None, dict[str, Scalar]]:
        loss, metrics = original(server_round, results, failures)
        abs_round = server_round + round_offset
        num_examples_total = int(sum(r.num_examples for _, r in results))
        row: dict[str, Any] = {
            "round": abs_round,
            "phase": "federated",
            "num_examples_total": num_examples_total,
            "num_clients": len(results),
        }
        if loss is not None:
            row["loss"] = float(loss)
        for k in _METRIC_KEYS:
            if k == "loss":
                continue
            v = metrics.get(k)
            if isinstance(v, (int, float)) and v == v:  # filter NaN
                row[k] = float(v)
        _logger.info(
            "[server] round %d federated: clients=%d n=%d loss=%s auc=%s f1=%s",
            abs_round,
            row["num_clients"],
            num_examples_total,
            row.get("loss", "n/a"),
            row.get("roc_auc", "n/a"),
            row.get("f1", "n/a"),
        )
        if tb_writer is not None:
            numeric = {k: v for k, v in row.items() if isinstance(v, (int, float))}
            tb_writer.log_scalars("server/federated", numeric, abs_round)
        if csv_logger is not None:
            csv_logger.append(row)
        return loss, metrics

    strategy.aggregate_evaluate = wrapped  # type: ignore[method-assign]


def _maybe_attach_server_training(
    cfg: ExperimentConfig,
    strategy: Any,
    tb_writer: MetricWriter | None,
) -> None:
    """Attach hybrid server-side training to ``strategy`` when enabled.

    No-op unless ``cfg.federated.server_training.enabled``. Imported lazily so
    the (torch-heavy) trainer is only constructed when actually requested.
    """
    st = cfg.federated.server_training
    if not st.enabled:
        return
    from fedmammobench.federated.server_training import (
        ServerTrainer,
        attach_server_training,
    )

    trainer = ServerTrainer(cfg)
    attach_server_training(
        strategy,
        trainer,
        server_weight=st.server_weight,
        tb_writer=tb_writer,
    )


def _initial_parameters(
    cfg: ExperimentConfig, *, resume_checkpoint_path: str | Path | None = None
):
    """Build the model and serialize it as Flower initial parameters.

    When ``resume_checkpoint_path`` is given and exists, loads that
    checkpoint's weights into the model (strict) instead of leaving it at
    random init — this is the "load the last aggregated weights" half of
    federated resume. Returns ``(Parameters, payload | None)``; ``payload``
    is the full checkpoint dict (for reading ``epoch``/``extra``) or ``None``
    if no resume was requested/found, so callers can tell the two cases apart.
    """
    device = resolve_device(cfg.device)
    model = build_model(cfg.model).to(device)
    payload = None
    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).is_file():
        payload = load_checkpoint(resume_checkpoint_path, model, strict=True)
    ndarrays = state_dict_to_ndarrays(model)
    return fl.common.ndarrays_to_parameters(ndarrays), payload


def run_simulation(
    cfg: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> Any:
    """Run a federated simulation according to ``cfg``.

    Args:
        cfg: Fully populated :class:`ExperimentConfig`.
        output_dir: Directory for TB / CSV artifacts. Defaults to
            ``<cfg.output_dir>/<cfg.name>``.
        resume: When True, resume from ``weights/global_model.pt`` if present
            (loads its weights as the new initial parameters, and offsets all
            round numbers in logs/checkpoints by the last completed round —
            Flower itself always restarts its internal round counter at 1, so
            this process runs only the REMAINING rounds). If no checkpoint
            exists yet, starts fresh but writes a per-round safety-net
            checkpoint to ``weights/global_model.pt`` so a later crash can be
            recovered from by re-running with ``resume=True`` again.

    Returns:
        The :class:`flwr.server.History` produced by ``start_simulation``, or
        ``None`` if the run stopped early via
        ``cfg.federated.early_stopping_patience``, or if ``resume=True`` found
        ``federated.rounds`` already reached by the checkpoint.
    """
    out_root = Path(output_dir or Path(cfg.output_dir) / cfg.name).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Resume resolution up front — before the (expensive) build_dataset(cfg)
    # call below — so a resume that's already fully done bails out cheaply.
    weights_dir = out_root.parent / "weights"
    global_ckpt_path = weights_dir / "global_model.pt"
    initial_parameters, resume_payload = _initial_parameters(
        cfg, resume_checkpoint_path=(global_ckpt_path if resume else None)
    )
    round_offset = int(resume_payload.get("epoch", 0)) if resume_payload else 0
    remaining_rounds = cfg.federated.rounds - round_offset
    recorder_resume_state = (
        resume_payload.get("extra", {}).get("resume_state") if resume_payload else None
    )
    if resume:
        _logger.info(
            "Resume: round_offset=%d remaining_rounds=%d (checkpoint %s)",
            round_offset,
            remaining_rounds,
            "found" if resume_payload else "not found — starting fresh",
        )
        if remaining_rounds <= 0:
            _logger.info(
                "federated.rounds (%d) already reached by checkpoint (last completed "
                "round %d) — nothing to do.",
                cfg.federated.rounds,
                round_offset,
            )
            return None

    # tb_writer fans out to TensorBoard + (if cfg.wandb.enabled) W&B — every
    # existing consumer below only calls log_scalar/log_scalars/flush/close,
    # so nothing else in this function needs to change. See metrics_sink.py.
    tb_writer = build_metric_sink(cfg, out_root, job_type="fl-server-sim")
    csv_logger = CSVLogger(out_root / "server_metrics.csv")
    # Federated phase metrics live in a separate CSV so the two phases do not
    # fight over CSVLogger's locked-header schema (centralized vs federated
    # have a different set of columns).
    fed_csv_logger = CSVLogger(out_root / "server_federated_metrics.csv")

    _logger.info("Building datasets for federated simulation (%s)", cfg.data.name)
    datasets = build_dataset(cfg)
    if not datasets or "train" not in datasets:
        raise ValueError(
            "run_simulation requires a 'train' dataset for clients. "
            "data.name='none' is only valid for the gRPC server, where clients "
            "bring their own data. Use a real dataset name (cbis_ddsm, "
            "mammo_bench, vindr_mammo, ...) for simulation runs."
        )

    # Strategy ----------------------------------------------------------
    strategy_kwargs: dict[str, Any] = {
        "fraction_fit": cfg.federated.fraction_fit,
        "fraction_evaluate": cfg.federated.fraction_evaluate,
        "min_fit_clients": cfg.federated.min_fit_clients,
        "min_evaluate_clients": cfg.federated.min_evaluate_clients,
        "min_available_clients": cfg.federated.min_available_clients,
        "accept_failures": cfg.federated.accept_failures,
        "initial_parameters": initial_parameters,
        "evaluate_fn": _build_evaluate_fn(
            cfg,
            datasets.get("test"),
            datasets["train"].labels,
            tb_writer,
            csv_logger,
            round_offset=round_offset,
        ),
        "on_fit_config_fn": _make_on_fit_config_fn(cfg, round_offset=round_offset),
    }
    strategy_kwargs.update(cfg.federated.strategy.params)
    strategy = build_strategy(cfg.federated.strategy.name, **strategy_kwargs)
    _attach_federated_logging(strategy, tb_writer, fed_csv_logger, round_offset=round_offset)
    _maybe_attach_server_training(cfg, strategy, tb_writer)
    # Per-node logging + timing + global-model (+ best-checkpoint) capture.
    # Attached last so the captured global parameters reflect any
    # server-side training step. cfg is threaded through so the recorder can
    # materialize/save weights/global_best.pt when federated.save_best_checkpoint
    # is enabled (see node_logging.py record_eval's best-checkpoint hook).
    recorder = NodeMetricsRecorder(
        out_root,
        tb_writer,
        cfg=cfg,
        round_offset=round_offset,
        resume_state=recorder_resume_state,
        save_every_round=resume,
    )
    recorder.wrap(strategy)

    client_fn = client_fn_factory(cfg, datasets, out_root=out_root / "clients")

    round_timeout = cfg.federated.round_timeout_seconds or None
    server_config = ServerConfig(
        num_rounds=remaining_rounds,
        round_timeout=float(round_timeout) if round_timeout else None,
    )

    _logger.info(
        "Starting simulation: %d clients, %d rounds, strategy=%s",
        cfg.federated.num_clients,
        remaining_rounds,
        cfg.federated.strategy.name,
    )
    recorder.start()
    t0 = time.perf_counter()
    history = None
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.federated.num_clients,
            config=server_config,
            strategy=strategy,
            client_resources=cfg.federated.client_resources,
            ray_init_args=cfg.federated.ray_init_args or None,
        )
    except RuntimeError as e:
        # fl.simulation.start_simulation catches any Exception raised inside
        # the strategy (e.g. from our wrapped aggregate_evaluate) and
        # re-raises it as RuntimeError("Simulation crashed.") from ex — so an
        # intentional EarlyStoppingTriggered arrives here already rewrapped,
        # not as its own type. Unwrap via __cause__ to tell a clean, expected
        # stop from a real crash; a genuine crash still propagates.
        if isinstance(e.__cause__, EarlyStoppingTriggered):
            _logger.info("Simulation stopped by early stopping: %s", e.__cause__)
        else:
            raise
    finally:
        total_seconds = time.perf_counter() - t0
        recorder.save_global_model(cfg)
        recorder.write_timing_summary(cfg, total_seconds)
        recorder.close()
        tb_writer.close()
        _autoplot(out_root)
        _logger.info("Simulation complete. Artifacts in %s", out_root)
    return history


def _make_on_fit_config_fn(cfg: ExperimentConfig, round_offset: int = 0):
    """Return a per-round config function sent to clients in ``fit``.

    ``round_offset`` shifts the round number CLIENTS see (``current_round``,
    read by ``apply_freeze_policy`` for progressive backbone unfreezing) —
    without this, a resumed run's unfreeze schedule would desync, since
    Flower's internal round counter always restarts at 1 on a new process.
    """

    def fn(server_round: int) -> dict[str, Scalar]:
        abs_round = server_round + round_offset
        return {
            "server_round": abs_round,
            "current_round": abs_round,  # read by apply_freeze_policy progressive unfreeze
            "local_epochs": cfg.training.local_epochs,
        }

    return fn


def _make_on_evaluate_config_fn(round_offset: int = 0):
    """Return a per-round evaluate config function (gRPC server only).

    Extracted from an inline lambda so it can be unit-tested and offset like
    ``_make_on_fit_config_fn``.
    """

    def fn(server_round: int) -> dict[str, Scalar]:
        return {"current_round": server_round + round_offset}

    return fn


# ----------------------------------------------------------------------
# Real gRPC server (multi-device deployment)
# ----------------------------------------------------------------------

def run_grpc_server(
    cfg: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> None:
    """Run the Flower gRPC server for a real multi-device deployment.

    Unlike :func:`run_simulation`, this function does not spawn clients and
    does not initialize Ray. It opens a gRPC port (``cfg.federated.server_address``)
    and waits for physical clients to connect. The clients must run
    ``scripts/run_client.py`` against this server's address.

    The server's local data (built from ``cfg``) is used exclusively for
    *centralized* evaluation via ``evaluate_fn``. In the canonical Mammo-Bench
    setup, the server YAML sets ``data.test_fraction=1.0`` and
    ``data.val_fraction=0.0`` so the entire local manifest becomes the held-out
    test set.

    Args:
        cfg: Fully populated :class:`ExperimentConfig`.
        output_dir: Directory for TB / CSV artifacts. Defaults to
            ``<cfg.output_dir>/<cfg.name>``.
        resume: Same semantics as :func:`run_simulation`'s ``resume`` — resume
            from ``weights/global_model.pt`` if present, else start fresh but
            write the per-round safety-net checkpoint there.

    Returns:
        None. Blocks until ``cfg.federated.rounds`` (absolute; fewer rounds
        run this process if resuming) rounds complete or the server is
        interrupted. Returns immediately, without opening the gRPC port, if
        ``resume=True`` found ``federated.rounds`` already reached.
    """
    out_root = Path(output_dir or Path(cfg.output_dir) / cfg.name).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Resume resolution up front — see the matching block in run_simulation()
    # for the full rationale.
    weights_dir = out_root.parent / "weights"
    global_ckpt_path = weights_dir / "global_model.pt"
    initial_parameters, resume_payload = _initial_parameters(
        cfg, resume_checkpoint_path=(global_ckpt_path if resume else None)
    )
    round_offset = int(resume_payload.get("epoch", 0)) if resume_payload else 0
    remaining_rounds = cfg.federated.rounds - round_offset
    recorder_resume_state = (
        resume_payload.get("extra", {}).get("resume_state") if resume_payload else None
    )
    if resume:
        _logger.info(
            "Resume: round_offset=%d remaining_rounds=%d (checkpoint %s)",
            round_offset,
            remaining_rounds,
            "found" if resume_payload else "not found — starting fresh",
        )
        if remaining_rounds <= 0:
            _logger.info(
                "federated.rounds (%d) already reached by checkpoint (last completed "
                "round %d) — nothing to do.",
                cfg.federated.rounds,
                round_offset,
            )
            return

    # tb_writer fans out to TensorBoard + (if cfg.wandb.enabled) W&B — see
    # the matching comment in run_simulation() above.
    tb_writer = build_metric_sink(cfg, out_root, job_type="fl-server-grpc")
    csv_logger = CSVLogger(out_root / "server_metrics.csv")
    # Federated phase metrics live in a separate CSV so the two phases do not
    # fight over CSVLogger's locked-header schema (centralized vs federated
    # have a different set of columns).
    fed_csv_logger = CSVLogger(out_root / "server_federated_metrics.csv")

    if cfg.data.name == "none":
        _logger.info(
            "data.name='none' — skipping server-side dataset construction. "
            "Validation will be 100%% federated (aggregated client metrics)."
        )
        test_dataset = None
        train_labels_for_loss = None
    else:
        _logger.info(
            "Building server-side test dataset for centralized evaluation (%s)",
            cfg.data.name,
        )
        datasets = build_dataset(cfg)
        test_dataset = datasets.get("test")
        if test_dataset is None or len(test_dataset) == 0:
            _logger.warning(
                "Server build_dataset(cfg) produced no test split. Centralized "
                "evaluation will be disabled; metrics will come from federated "
                "client evaluation only."
            )
            train_labels_for_loss = None
        else:
            train_labels_for_loss = test_dataset.labels

    strategy_kwargs: dict[str, Any] = {
        "fraction_fit": cfg.federated.fraction_fit,
        "fraction_evaluate": cfg.federated.fraction_evaluate,
        "min_fit_clients": cfg.federated.min_fit_clients,
        "min_evaluate_clients": cfg.federated.min_evaluate_clients,
        "min_available_clients": cfg.federated.min_available_clients,
        "accept_failures": cfg.federated.accept_failures,
        "initial_parameters": initial_parameters,
        "evaluate_fn": _build_evaluate_fn(
            cfg,
            test_dataset,
            train_labels_for_loss,
            tb_writer,
            csv_logger,
            round_offset=round_offset,
        ),
        "on_fit_config_fn": _make_on_fit_config_fn(cfg, round_offset=round_offset),
        "on_evaluate_config_fn": _make_on_evaluate_config_fn(round_offset=round_offset),
    }
    strategy_kwargs.update(cfg.federated.strategy.params)
    strategy = build_strategy(cfg.federated.strategy.name, **strategy_kwargs)
    _attach_federated_logging(strategy, tb_writer, fed_csv_logger, round_offset=round_offset)
    _maybe_attach_server_training(cfg, strategy, tb_writer)
    recorder = NodeMetricsRecorder(
        out_root,
        tb_writer,
        cfg=cfg,
        round_offset=round_offset,
        resume_state=recorder_resume_state,
        save_every_round=resume,
    )
    recorder.wrap(strategy)

    round_timeout = cfg.federated.round_timeout_seconds or None
    server_config = ServerConfig(
        num_rounds=remaining_rounds,
        round_timeout=float(round_timeout) if round_timeout else None,
    )

    _logger.info(
        "Starting gRPC server on %s for %d rounds (strategy=%s, "
        "min_available_clients=%d, grpc_max_msg=%d MB, round_timeout=%s). "
        "Waiting for clients to connect ...",
        cfg.federated.server_address,
        remaining_rounds,
        cfg.federated.strategy.name,
        cfg.federated.min_available_clients,
        cfg.federated.grpc_max_message_length // (1024 * 1024),
        f"{round_timeout}s" if round_timeout else "none",
    )
    # Inject gRPC keepalive so long-running fit rounds (>2 min) don't cause
    # GrpcBridgeClosed errors. Flower's start_server() doesn't expose these
    # options directly, so we patch grpc.server() before calling it.
    import grpc as _grpc
    _orig_grpc_server = _grpc.server

    def _grpc_server_with_keepalive(thread_pool, **kwargs):
        opts = list(kwargs.pop("options", []))
        opts += [
            ("grpc.keepalive_time_ms", 120_000),         # ping every 2 min
            ("grpc.keepalive_timeout_ms", 20_000),        # 20 s to ack ping
            ("grpc.http2.max_pings_without_data", 0),     # no ping limit
            ("grpc.keepalive_permit_without_calls", 1),   # ping even when idle
            ("grpc.max_connection_idle_ms", 3_600_000),   # 1 h idle before close
        ]
        return _orig_grpc_server(thread_pool, options=opts, **kwargs)

    _grpc.server = _grpc_server_with_keepalive
    recorder.start()
    t0 = time.perf_counter()
    try:
        fl.server.start_server(
            server_address=cfg.federated.server_address,
            config=server_config,
            strategy=strategy,
            grpc_max_message_length=cfg.federated.grpc_max_message_length,
        )
    except EarlyStoppingTriggered as e:
        # Unlike fl.simulation.start_simulation, fl.server.start_server has no
        # internal try/except of its own — the exception from our wrapped
        # aggregate_evaluate propagates here raw, not rewrapped.
        _logger.info("gRPC server stopped by early stopping: %s", e)
    finally:
        _grpc.server = _orig_grpc_server  # restore original
        total_seconds = time.perf_counter() - t0
        recorder.save_global_model(cfg)
        recorder.write_timing_summary(cfg, total_seconds)
        recorder.close()
        tb_writer.close()
        _autoplot(out_root)
        _logger.info("gRPC server stopped. Artifacts in %s", out_root)


__all__ = ["run_simulation", "run_grpc_server"]
