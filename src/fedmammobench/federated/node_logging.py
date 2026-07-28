"""Per-node metric logging, timing, and global-model capture.

Wraps a Flower strategy so that, on top of the aggregated server metrics:

- every participating node's per-round **fit** and **evaluate** metrics are
  written to its own subfolder ``runs/<name>/clients/client_<id>/`` (a CSV per
  phase plus a TensorBoard run);
- **wall-clock time** is recorded per node (``fit_seconds`` / ``eval_seconds``,
  reported by the client) and per round (server-side), in
  ``runs/<name>/server_timing.csv``, with an overall ``timing_summary.csv``;
- the latest **aggregated (global) parameters** are captured so the final global
  model can be checkpointed for post-hoc verification.

All file writes happen in the **server process**, driven by the per-client
results Flower already hands to ``aggregate_fit`` / ``aggregate_evaluate``. No
files are written from the (possibly Ray-distributed) client workers, so there
is no write contention and the layout is identical for simulation and gRPC.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fedmammobench.configs.schema import ExperimentConfig
from fedmammobench.federated.param_utils import load_ndarrays_to_state_dict
from fedmammobench.models import build_model
from fedmammobench.utils.checkpoint import save_checkpoint
from fedmammobench.utils.csv_logger import CSVLogger
from fedmammobench.utils.device import resolve_device
from fedmammobench.utils.logging_utils import get_logger
from fedmammobench.utils.tensorboard_utils import TensorBoardWriter

_logger = get_logger(__name__)

# Eval metric columns kept in a fixed order so the per-client CSV has a stable
# schema even when a node omits a NaN metric (e.g. roc_auc on a single-class batch).
_EVAL_KEYS: tuple[str, ...] = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "sensitivity",
    "specificity",
)


class EarlyStoppingTriggered(Exception):
    """Raised from ``record_eval`` when ``federated.early_stopping_patience``
    rounds pass with no improvement in the tracked metric.

    Flower's ``Server.fit`` round loop has no native "stop early" hook — the
    only lever a wrapped strategy has is raising an exception. ``server.py``
    catches this specific type around ``fl.simulation.start_simulation`` /
    ``fl.server.start_server`` and treats it as a clean, expected stop (an
    INFO log line) rather than a crash; the existing ``finally`` blocks there
    still save ``global_model.pt``, write the summary, close sinks, and
    autoplot regardless of how the round loop ends.
    """

    def __init__(
        self,
        server_round: int,
        metric: str,
        patience: int,
        best_round: int | None,
        best_value: float | None,
    ) -> None:
        self.server_round = server_round
        self.metric = metric
        self.patience = patience
        self.best_round = best_round
        self.best_value = best_value
        best_value_str = f"{best_value:.4f}" if best_value is not None else "n/a"
        super().__init__(
            f"Early stopping at round {server_round}: no improvement in {metric} "
            f"for {patience} rounds (best={best_value_str} at round {best_round})"
        )


class NodeMetricsRecorder:
    """Owns per-node CSV/TB sinks, timing, and the latest global parameters."""

    def __init__(
        self,
        out_root: str | Path,
        tb_writer: TensorBoardWriter | None = None,
        cfg: ExperimentConfig | None = None,
        *,
        round_offset: int = 0,
        resume_state: dict[str, Any] | None = None,
        save_every_round: bool = False,
    ) -> None:
        self.out_root = Path(out_root).expanduser().resolve()
        self.clients_dir = self.out_root / "clients"
        self.tb_writer = tb_writer

        self._fit_csv: dict[int, CSVLogger] = {}
        self._eval_csv: dict[int, CSVLogger] = {}
        self._client_tb: dict[int, TensorBoardWriter] = {}
        self._timing_csv = CSVLogger(self.out_root / "server_timing.csv")

        self._latest_parameters: Parameters | None = None
        self._start_time: float | None = None
        self._last_round_end: float | None = None

        # Per-node accumulated fit time and samples (for final summary)
        self._node_fit_seconds_total: dict[int, float] = {}
        self._node_fit_samples_total: dict[int, int] = {}
        # Aggregated federated eval metrics per round (for final summary)
        self._round_agg_metrics: list[dict] = []

        # Best-checkpoint tracking (federated.save_best_checkpoint). `cfg` is
        # optional (kept for tests/callers that only need CSV/TB recording,
        # e.g. tests/test_node_logging.py constructs this with just out_root)
        # but is required for both save_global_model() and best-checkpoint
        # saving, since materializing a checkpoint needs cfg.model/cfg.device.
        self.cfg = cfg
        _fed = getattr(cfg, "federated", None)
        self._track_best: bool = bool(_fed is not None and _fed.save_best_checkpoint)
        self._best_metric: str | None = _fed.best_checkpoint_metric if self._track_best else None
        self._best_value: float | None = resume_state.get("best_value") if resume_state else None
        self._best_round: int | None = resume_state.get("best_round") if resume_state else None
        self._best_path: Path | None = None

        # Early stopping (federated.early_stopping_patience). Reuses the same
        # best-value tracking as above; see EarlyStoppingTriggered.
        self._patience: int = int(_fed.early_stopping_patience) if _fed is not None else 0
        self._rounds_no_improve: int = (
            int(resume_state.get("rounds_no_improve", 0)) if resume_state else 0
        )
        self._stopped_early_round: int | None = None

        # Resume support (see server.py's run_simulation/run_grpc_server).
        # round_offset shifts every round number this recorder logs (CSVs,
        # TB, checkpoint `epoch=`/extra["round"]) so a resumed run continues
        # absolute round numbering instead of restarting at 1 — Flower's own
        # internal round counter always restarts at 1 for a new process, no
        # way around that. Applied once, at wrap()'s boundary; nothing below
        # this point needs to know about it.
        self._round_offset: int = int(round_offset)
        self._save_every_round: bool = bool(save_every_round)

    # -- lazy per-client sinks --------------------------------------------

    def _client_root(self, cid: int) -> Path:
        return self.clients_dir / f"client_{cid}"

    def _fit_logger(self, cid: int) -> CSVLogger:
        if cid not in self._fit_csv:
            self._fit_csv[cid] = CSVLogger(self._client_root(cid) / "fit_metrics.csv")
        return self._fit_csv[cid]

    def _eval_logger(self, cid: int) -> CSVLogger:
        if cid not in self._eval_csv:
            self._eval_csv[cid] = CSVLogger(self._client_root(cid) / "eval_metrics.csv")
        return self._eval_csv[cid]

    def _client_tb_writer(self, cid: int) -> TensorBoardWriter:
        if cid not in self._client_tb:
            self._client_tb[cid] = TensorBoardWriter(self._client_root(cid) / "tb")
        return self._client_tb[cid]

    @staticmethod
    def _cid_of(proxy: ClientProxy, metrics: dict[str, Any], fallback: int) -> int:
        raw = metrics.get("client_id") if metrics else None
        if raw is None:
            raw = getattr(proxy, "cid", fallback)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return fallback

    # -- recording --------------------------------------------------------

    def start(self) -> None:
        """Mark the start of the run so the first round's wall time is correct."""
        self._start_time = time.perf_counter()
        self._last_round_end = self._start_time

    def record_fit(
        self, server_round: int, results: Sequence[tuple[ClientProxy, FitRes]]
    ) -> None:
        now = time.perf_counter()
        fit_secs: list[float] = []
        for i, (proxy, res) in enumerate(results):
            m = dict(res.metrics or {})
            cid = self._cid_of(proxy, m, i)
            fs = float(m.get("fit_seconds", float("nan")))
            row = {
                "round": server_round,
                "client_id": cid,
                "num_samples": int(res.num_examples),
                "train_loss": float(m["train_loss"]) if "train_loss" in m else "",
                "task_loss": float(m["task_loss"]) if "task_loss" in m else "",
                "local_epochs": int(m["local_epochs"]) if "local_epochs" in m else "",
                "fit_seconds": fs if fs == fs else "",  # NaN -> ""
            }
            self._fit_logger(cid).append(row)
            tb = self._client_tb_writer(cid)
            numeric_row = {k: v for k, v in row.items() if isinstance(v, float)}
            tb.log_scalars("fit", numeric_row, server_round)
            tb.flush()
            if self.tb_writer is not None:
                # Same numbers, also pushed to the shared server sink (TB +
                # wandb, see MetricSink) under a per-node prefix, so
                # node_<cid>/train_loss shows up next to the other server
                # scalars — the per-client writer above only feeds that
                # node's own isolated TB run.
                self.tb_writer.log_scalars(f"node_{cid}", numeric_row, server_round)
            if fs == fs:
                fit_secs.append(fs)
                self._node_fit_seconds_total[cid] = self._node_fit_seconds_total.get(cid, 0.0) + fs
            self._node_fit_samples_total[cid] = (
                self._node_fit_samples_total.get(cid, 0) + int(res.num_examples)
            )

        wall = (now - self._last_round_end) if self._last_round_end is not None else float("nan")
        self._last_round_end = now
        timing_row = {
            "round": server_round,
            "round_wall_seconds": float(wall) if wall == wall else "",
            "num_clients_fit": len(results),
            "sum_fit_seconds": sum(fit_secs) if fit_secs else "",
            "max_fit_seconds": max(fit_secs) if fit_secs else "",
            "mean_fit_seconds": (sum(fit_secs) / len(fit_secs)) if fit_secs else "",
        }
        self._timing_csv.append(timing_row)
        if self.tb_writer is not None:
            self.tb_writer.log_scalars(
                "server/timing",
                {k: v for k, v in timing_row.items() if isinstance(v, float)},
                server_round,
            )
        _logger.info(
            "[server] round %d timing: wall=%.2fs, %d nodes, sum_fit=%.2fs, max_fit=%.2fs",
            server_round,
            wall if wall == wall else float("nan"),
            len(results),
            timing_row["sum_fit_seconds"] if fit_secs else float("nan"),
            timing_row["max_fit_seconds"] if fit_secs else float("nan"),
        )

    def record_eval(
        self, server_round: int, results: Sequence[tuple[ClientProxy, EvaluateRes]]
    ) -> None:
        for i, (proxy, res) in enumerate(results):
            m = dict(res.metrics or {})
            cid = self._cid_of(proxy, m, i)
            row: dict[str, Any] = {
                "round": server_round,
                "client_id": cid,
                "num_samples": int(res.num_examples),
                "eval_seconds": float(m["eval_seconds"]) if "eval_seconds" in m else "",
            }
            for k in _EVAL_KEYS:
                if k == "loss":
                    row["loss"] = float(res.loss)
                else:
                    row[k] = float(m[k]) if k in m else ""
            self._eval_logger(cid).append(row)
            tb = self._client_tb_writer(cid)
            numeric_row = {k: v for k, v in row.items() if isinstance(v, float)}
            tb.log_scalars("eval", numeric_row, server_round)
            tb.flush()
            if self.tb_writer is not None:
                # val_-prefix so it doesn't collide with record_fit's node_<cid>/
                # train-side keys on the shared server sink (e.g. node_0/val_loss
                # next to node_0/train_loss).
                self.tb_writer.log_scalars(
                    f"node_{cid}",
                    {f"val_{k}": v for k, v in numeric_row.items()},
                    server_round,
                )

        # Accumulate weighted-average metrics across clients for this round
        total_samples = sum(int(r.num_examples) for _, r in results)
        if total_samples > 0:
            agg: dict[str, Any] = {
                "round": server_round,
                "num_clients": len(results),
                "total_samples": total_samples,
            }
            for k in _EVAL_KEYS:
                weighted_vals = []
                weights = []
                for _, res in results:
                    m_r = dict(res.metrics or {})
                    v = float(res.loss) if k == "loss" else float(m_r.get(k, float("nan")))
                    if v == v:  # skip NaN
                        weighted_vals.append(v * int(res.num_examples))
                        weights.append(int(res.num_examples))
                if weights:
                    agg[k] = sum(weighted_vals) / sum(weights)
            self._round_agg_metrics.append(agg)

            # Best-checkpoint tracking (federated.save_best_checkpoint). The
            # ordering guarantee this relies on: within round R, Flower calls
            # aggregate_fit(R) (which sets self._latest_parameters, see
            # wrap()/fit_wrapped below) BEFORE distributing those exact params
            # to clients and calling aggregate_evaluate(R) -> record_eval(R).
            # So at this point _latest_parameters IS the checkpoint `agg`
            # describes, with no extra state needed to prove it.
            #
            # The else branch below (no improvement, INCLUDING the metric
            # being absent from `agg` entirely — e.g. an all-NaN round) feeds
            # early_stopping_patience. Treating an absent metric as
            # "no improvement" matters: otherwise a NaN streak would freeze
            # _rounds_no_improve and silently defeat early stopping.
            if self._track_best:
                current = agg.get(self._best_metric)
                if current is not None and current == current and (  # not NaN
                    self._best_value is None or current > self._best_value
                ):
                    self._best_value, self._best_round = current, server_round
                    self._rounds_no_improve = 0
                    self._save_best_global_model(server_round, current)
                else:
                    self._rounds_no_improve += 1

            # Resume safety net: overwrite weights/global_model.pt EVERY round
            # (not just on improvement, unlike _save_best_global_model above),
            # so a hard process kill leaves a checkpoint at most one round
            # stale. Runs regardless of federated.save_best_checkpoint. Placed
            # AFTER the best-tracking update above (so extra["resume_state"]
            # reflects THIS round's contribution, not a stale one-round-behind
            # snapshot) but BEFORE the early-stop raise below (so the round
            # that triggers the stop is still captured, not skipped).
            if self._save_every_round:
                self._save_progressive_global_model(server_round)

            if (
                self._track_best
                and self._patience > 0
                and self._rounds_no_improve >= self._patience
            ):
                self._stopped_early_round = server_round
                raise EarlyStoppingTriggered(
                    server_round,
                    self._best_metric,
                    self._patience,
                    self._best_round,
                    self._best_value,
                )

    # -- strategy wrapping ------------------------------------------------

    def wrap(self, strategy: Any) -> None:
        """Wrap ``aggregate_fit`` / ``aggregate_evaluate`` to record per node.

        Attach this **last** (outermost) so the captured global parameters are
        whatever clients will actually receive next round — i.e. after any
        server-side training step has post-processed the aggregated weights.
        """
        orig_fit = strategy.aggregate_fit
        orig_eval = strategy.aggregate_evaluate

        def fit_wrapped(server_round, results, failures):  # noqa: ANN001, ANN202
            # Flower's own strategy always sees its own raw round number —
            # the offset is applied only to what WE log, never to what we
            # pass to/receive from Flower's real aggregate_fit.
            aggregated, metrics = orig_fit(server_round, results, failures)
            if aggregated is not None:
                self._latest_parameters = aggregated
            self.record_fit(server_round + self._round_offset, results)
            return aggregated, metrics

        def eval_wrapped(server_round, results, failures):  # noqa: ANN001, ANN202
            out = orig_eval(server_round, results, failures)
            self.record_eval(server_round + self._round_offset, results)
            return out

        strategy.aggregate_fit = fit_wrapped  # type: ignore[method-assign]
        strategy.aggregate_evaluate = eval_wrapped  # type: ignore[method-assign]

    # -- finalization -----------------------------------------------------

    def _materialize_and_save(
        self, cfg: ExperimentConfig, out_path: Path, *, epoch: int, extra: dict[str, Any]
    ) -> Path | None:
        """Shared body: latest captured params -> model -> checkpoint on disk.

        Builds a fresh model each call rather than caching one, so this stays
        bit-identical to what save_global_model always did (including
        `.to(device)`) — the cost (~1s: build_model + strict state_dict load)
        is negligible against a multi-minute FL round.
        """
        if self._latest_parameters is None:
            return None
        ndarrays = parameters_to_ndarrays(self._latest_parameters)
        device = resolve_device(cfg.device)
        model = build_model(cfg.model).to(device)
        load_ndarrays_to_state_dict(model, ndarrays, strict=True)
        save_checkpoint(out_path, model, epoch=epoch, extra=extra)
        return out_path

    def save_global_model(
        self, cfg: ExperimentConfig, path: str | Path | None = None
    ) -> Path | None:
        """Materialize the latest aggregated parameters into a checkpoint.

        Always the LAST round's weights — unaffected by best-checkpoint
        tracking, which writes a separate `global_best.pt` (see
        _save_best_global_model). Returns the path written, or None if no
        parameters were captured (e.g. the run produced zero successful fit
        rounds).
        """
        if self._latest_parameters is None:
            _logger.warning(
                "No aggregated parameters were captured; global model not saved. "
                "Did any fit round succeed?"
            )
            return None
        # Weights live in a "weights/" sibling of self.out_root (which is
        # normally <output_dir>/<name>/), not inside it, so the (large)
        # checkpoint can be excluded from a sync/upload of the rest of the
        # run's metrics/logs by folder alone. Derived from self.out_root
        # (not cfg.output_dir) so it stays correct even when the recorder was
        # constructed with an explicit/overridden out_root.
        out = (
            Path(path)
            if path is not None
            else (self.out_root.parent / "weights" / "global_model.pt")
        )
        # epoch= is the actual last round that ran, not the configured budget
        # (cfg.federated.rounds) — they coincide unless the run stopped early
        # (early_stopping_patience) or crashed after at least one round. The
        # `self._round_offset or ...` fallback covers the narrow case of a
        # resumed run that crashes again before any new round completes: the
        # last known-good absolute round is the offset itself, not the full
        # target — metadata-only, never affects the model weights themselves.
        last_round = (
            self._round_agg_metrics[-1]["round"]
            if self._round_agg_metrics
            else (self._round_offset or cfg.federated.rounds)
        )
        self._materialize_and_save(
            cfg,
            out,
            epoch=last_round,
            extra={
                "source": "federated_global",
                "rounds": cfg.federated.rounds,
                "resume_state": {
                    "best_value": self._best_value,
                    "best_round": self._best_round,
                    "rounds_no_improve": self._rounds_no_improve,
                },
            },
        )
        _logger.info(
            "Saved final global model to %s — verify it with "
            "`fedmammobench-evaluate --checkpoint %s` on a held-out test set.",
            out,
            out,
        )
        return out

    def _save_progressive_global_model(self, server_round: int) -> None:
        """Overwrite weights/global_model.pt after every round — the actual
        resume safety net (federated.enable via server.py's ``resume=True``
        -> ``save_every_round``).

        Unlike save_global_model (only called once, in the caller's
        ``finally`` block — which a hard process kill bypasses entirely,
        since it skips Python's exception machinery), this runs every round,
        wrapped in try/except so a transient disk error never takes down
        round N+1. Shares the same path/extra shape as save_global_model so
        either one can be the "last" write.
        """
        cfg = self.cfg
        if cfg is None or self._latest_parameters is None:
            return
        out = self.out_root.parent / "weights" / "global_model.pt"
        try:
            self._materialize_and_save(
                cfg,
                out,
                epoch=server_round,
                extra={
                    "source": "federated_global",
                    "rounds": cfg.federated.rounds,
                    "resume_state": {
                        "best_value": self._best_value,
                        "best_round": self._best_round,
                        "rounds_no_improve": self._rounds_no_improve,
                    },
                },
            )
        except Exception:
            _logger.warning(
                "Failed to save progressive global checkpoint at round %d",
                server_round,
                exc_info=True,
            )

    def _save_best_global_model(self, server_round: int, value: float) -> None:
        """Save weights/global_best.pt when the tracked metric improves mid-run.

        Unlike save_global_model (called once, after the run loop finishes),
        this runs mid-run — wrapped in try/except so a transient disk error
        can never take down round N+1. Named `global_best.pt` (shares the
        `global_` prefix with `global_model.pt` so they sort together in an
        `ls`, and is distinct from the centralized path's `weights/best.pt`).
        """
        cfg = self.cfg
        if cfg is None or self._latest_parameters is None:
            return
        out = self.out_root.parent / "weights" / "global_best.pt"
        try:
            self._materialize_and_save(
                cfg,
                out,
                epoch=server_round,
                extra={
                    "source": "federated_global_best",
                    "round": server_round,
                    "metric": self._best_metric,
                    "value": float(value),
                    "rounds_total": cfg.federated.rounds,
                },
            )
        except Exception:
            _logger.warning(
                "Failed to save best global checkpoint at round %d", server_round, exc_info=True
            )
            return
        self._best_path = out
        _logger.info(
            "[server] round %d: new best %s=%.4f -> saved %s",
            server_round,
            self._best_metric,
            value,
            out,
        )

    def write_timing_summary(self, cfg: ExperimentConfig, total_seconds: float) -> None:
        """Write timing summary CSVs and a human-readable final_summary.txt."""
        # Actual rounds completed, not the configured budget — they coincide
        # unless the run stopped early (early_stopping_patience) or crashed
        # after at least one round completed.
        rounds_completed = len(self._round_agg_metrics) or int(cfg.federated.rounds)
        rounds = max(rounds_completed, 1)

        # 1. Overall timing CSV
        summary = CSVLogger(self.out_root / "timing_summary.csv")
        summary.append(
            {
                "total_seconds": round(total_seconds, 3),
                "total_minutes": round(total_seconds / 60, 2),
                "num_rounds": rounds,
                "num_clients": cfg.federated.num_clients,
                "avg_seconds_per_round": round(float(total_seconds) / rounds, 3),
            }
        )

        # 2. Per-node timing CSV
        if self._node_fit_seconds_total:
            node_csv = CSVLogger(self.out_root / "per_node_timing.csv")
            for cid in sorted(self._node_fit_seconds_total):
                total_fit = self._node_fit_seconds_total[cid]
                total_samp = self._node_fit_samples_total.get(cid, 0)
                node_csv.append(
                    {
                        "client_id": cid,
                        "total_fit_seconds": round(total_fit, 3),
                        "total_fit_minutes": round(total_fit / 60, 2),
                        "total_samples_trained": total_samp,
                        "avg_seconds_per_sample": (
                            round(total_fit / total_samp, 6) if total_samp else ""
                        ),
                    }
                )

        # 3. Human-readable final summary
        def _fmt(val: object, decimals: int = 4) -> str:
            if isinstance(val, float) and val != val:
                return "n/a"
            if isinstance(val, float):
                return f"{val:.{decimals}f}"
            return str(val)

        # When resumed (round_offset > 0), `rounds` only counts THIS process's
        # rounds — the wall-time/avg-per-round math below is correctly scoped
        # to this session, but "Rondas completadas" needs a distinct label so
        # it doesn't read as contradicting the absolute round number shown
        # further down (e.g. "Última ronda (#20)"). Gated so the default
        # (round_offset=0, i.e. every non-resumed run) stays byte-identical.
        session_label = (
            "Rondas completadas (sesión)" if self._round_offset > 0 else "Rondas completadas"
        )
        lines: list[str] = [
            "=" * 62,
            f"  RESUMEN FINAL — {cfg.name}",
            "=" * 62,
            f"  Estrategia         : {cfg.federated.strategy.name}",
            f"  {session_label} : {rounds}",
        ]
        if self._round_offset > 0 and self._round_agg_metrics:
            lines.append(
                f"  Ronda absoluta final : #{self._round_agg_metrics[-1]['round']}"
            )
        lines += [
            f"  Clientes           : {cfg.federated.num_clients}",
            f"  Tiempo total       : {total_seconds:.2f}s  ({total_seconds / 60:.1f} min)",
            f"  Promedio por ronda : {total_seconds / rounds:.2f}s",
            "",
            "  TIEMPO POR NODO (fit acumulado):",
        ]
        for cid in sorted(self._node_fit_seconds_total):
            tf = self._node_fit_seconds_total[cid]
            ts = self._node_fit_samples_total.get(cid, 0)
            lines.append(
                f"    Node {cid:>2d}: {tf:>8.2f}s  ({tf / 60:.1f} min)  "
                f"{ts} muestras entrenadas"
            )

        if self._round_agg_metrics:
            # Same key best-checkpoint tracking uses when enabled (self._best_metric),
            # else "roc_auc" — this is cosmetic-only (no weights kept for it) unless
            # federated.save_best_checkpoint is on, in which case it matches exactly.
            metric_key = self._best_metric or "roc_auc"
            best = max(
                self._round_agg_metrics, key=lambda r: float(r.get(metric_key, 0.0))
            )
            last = self._round_agg_metrics[-1]
            lines += [
                "",
                "  MÉTRICAS FEDERADAS (promedio ponderado de clientes):",
                f"    Última ronda (#{last['round']:>2d}) — "
                f"AUC={_fmt(last.get('roc_auc'))}  "
                f"F1={_fmt(last.get('f1'))}  "
                f"Loss={_fmt(last.get('loss'))}",
                f"    Mejor ronda  (#{best['round']:>2d}) — "
                f"AUC={_fmt(best.get('roc_auc'))}  "
                f"F1={_fmt(best.get('f1'))}  "
                f"Loss={_fmt(best.get('loss'))}",
            ]
            if self._best_path is not None:
                lines.append(
                    f"    Checkpoint mejor   : {self._best_path}  "
                    f"(ronda {self._best_round}, {self._best_metric}="
                    f"{_fmt(self._best_value)})"
                )
            if self._stopped_early_round is not None:
                lines.append(
                    f"    Parada temprana    : ronda {self._stopped_early_round} "
                    f"(paciencia={self._patience} rondas sin mejorar {self._best_metric})"
                )
            lines += [
                "",
                "  MÉTRICAS COMPLETAS POR RONDA:",
            ]
            for r in self._round_agg_metrics:
                lines.append(
                    f"    Ronda {r['round']:>2d}: "
                    f"AUC={_fmt(r.get('roc_auc'))}  "
                    f"F1={_fmt(r.get('f1'))}  "
                    f"Sens={_fmt(r.get('sensitivity'))}  "
                    f"Spec={_fmt(r.get('specificity'))}  "
                    f"Loss={_fmt(r.get('loss'))}"
                )

        lines += ["", "=" * 62]
        summary_text = "\n".join(lines) + "\n"
        summary_path = self.out_root / "final_summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        _logger.info("\n%s", summary_text)
        _logger.info(
            "Artefactos guardados en %s  (final_summary.txt, per_node_timing.csv, "
            "timing_summary.csv)",
            self.out_root,
        )

    def close(self) -> None:
        for tb in self._client_tb.values():
            tb.close()


__all__ = ["EarlyStoppingTriggered", "NodeMetricsRecorder"]
