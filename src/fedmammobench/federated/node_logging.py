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


class NodeMetricsRecorder:
    """Owns per-node CSV/TB sinks, timing, and the latest global parameters."""

    def __init__(self, out_root: str | Path, tb_writer: TensorBoardWriter | None = None) -> None:
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
            tb.log_scalars(
                "fit",
                {k: v for k, v in row.items() if isinstance(v, float)},
                server_round,
            )
            tb.flush()
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
            tb.log_scalars(
                "eval",
                {k: v for k, v in row.items() if isinstance(v, float)},
                server_round,
            )
            tb.flush()

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
            aggregated, metrics = orig_fit(server_round, results, failures)
            if aggregated is not None:
                self._latest_parameters = aggregated
            self.record_fit(server_round, results)
            return aggregated, metrics

        def eval_wrapped(server_round, results, failures):  # noqa: ANN001, ANN202
            out = orig_eval(server_round, results, failures)
            self.record_eval(server_round, results)
            return out

        strategy.aggregate_fit = fit_wrapped  # type: ignore[method-assign]
        strategy.aggregate_evaluate = eval_wrapped  # type: ignore[method-assign]

    # -- finalization -----------------------------------------------------

    def save_global_model(
        self, cfg: ExperimentConfig, path: str | Path | None = None
    ) -> Path | None:
        """Materialize the latest aggregated parameters into a checkpoint.

        Returns the path written, or None if no parameters were captured (e.g.
        the run produced zero successful fit rounds).
        """
        if self._latest_parameters is None:
            _logger.warning(
                "No aggregated parameters were captured; global model not saved. "
                "Did any fit round succeed?"
            )
            return None
        ndarrays = parameters_to_ndarrays(self._latest_parameters)
        device = resolve_device(cfg.device)
        model = build_model(cfg.model).to(device)
        load_ndarrays_to_state_dict(model, ndarrays, strict=True)
        out = Path(path) if path is not None else (self.out_root / "global_model.pt")
        save_checkpoint(
            out,
            model,
            epoch=cfg.federated.rounds,
            extra={"source": "federated_global", "rounds": cfg.federated.rounds},
        )
        _logger.info(
            "Saved final global model to %s — verify it with "
            "`fedmammobench-evaluate --checkpoint %s` on a held-out test set.",
            out,
            out,
        )
        return out

    def write_timing_summary(self, cfg: ExperimentConfig, total_seconds: float) -> None:
        """Write timing summary CSVs and a human-readable final_summary.txt."""
        rounds = max(int(cfg.federated.rounds), 1)

        # 1. Overall timing CSV
        summary = CSVLogger(self.out_root / "timing_summary.csv")
        summary.append(
            {
                "total_seconds": round(total_seconds, 3),
                "total_minutes": round(total_seconds / 60, 2),
                "num_rounds": cfg.federated.rounds,
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

        lines: list[str] = [
            "=" * 62,
            f"  RESUMEN FINAL — {cfg.name}",
            "=" * 62,
            f"  Estrategia         : {cfg.federated.strategy.name}",
            f"  Rondas completadas : {cfg.federated.rounds}",
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
            best = max(
                self._round_agg_metrics, key=lambda r: float(r.get("roc_auc", 0.0))
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


__all__ = ["NodeMetricsRecorder"]
