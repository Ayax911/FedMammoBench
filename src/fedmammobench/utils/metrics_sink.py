"""Fan-out metric writer + the ``ExperimentConfig -> writers`` factory.

Every existing consumer of a ``tb_writer`` (``Trainer``, ``server.py``,
``node_logging.py``, ``server_training.py``) already only calls
``log_scalar``/``log_scalars``/``flush``/``close`` — it duck-types, never
checks ``isinstance(..., TensorBoardWriter)``. That means a "log to N
sinks at once" fan-out can be dropped in as *the* writer at each
construction site with **zero signature changes** anywhere else: pass a
:class:`MetricSink` in place of the bare ``TensorBoardWriter`` and every
downstream ``tb_writer.log_scalars(...)`` call now reaches all of them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from fedmammobench.utils.logging_utils import get_logger
from fedmammobench.utils.tensorboard_utils import TensorBoardWriter
from fedmammobench.utils.wandb_utils import WandbWriter

if TYPE_CHECKING:
    from fedmammobench.configs.schema import ExperimentConfig

_logger = get_logger(__name__)


class MetricWriter(Protocol):
    """Structural type satisfied by TensorBoardWriter, WandbWriter, and MetricSink."""

    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_scalars(self, prefix: str, metrics: dict[str, float], step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class MetricSink:
    """Fans a metric call out to N writers. One failing child never affects the rest."""

    def __init__(self, writers: Sequence[MetricWriter]) -> None:
        self._writers = list(writers)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._each("log_scalar", tag, value, step)

    def log_scalars(self, prefix: str, metrics: dict[str, float], step: int) -> None:
        self._each("log_scalars", prefix, metrics, step)

    def flush(self) -> None:
        self._each("flush")

    def close(self) -> None:
        # Close ALL children even if one raises — otherwise a broken
        # TensorBoardWriter could leave a wandb run permanently "running".
        self._each("close")

    def _each(self, method: str, *args: Any) -> None:
        for w in self._writers:
            try:
                getattr(w, method)(*args)
            except Exception:
                _logger.debug("MetricSink: %s.%s failed", type(w).__name__, method, exc_info=True)

    def __enter__(self) -> MetricSink:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def build_metric_sink(cfg: ExperimentConfig, out_root: str | Path, *, job_type: str) -> MetricSink:
    """Build the standard sink: always TensorBoard, plus W&B when ``cfg.wandb.enabled``.

    Args:
        cfg: Full experiment config (used for ``cfg.wandb.*`` and to log the
            resolved config as the wandb run's ``config`` — safe to dump in
            full since ``WandbConfig`` never carries a secret field).
        out_root: The run's output directory (``<output_dir>/<name>``).
            TensorBoard writes to ``out_root/tb``; wandb writes to
            ``cfg.wandb.log_dir`` or ``out_root/wandb``.
        job_type: Short label distinguishing entry points in the wandb UI
            (e.g. ``"centralized"``, ``"fl-server-sim"``, ``"fl-server-grpc"``).
    """
    out_root = Path(out_root)
    writers: list[MetricWriter] = [TensorBoardWriter(out_root / "tb")]

    wb = getattr(cfg, "wandb", None)
    if wb is not None and wb.enabled:
        writers.append(
            WandbWriter(
                project=wb.project,
                entity=wb.entity,
                run_name=wb.run_name or cfg.name,
                group=wb.group or cfg.name,
                tags=[*wb.tags, job_type, cfg.mode],
                mode=wb.mode,
                enabled=True,
                config=dataclasses.asdict(cfg),
                log_dir=Path(wb.log_dir) if wb.log_dir else (out_root / "wandb"),
            )
        )

    return MetricSink(writers)


__all__ = ["MetricSink", "MetricWriter", "build_metric_sink"]
