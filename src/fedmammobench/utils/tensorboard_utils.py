"""Thin wrapper around ``torch.utils.tensorboard.SummaryWriter``.

The wrapper degrades gracefully if TensorBoard isn't installed — useful when
running inside very lean Docker images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)


class TensorBoardWriter:
    """Optional TensorBoard SummaryWriter wrapper.

    If TensorBoard is unavailable, all methods become no-ops so callers don't
    need to special-case it.
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self._enabled = enabled
        self._writer: Any | None = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            path = Path(log_dir).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(path))
        except ImportError:  # pragma: no cover
            _logger.warning("TensorBoard not available; TensorBoardWriter is a no-op.")
            self._enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar(tag, float(value), step)

    def log_scalars(self, prefix: str, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return
        for key, value in metrics.items():
            try:
                self._writer.add_scalar(f"{prefix}/{key}", float(value), step)
            except (TypeError, ValueError):
                # Skip non-numeric metrics rather than crash a long run.
                _logger.debug("Skipping non-numeric metric %s=%r", key, value)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "TensorBoardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["TensorBoardWriter"]
