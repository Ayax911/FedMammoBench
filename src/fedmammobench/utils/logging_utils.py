"""Stdlib-logging configuration for fedmammobench.

Two entry points:

- :func:`setup_logging` installs a console handler (and optional file handler)
  on the root logger with a consistent format. Call once at program startup.
- :func:`get_logger` returns a namespaced child logger for a module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root logger.

    Safe to call multiple times — pre-existing handlers are cleared first so
    re-invocation (e.g. inside Ray workers) produces a single line per record.

    Args:
        level: Logging level for the root logger (default INFO).
        log_file: Optional file path to also tee logs to.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers — important under Ray, which fork-imports.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence excessively verbose third-party loggers.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Call this in modules instead of ``logging.getLogger``."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
