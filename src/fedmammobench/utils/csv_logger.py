"""Append-only CSV metrics logger.

Writes a header on first append, then one row per call. Designed to be
robust to crashes mid-experiment: every row is flushed immediately.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    """Streaming CSV logger.

    The set of columns is locked once the first row is appended; subsequent
    rows with extra keys raise :class:`ValueError`. This is a feature, not a
    bug — silently dropping metrics is a worse failure mode in research.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        # If the file exists and has content, recover the header.
        if self._path.is_file() and self._path.stat().st_size > 0:
            with self._path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fieldnames = header

    def append(self, row: dict[str, Any]) -> None:
        """Append a single row. Header is written on the first call."""
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self._path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        unknown = set(row.keys()) - set(self._fieldnames)
        if unknown:
            raise ValueError(
                f"CSVLogger at {self._path}: row contains unknown fields {sorted(unknown)}. "
                f"Existing columns are {self._fieldnames}."
            )
        # Fill missing keys with empty strings to keep columns aligned.
        complete = {k: row.get(k, "") for k in self._fieldnames}
        with self._path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(complete)

    @property
    def path(self) -> Path:
        return self._path


__all__ = ["CSVLogger"]
