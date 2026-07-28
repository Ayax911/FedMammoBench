"""Thin, never-raising wrapper around the ``wandb`` SDK.

Mirrors :class:`~fedmammobench.utils.tensorboard_utils.TensorBoardWriter`'s
contract exactly (same 4-method surface: ``log_scalar``, ``log_scalars``,
``flush``, ``close``) so it can be dropped in as a second sink anywhere a
``TensorBoardWriter`` is used today — see
:mod:`fedmammobench.utils.metrics_sink` for the fan-out that does that.

``WandbConfig.enabled`` defaults to True (``configs/wandb_config.py``), so a
``WandbWriter`` is constructed on **every** training run, including keyless
Docker containers with no network egress restrictions but no credentials
either. Every failure mode below degrades to "keep training, no W&B data
this run" — never a hang, never an interactive prompt, never a raised
exception out of any public method.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# After this many consecutive log() failures mid-run (e.g. network dropped),
# stop trying — avoids spamming the log once per epoch/round for the rest of
# a long run.
_MAX_LOG_FAILURES = 10


class WandbWriter:
    """Optional Weights & Biases run wrapper. Never raises, never prompts.

    Degradation ladder (each step only reached if the previous succeeded):

    1. ``enabled=False`` or ``mode="disabled"`` -> return immediately,
       ``wandb`` is never imported at all.
    2. ``wandb`` not installed (``ImportError``) -> log one INFO line, no-op.
    3. ``mode="online"`` but no ``WANDB_API_KEY`` env var (empty string
       counts as absent — that's what ``-e WANDB_API_KEY="${WANDB_API_KEY:-}"``
       passes when the host var is unset) and no cached credentials
       (``~/.netrc`` via ``wandb.api.api_key``) -> downgrade to
       ``mode="offline"`` *before* ever calling ``wandb.init`` — this is what
       guarantees no network call and no prompt on a keyless container.
    4. ``wandb.init`` raises (bad entity, network down, ...) -> retry once in
       offline mode; if that also raises, disable for this run.
    5. A later ``run.log()`` call raises -> swallowed; writer disables
       itself after ``_MAX_LOG_FAILURES`` consecutive failures.

    ``WANDB_SILENT`` and ``WANDB_CONSOLE=off`` are forced (via
    ``os.environ.setdefault``, so an operator's explicit setting wins) before
    ``import wandb`` — wandb's default ``console="auto"`` wraps
    ``sys.stdout``, which would corrupt the plain-text readiness greps
    Docker deploy scripts run against container logs
    (``docker logs ... | grep "gRPC server running"``).
    """

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | tuple[str, ...] = (),
        mode: str = "online",
        enabled: bool = True,
        config: dict[str, Any] | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self._run: Any = None
        self._fail_count = 0

        if not enabled or mode == "disabled":
            return

        # Env overrides config — lets an operator force offline/disabled
        # without touching YAML (e.g. `WANDB_MODE=offline` on the host).
        mode = os.environ.get("WANDB_MODE") or mode

        # Must be set BEFORE `import wandb` — see class docstring.
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")

        try:
            import wandb
        except ImportError:
            _logger.info("wandb not installed; W&B logging disabled for this run (pip install wandb).")
            return

        if mode == "online" and not self._has_credentials(wandb):
            _logger.info(
                "No WANDB_API_KEY and no cached wandb credentials; falling back to "
                "offline mode. Sync later with `wandb sync %s/wandb/offline-run-*`.",
                log_dir,
            )
            mode = "offline"

        log_dir_path = Path(log_dir).expanduser().resolve() if log_dir is not None else None
        if log_dir_path is not None:
            log_dir_path.mkdir(parents=True, exist_ok=True)

        def _init(m: str) -> Any:
            return wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                tags=list(tags) or None,
                config=config,
                dir=str(log_dir_path) if log_dir_path is not None else None,
                mode=m,
                reinit=False,
                settings=wandb.Settings(
                    init_timeout=30,  # bounds a hung handshake to a keyless/unreachable host
                    silent=True,
                    console="off",
                    save_code=False,
                    disable_git=True,
                ),
            )

        try:
            self._run = _init(mode)
        except Exception:
            _logger.warning("wandb.init(mode=%s) failed; retrying offline.", mode, exc_info=True)
            try:
                self._run = _init("offline")
            except Exception:
                _logger.warning(
                    "wandb.init(offline) also failed; W&B logging disabled for this run.",
                    exc_info=True,
                )
                self._run = None

    @staticmethod
    def _has_credentials(wandb_module: Any) -> bool:
        if os.environ.get("WANDB_API_KEY", "").strip():
            return True
        try:
            return bool(wandb_module.api.api_key)  # reads ~/.netrc / wandb settings file
        except Exception:
            return False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars("", {tag: value}, step)

    def log_scalars(self, prefix: str, metrics: dict[str, float], step: int) -> None:
        if self._run is None:
            return
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                payload[f"{prefix}/{key}" if prefix else key] = float(value)
            except (TypeError, ValueError):
                # Same policy as TensorBoardWriter: skip non-numeric metrics
                # rather than crash a long run.
                continue
        if not payload:
            return
        try:
            self._run.log(payload, step=int(step))
            self._fail_count = 0
        except Exception:
            self._fail_count += 1
            if self._fail_count >= _MAX_LOG_FAILURES:
                _logger.warning(
                    "wandb.log failed %d times in a row; disabling W&B for the rest of this run.",
                    self._fail_count,
                )
                self._run = None
            else:
                _logger.debug("wandb.log failed (%d/%d)", self._fail_count, _MAX_LOG_FAILURES, exc_info=True)

    def flush(self) -> None:
        # wandb batches and flushes internally on its own thread; nothing to do.
        return

    def close(self) -> None:
        if self._run is None:
            return
        run, self._run = self._run, None  # idempotent — a second close() is a no-op
        try:
            run.finish()
        except Exception:
            _logger.debug("wandb run.finish() failed", exc_info=True)

    def __enter__(self) -> WandbWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["WandbWriter"]
