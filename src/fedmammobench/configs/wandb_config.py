"""Weights & Biases experiment-tracking configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

_VALID_MODES = ("online", "offline", "disabled")


@dataclass
class WandbConfig:
    """Weights & Biases logging settings.

    ONE run per experiment, server-side only: in federated mode only the
    server process logs (aggregated metrics + one ``node_<id>/...`` series
    per client); FL client containers never construct a wandb run. See
    ``fedmammobench.utils.wandb_utils.WandbWriter`` for the degradation
    behavior when credentials/network/the package itself are unavailable —
    by design this can never block or prompt, since ``enabled`` defaults to
    True and every training run (including keyless Docker containers)
    constructs one of these.

    SECURITY: this dataclass must never gain an ``api_key`` (or similar
    secret) field. ``save_config`` (configs/loader.py) snapshots the fully
    resolved config to ``runs/<name>/config.snapshot.yaml`` on disk, so any
    secret placed here would be written out and synced with the rest of the
    run's artifacts. The key is read exclusively from the ``WANDB_API_KEY``
    environment variable at run time.

    Attributes:
        enabled: Master switch. True by default — set False (or
            ``mode: disabled``) to fully skip constructing a wandb run.
        project: W&B project name.
        entity: W&B entity (team/user). None uses the API key's default.
        run_name: Display name for the run. None defaults to ``cfg.name``.
        group: Groups related runs in the W&B UI. None defaults to
            ``cfg.name`` (so a run and its later re-evaluations group
            together).
        tags: Free-form tags attached to the run.
        mode: ``online`` (default, log to wandb.ai — downgrades to offline
            automatically if no credentials are found, see WandbWriter),
            ``offline`` (always write local-only, sync later with
            ``wandb sync``), or ``disabled`` (same as ``enabled: false``).
        log_dir: Directory wandb writes its local run files under. None
            defaults to ``<output_dir>/<name>/wandb``.
    """

    enabled: bool = True
    project: str = "fedmammobench"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] = "online"
    log_dir: str | None = None

    def validate(self) -> None:
        """Raise ValueError for invalid wandb settings."""
        if self.mode not in _VALID_MODES:
            raise ValueError(f"wandb.mode must be one of {_VALID_MODES}, got {self.mode!r}")
        if self.enabled and self.mode != "disabled" and not self.project:
            raise ValueError("wandb.project must be non-empty when wandb.enabled is True")


__all__ = ["WandbConfig"]
