"""Device resolution helpers."""

from __future__ import annotations

import logging
import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a string preference (``auto``/``cpu``/``cuda``) to a torch.device.

    ``auto`` picks CUDA when available, else CPU. Explicit ``cuda`` raises if
    CUDA isn't available — fail fast rather than silently fall back.
    """
    pref = preference.lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return torch.device("cuda")
    raise ValueError(f"Unknown device preference: {preference!r} (expected auto/cpu/cuda)")


def log_device_info(device: torch.device, logger: logging.Logger) -> None:
    """Log which device is active. Makes it easy to confirm GPU usage in docker logs."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_mb = props.total_memory / 1024 ** 2
        logger.info(
            "*** DEVICE: %s (%s, %.0f MB) — GPU ACTIVA ***",
            device,
            name,
            total_mb,
        )
    else:
        if torch.cuda.is_available():
            logger.warning(
                "*** DEVICE: cpu — CUDA disponible pero NO seleccionada "
                "(revisa cfg.device o el flag --gpus del contenedor) ***"
            )
        else:
            logger.warning(
                "*** DEVICE: cpu — CUDA no disponible en este contenedor "
                "(imagen CPU-only o falta --gpus / --runtime=nvidia) ***"
            )


__all__ = ["resolve_device", "log_device_info"]
