"""Device resolution helpers."""

from __future__ import annotations

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


__all__ = ["resolve_device"]
