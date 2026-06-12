"""Reproducibility helpers.

Bit-level reproducibility across machines is not guaranteed (cuDNN selects
algorithms that can vary by hardware), but identical hardware + identical
seed should reproduce metrics to within numerical tolerance.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed shared by all RNGs.
        deterministic: If True, set ``torch.backends.cudnn.deterministic`` and
            disable cuDNN benchmarking. This slows down convolutions on GPU
            but stabilizes runs across reruns.

    Notes:
        The ``PYTHONHASHSEED`` environment variable must be set *before* the
        Python process starts to affect hash randomization for the current
        process. We set it anyway so child processes (e.g. Ray workers
        spawned later) pick it up.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # use_deterministic_algorithms can raise on unsupported ops; we set
        # ``warn_only`` to avoid surprises in research code.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


__all__ = ["set_global_seed"]
