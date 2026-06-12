"""Convert between PyTorch state_dicts and Flower's list-of-ndarrays format.

Flower passes parameters as ``list[numpy.ndarray]`` with order determined by
the client. We use ``model.state_dict()`` key order, which is stable for a
given architecture across rounds.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def state_dict_to_ndarrays(model: nn.Module) -> list[np.ndarray]:
    """Return state_dict tensors as numpy arrays in deterministic key order."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def load_ndarrays_to_state_dict(
    model: nn.Module, parameters: list[np.ndarray], *, strict: bool = True
) -> None:
    """Load a Flower ndarrays list back into ``model`` in place."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"Parameter length mismatch: model has {len(keys)} tensors, "
            f"got {len(parameters)} ndarrays."
        )
    new_state = OrderedDict(
        (k, torch.as_tensor(v)) for k, v in zip(keys, parameters, strict=True)
    )
    model.load_state_dict(new_state, strict=strict)


__all__ = ["state_dict_to_ndarrays", "load_ndarrays_to_state_dict"]
