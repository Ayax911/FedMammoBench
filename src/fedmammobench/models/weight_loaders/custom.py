"""Custom checkpoint loader (arbitrary local .pth files)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from fedmammobench.configs.schema import ModelConfig
from fedmammobench.models.weight_loaders.base import LoadReport
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def _match_state_dict_prefix(
    state: dict[str, Any], model: nn.Module
) -> tuple[dict[str, Any], int]:
    """Normalize a checkpoint's key namespace to match ``model``'s state_dict.

    ``save_checkpoint`` serializes the *full wrapper*, so its keys carry a
    ``backbone.`` prefix (``backbone.conv1.weight``, ...). Other checkpoints may
    be bare (backbone-only: ``conv1.weight``) or carry a DataParallel
    ``module.`` prefix. This helper tries the state_dict as-is and under a few
    prefix transforms, returning whichever maximizes the key overlap with the
    target module, plus the number of keys that were renamed.

    Historic bug this guards against: a ``backbone.``-prefixed checkpoint was
    force-loaded into ``model.backbone`` (bare keys) → 0 tensors matched, and
    with ``strict_load=false`` the model silently kept random init.
    """
    target = set(model.state_dict().keys())

    def overlap(sd: dict[str, Any]) -> int:
        return len(target & set(sd.keys()))

    candidates: list[dict[str, Any]] = [state]
    if any(k.startswith("module.") for k in state):
        candidates.append(
            {k.removeprefix("module."): v for k, v in state.items()}
        )
    # bare-backbone checkpoint → wrapper target
    candidates.append({f"backbone.{k}": v for k, v in state.items()})
    # wrapper checkpoint → bare-backbone target
    if any(k.startswith("backbone.") for k in state):
        candidates.append(
            {k.removeprefix("backbone."): v for k, v in state.items()}
        )

    best = max(candidates, key=overlap)
    if best is state:
        return state, 0
    n_remapped = sum(1 for k in best if k not in state)
    return best, n_remapped


class CustomCheckpointLoader:
    """Load an arbitrary local checkpoint in fedmammobench format.

    The checkpoint must be a ``.pth`` file produced by
    :func:`fedmammobench.utils.checkpoint.save_checkpoint`, i.e. a dict with at
    least a ``"state_dict"`` key.  Because ``save_checkpoint`` serializes the
    full wrapper model, the state_dict is loaded into the wrapper (its keys are
    already ``backbone.``-prefixed); :func:`_match_state_dict_prefix` normalizes
    other layouts (bare-backbone, ``module.``-prefixed) transparently.

    ``cfg.checkpoint_path`` must be set when ``weight_source="custom"``.
    """

    def load(self, model: nn.Module, cfg: ModelConfig) -> LoadReport:
        if not cfg.checkpoint_path:
            raise ValueError(
                "weight_source='custom' requires checkpoint_path to be set. "
                "Set model.checkpoint_path in your YAML config."
            )

        src = Path(cfg.checkpoint_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"No custom checkpoint at {src}")

        payload = torch.load(str(src), map_location="cpu")
        state = (
            payload["state_dict"]
            if isinstance(payload, dict) and "state_dict" in payload
            else payload
        )

        # Normalize the checkpoint key namespace to the target module.
        state, n_remapped = _match_state_dict_prefix(state, model)

        # Drop tensors whose shape is incompatible with the target (e.g. a head
        # sized for a different num_classes). They are recorded as shape
        # mismatches and the model's fresh head is kept — the intended
        # warm-start behavior (the head is retrained anyway).
        model_sd = model.state_dict()
        shape_mismatches: list[tuple[str, tuple, tuple]] = []
        compatible: dict[str, Any] = {}
        for k, v in state.items():
            if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape):
                shape_mismatches.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            compatible[k] = v

        missing, unexpected = model.load_state_dict(compatible, strict=False)
        # Count the actual key intersection rather than `len(model_sd) -
        # len(missing)`: PyTorch's BatchNorm._load_from_state_dict silently
        # omits missing `num_batches_tracked` buffers from `missing_keys` (a
        # backward-compat carve-out for pre-0.4.1 checkpoints), which would
        # otherwise make an all-garbage checkpoint look partially loaded.
        n_loaded = len(set(compatible.keys()) & set(model_sd.keys()))
        if n_loaded == 0:
            raise RuntimeError(
                f"custom checkpoint {src} matched 0/{len(model_sd)} tensors "
                f"(key-namespace mismatch). Nothing was loaded — refusing to "
                f"train from random init. Check the checkpoint format."
            )

        # Honor strict_load: with strict, any residual mismatch is a hard error
        # (this is what makes a future prefix/shape regression loud, not silent).
        if cfg.strict_load and (missing or unexpected or shape_mismatches):
            raise RuntimeError(
                f"custom checkpoint {src} loaded with strict_load=True but had "
                f"missing={list(missing)} unexpected={list(unexpected)} "
                f"shape_mismatches={shape_mismatches}"
            )

        _logger.info(
            "Loaded custom checkpoint from %s (%d/%d tensors, remapped=%d, "
            "missing=%d unexpected=%d shape_mismatches=%d)",
            src,
            n_loaded,
            len(model_sd),
            n_remapped,
            len(missing),
            len(unexpected),
            len(shape_mismatches),
        )
        return LoadReport(
            source="custom",
            arch=cfg.name,
            missing_keys=list(missing),
            unexpected_keys=list(unexpected),
            remapped_keys=n_remapped,
            shape_mismatches=shape_mismatches,
            checkpoint_uri=str(src),
        )


__all__ = ["CustomCheckpointLoader"]
