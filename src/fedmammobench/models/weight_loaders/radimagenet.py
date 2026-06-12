"""RadImageNet weight loader.

Loads checkpoints from the RadImageNet project (BMEII-AI/RadImageNet) into a
fedmammo model backbone.  Checkpoints must be obtained manually from the
official repository and placed in one of two locations (see :meth:`_resolve_path`).

Supported architectures: ``resnet50``, ``densenet121``, ``inception_v3``.

Checkpoint file naming convention expected on disk::

    RadImageNet-{arch}.pth   (e.g. ``RadImageNet-resnet50.pth``)

Where to place them (in priority order):

1. ``cfg.model.checkpoint_path`` — absolute path to the ``.pth`` file.
2. The directory pointed to by the ``$FEDMAMMO_RADIMAGENET_DIR`` environment
   variable, named ``RadImageNet-{arch}.pth``.

Neither location is hard-coded; the loader raises :class:`FileNotFoundError`
with an actionable message if neither resolves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fedmammo.configs.schema import ModelConfig
from fedmammo.models._adapt import adapt_weight_tensor
from fedmammo.models.weight_loaders._keymaps import (
    FIRST_CONV_KEY,
    SUPPORTED_ARCHS,
    remap_radimagenet_keys,
)
from fedmammo.models.weight_loaders.base import LoadReport
from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)

_ENV_VAR = "FEDMAMMO_RADIMAGENET_DIR"


class RadImageNetLoader:
    """Load RadImageNet pretrained weights into a model's backbone.

    The loader:

    1. Resolves the checkpoint path from ``cfg.checkpoint_path`` or
       ``$FEDMAMMO_RADIMAGENET_DIR``.
    2. Unwraps nested dicts (``{state_dict: ...}`` or
       ``{model_state_dict: ...}`` wrappers).
    3. Strips DataParallel ``module.`` prefixes and drops head keys.
    4. Adapts the first-conv weight to ``cfg.in_channels`` using the
       *sum_preserving* strategy if a channel mismatch is detected.
    5. Loads into ``model.backbone`` with ``strict=False`` and logs the report.
    """

    def load(self, model: nn.Module, cfg: ModelConfig) -> LoadReport:
        arch = cfg.name.lower()
        if arch not in SUPPORTED_ARCHS:
            raise ValueError(
                f"RadImageNet weights are not published for {arch!r}. "
                f"Supported architectures: {sorted(SUPPORTED_ARCHS)}. "
                f"For {arch!r}, use weight_source='imagenet' or weight_source='none'."
            )

        path = self._resolve_path(cfg)
        _logger.info("Loading RadImageNet checkpoint: %s", path)

        raw = torch.load(str(path), map_location="cpu")
        state = self._unwrap(raw)
        state, n_remap = remap_radimagenet_keys(state, arch)
        state = self._adapt_first_conv(state, arch, cfg.in_channels)

        backbone = getattr(model, "backbone", model)
        missing, unexpected = backbone.load_state_dict(state, strict=False)

        # Filter non-head missing keys for the warning threshold.
        head_missing = [k for k in missing if _is_head_key(k, arch)]
        non_head_missing = [k for k in missing if not _is_head_key(k, arch)]
        if non_head_missing:
            _logger.warning(
                "RadImageNet load [%s]: unexpected missing keys: %s",
                arch, non_head_missing,
            )
        if unexpected:
            _logger.debug(
                "RadImageNet load [%s]: discarded unexpected keys: %s",
                arch, unexpected,
            )

        _logger.info(
            "Loaded RadImageNet weights for %s (in_channels=%d, remapped=%d, "
            "missing=%d [%d head], unexpected=%d)",
            arch, cfg.in_channels, n_remap,
            len(missing), len(head_missing), len(unexpected),
        )
        return LoadReport(
            source="radimagenet",
            arch=arch,
            missing_keys=missing,
            unexpected_keys=unexpected,
            remapped_keys=n_remap,
            checkpoint_uri=str(path),
        )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, cfg: ModelConfig) -> Path:
        """Resolve the checkpoint file, raising a clear error if not found."""
        arch = cfg.name.lower()
        filename = f"RadImageNet-{arch}.pth"

        # 1. Explicit cfg path
        if cfg.checkpoint_path:
            p = Path(cfg.checkpoint_path).expanduser()
            if p.is_file():
                return p
            raise FileNotFoundError(
                f"RadImageNet checkpoint not found at cfg.model.checkpoint_path="
                f"{str(p)!r}. Verify the path is correct."
            )

        # 2. Environment variable directory
        env_dir = os.environ.get(_ENV_VAR)
        if env_dir:
            p = Path(env_dir).expanduser() / filename
            if p.is_file():
                return p
            raise FileNotFoundError(
                f"RadImageNet checkpoint not found: {str(p)!r}. "
                f"${_ENV_VAR} is set to {env_dir!r} but {filename!r} "
                f"was not found there."
            )

        raise FileNotFoundError(
            f"RadImageNet checkpoint for {arch!r} not found. "
            f"Download {filename!r} from the official RadImageNet repository "
            f"(BMEII-AI/RadImageNet on GitHub) and either:\n"
            f"  (a) Set cfg.model.checkpoint_path to the absolute file path, or\n"
            f"  (b) Set the ${_ENV_VAR} environment variable to the directory "
            f"containing {filename!r}.\n"
            f"See docs/TRANSFER_LEARNING_GUIDE.md for step-by-step instructions."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(raw: Any) -> dict[str, Any]:
        """Unwrap checkpoint containers to a plain state dict."""
        if isinstance(raw, dict):
            # Common wrappers: {state_dict: ...}, {model_state_dict: ...}
            for key in ("state_dict", "model_state_dict", "model"):
                if key in raw and isinstance(raw[key], dict):
                    return raw[key]
        # Already a plain state dict
        return raw

    @staticmethod
    def _adapt_first_conv(
        state: dict[str, Any], arch: str, target_in_channels: int
    ) -> dict[str, Any]:
        """Adapt the first-conv weight to ``target_in_channels`` if needed."""
        if target_in_channels == 3:
            return state
        key = FIRST_CONV_KEY.get(arch)
        if key is None or key not in state:
            return state
        w = state[key]
        adapted = adapt_weight_tensor(w, target_in_channels, strategy="sum_preserving")
        state = dict(state)
        state[key] = adapted
        return state


# ---------------------------------------------------------------------------
# Head-key classifier (used for warning suppression)
# ---------------------------------------------------------------------------

_HEAD_PREFIXES_ALL = ("fc.", "classifier.", "AuxLogits.")


def _is_head_key(key: str, arch: str) -> bool:
    return any(key.startswith(p) for p in _HEAD_PREFIXES_ALL)


__all__ = ["RadImageNetLoader"]
