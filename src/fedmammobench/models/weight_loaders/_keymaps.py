"""Key-remapping tables for RadImageNet checkpoints.

RadImageNet checkpoints (BMEII-AI/RadImageNet) were trained with standard
torchvision architectures, so key names are largely identical.  The main
transformations needed are:

1. Strip ``module.`` prefix added by ``torch.nn.DataParallel`` training.
2. Drop classification head keys (not transferred to the target task).

Functions
---------
remap_radimagenet_keys(state_dict, arch)
    Strip DataParallel prefix + apply any arch-specific renames.
    Returns ``(new_dict, n_remapped)`` where ``n_remapped`` is the number of
    keys that were changed (for traceability in :class:`LoadReport`).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Supported architectures and their head-key prefixes
# ---------------------------------------------------------------------------

#: Architectures published by the RadImageNet project.
SUPPORTED_ARCHS: frozenset[str] = frozenset({"resnet50", "densenet121", "inception_v3"})

#: State-dict key prefixes that belong to the replaced classification head.
#: Keys matching any of these prefixes are dropped during loading.
_HEAD_PREFIXES_BY_ARCH: dict[str, tuple[str, ...]] = {
    "resnet50":     ("fc.",),
    "densenet121":  ("classifier.",),
    "inception_v3": ("fc.", "AuxLogits."),
}

# ---------------------------------------------------------------------------
# First conv key per arch (RadImageNet state dict, post-prefix-strip)
# ---------------------------------------------------------------------------

#: Key of the first conv weight in the RadImageNet state dict (after stripping
#: any DataParallel ``module.`` prefix).  Used by the loader to adapt the
#: pretrained 3-channel weight to ``cfg.in_channels``.
FIRST_CONV_KEY: dict[str, str] = {
    "resnet50":     "conv1.weight",
    "densenet121":  "features.conv0.weight",
    "inception_v3": "Conv2d_1a_3x3.conv.weight",
}

# ---------------------------------------------------------------------------
# Public remapping function
# ---------------------------------------------------------------------------


def remap_radimagenet_keys(
    state_dict: dict[str, Any],
    arch: str,
) -> tuple[dict[str, Any], int]:
    """Strip DataParallel prefix and drop classification head keys.

    Args:
        state_dict: Raw state dict loaded from a RadImageNet ``.pth`` file.
        arch: Architecture name (must be in :data:`SUPPORTED_ARCHS`).

    Returns:
        ``(new_state_dict, n_remapped)`` — the cleaned dict and the count of
        keys that were renamed (prefix stripped).  Head keys are dropped and
        **not** counted in ``n_remapped`` (they are discarded, not renamed).
    """
    head_prefixes = _HEAD_PREFIXES_BY_ARCH.get(arch, ())
    new_dict: dict[str, Any] = {}
    n_remapped = 0

    for key, value in state_dict.items():
        # 1. Strip DataParallel prefix
        clean_key = key
        if key.startswith("module."):
            clean_key = key[len("module."):]
            n_remapped += 1

        # 2. Drop head keys
        if any(clean_key.startswith(p) for p in head_prefixes):
            continue

        new_dict[clean_key] = value

    return new_dict, n_remapped


__all__ = [
    "SUPPORTED_ARCHS",
    "FIRST_CONV_KEY",
    "remap_radimagenet_keys",
]
