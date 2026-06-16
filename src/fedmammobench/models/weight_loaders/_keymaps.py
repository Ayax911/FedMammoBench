"""Key-remapping tables for RadImageNet checkpoints.

RadImageNet checkpoints (BMEII-AI/RadImageNet) were trained with standard
torchvision architectures, so key names are largely identical.  The main
transformations needed are:

1. Strip ``module.`` prefix added by ``torch.nn.DataParallel`` training.
2. Remap ``backbone.N`` Sequential indices to torchvision named children.
   The official RadImageNet ResNet checkpoints wrap the backbone as a
   Sequential, so keys are ``backbone.0.weight`` instead of
   ``conv1.weight``.  Without this remapping, ``load_state_dict`` with
   ``strict=False`` silently loads **nothing** because no key matches.
3. Drop classification head keys (not transferred to the target task).

Functions
---------
remap_radimagenet_keys(state_dict, arch)
    Strip DataParallel prefix + apply arch-specific backbone renames.
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
# backbone.N → torchvision named-child remapping per arch
# ---------------------------------------------------------------------------

# The official RadImageNet ResNet .pth files store the feature extractor as
# backbone = nn.Sequential([conv1, bn1, relu, maxpool, layer1..4, avgpool]).
# Indices without learned weights (relu=2, maxpool=3, avgpool=8) are absent
# from the state dict, so only the six entries below need remapping.
_BACKBONE_SEQUENTIAL_REMAP: dict[str, dict[str, str]] = {
    "resnet50": {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
    },
    "resnet18": {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
    },
}

# ---------------------------------------------------------------------------
# First conv key per arch (RadImageNet state dict, post-full-remap)
# ---------------------------------------------------------------------------

#: Key of the first conv weight **after** all remapping steps.
#: Used by the loader to adapt the pretrained 3-channel weight to
#: ``cfg.in_channels``.
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
    """Strip DataParallel prefix, remap backbone Sequential indices, drop head.

    Args:
        state_dict: Raw state dict loaded from a RadImageNet ``.pth`` file.
        arch: Architecture name (must be in :data:`SUPPORTED_ARCHS`).

    Returns:
        ``(new_state_dict, n_remapped)`` — the cleaned dict and the count of
        keys that were renamed (prefix stripped or backbone index remapped).
        Head keys are dropped and **not** counted in ``n_remapped``.
    """
    head_prefixes = _HEAD_PREFIXES_BY_ARCH.get(arch, ())
    backbone_remap = _BACKBONE_SEQUENTIAL_REMAP.get(arch, {})
    new_dict: dict[str, Any] = {}
    n_remapped = 0

    for key, value in state_dict.items():
        clean_key = key

        # 1. Strip DataParallel prefix
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
            n_remapped += 1

        # 2. Remap backbone.N Sequential indices → torchvision named children
        for old_prefix, new_prefix in backbone_remap.items():
            if clean_key.startswith(old_prefix):
                clean_key = new_prefix + clean_key[len(old_prefix):]
                n_remapped += 1
                break

        # 3. Drop head keys
        if any(clean_key.startswith(p) for p in head_prefixes):
            continue

        new_dict[clean_key] = value

    return new_dict, n_remapped


__all__ = [
    "SUPPORTED_ARCHS",
    "FIRST_CONV_KEY",
    "remap_radimagenet_keys",
]
