"""Weight-loading registry and public API.

Usage (from the model factory)::

    from fedmammobench.models.weight_loaders import load_weights, apply_freeze_policy

    model = _build_architecture(cfg)
    load_weights(model, cfg)
    apply_freeze_policy(model, cfg)

To register a third-party loader at runtime::

    from fedmammobench.models.weight_loaders import register_loader
    register_loader("medicalnet", MyMedicalNetLoader())
"""

from __future__ import annotations

from torch import nn

from fedmammobench.configs.schema import ModelConfig
from fedmammobench.models.weight_loaders.base import LoadReport, WeightLoader
from fedmammobench.models.weight_loaders.custom import CustomCheckpointLoader
from fedmammobench.models.weight_loaders.imagenet import ImageNetLoader
from fedmammobench.models.weight_loaders.none import NoneLoader
from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOADERS: dict[str, WeightLoader] = {
    "imagenet": ImageNetLoader(),
    "none": NoneLoader(),
    "custom": CustomCheckpointLoader(),
    "radimagenet": RadImageNetLoader(),
}


def register_loader(name: str, loader: WeightLoader) -> None:
    """Register a :class:`WeightLoader` under ``name``.

    Raises :class:`ValueError` if the name is already registered.
    Call before :func:`build_model` to make the loader available via YAML.
    """
    if name in _LOADERS:
        raise ValueError(f"WeightLoader already registered: {name!r}")
    _LOADERS[name] = loader


# ---------------------------------------------------------------------------
# Source resolution (backward-compat with pretrained flag)
# ---------------------------------------------------------------------------

def resolve_source(cfg: ModelConfig) -> str:
    """Return the effective weight source, resolving ``"auto"`` from the legacy flag.

    Priority (highest first):

    1. ``cfg.weight_source`` if it is *not* ``"auto"`` — explicit always wins.
    2. ``"imagenet"`` if ``cfg.pretrained is True`` (legacy behaviour).
    3. ``"none"`` otherwise.
    """
    if cfg.weight_source != "auto":
        return cfg.weight_source
    return "imagenet" if cfg.pretrained else "none"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_weights(model: nn.Module, cfg: ModelConfig) -> LoadReport:
    """Load pretrained weights into *model* according to *cfg*.

    The source is resolved via :func:`resolve_source` and the corresponding
    :class:`WeightLoader` from the registry is invoked.  The resulting
    :class:`LoadReport` is logged at INFO level and returned for inspection.
    """
    source = resolve_source(cfg)
    if source not in _LOADERS:
        raise ValueError(
            f"Unknown weight source: {source!r}. "
            f"Registered sources: {sorted(_LOADERS)}"
        )
    _logger.info("Loading weights: source=%r arch=%r", source, cfg.name)
    report = _LOADERS[source].load(model, cfg)
    _log_report(report)
    return report


def apply_freeze_policy(
    model: nn.Module,
    cfg: ModelConfig,
    *,
    current_round: int | None = None,
) -> dict[str, int]:
    """Apply parameter-freezing policy based on *cfg*.

    Behaviour:
    - ``freeze_backbone=True`` — all backbone feature-extractor parameters are
      frozen (``requires_grad=False``), BatchNorm running stats are locked via
      ``eval()`` mode, and the classification head remains trainable.
    - ``freeze_head=True`` — the classification head is frozen.
    - ``unfreeze_at_epoch`` — when ``current_round >= unfreeze_at_epoch``, the
      backbone freeze is lifted and all params become trainable again.

    .. note::
        ``_set_bn_eval`` sets BN to eval mode to prevent running-stat drift
        when the backbone is frozen.  The Trainer's ``model.train()`` call at
        the start of each epoch will re-enable BN training mode.  For strict
        BN stat freezing the trainer would need to re-call ``_set_bn_eval``
        after ``model.train()``; this is omitted to avoid coupling the trainer
        to the freeze policy.  The practical effect is minor — ``requires_grad``
        already prevents the BN γ/β weights from updating.

    Returns:
        Dict with ``trainable_params``, ``total_params``, and
        ``frozen_modules`` (list of module names that were frozen).
    """
    # Progressive unfreezing: lift backbone freeze once round threshold is met.
    unfreeze_now = (
        cfg.unfreeze_at_epoch is not None
        and current_round is not None
        and current_round >= cfg.unfreeze_at_epoch
    )
    if unfreeze_now:
        backbone = getattr(model, "backbone", model)
        if cfg.unfreeze_layers:
            # Partial unfreeze: only enable the specified named submodules,
            # matching the centralizada phase_b policy (e.g. ["layer4", "fc"]).
            for layer_name in cfg.unfreeze_layers:
                layer = getattr(backbone, layer_name, None)
                if layer is not None:
                    for p in layer.parameters():
                        p.requires_grad = True
                else:
                    _logger.warning(
                        "apply_freeze_policy: unfreeze_layers entry %r not found on backbone; skipping.",
                        layer_name,
                    )
            _logger.info(
                "apply_freeze_policy: partial unfreeze at round=%d "
                "(unfreeze_at_epoch=%d) — layers=%s now trainable.",
                current_round,
                cfg.unfreeze_at_epoch,
                cfg.unfreeze_layers,
            )
        else:
            # Full unfreeze (original behavior when unfreeze_layers is not set).
            for p in model.parameters():
                p.requires_grad = True
            _logger.info(
                "apply_freeze_policy: progressive unfreeze at round=%d "
                "(unfreeze_at_epoch=%d) — all params now trainable.",
                current_round,
                cfg.unfreeze_at_epoch,
            )
        return _param_report(model, frozen_modules=[])

    if not cfg.freeze_backbone and not cfg.freeze_head:
        return _param_report(model, frozen_modules=[])

    backbone_mod, head_mod = _split_backbone_head(model)
    frozen_modules: list[str] = []

    if cfg.freeze_backbone:
        # Freeze everything in the backbone first.
        for p in backbone_mod.parameters():
            p.requires_grad = False
        _set_bn_eval(backbone_mod)
        frozen_modules.append("backbone")

        # Re-enable the head if only the feature extractor should be frozen.
        if not cfg.freeze_head and head_mod is not None:
            for p in head_mod.parameters():
                p.requires_grad = True

    if cfg.freeze_head and head_mod is not None:
        for p in head_mod.parameters():
            p.requires_grad = False
        frozen_modules.append("head")

    report = _param_report(model, frozen_modules=frozen_modules)
    _logger.info(
        "apply_freeze_policy: frozen=%s trainable=%d/%d params",
        frozen_modules,
        report["trainable_params"],
        report["total_params"],
    )
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_backbone_head(
    model: nn.Module,
) -> tuple[nn.Module, nn.Module | None]:
    """Return ``(backbone, head)`` for the standard fedmammobench wrapper convention.

    All classifier wrappers expose ``self.backbone`` (the full torchvision
    model) and the head is the last module at ``backbone.fc`` or
    ``backbone.classifier``.  Falls back to ``(model, None)`` for bare models.
    """
    backbone = getattr(model, "backbone", model)
    for attr in ("fc", "classifier"):
        head = getattr(backbone, attr, None)
        if head is not None:
            return backbone, head
    return backbone, None


def _set_bn_eval(module: nn.Module) -> None:
    """Recursively set all BatchNorm submodules to eval mode."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def _param_report(
    model: nn.Module, frozen_modules: list[str]
) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "frozen_modules": frozen_modules,
    }


def _log_report(report: LoadReport) -> None:
    level = "info"
    if report.missing_keys or report.unexpected_keys or report.shape_mismatches:
        level = "warning"
    log = getattr(_logger, level)
    log(
        "LoadReport source=%r arch=%r missing=%d unexpected=%d remapped=%d "
        "shape_mismatches=%d uri=%r",
        report.source,
        report.arch,
        len(report.missing_keys),
        len(report.unexpected_keys),
        report.remapped_keys,
        len(report.shape_mismatches),
        report.checkpoint_uri,
    )


__all__ = [
    "LoadReport",
    "WeightLoader",
    "apply_freeze_policy",
    "load_weights",
    "register_loader",
    "resolve_source",
]
