"""RadImageNet integration tests.

All tests run CPU-only without real RadImageNet checkpoints.  Tests that
require ``torch`` or ``albumentations`` are skipped automatically when those
packages are not installed (``pytest.importorskip``).

Run:
    pytest tests/test_radimagenet.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides):
    """Return a minimal ModelConfig for testing."""
    from fedmammobench.configs.schema import ModelConfig

    defaults = dict(
        name="resnet18",
        pretrained=False,
        weight_source="auto",
        in_channels=1,
        num_classes=2,
        dropout=0.2,
        freeze_backbone=False,
        freeze_head=False,
        unfreeze_at_epoch=None,
        checkpoint_path=None,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# Schema / config — no torch needed
# ---------------------------------------------------------------------------

class TestNormalizePresets:
    """Tests for NORMALIZE_PRESETS and AugmentationConfig defaults."""

    def test_radimagenet_gray_preset_values(self):
        from fedmammobench.configs.schema import NORMALIZE_PRESETS

        p = NORMALIZE_PRESETS["radimagenet_gray"]
        assert p["mean"] == (0.5,)
        assert p["std"] == (0.5,)

    def test_imagenet_rgb_preset_length(self):
        from fedmammobench.configs.schema import NORMALIZE_PRESETS

        p = NORMALIZE_PRESETS["imagenet_rgb"]
        assert len(p["mean"]) == 3
        assert len(p["std"]) == 3

    def test_mammo_default_preset(self):
        from fedmammobench.configs.schema import NORMALIZE_PRESETS

        p = NORMALIZE_PRESETS["mammo_default"]
        assert p["mean"] == (0.5,)
        assert p["std"] == (0.25,)

    def test_five_presets_defined(self):
        from fedmammobench.configs.schema import NORMALIZE_PRESETS

        assert len(NORMALIZE_PRESETS) == 5

    # --- transforms._resolve_norm (requires albumentations at module load) ---

    def test_preset_resolution_correct_channels(self):
        pytest.importorskip("albumentations")
        from fedmammobench.configs.schema import AugmentationConfig
        from fedmammobench.datasets.transforms import _resolve_norm

        aug = AugmentationConfig(normalize_preset="radimagenet_gray")
        mean, std = _resolve_norm(aug, in_channels=1)
        assert mean == (0.5,)
        assert std == (0.5,)

    def test_preset_channel_mismatch_raises(self):
        pytest.importorskip("albumentations")
        from fedmammobench.configs.schema import AugmentationConfig
        from fedmammobench.datasets.transforms import _resolve_norm

        aug = AugmentationConfig(normalize_preset="imagenet_rgb")  # 3-channel preset
        with pytest.raises(ValueError, match="in_channels"):
            _resolve_norm(aug, in_channels=1)

    def test_unknown_preset_raises(self):
        pytest.importorskip("albumentations")
        from fedmammobench.configs.schema import AugmentationConfig
        from fedmammobench.datasets.transforms import _resolve_norm

        aug = AugmentationConfig(normalize_preset="bad_preset")
        with pytest.raises(ValueError, match="Unknown normalize_preset"):
            _resolve_norm(aug, in_channels=1)

    def test_scalar_mean_replicates_to_channels(self):
        pytest.importorskip("albumentations")
        from fedmammobench.configs.schema import AugmentationConfig
        from fedmammobench.datasets.transforms import _resolve_norm

        aug = AugmentationConfig(normalize_mean=0.5, normalize_std=0.25)
        mean, std = _resolve_norm(aug, in_channels=3)
        assert mean == (0.5, 0.5, 0.5)
        assert std == (0.25, 0.25, 0.25)

    def test_list_mean_passthrough(self):
        pytest.importorskip("albumentations")
        from fedmammobench.configs.schema import AugmentationConfig
        from fedmammobench.datasets.transforms import _resolve_norm

        aug = AugmentationConfig(
            normalize_mean=[0.485, 0.456, 0.406],
            normalize_std=[0.229, 0.224, 0.225],
        )
        mean, std = _resolve_norm(aug, in_channels=3)
        assert len(mean) == 3
        assert abs(mean[0] - 0.485) < 1e-6

    def test_deprecated_grayscale_kwarg_warns(self):
        pytest.importorskip("albumentations")
        from fedmammobench.configs.schema import AugmentationConfig
        from fedmammobench.datasets.transforms import build_transforms

        aug = AugmentationConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_transforms(32, aug, grayscale=True)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ---------------------------------------------------------------------------
# Key remapping — pure Python, requires torch indirectly via package __init__
# ---------------------------------------------------------------------------

class TestKeymaps:
    """Tests for _keymaps.py — pure dict operations, no tensor ops."""

    def test_strip_module_prefix(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders._keymaps import remap_radimagenet_keys

        raw = {
            "module.conv1.weight": "t1",
            "module.layer1.0.weight": "t2",
        }
        new, n = remap_radimagenet_keys(raw, "resnet50")
        assert "conv1.weight" in new
        assert "layer1.0.weight" in new
        assert n == 2

    def test_drop_fc_keys_resnet50(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders._keymaps import remap_radimagenet_keys

        raw = {"conv1.weight": "t", "fc.weight": "t2", "fc.bias": "t3"}
        new, _ = remap_radimagenet_keys(raw, "resnet50")
        assert "fc.weight" not in new
        assert "fc.bias" not in new
        assert "conv1.weight" in new

    def test_drop_classifier_keys_densenet121(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders._keymaps import remap_radimagenet_keys

        raw = {"features.conv0.weight": "t", "classifier.weight": "t2"}
        new, _ = remap_radimagenet_keys(raw, "densenet121")
        assert "classifier.weight" not in new
        assert "features.conv0.weight" in new

    def test_drop_auxlogits_inception_v3(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders._keymaps import remap_radimagenet_keys

        raw = {
            "Conv2d_1a_3x3.conv.weight": "t",
            "AuxLogits.conv0.weight": "t2",
            "fc.weight": "t3",
        }
        new, _ = remap_radimagenet_keys(raw, "inception_v3")
        assert "AuxLogits.conv0.weight" not in new
        assert "fc.weight" not in new
        assert "Conv2d_1a_3x3.conv.weight" in new

    def test_supported_archs_constant(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders._keymaps import SUPPORTED_ARCHS

        assert "resnet50" in SUPPORTED_ARCHS
        assert "densenet121" in SUPPORTED_ARCHS
        assert "inception_v3" in SUPPORTED_ARCHS
        assert "resnet18" not in SUPPORTED_ARCHS
        assert "efficientnet_b0" not in SUPPORTED_ARCHS


# ---------------------------------------------------------------------------
# resolve_source / BC — requires torch (models package chain)
# ---------------------------------------------------------------------------

class TestResolveSource:
    def test_auto_pretrained_true_resolves_imagenet(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders import resolve_source

        cfg = _make_cfg(pretrained=True, weight_source="auto")
        assert resolve_source(cfg) == "imagenet"

    def test_auto_pretrained_false_resolves_none(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders import resolve_source

        cfg = _make_cfg(pretrained=False, weight_source="auto")
        assert resolve_source(cfg) == "none"

    def test_explicit_radimagenet_wins_over_pretrained_flag(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders import resolve_source

        cfg = _make_cfg(pretrained=False, weight_source="radimagenet")
        assert resolve_source(cfg) == "radimagenet"

    def test_explicit_none_wins_over_pretrained_true(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders import resolve_source

        cfg = _make_cfg(pretrained=True, weight_source="none")
        assert resolve_source(cfg) == "none"

    def test_explicit_custom_resolves(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders import resolve_source

        cfg = _make_cfg(weight_source="custom", checkpoint_path="/some/path.pth")
        assert resolve_source(cfg) == "custom"


# ---------------------------------------------------------------------------
# Channel adaptation — requires torch (tensor ops)
# ---------------------------------------------------------------------------

class TestChannelAdaptation:
    def test_adapt_3_to_1_sum_preserving(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models._adapt import adapt_weight_tensor

        w = torch.ones(4, 3, 3, 3)
        adapted = adapt_weight_tensor(w, target_in_channels=1)
        assert adapted.shape == (4, 1, 3, 3)
        assert torch.allclose(adapted, torch.full_like(adapted, 3.0))

    def test_adapt_1_to_3_correct_scale(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models._adapt import adapt_weight_tensor

        w = torch.ones(4, 1, 3, 3)
        adapted = adapt_weight_tensor(w, target_in_channels=3)
        assert adapted.shape == (4, 3, 3, 3)
        assert torch.allclose(adapted, torch.full_like(adapted, 1.0 / 3.0))

    def test_adapt_no_op_when_channels_equal(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models._adapt import adapt_weight_tensor

        w = torch.randn(4, 3, 3, 3)
        adapted = adapt_weight_tensor(w, target_in_channels=3)
        assert adapted is w  # exact same object returned (no copy)

    def test_legacy_mean_no_scale_factor(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models._adapt import adapt_weight_tensor

        w = torch.ones(4, 3, 3, 3)
        adapted = adapt_weight_tensor(w, target_in_channels=1, strategy="legacy_mean")
        assert torch.allclose(adapted, torch.ones(4, 1, 3, 3))

    def test_sum_preserving_3x_magnitude_of_legacy_mean(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models._adapt import adapt_weight_tensor

        w = torch.ones(1, 3, 3, 3)
        sp = adapt_weight_tensor(w, 1, strategy="sum_preserving")
        lm = adapt_weight_tensor(w, 1, strategy="legacy_mean")
        # sum_preserving should be 3× the legacy_mean result for 3→1 adaptation
        assert torch.allclose(sp / lm, torch.full_like(sp, 3.0))


# ---------------------------------------------------------------------------
# RadImageNetLoader — path resolution (no real checkpoint needed)
# ---------------------------------------------------------------------------

class TestRadImageNetLoader:
    def test_unsupported_arch_raises_value_error(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        loader = RadImageNetLoader()

        class DummyModel:
            pass

        cfg = _make_cfg(name="resnet18", weight_source="radimagenet")
        with pytest.raises(ValueError, match="RadImageNet weights are not published"):
            loader.load(DummyModel(), cfg)

    def test_missing_checkpoint_raises_file_not_found(self, monkeypatch):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        loader = RadImageNetLoader()
        monkeypatch.delenv("FEDMAMMOBENCH_RADIMAGENET_DIR", raising=False)

        class DummyModel:
            pass

        cfg = _make_cfg(name="resnet50", weight_source="radimagenet")
        with pytest.raises(FileNotFoundError, match="FEDMAMMOBENCH_RADIMAGENET_DIR"):
            loader.load(DummyModel(), cfg)

    def test_error_message_is_actionable(self, monkeypatch):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        loader = RadImageNetLoader()
        monkeypatch.delenv("FEDMAMMOBENCH_RADIMAGENET_DIR", raising=False)

        class DummyModel:
            pass

        cfg = _make_cfg(name="resnet50", weight_source="radimagenet")
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load(DummyModel(), cfg)
        msg = str(exc_info.value)
        assert "RadImageNet-resnet50.pth" in msg
        assert "BMEII-AI" in msg or "TRANSFER_LEARNING_GUIDE" in msg

    def test_resolve_path_via_explicit_checkpoint_path(self, tmp_path):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        loader = RadImageNetLoader()
        fake_ckpt = tmp_path / "my_ckpt.pth"
        fake_ckpt.touch()
        cfg_obj = type("C", (), {"checkpoint_path": str(fake_ckpt), "name": "resnet50"})()
        resolved = loader._resolve_path(cfg_obj)
        assert resolved == fake_ckpt

    def test_resolve_path_via_env_var(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        loader = RadImageNetLoader()
        fake_ckpt = tmp_path / "RadImageNet-resnet50.pth"
        fake_ckpt.touch()
        monkeypatch.setenv("FEDMAMMOBENCH_RADIMAGENET_DIR", str(tmp_path))
        cfg_obj = type("C", (), {"checkpoint_path": None, "name": "resnet50"})()
        resolved = loader._resolve_path(cfg_obj)
        assert resolved == fake_ckpt

    def test_unwrap_state_dict_key(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        raw = {"state_dict": {"conv1.weight": "tensor"}}
        unwrapped = RadImageNetLoader._unwrap(raw)
        assert "conv1.weight" in unwrapped

    def test_unwrap_model_state_dict_key(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        raw = {"model_state_dict": {"conv1.weight": "tensor"}}
        unwrapped = RadImageNetLoader._unwrap(raw)
        assert "conv1.weight" in unwrapped

    def test_unwrap_plain_dict_passthrough(self):
        pytest.importorskip("torch")
        from fedmammobench.models.weight_loaders.radimagenet import RadImageNetLoader

        raw = {"conv1.weight": "tensor"}
        unwrapped = RadImageNetLoader._unwrap(raw)
        assert unwrapped is raw


# ---------------------------------------------------------------------------
# apply_freeze_policy — requires torch (nn.Module, parameters)
# ---------------------------------------------------------------------------

class TestFreezePolicy:
    def test_no_freeze_all_trainable(self):
        torch = pytest.importorskip("torch")
        from torch import nn
        from fedmammobench.models.weight_loaders import apply_freeze_policy

        model = nn.Linear(4, 2)
        cfg = _make_cfg(freeze_backbone=False, freeze_head=False)
        rep = apply_freeze_policy(model, cfg)
        assert rep["frozen_modules"] == []
        assert all(p.requires_grad for p in model.parameters())

    def test_freeze_backbone_report_keys(self):
        torch = pytest.importorskip("torch")
        from torch import nn
        from fedmammobench.models.weight_loaders import apply_freeze_policy

        model = nn.Linear(4, 2)
        cfg = _make_cfg(freeze_backbone=True)
        rep = apply_freeze_policy(model, cfg)
        assert "trainable_params" in rep
        assert "total_params" in rep
        assert "frozen_modules" in rep

    def test_progressive_unfreeze_below_threshold(self):
        torch = pytest.importorskip("torch")
        from torch import nn
        from fedmammobench.models.weight_loaders import apply_freeze_policy

        model = nn.Linear(4, 2)
        for p in model.parameters():
            p.requires_grad = False

        cfg = _make_cfg(freeze_backbone=True, unfreeze_at_epoch=5)
        # Round 4 < threshold 5 → unfreeze should NOT trigger
        rep = apply_freeze_policy(model, cfg, current_round=4)
        # frozen_modules should still be non-empty (freeze was re-applied)
        assert rep["frozen_modules"] != []

    def test_progressive_unfreeze_at_threshold(self):
        torch = pytest.importorskip("torch")
        from torch import nn
        from fedmammobench.models.weight_loaders import apply_freeze_policy

        model = nn.Linear(4, 2)
        for p in model.parameters():
            p.requires_grad = False

        cfg = _make_cfg(freeze_backbone=True, unfreeze_at_epoch=5)
        rep = apply_freeze_policy(model, cfg, current_round=5)
        assert rep["frozen_modules"] == []
        assert all(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# Config BC — no torch needed (config loader only)
# ---------------------------------------------------------------------------

class TestConfigBC:
    def test_new_radimagenet_configs_parse(self):
        configs_dir = Path(__file__).resolve().parent.parent / "configs"
        from fedmammobench.configs.loader import load_config

        for name in [
            "radimagenet_resnet50_centralized.yaml",
            "radimagenet_resnet50_fedavg.yaml",
            "radimagenet_densenet121_fedavg.yaml",
        ]:
            cfg = load_config(str(configs_dir / name))
            assert cfg.model.weight_source == "radimagenet"
            assert cfg.model.freeze_backbone is True
            assert cfg.training.augmentation.normalize_preset == "radimagenet_gray"

    def test_existing_configs_not_broken(self):
        configs_dir = Path(__file__).resolve().parent.parent / "configs"
        from fedmammobench.configs.loader import load_config

        for name in ["base.yaml", "fedavg_cbis_ddsm.yaml"]:
            cfg = load_config(str(configs_dir / name))
            assert cfg.model.weight_source == "auto"
            assert cfg.model.freeze_backbone is False
