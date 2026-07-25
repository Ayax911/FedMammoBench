"""Tests for the configurable multi-layer classification head.

Run:
    pytest tests/test_head.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

torch = pytest.importorskip("torch")
nn = torch.nn


def _make_cfg(**overrides):
    from fedmammobench.configs.schema import HeadConfig, ModelConfig

    head_overrides = overrides.pop("head", {})
    defaults = dict(
        name="resnet18",
        pretrained=False,
        weight_source="none",
        in_channels=1,
        num_classes=2,
        dropout=0.2,
        freeze_backbone=False,
        freeze_head=False,
        unfreeze_at_epoch=None,
        checkpoint_path=None,
    )
    defaults.update(overrides)
    return ModelConfig(head=HeadConfig(**head_overrides), **defaults)


# ---------------------------------------------------------------------------
# build_head — retrocompatibility with the original single-Linear head
# ---------------------------------------------------------------------------

class TestBuildHeadRetrocompat:
    def test_empty_hidden_dims_zero_dropout_is_plain_linear(self):
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(dropout=0.0, num_classes=2)
        head = build_head(128, cfg)

        assert isinstance(head, nn.Linear)
        assert head.in_features == 128
        assert head.out_features == 2

    def test_empty_hidden_dims_with_dropout_is_sequential_dropout_linear(self):
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(dropout=0.3, num_classes=1)
        head = build_head(128, cfg)

        assert isinstance(head, nn.Sequential)
        assert len(head) == 2
        assert isinstance(head[0], nn.Dropout)
        assert head[0].p == pytest.approx(0.3)
        assert isinstance(head[1], nn.Linear)
        assert head[1].in_features == 128
        assert head[1].out_features == 1

    def test_state_dict_keys_match_legacy_layout(self):
        """fc.1.weight / fc.1.bias — same indices the project's checkpoints use."""
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(dropout=0.2, num_classes=2)
        head = build_head(64, cfg)
        keys = set(head.state_dict().keys())
        assert keys == {"1.weight", "1.bias"}


# ---------------------------------------------------------------------------
# build_head — multi-layer MLP
# ---------------------------------------------------------------------------

class TestBuildHeadMultilayer:
    def test_hidden_dims_output_shape(self):
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(dropout=0.2, num_classes=1, head={"hidden_dims": [64, 32]})
        head = build_head(128, cfg)

        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 1)

    def test_final_layer_has_no_activation_after_it(self):
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(dropout=0.0, num_classes=2, head={"hidden_dims": [32]})
        head = build_head(16, cfg)

        assert isinstance(head, nn.Sequential)
        assert isinstance(head[-1], nn.Linear)
        assert head[-1].out_features == 2

    @pytest.mark.parametrize("activation", ["relu", "gelu", "tanh", "leaky_relu"])
    def test_each_activation_builds_and_runs(self, activation):
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(num_classes=2, head={"hidden_dims": [16], "activation": activation})
        head = build_head(8, cfg)
        out = head(torch.randn(2, 8))
        assert out.shape == (2, 2)

    def test_batch_norm_inserted_with_correct_dims(self):
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(num_classes=2, head={"hidden_dims": [32, 16], "batch_norm": True})
        head = build_head(64, cfg)

        bn_layers = [m for m in head if isinstance(m, nn.BatchNorm1d)]
        assert [bn.num_features for bn in bn_layers] == [32, 16]

    def test_no_batch_norm_by_default(self):
        from fedmammobench.models._head import build_head

        cfg = _make_cfg(num_classes=2, head={"hidden_dims": [32]})
        head = build_head(64, cfg)
        assert not any(isinstance(m, nn.BatchNorm1d) for m in head)


# ---------------------------------------------------------------------------
# HeadConfig.validate()
# ---------------------------------------------------------------------------

class TestHeadConfigValidate:
    def test_zero_dim_raises(self):
        from fedmammobench.configs.schema import HeadConfig

        with pytest.raises(ValueError):
            HeadConfig(hidden_dims=[0]).validate()

    def test_negative_dim_raises(self):
        from fedmammobench.configs.schema import HeadConfig

        with pytest.raises(ValueError):
            HeadConfig(hidden_dims=[16, -1]).validate()

    def test_valid_dims_ok(self):
        from fedmammobench.configs.schema import HeadConfig

        HeadConfig(hidden_dims=[512, 128]).validate()

    def test_model_config_validate_delegates_to_head(self):
        cfg = _make_cfg(num_classes=2, head={"hidden_dims": [0]})
        with pytest.raises(ValueError):
            cfg.validate()


# ---------------------------------------------------------------------------
# config_hash_fields — server/client consistency check must see head changes
# ---------------------------------------------------------------------------

class TestConfigHashFields:
    def test_differs_between_distinct_heads(self):
        cfg_a = _make_cfg(head={"hidden_dims": [64]})
        cfg_b = _make_cfg(head={"hidden_dims": [128]})
        assert cfg_a.config_hash_fields() != cfg_b.config_hash_fields()

    def test_same_for_identical_heads(self):
        cfg_a = _make_cfg(head={"hidden_dims": [64, 32], "activation": "gelu"})
        cfg_b = _make_cfg(head={"hidden_dims": [64, 32], "activation": "gelu"})
        assert cfg_a.config_hash_fields() == cfg_b.config_hash_fields()

    def test_default_empty_head_hash_stable(self):
        cfg = _make_cfg()
        fields = cfg.config_hash_fields()
        assert fields["head_hidden_dims"] == ()
        assert fields["head_activation"] == "relu"
        assert fields["head_batch_norm"] is False


# ---------------------------------------------------------------------------
# Forward shape across all five architectures, with a multi-layer head
# ---------------------------------------------------------------------------

class TestForwardShapeAllArchitectures:
    @pytest.mark.parametrize(
        "name,image_size",
        [
            ("resnet18", 64),
            ("resnet50", 64),
            ("densenet121", 64),
            ("efficientnet_b0", 64),
            ("inception_v3", 299),
        ],
    )
    def test_forward_shape_with_mlp_head(self, name, image_size):
        import fedmammobench.models  # noqa: F401  (side-effect registration)
        from fedmammobench.models.factory import build_model

        cfg = _make_cfg(
            name=name,
            in_channels=1,
            num_classes=1,
            dropout=0.1,
            head={"hidden_dims": [32], "activation": "leaky_relu"},
        )
        model = build_model(cfg, load_pretrained_weights=False)
        model.eval()

        x = torch.randn(2, 1, image_size, image_size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)


# ---------------------------------------------------------------------------
# Freeze policy treats a multi-layer head as a single block
# ---------------------------------------------------------------------------

class TestFreezePolicyWithMultilayerHead:
    def test_freeze_backbone_keeps_all_head_params_trainable(self):
        import fedmammobench.models  # noqa: F401
        from fedmammobench.models.factory import build_model
        from fedmammobench.models.weight_loaders import apply_freeze_policy

        cfg = _make_cfg(
            name="resnet18",
            num_classes=1,
            head={"hidden_dims": [32, 16]},
            freeze_backbone=True,
        )
        model = build_model(cfg, load_pretrained_weights=False)
        apply_freeze_policy(model, cfg)

        backbone = model.backbone
        head = backbone.fc
        assert all(not p.requires_grad for p in backbone.parameters() if p not in set(head.parameters()))
        assert all(p.requires_grad for p in head.parameters())
