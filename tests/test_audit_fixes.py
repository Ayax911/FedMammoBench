"""Regression tests for the federated-collapse audit fixes.

Covers the three confirmed bugs behind the exp12/13/14 collapse:

- **B0** — the ``custom`` warm-start loader silently loaded 0/N tensors
  (``backbone.``-prefixed checkpoint force-loaded into ``model.backbone``),
  leaving the federation training from random init.
- **B1** — frozen-backbone BatchNorm running stats drifted because
  ``model.train()`` re-enabled them every epoch.
- **C4.2** — params unfrozen mid-round (cyclic unfreeze) were never registered
  with the optimizer, so ``optimizer.step()`` never updated them.

Plus a note pinning the **B2 = NO-BUG** conclusion (FedAvg param aggregation is
sample-weighted by Flower's default ``aggregate_fit``).

Uses ``resolve_device("auto")`` like the rest of the codebase — runs on GPU
when available, falls back to CPU otherwise. Skipped automatically when
``torch`` is not installed.

Run:
    pytest tests/test_audit_fixes.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _model_cfg(**overrides):
    from fedmammobench.configs.schema import ModelConfig

    defaults = dict(
        name="resnet18",
        weight_source="none",
        pretrained=False,
        in_channels=3,
        num_classes=2,
        dropout=0.0,
        freeze_backbone=False,
        freeze_head=False,
        unfreeze_at_epoch=None,
        checkpoint_path=None,
        strict_load=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# B0 — custom warm-start loader
# ---------------------------------------------------------------------------

class TestCustomWarmStartLoader:
    """The historic bug: a wrapper checkpoint (keys 'backbone.*') was loaded into
    model.backbone (bare keys) → 0 tensors matched, silently, under
    strict_load=false. These tests lock in the repaired behavior."""

    def test_roundtrip_loads_all_tensors(self, tmp_path):
        torch = pytest.importorskip("torch")
        from fedmammobench.models.factory import build_model
        from fedmammobench.utils.checkpoint import save_checkpoint

        # Build a distinctive source model (shift every param so it is clearly
        # NOT the default random init) and serialize it like the project does.
        model_src = build_model(_model_cfg(), load_pretrained_weights=False)
        with torch.no_grad():
            for p in model_src.parameters():
                p.add_(1.0)
        ckpt = tmp_path / "final.pt"
        save_checkpoint(ckpt, model_src)

        # A fresh model built with weight_source=custom must recover the SAME
        # weights (the old buggy loader left it at its own random init).
        cfg = _model_cfg(weight_source="custom", checkpoint_path=str(ckpt))
        model_loaded = build_model(cfg, load_pretrained_weights=True)

        src_sd = model_src.state_dict()
        for k, v in model_loaded.state_dict().items():
            assert torch.equal(v, src_sd[k]), f"tensor {k} was not loaded from checkpoint"

    def test_bare_backbone_checkpoint_is_prefix_normalized(self, tmp_path):
        torch = pytest.importorskip("torch")
        from fedmammobench.models.factory import build_model
        from fedmammobench.models.weight_loaders.custom import CustomCheckpointLoader

        model_src = build_model(_model_cfg(), load_pretrained_weights=False)
        with torch.no_grad():
            for p in model_src.parameters():
                p.add_(2.0)
        # Save ONLY the backbone (bare keys: conv1.weight, ... — no 'backbone.').
        bare = model_src.backbone.state_dict()
        ckpt = tmp_path / "bare.pt"
        torch.save({"state_dict": bare}, ckpt)

        model_dst = build_model(_model_cfg(), load_pretrained_weights=False)
        cfg = _model_cfg(weight_source="custom", checkpoint_path=str(ckpt))
        report = CustomCheckpointLoader().load(model_dst, cfg)

        # Every backbone tensor should have been loaded (prefix added back).
        assert report.remapped_keys > 0
        for k, v in model_dst.backbone.state_dict().items():
            assert torch.equal(v, bare[k]), f"backbone tensor {k} not loaded"

    def test_zero_match_raises_instead_of_silent_random_init(self, tmp_path):
        torch = pytest.importorskip("torch")
        from fedmammobench.models.factory import build_model
        from fedmammobench.models.weight_loaders.custom import CustomCheckpointLoader

        # A checkpoint whose keys match nothing in the model.
        torch.save({"state_dict": {"totally.unrelated.key": torch.zeros(2)}},
                   tmp_path / "junk.pt")
        model = build_model(_model_cfg(), load_pretrained_weights=False)
        cfg = _model_cfg(weight_source="custom",
                         checkpoint_path=str(tmp_path / "junk.pt"),
                         strict_load=False)
        with pytest.raises(RuntimeError, match="matched 0/"):
            CustomCheckpointLoader().load(model, cfg)

    def test_strict_load_raises_on_partial_mismatch(self, tmp_path):
        torch = pytest.importorskip("torch")
        from fedmammobench.models.factory import build_model
        from fedmammobench.models.weight_loaders.custom import CustomCheckpointLoader

        model_src = build_model(_model_cfg(), load_pretrained_weights=False)
        sd = dict(model_src.state_dict())
        sd.pop(next(iter(sd)))  # drop one key → strict must complain
        torch.save({"state_dict": sd}, tmp_path / "partial.pt")

        model = build_model(_model_cfg(), load_pretrained_weights=False)
        cfg = _model_cfg(weight_source="custom",
                         checkpoint_path=str(tmp_path / "partial.pt"),
                         strict_load=True)
        with pytest.raises(RuntimeError):
            CustomCheckpointLoader().load(model, cfg)


# ---------------------------------------------------------------------------
# B1 — frozen BatchNorm running-stat drift
# ---------------------------------------------------------------------------

def _tiny_loader(torch, n=8):
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _make_trainer(torch, model):
    from fedmammobench.training.trainer import Trainer
    from fedmammobench.training.optim import build_optimizer
    from fedmammobench.configs.schema import OptimizerConfig
    from fedmammobench.utils.device import resolve_device

    device = resolve_device("auto")
    model.to(device)
    opt = build_optimizer(model, OptimizerConfig(name="adamw", lr=1e-3))
    crit = torch.nn.CrossEntropyLoss()
    return Trainer(model, opt, crit, device)


class TestFrozenBatchNorm:
    def test_frozen_bn_stats_do_not_drift(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models.factory import build_model
        from fedmammobench.models.weight_loaders import apply_freeze_policy
        from fedmammobench.utils.device import resolve_device

        model = build_model(_model_cfg(freeze_backbone=True),
                            load_pretrained_weights=False)
        apply_freeze_policy(model, _model_cfg(freeze_backbone=True))
        # Move to the resolved device (GPU if available) before snapshotting,
        # so the "before" and "after" tensors live on the same device.
        model.to(resolve_device("auto"))
        bn = model.backbone.bn1
        before_mean = bn.running_mean.clone()

        trainer = _make_trainer(torch, model)
        trainer.train_one_epoch(_tiny_loader(torch), epoch=0)

        assert bn.training is False, "frozen BN should be pinned to eval mode"
        assert torch.equal(bn.running_mean, before_mean), \
            "frozen BN running_mean drifted despite freeze"

    def test_trainable_bn_stats_do_update(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models.factory import build_model
        from fedmammobench.utils.device import resolve_device

        # Control: nothing frozen → the fix is a no-op and BN stats update.
        model = build_model(_model_cfg(freeze_backbone=False),
                            load_pretrained_weights=False)
        model.to(resolve_device("auto"))
        bn = model.backbone.bn1
        before_mean = bn.running_mean.clone()

        trainer = _make_trainer(torch, model)
        trainer.train_one_epoch(_tiny_loader(torch), epoch=0)

        assert bn.training is True
        assert not torch.equal(bn.running_mean, before_mean), \
            "trainable BN running_mean should update during training"


# ---------------------------------------------------------------------------
# C4.2 — optimizer must gain the layers unfrozen mid-round
# ---------------------------------------------------------------------------

class TestOptimizerUnfreeze:
    def test_optimizer_excludes_frozen_then_includes_after_unfreeze(self):
        torch = pytest.importorskip("torch")
        from fedmammobench.models.factory import build_model
        from fedmammobench.models.weight_loaders import apply_freeze_policy
        from fedmammobench.training.optim import build_optimizer
        from fedmammobench.configs.schema import OptimizerConfig
        from fedmammobench.utils.device import resolve_device

        device = resolve_device("auto")
        model = build_model(_model_cfg(freeze_backbone=True),
                            load_pretrained_weights=False)
        apply_freeze_policy(model, _model_cfg(freeze_backbone=True))
        model.to(device)

        opt_cfg = OptimizerConfig(name="adamw", lr=1e-4,
                                  lr_head=1e-3, lr_backbone=1e-4)
        optimizer = build_optimizer(model, opt_cfg)

        layer4_ids = {id(p) for p in model.backbone.layer4.parameters()}
        in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
        # Documents the bug: while frozen, the optimizer holds none of layer4.
        assert layer4_ids.isdisjoint(in_opt)

        # Apply the fix's logic: unfreeze layer4, then register the new params.
        existing = {id(p) for g in optimizer.param_groups for p in g["params"]}
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
        newly = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in existing]
        assert newly, "layer4 params should now be trainable"
        optimizer.add_param_group({"params": newly, "lr": opt_cfg.lr_backbone})

        in_opt_after = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert layer4_ids.issubset(in_opt_after)

        # And a step actually moves a layer4 weight.
        w = next(model.backbone.layer4.parameters())
        before = w.detach().clone()
        crit = torch.nn.CrossEntropyLoss()
        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        optimizer.zero_grad()
        crit(model(x), y).backward()
        optimizer.step()
        assert not torch.equal(w.detach(), before), \
            "unfrozen layer4 weight did not update after optimizer.step()"


# ---------------------------------------------------------------------------
# B2 — NOT a bug: FedAvg parameter aggregation is sample-weighted
# ---------------------------------------------------------------------------

class TestFedAvgWeighting:
    def test_build_fedavg_uses_stock_flower_weighted_aggregation(self):
        pytest.importorskip("flwr")
        from flwr.server.strategy import FedAvg
        from fedmammobench.federated.strategies.fedavg import (
            build_fedavg,
            _weighted_average,
        )

        strat = build_fedavg()
        # Stock Flower FedAvg.aggregate_fit weights parameters by num_examples;
        # the repo's _weighted_average is wired ONLY to metric aggregation, not
        # to parameter aggregation. So B2 (unweighted params) does not exist.
        assert isinstance(strat, FedAvg)
        assert strat.fit_metrics_aggregation_fn is _weighted_average
        assert FedAvg.aggregate_fit is type(strat).aggregate_fit

    def test_weighted_average_helper_is_sample_weighted(self):
        from fedmammobench.federated.strategies.fedavg import _weighted_average

        # 1 example @ acc=0.0 and 3 examples @ acc=1.0 → weighted mean 0.75.
        out = _weighted_average([(1, {"accuracy": 0.0}), (3, {"accuracy": 1.0})])
        assert abs(out["accuracy"] - 0.75) < 1e-9
