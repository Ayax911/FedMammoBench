"""Regression tests for centralized progressive backbone unfreeze.

``ModelConfig.unfreeze_at_epoch``/``unfreeze_layers`` were documented as
working for "federated round (or centralized epoch)" but ``Trainer.fit`` never
actually called ``apply_freeze_policy`` per epoch — only the federated client
did, once per round. A centralized config setting ``unfreeze_at_epoch`` (e.g.
to fine-tune ``layer4`` after a period of linear probing) silently trained
with the backbone permanently frozen instead. These tests cover the fix:
``Trainer.fit(model_cfg=..., unfreeze_lr=...)``.

Run::

    pytest tests/test_progressive_unfreeze_centralized.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _partial_freeze_trainer(torch):
    """A tiny backbone/head wrapper matching the project's naming convention.

    ``_split_backbone_head`` (models/weight_loaders/__init__.py) resolves the
    backbone as ``model.backbone`` and the head as ``backbone.fc`` — mirrored
    here with ``layer1``/``layer4`` standing in for a real resnet's blocks.
    Starts already statically frozen (backbone all frozen except ``fc``),
    matching what ``models/factory.py`` does at build time for
    ``freeze_backbone: true``.
    """
    from fedmammobench.training.trainer import Trainer

    torch.manual_seed(0)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(4, 4)
            self.layer4 = torch.nn.Linear(4, 4)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(self.layer4(self.layer1(x)))

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()

        def forward(self, x):
            return self.backbone(x)

    model = Wrapper()
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    trainer = Trainer(model, optimizer, criterion, device)

    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    return trainer, model, optimizer, loader


def _model_cfg(**overrides):
    from fedmammobench.configs.schema import ModelConfig

    cfg = ModelConfig(freeze_backbone=True)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestProgressiveUnfreezeCentralized:
    def test_unfreeze_at_epoch_zero_unfreezes_from_first_epoch(self, torch=__import__("torch")):
        trainer, model, optimizer, loader = _partial_freeze_trainer(torch)
        cfg = _model_cfg(unfreeze_at_epoch=0, unfreeze_layers=["layer4"])

        assert model.backbone.layer4.weight.requires_grad is False
        trainer.fit(loader, epochs=1, model_cfg=cfg, unfreeze_lr=0.01)

        assert model.backbone.layer4.weight.requires_grad is True
        # layer1 was never in unfreeze_layers -> stays frozen.
        assert model.backbone.layer1.weight.requires_grad is False
        assert len(optimizer.param_groups) == 2
        new_group = optimizer.param_groups[-1]
        assert new_group["lr"] == 0.01
        assert any(p is model.backbone.layer4.weight for p in new_group["params"])

    def test_no_model_cfg_is_byte_identical_to_before(self, torch=__import__("torch")):
        trainer, model, optimizer, loader = _partial_freeze_trainer(torch)

        trainer.fit(loader, epochs=1)  # model_cfg defaults to None

        assert model.backbone.layer4.weight.requires_grad is False
        assert len(optimizer.param_groups) == 1

    def test_unconfigured_unfreeze_is_inert(self, torch=__import__("torch")):
        trainer, model, optimizer, loader = _partial_freeze_trainer(torch)
        cfg = _model_cfg()  # unfreeze_at_epoch stays None

        trainer.fit(loader, epochs=1, model_cfg=cfg, unfreeze_lr=0.01)

        assert model.backbone.layer4.weight.requires_grad is False
        assert len(optimizer.param_groups) == 1

    def test_unfreeze_at_later_epoch_stays_frozen_until_reached(self, torch=__import__("torch")):
        trainer, model, optimizer, loader = _partial_freeze_trainer(torch)
        cfg = _model_cfg(unfreeze_at_epoch=2, unfreeze_layers=["layer4"])

        trainer.fit(loader, epochs=2, model_cfg=cfg, unfreeze_lr=0.01)  # epochs 0, 1
        assert model.backbone.layer4.weight.requires_grad is False
        assert len(optimizer.param_groups) == 1

        trainer.fit(loader, epochs=1, start_epoch=2, model_cfg=cfg, unfreeze_lr=0.01)  # epoch 2
        assert model.backbone.layer4.weight.requires_grad is True
        assert len(optimizer.param_groups) == 2

    def test_unfreeze_lr_defaults_to_first_group_lr(self, torch=__import__("torch")):
        trainer, model, optimizer, loader = _partial_freeze_trainer(torch)
        cfg = _model_cfg(unfreeze_at_epoch=0, unfreeze_layers=["layer4"])

        trainer.fit(loader, epochs=1, model_cfg=cfg)  # unfreeze_lr omitted -> None

        assert optimizer.param_groups[-1]["lr"] == optimizer.param_groups[0]["lr"]

    def test_repeated_calls_do_not_duplicate_param_group(self, torch=__import__("torch")):
        trainer, model, optimizer, loader = _partial_freeze_trainer(torch)
        cfg = _model_cfg(unfreeze_at_epoch=0, unfreeze_layers=["layer4"])

        trainer.fit(loader, epochs=1, model_cfg=cfg, unfreeze_lr=0.01)
        assert len(optimizer.param_groups) == 2

        trainer.fit(loader, epochs=1, start_epoch=1, model_cfg=cfg, unfreeze_lr=0.01)
        assert len(optimizer.param_groups) == 2  # idempotent, no duplicate group
