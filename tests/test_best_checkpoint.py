"""Regression tests for Trainer.fit's best-checkpoint tracking.

Added after analyzing exp50-55: all six linear-probing runs reached their best
val_roc_auc within the first 3-15 (of 100) epochs and then degraded from
overfitting, but the pipeline had no mechanism to save or reuse that better
checkpoint — ``final.pt`` was always the last epoch, and test evaluation
always ran on those same overfit weights (see memory
``training_pipeline_no_early_stopping``). These tests cover the fix:
``Trainer.fit(best_checkpoint_metric=..., best_checkpoint_path=...)``.

Run::

    pytest tests/test_best_checkpoint.py -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class _ScriptedEvaluator:
    """Fake Evaluator returning a scripted val_roc_auc sequence, one per call.

    Ignores the real loader/criterion (Trainer only reads the returned dict).
    Snapshots the live model's state_dict on every call so the test can prove
    *which* epoch's weights a saved checkpoint actually corresponds to.
    """

    def __init__(self, model, roc_auc_sequence: list[float]) -> None:
        self.model = model
        self._sequence = roc_auc_sequence
        self._call = 0
        self.snapshots: list[dict] = []

    def evaluate(self, loader, *, criterion=None):
        value = self._sequence[self._call]
        self.snapshots.append(copy.deepcopy(self.model.state_dict()))
        self._call += 1
        return {"roc_auc": value, "f1": value, "loss": 1.0 - value}


def _trainer_and_loader(torch):
    from fedmammobench.training.trainer import Trainer

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    trainer = Trainer(model, optimizer, criterion, device)

    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    return trainer, model, loader


def _state_dicts_equal(torch, a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


class TestBestCheckpointTracking:
    def test_best_epoch_and_value_are_reported(self, tmp_path):
        torch = pytest.importorskip("torch")
        trainer, model, loader = _trainer_and_loader(torch)
        evaluator = _ScriptedEvaluator(model, [0.5, 0.7, 0.6, 0.9, 0.8])
        best_path = tmp_path / "best.pt"

        result = trainer.fit(
            loader,
            epochs=5,
            val_loader=loader,
            evaluator=evaluator,
            best_checkpoint_metric="roc_auc",
            best_checkpoint_path=best_path,
        )

        assert result["best_epoch"] == 3
        assert result["best_val_roc_auc"] == pytest.approx(0.9)
        assert best_path.is_file()

    def test_saved_checkpoint_matches_best_epoch_not_final_epoch(self, tmp_path):
        """The .pt on disk must hold epoch 3's weights, not epoch 4's."""
        torch = pytest.importorskip("torch")
        from fedmammobench.utils.checkpoint import load_checkpoint

        trainer, model, loader = _trainer_and_loader(torch)
        evaluator = _ScriptedEvaluator(model, [0.5, 0.7, 0.6, 0.9, 0.8])
        best_path = tmp_path / "best.pt"

        trainer.fit(
            loader,
            epochs=5,
            val_loader=loader,
            evaluator=evaluator,
            best_checkpoint_metric="roc_auc",
            best_checkpoint_path=best_path,
        )

        # Reload into a fresh module of the same shape and compare tensors.
        reloaded = torch.nn.Linear(4, 2)
        load_checkpoint(best_path, reloaded, strict=True)
        reloaded_sd = reloaded.state_dict()

        assert _state_dicts_equal(torch, reloaded_sd, evaluator.snapshots[3]), (
            "best.pt should hold epoch 3's weights (the scripted best)"
        )
        assert not _state_dicts_equal(torch, reloaded_sd, evaluator.snapshots[4]), (
            "best.pt should NOT hold epoch 4's (final, worse) weights"
        )

    def test_no_tracking_when_metric_or_path_omitted(self, tmp_path):
        """Default behavior (no kwargs passed) must stay exactly as before."""
        torch = pytest.importorskip("torch")
        trainer, model, loader = _trainer_and_loader(torch)
        evaluator = _ScriptedEvaluator(model, [0.5, 0.7, 0.6])

        result = trainer.fit(loader, epochs=3, val_loader=loader, evaluator=evaluator)

        assert "best_epoch" not in result
        assert not (tmp_path / "best.pt").exists()

    def test_metric_absent_from_val_dict_is_skipped_safely(self, tmp_path):
        """An evaluator that never returns the tracked key must not crash fit()."""
        torch = pytest.importorskip("torch")
        trainer, model, loader = _trainer_and_loader(torch)

        class _NoAucEvaluator:
            def evaluate(self, loader, *, criterion=None):
                return {"f1": 0.8, "loss": 0.2}

        best_path = tmp_path / "best.pt"
        result = trainer.fit(
            loader,
            epochs=2,
            val_loader=loader,
            evaluator=_NoAucEvaluator(),
            best_checkpoint_metric="roc_auc",
            best_checkpoint_path=best_path,
        )

        assert result["best_epoch"] is None
        assert result["best_val_roc_auc"] is None
        assert not best_path.exists()
