"""Tests for centralized resume-after-a-crash (Trainer.fit's resume_checkpoint_path/
resume_state, scheduler round-trip through save_checkpoint/load_checkpoint, and
scripts/run_centralized.py's _resolve_centralized_resume helper).

A hard crash on the shared workstation (segfault, OOM-killer, CUDA driver reset)
skips Python's finally blocks entirely, so the pre-existing end-of-fit() checkpoint
save gives zero protection against it — these tests cover the new per-epoch
safety-net checkpoint that fixes that gap. See CHANGELOG.md [0.8.0] and memory
training_pipeline_no_early_stopping.

Run::

    pytest tests/test_resume_centralized.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

REPO_ROOT = Path(__file__).resolve().parent.parent


class _ScriptedEvaluator:
    """Fake Evaluator returning a scripted val_roc_auc sequence, one per call."""

    def __init__(self, seq: list[float]) -> None:
        self._seq = seq
        self.call = 0

    def evaluate(self, loader, *, criterion=None):
        value = self._seq[self.call]
        self.call += 1
        return {"roc_auc": value, "f1": value, "loss": 1.0 - value}


def _trainer_and_loader(torch, *, with_scheduler: bool = False):
    from fedmammobench.training.trainer import Trainer

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    scheduler = None
    if with_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(model, optimizer, criterion, device, scheduler=scheduler)

    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    return trainer, model, optimizer, scheduler, loader


def _load_run_centralized_module():
    """Load scripts/run_centralized.py as a module, mirroring
    cli/centralized.py::_load_script_main's own pattern, so
    _resolve_centralized_resume can be unit-tested directly."""
    script = REPO_ROOT / "scripts" / "run_centralized.py"
    spec = importlib.util.spec_from_file_location("_test_run_centralized", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTrainerFitResumeCheckpoint:
    def test_epochs_zero_guard_returns_without_crash(self, tmp_path):
        torch = pytest.importorskip("torch")
        trainer, model, optimizer, scheduler, loader = _trainer_and_loader(torch)

        result = trainer.fit(loader, epochs=0, val_loader=loader, evaluator=_ScriptedEvaluator([]))

        assert result["epochs_run"] == 0
        assert result["early_stopped"] is False

    def test_epochs_zero_with_resume_state_reports_it(self, tmp_path):
        """A 0-epoch call (already-complete resume) still surfaces the
        resumed best_epoch/best_value, not None, so downstream test-eval
        reload logic in run_centralized.py stays sensible."""
        torch = pytest.importorskip("torch")
        trainer, model, optimizer, scheduler, loader = _trainer_and_loader(torch)

        result = trainer.fit(
            loader,
            epochs=0,
            val_loader=loader,
            evaluator=_ScriptedEvaluator([]),
            best_checkpoint_metric="roc_auc",
            best_checkpoint_path=tmp_path / "best.pt",
            resume_state={"best_value": 0.9, "best_epoch": 1, "rounds_no_improve": 0},
        )

        assert result["epochs_run"] == 0
        assert result["best_epoch"] == 1
        assert result["best_val_roc_auc"] == pytest.approx(0.9)

    def test_resume_checkpoint_records_absolute_epoch_index_not_count(self, tmp_path):
        """The safety-net checkpoint must store the raw loop index (matching
        best.pt's existing convention), not a since-start_epoch count — the
        exact bug that would silently miscompute a second, chained resume."""
        torch = pytest.importorskip("torch")
        trainer, model, optimizer, scheduler, loader = _trainer_and_loader(torch)
        resume_path = tmp_path / "final.pt"

        trainer.fit(
            loader,
            epochs=3,
            val_loader=loader,
            evaluator=_ScriptedEvaluator([0.5, 0.6, 0.7]),
            resume_checkpoint_path=resume_path,
        )

        payload = torch.load(resume_path, map_location="cpu")
        assert payload["epoch"] == 2  # absolute 0-based index of the 3rd (last) epoch

    def test_resume_checkpoint_written_even_without_best_tracking(self, tmp_path):
        """resume_checkpoint_path is independent of best_checkpoint_metric/path —
        the safety net should work even when best-checkpoint tracking is off."""
        torch = pytest.importorskip("torch")
        trainer, model, optimizer, scheduler, loader = _trainer_and_loader(torch)
        resume_path = tmp_path / "final.pt"

        trainer.fit(
            loader, epochs=2, val_loader=loader, evaluator=_ScriptedEvaluator([0.5, 0.6]),
            resume_checkpoint_path=resume_path,
        )

        assert resume_path.is_file()
        payload = torch.load(resume_path, map_location="cpu")
        assert payload["epoch"] == 1
        assert payload["extra"]["resume_state"] == {
            "best_value": None,
            "best_epoch": None,
            "rounds_no_improve": 0,
        }

    def test_early_stopping_across_simulated_crash(self, tmp_path):
        """Splitting [0.5, 0.9, 0.7, 0.65, 0.6] (patience=2) into two fit()
        calls (simulated crash after epoch 1) must stop at the same absolute
        epoch (3) and report the same best_epoch (1) as an uninterrupted run —
        proving rounds_no_improve/best_value continuity across the crash."""
        torch = pytest.importorskip("torch")
        sequence = [0.5, 0.9, 0.7, 0.65, 0.6]

        # Reference: uninterrupted run.
        trainer_full, _, _, _, loader_full = _trainer_and_loader(torch)
        result_full = trainer_full.fit(
            loader_full, epochs=5, val_loader=loader_full, evaluator=_ScriptedEvaluator(sequence),
            best_checkpoint_metric="roc_auc", best_checkpoint_path=tmp_path / "best_full.pt",
            early_stopping_patience=2,
        )
        assert result_full["early_stopped"] is True
        assert result_full["epochs_run"] == 4
        assert result_full["best_epoch"] == 1

        # Crash-simulated: only epochs 0-1 run, with the safety net active.
        resume_path = tmp_path / "final_resumed.pt"
        best_path = tmp_path / "best_resumed.pt"
        trainer1, _, opt1, sch1, loader1 = _trainer_and_loader(torch)
        result_part1 = trainer1.fit(
            loader1, epochs=2, val_loader=loader1, evaluator=_ScriptedEvaluator(sequence[:2]),
            best_checkpoint_metric="roc_auc", best_checkpoint_path=best_path,
            early_stopping_patience=2, resume_checkpoint_path=resume_path,
        )
        assert result_part1["epochs_run"] == 2
        assert result_part1["early_stopped"] is False

        from fedmammobench.utils.checkpoint import load_checkpoint

        payload = load_checkpoint(resume_path, trainer1.model, strict=True)
        resume_state = payload["extra"]["resume_state"]

        # Resume: fresh trainer/optimizer/scheduler, reload state, continue.
        trainer2, model2, opt2, sch2, loader2 = _trainer_and_loader(torch)
        load_checkpoint(resume_path, model2, optimizer=opt2, strict=True)
        result_part2 = trainer2.fit(
            loader2, epochs=3, start_epoch=2, val_loader=loader2,
            evaluator=_ScriptedEvaluator(sequence[2:]),
            best_checkpoint_metric="roc_auc", best_checkpoint_path=best_path,
            early_stopping_patience=2, resume_state=resume_state,
            resume_checkpoint_path=resume_path,
        )

        assert result_part2["epoch"] == result_full["epoch"] == 3
        assert result_part2["early_stopped"] is True
        assert result_part2["best_epoch"] == 1

    def test_early_stopping_negative_control_without_resume_state(self, tmp_path):
        """Without passing resume_state back, the resumed call's patience
        counter incorrectly restarts at 0 -- documents exactly the bug the
        continuity test above guards against."""
        torch = pytest.importorskip("torch")
        sequence = [0.5, 0.9, 0.7, 0.65, 0.6]
        trainer, model, optimizer, scheduler, loader = _trainer_and_loader(torch)

        result = trainer.fit(
            loader, epochs=3, start_epoch=2, val_loader=loader,
            evaluator=_ScriptedEvaluator(sequence[2:]),
            best_checkpoint_metric="roc_auc", best_checkpoint_path=tmp_path / "best.pt",
            early_stopping_patience=2, resume_state=None,
        )

        # Without resume_state, best_value starts at None -> epoch 2's 0.7
        # looks like a fresh "best", so the patience clock restarts and the
        # run doesn't stop until 2 rounds after THAT (epoch 4, one epoch
        # later than the correct absolute epoch 3).
        assert result["early_stopped"] is True
        assert result["epoch"] == 4
        assert result["best_epoch"] == 2  # wrong: should have been 1

    def test_resume_kwargs_default_to_byte_identical_behavior(self, tmp_path):
        """No resume_checkpoint_path/resume_state passed -> identical to the
        pre-resume-feature behavior (regression guard)."""
        torch = pytest.importorskip("torch")
        trainer, model, optimizer, scheduler, loader = _trainer_and_loader(torch)

        result = trainer.fit(
            loader, epochs=3, val_loader=loader, evaluator=_ScriptedEvaluator([0.5, 0.7, 0.6]),
        )

        assert result["epochs_run"] == 3
        assert result["early_stopped"] is False
        assert "best_epoch" not in result


class TestSchedulerCheckpointRoundTrip:
    def test_scheduler_state_round_trips(self, tmp_path):
        torch = pytest.importorskip("torch")
        from fedmammobench.utils.checkpoint import load_checkpoint, save_checkpoint

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler, epoch=2)

        model2 = torch.nn.Linear(4, 2)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.5)
        load_checkpoint(path, model2, optimizer=optimizer2, scheduler=scheduler2, strict=True)

        assert scheduler2.state_dict() == scheduler.state_dict()
        assert scheduler2.get_last_lr() == scheduler.get_last_lr()

    def test_scheduler_none_is_safe(self, tmp_path):
        """save_checkpoint(scheduler=None)/load_checkpoint(scheduler=None) must
        not add/expect a 'scheduler' key — mirrors optimizer=None's contract."""
        torch = pytest.importorskip("torch")
        from fedmammobench.utils.checkpoint import load_checkpoint, save_checkpoint

        model = torch.nn.Linear(4, 2)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, epoch=0)
        payload = torch.load(path, map_location="cpu")
        assert "scheduler" not in payload

        model2 = torch.nn.Linear(4, 2)
        load_checkpoint(path, model2, strict=True)  # must not raise


class TestResolveCentralizedResume:
    """Unit tests for scripts/run_centralized.py::_resolve_centralized_resume."""

    def _cfg(self, *, epochs: int):
        from fedmammobench.configs.schema import ExperimentConfig, TrainingConfig

        cfg = ExperimentConfig(name="resume_test", mode="centralized")
        cfg.training = TrainingConfig(epochs=epochs)
        return cfg

    def test_no_resume_flag_returns_full_budget(self, tmp_path):
        torch = pytest.importorskip("torch")
        mod = _load_run_centralized_module()
        cfg = self._cfg(epochs=10)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        start_epoch, remaining, resume_state = mod._resolve_centralized_resume(
            cfg, tmp_path / "final.pt", resume=False, model=model, optimizer=optimizer, scheduler=None
        )

        assert (start_epoch, remaining, resume_state) == (0, 10, None)

    def test_resume_flag_but_no_checkpoint_returns_full_budget(self, tmp_path):
        torch = pytest.importorskip("torch")
        mod = _load_run_centralized_module()
        cfg = self._cfg(epochs=10)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        start_epoch, remaining, resume_state = mod._resolve_centralized_resume(
            cfg, tmp_path / "final.pt", resume=True, model=model, optimizer=optimizer, scheduler=None
        )

        assert (start_epoch, remaining, resume_state) == (0, 10, None)

    def test_resume_with_existing_checkpoint_computes_remaining(self, tmp_path):
        torch = pytest.importorskip("torch")
        from fedmammobench.utils.checkpoint import save_checkpoint

        mod = _load_run_centralized_module()
        cfg = self._cfg(epochs=10)
        ckpt_model = torch.nn.Linear(4, 2)
        ckpt_path = tmp_path / "final.pt"
        save_checkpoint(
            ckpt_path, ckpt_model, epoch=4,
            extra={"resume_state": {"best_value": 0.8, "best_epoch": 3, "rounds_no_improve": 1}},
        )

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        start_epoch, remaining, resume_state = mod._resolve_centralized_resume(
            cfg, ckpt_path, resume=True, model=model, optimizer=optimizer, scheduler=None
        )

        assert start_epoch == 5
        assert remaining == 5  # epochs=10, start_epoch=5 -> 5 left
        assert resume_state == {"best_value": 0.8, "best_epoch": 3, "rounds_no_improve": 1}

    def test_resume_with_shrunk_epochs_yields_remaining_le_zero(self, tmp_path):
        """cfg.training.epochs shrunk below what the checkpoint already
        reached -> remaining <= 0, the caller's cue to skip fit() entirely."""
        torch = pytest.importorskip("torch")
        from fedmammobench.utils.checkpoint import save_checkpoint

        mod = _load_run_centralized_module()
        cfg = self._cfg(epochs=5)
        ckpt_model = torch.nn.Linear(4, 2)
        ckpt_path = tmp_path / "final.pt"
        save_checkpoint(ckpt_path, ckpt_model, epoch=9)  # already ran 10 epochs

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        start_epoch, remaining, _ = mod._resolve_centralized_resume(
            cfg, ckpt_path, resume=True, model=model, optimizer=optimizer, scheduler=None
        )

        assert remaining <= 0
