"""Tests for federated resume-after-a-crash: NodeMetricsRecorder's
round_offset/resume_state/save_every_round, server.py's round-offset
plumbing (_make_on_fit_config_fn, _make_on_evaluate_config_fn,
_attach_federated_logging, _build_evaluate_fn), _initial_parameters'
checkpoint loading, and run_simulation's early bail-out when a resumed
checkpoint already reached federated.rounds.

Mirrors tests/test_federated_best_checkpoint.py's approach (drive
NodeMetricsRecorder directly with synthetic Flower Fit/Evaluate results, no
Ray/real clients). Flower's round loop can't be told to "start counting at
round N" (its internal counter always restarts at 1 for a new process) — the
round_offset mechanism compensates by shifting only what THIS project logs,
never what Flower's own strategy sees. See CHANGELOG.md [0.8.0].

Run::

    pytest tests/test_resume_federated.py -v
"""

from __future__ import annotations

import csv as csv_module
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _cfg(
    *,
    rounds: int = 10,
    save_best_checkpoint: bool = True,
    best_checkpoint_metric: str = "roc_auc",
    early_stopping_patience: int = 0,
):
    from fedmammobench.configs.schema import (
        DataConfig,
        ExperimentConfig,
        FederatedConfig,
        ModelConfig,
    )

    return ExperimentConfig(
        name="fedresume",
        mode="federated",
        seed=42,
        device="cpu",
        data=DataConfig(name="cbis_ddsm", image_size=32),
        model=ModelConfig(name="resnet18", pretrained=False, in_channels=1),
        federated=FederatedConfig(
            num_clients=2,
            rounds=rounds,
            save_best_checkpoint=save_best_checkpoint,
            best_checkpoint_metric=best_checkpoint_metric,
            early_stopping_patience=early_stopping_patience,
        ),
    )


def _make_fit_results(params):
    from flwr.common import Code, FitRes, Status

    status = Status(code=Code.OK, message="ok")
    return [
        (
            object(),
            FitRes(
                status=status,
                parameters=params,
                num_examples=10,
                metrics={"client_id": 0, "train_loss": 0.5, "fit_seconds": 1.0, "local_epochs": 1},
            ),
        )
    ]


def _make_eval_results(roc_auc: float | None):
    from flwr.common import Code, EvaluateRes, Status

    status = Status(code=Code.OK, message="ok")
    m = {"client_id": 0}
    if roc_auc is not None:
        m["roc_auc"] = roc_auc
    return [(object(), EvaluateRes(status=status, loss=0.5, num_examples=5, metrics=m))]


class _DummyStrategy:
    """aggregate_fit returns round-scaled parameters; also records the raw
    round number it was called with, so tests can prove the offset never
    reaches Flower's own strategy — only NodeMetricsRecorder's bookkeeping."""

    def __init__(self, base_ndarrays):
        self.base_ndarrays = base_ndarrays
        self.seen_fit_rounds: list[int] = []
        self.seen_eval_rounds: list[int] = []

    def aggregate_fit(self, server_round, results, failures):
        from flwr.common import ndarrays_to_parameters

        self.seen_fit_rounds.append(server_round)
        scaled = [arr * server_round for arr in self.base_ndarrays]
        return ndarrays_to_parameters(scaled), {}

    def aggregate_evaluate(self, server_round, results, failures):
        self.seen_eval_rounds.append(server_round)
        return 0.5, {}


def _run_rounds(strategy, recorder, roc_aucs, *, start=1):
    for i, auc in enumerate(roc_aucs, start=start):
        strategy.aggregate_fit(i, _make_fit_results(None), [])
        strategy.aggregate_evaluate(i, _make_eval_results(auc), [])


@pytest.fixture
def setup(tmp_path):
    pytest.importorskip("torch")
    from fedmammobench.federated.node_logging import NodeMetricsRecorder
    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.models import build_model

    cfg = _cfg()
    model = build_model(cfg.model)
    base_ndarrays = state_dict_to_ndarrays(model)
    return cfg, base_ndarrays, tmp_path


class TestNodeMetricsRecorderResume:
    def test_round_offset_applied_to_recorded_rows_not_to_wrapped_strategy(self, setup):
        cfg, base_ndarrays, tmp_path = setup
        from fedmammobench.federated.node_logging import NodeMetricsRecorder

        strategy = _DummyStrategy(base_ndarrays)
        recorder = NodeMetricsRecorder(tmp_path, cfg=cfg, round_offset=7)
        recorder.wrap(strategy)
        recorder.start()

        _run_rounds(strategy, recorder, [0.85])

        assert strategy.seen_fit_rounds == [1]
        assert strategy.seen_eval_rounds == [1]

        fit_csv = tmp_path / "clients" / "client_0" / "fit_metrics.csv"
        eval_csv = tmp_path / "clients" / "client_0" / "eval_metrics.csv"
        with fit_csv.open() as f:
            assert list(csv_module.DictReader(f))[0]["round"] == "8"
        with eval_csv.open() as f:
            assert list(csv_module.DictReader(f))[0]["round"] == "8"

    def test_resume_state_seeds_patience_counter(self, setup):
        cfg, base_ndarrays, tmp_path = setup
        from fedmammobench.federated.node_logging import (
            EarlyStoppingTriggered,
            NodeMetricsRecorder,
        )

        cfg.federated.early_stopping_patience = 2
        strategy = _DummyStrategy(base_ndarrays)
        recorder = NodeMetricsRecorder(
            tmp_path,
            cfg=cfg,
            round_offset=7,
            resume_state={"best_value": 0.85, "best_round": 2, "rounds_no_improve": 1},
        )
        recorder.wrap(strategy)
        recorder.start()

        # rounds_no_improve starts at 1 (seeded); one more no-improvement
        # round (0.80 < 0.85) should hit patience=2 immediately.
        with pytest.raises(EarlyStoppingTriggered) as excinfo:
            _run_rounds(strategy, recorder, [0.80])

        assert excinfo.value.server_round == 8  # absolute, offset applied
        assert excinfo.value.best_round == 2

    def test_save_every_round_writes_global_model_pt_independent_of_best_tracking(self, setup):
        cfg, base_ndarrays, tmp_path = setup
        from fedmammobench.federated.node_logging import NodeMetricsRecorder

        cfg.federated.save_best_checkpoint = False
        strategy = _DummyStrategy(base_ndarrays)
        recorder = NodeMetricsRecorder(tmp_path, cfg=cfg, save_every_round=True)
        recorder.wrap(strategy)
        recorder.start()

        _run_rounds(strategy, recorder, [0.5])  # no improvement possible w/ tracking off

        weights_dir = tmp_path.parent / "weights"
        assert (weights_dir / "global_model.pt").is_file()

    def test_progressive_save_extra_includes_live_resume_state(self, setup):
        cfg, base_ndarrays, tmp_path = setup
        import torch

        from fedmammobench.federated.node_logging import NodeMetricsRecorder

        strategy = _DummyStrategy(base_ndarrays)
        recorder = NodeMetricsRecorder(tmp_path, cfg=cfg, save_every_round=True)
        recorder.wrap(strategy)
        recorder.start()

        _run_rounds(strategy, recorder, [0.60, 0.85])  # best = round 2

        weights_dir = tmp_path.parent / "weights"
        payload = torch.load(weights_dir / "global_model.pt", map_location="cpu")
        assert payload["epoch"] == 2
        assert payload["extra"]["resume_state"] == {
            "best_value": pytest.approx(0.85),
            "best_round": 2,
            "rounds_no_improve": 0,
        }

    def test_write_timing_summary_distinguishes_session_from_absolute_rounds(self, setup):
        cfg, base_ndarrays, tmp_path = setup
        from fedmammobench.federated.node_logging import NodeMetricsRecorder

        strategy = _DummyStrategy(base_ndarrays)
        recorder = NodeMetricsRecorder(tmp_path, cfg=cfg, round_offset=14)
        recorder.wrap(strategy)
        recorder.start()
        _run_rounds(strategy, recorder, [0.7, 0.8, 0.75])  # 3 rounds this session -> #15,16,17

        recorder.write_timing_summary(cfg, total_seconds=12.0)
        text = (tmp_path / "final_summary.txt").read_text()
        assert "Rondas completadas (sesión) : 3" in text
        assert "Ronda absoluta final : #17" in text

    def test_write_timing_summary_default_offset_byte_identical_label(self, setup):
        """round_offset=0 (every non-resumed run) must keep the original,
        unqualified 'Rondas completadas' label and add no absolute-round line."""
        cfg, base_ndarrays, tmp_path = setup
        from fedmammobench.federated.node_logging import NodeMetricsRecorder

        strategy = _DummyStrategy(base_ndarrays)
        recorder = NodeMetricsRecorder(tmp_path, cfg=cfg)
        recorder.wrap(strategy)
        recorder.start()
        _run_rounds(strategy, recorder, [0.7, 0.8])

        recorder.write_timing_summary(cfg, total_seconds=5.0)
        text = (tmp_path / "final_summary.txt").read_text()
        assert "Rondas completadas : 2" in text
        assert "(sesión)" not in text
        assert "Ronda absoluta final" not in text


class TestServerRoundOffsetPlumbing:
    def test_make_on_fit_config_fn_applies_offset(self):
        import fedmammobench.federated.server as server_mod
        from fedmammobench.configs.schema import ExperimentConfig, TrainingConfig

        cfg = ExperimentConfig(name="x", mode="federated")
        cfg.training = TrainingConfig(local_epochs=3)

        fn = server_mod._make_on_fit_config_fn(cfg, round_offset=7)
        config = fn(1)

        assert config["server_round"] == 8
        assert config["current_round"] == 8
        assert config["local_epochs"] == 3

    def test_make_on_fit_config_fn_default_offset_zero(self):
        import fedmammobench.federated.server as server_mod
        from fedmammobench.configs.schema import ExperimentConfig

        fn = server_mod._make_on_fit_config_fn(ExperimentConfig(name="x", mode="federated"))
        assert fn(5)["current_round"] == 5

    def test_make_on_evaluate_config_fn_applies_offset(self):
        import fedmammobench.federated.server as server_mod

        fn = server_mod._make_on_evaluate_config_fn(round_offset=7)
        assert fn(1) == {"current_round": 8}
        fn0 = server_mod._make_on_evaluate_config_fn()
        assert fn0(5) == {"current_round": 5}

    def test_attach_federated_logging_applies_offset_not_to_wrapped_strategy(self, tmp_path):
        import fedmammobench.federated.server as server_mod
        from fedmammobench.utils.csv_logger import CSVLogger

        class FakeStrategy:
            def __init__(self):
                self.seen_rounds: list[int] = []

            def aggregate_evaluate(self, server_round, results, failures):
                self.seen_rounds.append(server_round)
                return 0.5, {"roc_auc": 0.7}

        strategy = FakeStrategy()
        csv_logger = CSVLogger(tmp_path / "server_federated_metrics.csv")
        server_mod._attach_federated_logging(strategy, None, csv_logger, round_offset=7)

        strategy.aggregate_evaluate(1, [], [])

        assert strategy.seen_rounds == [1]  # Flower's own strategy sees the raw round
        with (tmp_path / "server_federated_metrics.csv").open() as f:
            row = next(csv_module.DictReader(f))
        assert row["round"] == "8"

    def test_build_evaluate_fn_applies_offset(self, tmp_path):
        pytest.importorskip("torch")
        import fedmammobench.federated.server as server_mod
        from fedmammobench.utils.csv_logger import CSVLogger

        cfg = _cfg()
        fake_evaluator = MagicMock()
        fake_evaluator.evaluate.return_value = {"loss": 0.5, "roc_auc": 0.7}
        csv_logger = CSVLogger(tmp_path / "server_metrics.csv")

        class _FakeDataset:
            def __len__(self):
                return 5

        with (
            patch.object(server_mod, "build_model", return_value=MagicMock(to=lambda d: MagicMock())),
            patch.object(server_mod, "build_loss", return_value=MagicMock(to=lambda d: MagicMock())),
            patch.object(server_mod, "build_dataloader", return_value=MagicMock()),
            patch.object(server_mod, "Evaluator", return_value=fake_evaluator),
            patch.object(server_mod, "load_ndarrays_to_state_dict"),
        ):
            evaluate_fn = server_mod._build_evaluate_fn(
                cfg, _FakeDataset(), None, None, csv_logger, round_offset=7
            )
            assert evaluate_fn is not None
            evaluate_fn(1, [], {})

        with (tmp_path / "server_metrics.csv").open() as f:
            row = next(csv_module.DictReader(f))
        assert row["round"] == "8"
        assert row["phase"] == "centralized"


class TestInitialParametersResume:
    def test_no_resume_path_returns_none_payload(self):
        pytest.importorskip("torch")
        from fedmammobench.federated.server import _initial_parameters

        cfg = _cfg()
        _, payload = _initial_parameters(cfg, resume_checkpoint_path=None)
        assert payload is None

    def test_resume_path_loads_checkpoint_weights(self, tmp_path):
        pytest.importorskip("torch")
        import torch
        from flwr.common import parameters_to_ndarrays

        from fedmammobench.federated.server import _initial_parameters
        from fedmammobench.models import build_model
        from fedmammobench.utils.checkpoint import save_checkpoint

        cfg = _cfg()
        model = build_model(cfg.model)
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(0.0).add_(3.0)
        ckpt_path = tmp_path / "global_model.pt"
        save_checkpoint(
            ckpt_path, model, epoch=5,
            extra={"resume_state": {"best_value": 0.77, "best_round": 3, "rounds_no_improve": 1}},
        )

        params, payload = _initial_parameters(cfg, resume_checkpoint_path=ckpt_path)

        assert payload is not None
        assert payload["epoch"] == 5
        assert payload["extra"]["resume_state"]["best_round"] == 3
        nd = parameters_to_ndarrays(params)
        assert (nd[0] == 3.0).all()  # first array (conv1.weight) is a real param, not a BN buffer

    def test_nonexistent_resume_path_falls_back_to_fresh_init(self, tmp_path):
        pytest.importorskip("torch")
        from fedmammobench.federated.server import _initial_parameters

        cfg = _cfg()
        _, payload = _initial_parameters(cfg, resume_checkpoint_path=tmp_path / "missing.pt")
        assert payload is None


class TestRunSimulationResumeBailout:
    def test_remaining_rounds_le_zero_skips_start_simulation(self, tmp_path):
        pytest.importorskip("torch")
        import fedmammobench.federated.server as server_mod
        from fedmammobench.configs.schema import (
            DataConfig,
            ExperimentConfig,
            FederatedConfig,
            ModelConfig,
            WandbConfig,
        )
        from fedmammobench.models import build_model
        from fedmammobench.utils.checkpoint import save_checkpoint

        cfg = ExperimentConfig(
            name="bailout_test",
            mode="federated",
            device="cpu",
            data=DataConfig(name="none"),
            model=ModelConfig(name="resnet18", pretrained=False, in_channels=1),
            federated=FederatedConfig(
                num_clients=2, rounds=5,
                min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
            ),
            wandb=WandbConfig(enabled=False),
        )
        out_dir = tmp_path / "run"
        weights_dir = out_dir.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model = build_model(cfg.model)
        # Checkpoint already at round 5 == cfg.federated.rounds -> remaining = 0.
        save_checkpoint(weights_dir / "global_model.pt", model, epoch=5)

        def _fail_if_called(*a, **k):
            raise AssertionError("fl.simulation.start_simulation should not be called")

        with (
            patch.object(server_mod.fl.simulation, "start_simulation", side_effect=_fail_if_called),
            patch.object(server_mod, "build_dataset") as mock_build_dataset,
        ):
            history = server_mod.run_simulation(cfg, output_dir=out_dir, resume=True)

        assert history is None
        assert not mock_build_dataset.called
