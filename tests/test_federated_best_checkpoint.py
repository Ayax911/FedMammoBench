"""Tests for federated best-checkpoint tracking (federated.save_best_checkpoint).

Mirrors tests/test_node_logging.py's approach (drive NodeMetricsRecorder
directly with synthetic Flower Fit/Evaluate results, no Ray/real clients) and
tests/test_best_checkpoint.py's approach for the centralized analogue (prove
weight *identity*, not just that a file exists).

Added alongside exp56 / the centralized save_best_checkpoint feature: exp50-55
showed the federated path has the exact same "last round can be worse than an
earlier one" problem, but had no mechanism at all to recover the better
checkpoint — this closes that gap. See memory training_pipeline_no_early_stopping.

Run::

    pytest tests/test_federated_best_checkpoint.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _cfg(*, save_best_checkpoint: bool = True, best_checkpoint_metric: str = "roc_auc"):
    from fedmammobench.configs.schema import (
        DataConfig,
        ExperimentConfig,
        FederatedConfig,
        ModelConfig,
    )

    return ExperimentConfig(
        name="fedbest",
        mode="federated",
        seed=42,
        device="cpu",
        data=DataConfig(name="cbis_ddsm", image_size=32),
        model=ModelConfig(name="resnet18", pretrained=False, in_channels=1),
        federated=FederatedConfig(
            num_clients=2,
            rounds=3,
            save_best_checkpoint=save_best_checkpoint,
            best_checkpoint_metric=best_checkpoint_metric,
        ),
    )


def _make_fit_results(params):
    from flwr.common import Code, FitRes, Status

    status = Status(code=Code.OK, message="ok")
    results = []
    for cid in (0, 1):
        res = FitRes(
            status=status,
            parameters=params,
            num_examples=10,
            metrics={"client_id": cid, "train_loss": 0.5, "fit_seconds": 1.0, "local_epochs": 1},
        )
        results.append((object(), res))
    return results


def _make_eval_results(roc_auc: float | None):
    """Both clients report the same roc_auc (equal weights) so the weighted
    average record_eval computes lands exactly on the value passed in —
    makes the "which round is best" assertions unambiguous.
    """
    from flwr.common import Code, EvaluateRes, Status

    status = Status(code=Code.OK, message="ok")
    results = []
    for cid in (0, 1):
        m = {"client_id": cid}
        if roc_auc is not None:
            m["roc_auc"] = roc_auc
        res = EvaluateRes(status=status, loss=0.5, num_examples=5, metrics=m)
        results.append((object(), res))
    return results


class _DummyStrategy:
    """aggregate_fit returns round-scaled parameters (base * server_round) so
    each round's weights are distinguishable — lets tests prove a saved
    checkpoint holds a *specific* round's weights, not just *some* round's.
    """

    def __init__(self, base_ndarrays):
        self.base_ndarrays = base_ndarrays

    def aggregate_fit(self, server_round, results, failures):
        from flwr.common import ndarrays_to_parameters

        scaled = [arr * server_round for arr in self.base_ndarrays]
        return ndarrays_to_parameters(scaled), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return 0.5, {}


@pytest.fixture
def setup(tmp_path):
    pytest.importorskip("torch")
    from fedmammobench.federated.node_logging import NodeMetricsRecorder
    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.models import build_model

    cfg = _cfg()
    model = build_model(cfg.model)
    base_ndarrays = state_dict_to_ndarrays(model)

    strategy = _DummyStrategy(base_ndarrays)
    recorder = NodeMetricsRecorder(tmp_path, cfg=cfg)
    recorder.wrap(strategy)
    recorder.start()
    return cfg, strategy, recorder, base_ndarrays, tmp_path


def _first_param_name(cfg) -> str:
    from fedmammobench.models import build_model

    model = build_model(cfg.model)
    return next(iter(model.state_dict()))


def _run_rounds(strategy, recorder, roc_aucs: list[float | None]) -> None:
    for i, auc in enumerate(roc_aucs, start=1):
        strategy.aggregate_fit(i, _make_fit_results(None), [])
        strategy.aggregate_evaluate(i, _make_eval_results(auc), [])


class TestBestGlobalCheckpointSaved:
    def test_saved_on_improvement_with_correct_metadata(self, setup) -> None:
        _, strategy, recorder, _, _ = setup
        _run_rounds(strategy, recorder, [0.60, 0.85, 0.70])  # best = round 2

        assert recorder._best_round == 2
        assert recorder._best_value == pytest.approx(0.85)
        assert recorder._best_path is not None
        assert recorder._best_path.is_file()

        import torch

        payload = torch.load(recorder._best_path, map_location="cpu")
        assert payload["extra"]["source"] == "federated_global_best"
        assert payload["extra"]["round"] == 2
        assert payload["extra"]["metric"] == "roc_auc"
        assert payload["extra"]["value"] == pytest.approx(0.85)

    def test_checkpoint_holds_best_round_weights_not_last(self, setup) -> None:
        """The .pt on disk must hold round 2's weights, not round 3's."""
        cfg, strategy, recorder, base_ndarrays, _ = setup
        _run_rounds(strategy, recorder, [0.60, 0.85, 0.70])  # best = round 2, last = round 3

        import torch

        from fedmammobench.models import build_model
        from fedmammobench.utils.checkpoint import load_checkpoint

        reloaded = build_model(cfg.model)
        load_checkpoint(recorder._best_path, reloaded, strict=True)
        key = _first_param_name(cfg)
        expected_round2 = torch.as_tensor(base_ndarrays[0]) * 2
        expected_round3 = torch.as_tensor(base_ndarrays[0]) * 3

        actual = reloaded.state_dict()[key]
        assert torch.allclose(actual, expected_round2), "global_best.pt should hold round 2's weights"
        assert not torch.allclose(actual, expected_round3), "global_best.pt should NOT hold round 3's weights"

    def test_final_global_model_unaffected_by_best_tracking(self, setup) -> None:
        """save_global_model always writes the LAST round, regardless of tracking."""
        cfg, strategy, recorder, base_ndarrays, _ = setup
        _run_rounds(strategy, recorder, [0.60, 0.85, 0.70])

        out = recorder.save_global_model(cfg)
        assert out is not None and out.is_file()

        import torch

        from fedmammobench.models import build_model
        from fedmammobench.utils.checkpoint import load_checkpoint

        reloaded = build_model(cfg.model)
        payload = load_checkpoint(out, reloaded, strict=True)
        assert payload["extra"]["source"] == "federated_global"  # unchanged from before this feature
        key = _first_param_name(cfg)
        expected_round3 = torch.as_tensor(base_ndarrays[0]) * 3
        assert torch.allclose(reloaded.state_dict()[key], expected_round3)

    def test_no_best_checkpoint_when_disabled(self, tmp_path) -> None:
        pytest.importorskip("torch")
        from fedmammobench.federated.node_logging import NodeMetricsRecorder
        from fedmammobench.federated.param_utils import state_dict_to_ndarrays
        from fedmammobench.models import build_model

        cfg = _cfg(save_best_checkpoint=False)
        model = build_model(cfg.model)
        strategy = _DummyStrategy(state_dict_to_ndarrays(model))
        recorder = NodeMetricsRecorder(tmp_path, cfg=cfg)
        recorder.wrap(strategy)
        recorder.start()

        _run_rounds(strategy, recorder, [0.60, 0.85, 0.70])

        assert recorder._best_path is None
        assert recorder._best_round is None
        # write_timing_summary must not crash and must fall back to "roc_auc"
        # (cosmetic max-by-round in the text summary) even with tracking off.
        recorder.write_timing_summary(cfg, total_seconds=1.0)
        summary_text = (tmp_path / "final_summary.txt").read_text()
        assert "Checkpoint mejor" not in summary_text

    def test_metric_missing_from_agg_is_safe(self, setup) -> None:
        """Clients that never report the tracked metric must not crash anything."""
        _, strategy, recorder, _, _ = setup
        _run_rounds(strategy, recorder, [None, None, None])  # no client ever reports roc_auc

        assert recorder._best_path is None
        assert recorder._best_round is None

    def test_recorder_without_cfg_disables_tracking(self, tmp_path) -> None:
        """cfg=None (e.g. existing callers/tests) must not track or crash."""
        pytest.importorskip("torch")
        from fedmammobench.federated.node_logging import NodeMetricsRecorder
        from fedmammobench.federated.param_utils import state_dict_to_ndarrays
        from fedmammobench.models import build_model

        cfg = _cfg()
        model = build_model(cfg.model)
        strategy = _DummyStrategy(state_dict_to_ndarrays(model))
        recorder = NodeMetricsRecorder(tmp_path)  # no cfg=
        recorder.wrap(strategy)
        recorder.start()

        _run_rounds(strategy, recorder, [0.60, 0.85, 0.70])

        assert recorder._track_best is False
        assert recorder._best_path is None


class TestFederatedBestCheckpointConfig:
    def test_validate_rejects_auc_pr(self) -> None:
        from fedmammobench.configs.federated_config import FederatedConfig

        cfg = FederatedConfig(save_best_checkpoint=True, best_checkpoint_metric="auc_pr")
        with pytest.raises(ValueError, match="best_checkpoint_metric"):
            cfg.validate()

    def test_validate_accepts_roc_auc(self) -> None:
        from fedmammobench.configs.federated_config import FederatedConfig

        FederatedConfig(save_best_checkpoint=True, best_checkpoint_metric="roc_auc").validate()

    def test_invalid_metric_ignored_when_disabled(self) -> None:
        from fedmammobench.configs.federated_config import FederatedConfig

        # Same convention as TrainingConfig: only validated when the feature is on.
        FederatedConfig(save_best_checkpoint=False, best_checkpoint_metric="auc_pr").validate()

    def test_federated_best_checkpoint_metrics_excludes_auc_pr(self) -> None:
        from fedmammobench.configs.federated_config import FEDERATED_BEST_CHECKPOINT_METRICS
        from fedmammobench.configs.training_config import BEST_CHECKPOINT_METRICS

        assert "auc_pr" in BEST_CHECKPOINT_METRICS
        assert "auc_pr" not in FEDERATED_BEST_CHECKPOINT_METRICS
        assert set(FEDERATED_BEST_CHECKPOINT_METRICS) == set(BEST_CHECKPOINT_METRICS) - {"auc_pr"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
