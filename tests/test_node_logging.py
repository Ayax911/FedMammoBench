"""Tests for per-node metric logging, timing, and global-model capture.

Drives NodeMetricsRecorder directly with synthetic Flower Fit/Evaluate results
(no Ray, no real clients), then checks the per-client subfolders, timing CSVs,
and the saved global checkpoint.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _minimal_cfg():
    from fedmammobench.configs.schema import (
        DataConfig,
        ExperimentConfig,
        FederatedConfig,
        ModelConfig,
    )

    return ExperimentConfig(
        name="nodelog",
        mode="federated",
        seed=42,
        device="cpu",
        data=DataConfig(name="cbis_ddsm", image_size=32),
        model=ModelConfig(name="resnet18", pretrained=False, in_channels=1),
        federated=FederatedConfig(num_clients=2, rounds=2),
    )


def _make_fit_results(params):
    from flwr.common import Code, FitRes, Status

    status = Status(code=Code.OK, message="ok")
    results = []
    for cid in (0, 1):
        res = FitRes(
            status=status,
            parameters=params,
            num_examples=10 + cid,
            metrics={
                "client_id": cid,
                "train_loss": 0.5 + cid,
                "task_loss": 0.4 + cid,
                "fit_seconds": 1.0 + cid,
                "local_epochs": 1,
            },
        )
        results.append((object(), res))
    return results


def _make_eval_results(*, with_auc: bool):
    from flwr.common import Code, EvaluateRes, Status

    status = Status(code=Code.OK, message="ok")
    results = []
    for cid in (0, 1):
        m = {"client_id": cid, "accuracy": 0.8, "eval_seconds": 0.3 + cid}
        if with_auc:
            m["roc_auc"] = 0.75
        res = EvaluateRes(status=status, loss=0.6, num_examples=5 + cid, metrics=m)
        results.append((object(), res))
    return results


@pytest.fixture
def recorder_setup(tmp_path):
    pytest.importorskip("torch")
    from flwr.common import ndarrays_to_parameters

    from fedmammobench.federated.node_logging import NodeMetricsRecorder
    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.models import build_model

    cfg = _minimal_cfg()
    model = build_model(cfg.model)
    params = ndarrays_to_parameters(state_dict_to_ndarrays(model))

    class DummyStrategy:
        def aggregate_fit(self, server_round, results, failures):
            return params, {"agg": 1.0}

        def aggregate_evaluate(self, server_round, results, failures):
            return 0.6, {"accuracy": 0.8}

    strategy = DummyStrategy()
    recorder = NodeMetricsRecorder(tmp_path)
    recorder.wrap(strategy)
    recorder.start()
    return cfg, strategy, recorder, params, tmp_path


def test_per_node_subfolders_and_metrics(recorder_setup) -> None:
    cfg, strategy, recorder, params, root = recorder_setup
    fit_results = _make_fit_results(params)

    strategy.aggregate_fit(1, fit_results, [])
    strategy.aggregate_evaluate(1, _make_eval_results(with_auc=False), [])
    strategy.aggregate_fit(2, fit_results, [])
    strategy.aggregate_evaluate(2, _make_eval_results(with_auc=True), [])

    for cid in (0, 1):
        cdir = root / "clients" / f"client_{cid}"
        assert (cdir / "fit_metrics.csv").is_file(), f"missing fit csv for client {cid}"
        assert (cdir / "eval_metrics.csv").is_file(), f"missing eval csv for client {cid}"

        with (cdir / "fit_metrics.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2  # two rounds
        assert {r["round"] for r in rows} == {"1", "2"}
        assert all(r["client_id"] == str(cid) for r in rows)
        assert float(rows[0]["fit_seconds"]) == pytest.approx(1.0 + cid)

        # eval CSV has a stable schema even though round 1 omitted roc_auc.
        with (cdir / "eval_metrics.csv").open() as fh:
            erows = list(csv.DictReader(fh))
        assert len(erows) == 2
        assert "roc_auc" in erows[0]  # column present
        assert erows[0]["roc_auc"] == ""  # round 1: omitted -> empty
        assert erows[1]["roc_auc"] == "0.75"  # round 2: present
        assert float(erows[0]["eval_seconds"]) == pytest.approx(0.3 + cid)


def test_timing_csvs_written(recorder_setup) -> None:
    cfg, strategy, recorder, params, root = recorder_setup
    strategy.aggregate_fit(1, _make_fit_results(params), [])
    strategy.aggregate_fit(2, _make_fit_results(params), [])

    timing = root / "server_timing.csv"
    assert timing.is_file()
    with timing.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["num_clients_fit"] == "2"
    # sum_fit_seconds = (1.0) + (2.0) for cids 0 and 1
    assert float(rows[0]["sum_fit_seconds"]) == pytest.approx(3.0)
    assert float(rows[0]["max_fit_seconds"]) == pytest.approx(2.0)

    recorder.write_timing_summary(cfg, total_seconds=12.5)
    summary = root / "timing_summary.csv"
    assert summary.is_file()
    with summary.open() as fh:
        srow = next(csv.DictReader(fh))
    assert float(srow["total_seconds"]) == pytest.approx(12.5)
    assert srow["num_rounds"] == "2"
    assert float(srow["avg_seconds_per_round"]) == pytest.approx(6.25)


def test_global_model_checkpoint_saved(recorder_setup) -> None:
    cfg, strategy, recorder, params, root = recorder_setup
    strategy.aggregate_fit(1, _make_fit_results(params), [])

    out = recorder.save_global_model(cfg)
    assert out is not None and out.is_file()

    # The checkpoint must load back into a fresh model of the same config.
    import torch

    from fedmammobench.models import build_model
    from fedmammobench.utils.checkpoint import load_checkpoint

    model = build_model(cfg.model)
    payload = load_checkpoint(out, model)
    assert payload["extra"]["source"] == "federated_global"
    assert isinstance(model, torch.nn.Module)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
