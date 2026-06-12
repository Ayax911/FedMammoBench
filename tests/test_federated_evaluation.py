"""Tests for the pure-federated evaluation path.

Covers two pieces:

1. ``data.name='none'`` lets the server skip dataset construction entirely
   — no manifest, no images, no errors. This is the "no holdout on the
   server" deployment mode.
2. ``_attach_federated_logging`` wraps a strategy's ``aggregate_evaluate``
   so that per-round client-side metrics are weighted-averaged by the
   underlying strategy *and* written to ``server_federated_metrics.csv``
   with ``phase='federated'``. The wrapper must preserve the original
   return value.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# fedmammobench pulls torch transitively (datasets/base.py, federated/server.py).
# Skip the whole module if torch isn't installed in the current env — the
# tests still cover everything end-to-end where torch is available.
pytest.importorskip("torch")
pytest.importorskip("flwr")


def test_build_dataset_none_returns_empty() -> None:
    """``data.name='none'`` is a sentinel that disables dataset construction."""
    from fedmammobench.configs.schema import DataConfig, ExperimentConfig
    from fedmammobench.datasets import build_dataset

    cfg = ExperimentConfig(name="server_no_holdout", data=DataConfig(name="none"))
    out = build_dataset(cfg)
    assert out == {}, f"expected empty mapping, got {out!r}"


def test_attach_federated_logging_writes_csv(tmp_path: Path) -> None:
    """Wrapper logs a row with phase='federated' and preserves return value."""
    from fedmammobench.federated.server import _attach_federated_logging
    from fedmammobench.utils.csv_logger import CSVLogger

    csv_path = tmp_path / "server_federated_metrics.csv"
    csv_logger = CSVLogger(csv_path)

    captured: dict = {}

    def fake_aggregate_evaluate(server_round, results, failures):
        captured["round"] = server_round
        captured["n_results"] = len(results)
        return 0.42, {"roc_auc": 0.9, "f1": 0.8, "accuracy": 0.85}

    strategy = SimpleNamespace(aggregate_evaluate=fake_aggregate_evaluate)
    _attach_federated_logging(strategy, tb_writer=None, csv_logger=csv_logger)

    # Fake EvaluateRes-like objects: only need a `num_examples` attribute.
    fake_results = [
        ("client0_proxy", SimpleNamespace(num_examples=10)),
        ("client1_proxy", SimpleNamespace(num_examples=20)),
    ]
    loss, metrics = strategy.aggregate_evaluate(1, fake_results, [])

    # Return value preserved verbatim
    assert loss == pytest.approx(0.42)
    assert metrics == {"roc_auc": 0.9, "f1": 0.8, "accuracy": 0.85}
    assert captured == {"round": 1, "n_results": 2}

    # CSV row contains the federated metrics + bookkeeping fields
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["phase"] == "federated"
    assert int(row["round"]) == 1
    assert int(row["num_examples_total"]) == 30
    assert int(row["num_clients"]) == 2
    assert float(row["loss"]) == pytest.approx(0.42)
    assert float(row["roc_auc"]) == pytest.approx(0.9)
    assert float(row["f1"]) == pytest.approx(0.8)
    assert float(row["accuracy"]) == pytest.approx(0.85)


def test_attach_federated_logging_handles_none_loss(tmp_path: Path) -> None:
    """When the strategy reports loss=None (no clients evaluated), do not crash."""
    from fedmammobench.federated.server import _attach_federated_logging
    from fedmammobench.utils.csv_logger import CSVLogger

    csv_logger = CSVLogger(tmp_path / "server_federated_metrics.csv")

    def fake_aggregate_evaluate(server_round, results, failures):  # noqa: ARG001
        return None, {}

    strategy = SimpleNamespace(aggregate_evaluate=fake_aggregate_evaluate)
    _attach_federated_logging(strategy, tb_writer=None, csv_logger=csv_logger)

    loss, metrics = strategy.aggregate_evaluate(1, [], [])
    assert loss is None
    assert metrics == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
