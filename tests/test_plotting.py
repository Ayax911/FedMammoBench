"""Tests for fedmammobench.plotting.

The main regression covered here: _collect_node_dfs used to strip only
"cid_"/"node_" prefixes, but NodeMetricsRecorder._client_root
(federated/node_logging.py) actually names directories "client_<id>" — so
int("client_0") raised ValueError, the except clause did `continue`, and
EVERY node directory was silently skipped. No federated run anywhere in
runs/ had a single nodes_*.png before this fix.

Run::

    pytest tests/test_plotting.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class TestCollectNodeDfs:
    def test_accepts_real_client_prefix(self, tmp_path) -> None:
        """The confirmed bug: real dirs are "client_<id>", not "cid_<id>"."""
        pd = pytest.importorskip("pandas")
        from fedmammobench.plotting import _collect_node_dfs

        clients_dir = tmp_path / "clients"
        for cid in (0, 1, 3):
            d = clients_dir / f"client_{cid}"
            d.mkdir(parents=True)
            pd.DataFrame({"round": [1, 2], "train_loss": [0.5, 0.4]}).to_csv(
                d / "fit_metrics.csv", index=False
            )

        dfs = _collect_node_dfs(tmp_path, "fit_metrics.csv")
        assert set(dfs) == {0, 1, 3}

    def test_still_accepts_legacy_prefixes(self, tmp_path) -> None:
        """cid_<N> and node_<N> (the previously-assumed, never-actual names)."""
        pd = pytest.importorskip("pandas")
        from fedmammobench.plotting import _collect_node_dfs

        clients_dir = tmp_path / "clients"
        for name in ("cid_2", "node_4"):
            d = clients_dir / name
            d.mkdir(parents=True)
            pd.DataFrame({"round": [1], "train_loss": [0.5]}).to_csv(
                d / "fit_metrics.csv", index=False
            )

        dfs = _collect_node_dfs(tmp_path, "fit_metrics.csv")
        assert set(dfs) == {2, 4}

    def test_no_clients_dir_returns_empty(self, tmp_path) -> None:
        pytest.importorskip("pandas")
        from fedmammobench.plotting import _collect_node_dfs

        assert _collect_node_dfs(tmp_path, "fit_metrics.csv") == {}

    def test_unparseable_dir_name_skipped_not_raised(self, tmp_path) -> None:
        pytest.importorskip("pandas")
        from fedmammobench.plotting import _collect_node_dfs

        (tmp_path / "clients" / "not_a_node_dir").mkdir(parents=True)
        assert _collect_node_dfs(tmp_path, "fit_metrics.csv") == {}


class TestNodeLossFrame:
    def test_joins_train_and_val_on_round(self) -> None:
        pd = pytest.importorskip("pandas")
        from fedmammobench.plotting import _node_loss_frame

        fit_df = pd.DataFrame({"round": [1, 2, 3], "train_loss": [0.5, 0.4, 0.3]})
        eval_df = pd.DataFrame({"round": [1, 2, 3], "loss": [0.6, 0.55, 0.5]})

        merged = _node_loss_frame(fit_df, eval_df)
        assert list(merged["round"]) == [1, 2, 3]
        assert list(merged["train_loss"]) == [0.5, 0.4, 0.3]
        assert list(merged["val_loss"]) == [0.6, 0.55, 0.5]

    def test_missing_rounds_produce_nan_not_exception(self) -> None:
        pd = pytest.importorskip("pandas")
        from fedmammobench.plotting import _node_loss_frame

        fit_df = pd.DataFrame({"round": [1, 2], "train_loss": [0.5, 0.4]})
        eval_df = pd.DataFrame({"round": [1, 3], "loss": [0.6, 0.5]})  # round 3 has no fit row

        merged = _node_loss_frame(fit_df, eval_df)
        assert set(merged["round"]) == {1, 2, 3}
        assert merged.set_index("round").loc[3, "train_loss"] != merged.set_index("round").loc[3, "train_loss"]  # NaN

    def test_both_none_returns_none(self) -> None:
        pytest.importorskip("pandas")
        from fedmammobench.plotting import _node_loss_frame

        assert _node_loss_frame(None, None) is None

    def test_eval_only_still_returns_frame(self) -> None:
        pd = pytest.importorskip("pandas")
        from fedmammobench.plotting import _node_loss_frame

        eval_df = pd.DataFrame({"round": [1, 2], "loss": [0.6, 0.5]})
        merged = _node_loss_frame(None, eval_df)
        assert merged is not None
        assert list(merged["val_loss"]) == [0.6, 0.5]


class TestPlotNodesWritesExpectedFiles:
    def _write_run(self, tmp_path, n_nodes=2, n_rounds=3):
        pd = pytest.importorskip("pandas")
        for cid in range(n_nodes):
            d = tmp_path / "clients" / f"client_{cid}"
            d.mkdir(parents=True)
            pd.DataFrame(
                {
                    "round": range(1, n_rounds + 1),
                    "train_loss": [0.5 - 0.05 * r for r in range(n_rounds)],
                }
            ).to_csv(d / "fit_metrics.csv", index=False)
            pd.DataFrame(
                {
                    "round": range(1, n_rounds + 1),
                    "loss": [0.6 - 0.02 * r for r in range(n_rounds)],
                    "roc_auc": [0.7 + 0.01 * r for r in range(n_rounds)],
                    "f1": [0.6 + 0.01 * r for r in range(n_rounds)],
                }
            ).to_csv(d / "eval_metrics.csv", index=False)
        return tmp_path

    def test_writes_overlay_and_per_node_loss_plots(self, tmp_path) -> None:
        pytest.importorskip("matplotlib")
        from fedmammobench.plotting import plot_nodes

        run_dir = self._write_run(tmp_path)
        out_dir = tmp_path / "plots"
        written = plot_nodes(run_dir, out_dir, dpi=50)

        names = {p.name for p in written}
        assert "nodes_train_loss.png" in names
        assert "nodes_val_auc.png" in names
        assert "nodes_val_f1.png" in names
        assert "nodes_loss_train_val.png" in names
        assert "node_0_loss.png" in names
        assert "node_1_loss.png" in names
        for p in written:
            assert p.is_file() and p.stat().st_size > 0

    def test_no_client_dirs_returns_empty_without_raising(self, tmp_path) -> None:
        pytest.importorskip("matplotlib")
        from fedmammobench.plotting import plot_nodes

        assert plot_nodes(tmp_path, tmp_path / "plots", dpi=50) == []


class TestAutoplot:
    def test_never_raises_on_empty_dir(self, tmp_path) -> None:
        from fedmammobench.plotting import autoplot

        autoplot(tmp_path)  # no metrics.csv / server_metrics.csv / clients -> ValueError internally, swallowed

    def test_never_raises_on_missing_run_dir(self, tmp_path) -> None:
        from fedmammobench.plotting import autoplot

        autoplot(tmp_path / "does_not_exist")  # FileNotFoundError internally, swallowed

    def test_survives_missing_matplotlib(self, tmp_path, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("simulated: matplotlib not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        from fedmammobench.plotting import autoplot

        (tmp_path / "metrics.csv").write_text("epoch,train_loss\n0,0.5\n")
        autoplot(tmp_path)  # must not raise despite matplotlib being "missing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
