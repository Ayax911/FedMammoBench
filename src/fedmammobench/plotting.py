"""Generate training/evaluation plots for a completed experiment run.

Supports both centralized and federated runs. Auto-detects the mode based on
which CSV files are present in the run directory. ``scripts/plot_experiment.py``
is a thin CLI wrapper around this module; ``autoplot()`` is the hook called
automatically at the end of a training run (centralized and federated).

Centralized outputs  (reads ``metrics.csv``)
--------------------------------------------
  - loss_curves.png       : train + val loss per epoch
  - metric_curves.png     : val AUC, F1, sensitivity, specificity per epoch

Federated outputs
-----------------
  Server-level (reads ``server_metrics.csv``, ``server_federated_metrics.csv``):
  - server_centralized.png : centralized eval metrics per round
  - server_federated.png   : federated (client-averaged) metrics per round

  Per-node  (reads ``clients/client_<N>/fit_metrics.csv`` and ``eval_metrics.csv``):
  - nodes_train_loss.png     : train loss per round, all nodes overlaid
  - nodes_val_auc.png        : val AUC per round, all nodes overlaid
  - nodes_val_f1.png         : val F1 per round, all nodes overlaid
  - nodes_loss_train_val.png : train loss (left) + val loss (right), all nodes overlaid
  - node_<N>_loss.png        : one node's train + val loss on a single axes
  - node_<N>_curves.png      : per-node detailed curves (loss + metrics)

matplotlib/pandas are imported lazily (inside functions, not at module scope)
so importing this module never requires them — ``autoplot()`` degrades to a
logged warning if they aren't installed, which matters because matplotlib is
in ``requirements.txt`` (so it's in the Docker image) but not in
``pyproject.toml`` (so CI doesn't have it).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from fedmammobench.utils.logging_utils import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metrics to plot (column suffix in the CSV -> display label)
# ---------------------------------------------------------------------------
_TRAIN_VAL_PAIRS: list[tuple[str, str]] = [
    ("loss", "Loss"),
    ("roc_auc", "AUC-ROC"),
    ("f1", "F1"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("accuracy", "Accuracy"),
]

# Fixed palette (not sourced from matplotlib.rcParams) so a module-scope
# import never touches matplotlib, and so node colors are stable across runs
# and across processes. Same 10-color sequence as matplotlib's default
# "tab10" cycle.
_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Lazy matplotlib import
# ---------------------------------------------------------------------------

def _mpl():
    """Import matplotlib (headless) + pyplot + ticker on first use."""
    import matplotlib

    matplotlib.use("Agg")  # headless — no display required
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    return plt, ticker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    import pandas as pd

    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def _savefig(fig: Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    _logger.info("  saved -> %s", path)


def _style_ax(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    _, ticker = _mpl()
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Centralized
# ---------------------------------------------------------------------------

def plot_centralized(run_dir: Path, out_dir: Path, dpi: int) -> list[Path]:
    plt, _ = _mpl()
    df = _load(run_dir / "metrics.csv")
    if df is None:
        _logger.info("  [centralized] metrics.csv not found — skipping.")
        return []

    written: list[Path] = []
    x = df["epoch"] if "epoch" in df.columns else df.index

    # --- loss curves ---
    fig, ax = plt.subplots(figsize=(8, 4))
    if "train_loss" in df.columns:
        ax.plot(x, df["train_loss"], label="Train loss", color=_COLORS[0])
    if "val_loss" in df.columns:
        ax.plot(x, df["val_loss"], label="Val loss", color=_COLORS[1], linestyle="--")
    _style_ax(ax, "Loss curves", "Epoch", "Loss")
    fig.tight_layout()
    out = out_dir / "loss_curves.png"
    _savefig(fig, out, dpi)
    written.append(out)

    # --- metric curves (one subplot per metric) ---
    available = [
        (col_suffix, label)
        for col_suffix, label in _TRAIN_VAL_PAIRS
        if col_suffix != "loss" and f"val_{col_suffix}" in df.columns
    ]
    if available:
        ncols = 2
        nrows = (len(available) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, (col_suffix, label) in enumerate(available):
            ax = axes_flat[i]
            if f"train_{col_suffix}" in df.columns:
                ax.plot(x, df[f"train_{col_suffix}"], label=f"Train {label}", color=_COLORS[0])
            ax.plot(x, df[f"val_{col_suffix}"], label=f"Val {label}", color=_COLORS[1], linestyle="--")
            _style_ax(ax, label, "Epoch", label)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("Validation metrics per epoch", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = out_dir / "metric_curves.png"
        _savefig(fig, out, dpi)
        written.append(out)

    _logger.info("  [centralized] plots written to %s", out_dir)
    return written


# ---------------------------------------------------------------------------
# Federated — server level
# ---------------------------------------------------------------------------

def plot_server(run_dir: Path, out_dir: Path, dpi: int) -> list[Path]:
    plt, _ = _mpl()
    written: list[Path] = []
    for csv_name, out_name, title_prefix in [
        ("server_metrics.csv", "server_centralized.png", "Server centralized eval"),
        ("server_federated_metrics.csv", "server_federated.png", "Server federated (client-avg)"),
    ]:
        df = _load(run_dir / csv_name)
        if df is None:
            continue

        x_col = "round" if "round" in df.columns else df.index
        x = df[x_col] if isinstance(x_col, str) else x_col

        available = [
            (col_suffix, label)
            for col_suffix, label in _TRAIN_VAL_PAIRS
            if col_suffix in df.columns
        ]
        if not available:
            continue

        ncols = 2
        nrows = (len(available) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, (col_suffix, label) in enumerate(available):
            ax = axes_flat[i]
            ax.plot(x, df[col_suffix], label=label, color=_COLORS[i % len(_COLORS)])
            _style_ax(ax, label, "Round", label)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{title_prefix} — per round", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = out_dir / out_name
        _savefig(fig, out, dpi)
        written.append(out)

    _logger.info("  [server] plots written to %s", out_dir)
    return written


# ---------------------------------------------------------------------------
# Federated — per-node overlaid comparisons
# ---------------------------------------------------------------------------

def _collect_node_dfs(run_dir: Path, csv_filename: str) -> dict[int, pd.DataFrame]:
    clients_dir = run_dir / "clients"
    if not clients_dir.is_dir():
        return {}
    node_dfs: dict[int, pd.DataFrame] = {}
    for node_dir in sorted(clients_dir.iterdir()):
        if not node_dir.is_dir():
            continue
        # NodeMetricsRecorder._client_root (federated/node_logging.py) names
        # directories "client_<id>". Pull the trailing integer regardless of
        # prefix so "client_0", "cid_0", and "node_0" all parse — the
        # previous implementation only stripped "cid_"/"node_" prefixes,
        # which meant int("client_0") raised and every node was silently
        # skipped (no nodes_*.png ever rendered for a federated run).
        m = re.search(r"(\d+)$", node_dir.name)
        if m is None:
            continue
        cid = int(m.group(1))
        df = _load(node_dir / csv_filename)
        if df is not None:
            node_dfs[cid] = df
    return node_dfs


def _x_col(df: pd.DataFrame) -> pd.Series:
    for col in ("round", "epoch", "server_round"):
        if col in df.columns:
            return df[col]
    import pandas as pd

    return pd.Series(range(len(df)))


def _node_loss_frame(
    fit_df: pd.DataFrame | None, eval_df: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Outer-join a node's fit (train_loss) and eval (loss -> val_loss) on round.

    ``fit_metrics.csv`` has ``round, train_loss, ...`` (written by
    ``NodeMetricsRecorder.record_fit``); ``eval_metrics.csv`` has
    ``round, loss, ...`` (the val loss, written by ``record_eval`` — see
    ``_EVAL_KEYS`` in ``federated/node_logging.py``). Missing cells were
    written as ``""`` by CSVLogger and read back as NaN by pandas, which
    matplotlib simply skips when plotting — no special handling needed.
    """
    import pandas as pd

    if fit_df is None and eval_df is None:
        return None
    train = None
    if fit_df is not None and "train_loss" in fit_df.columns:
        train = fit_df[["round", "train_loss"]]
    val = None
    if eval_df is not None and "loss" in eval_df.columns:
        val = eval_df[["round", "loss"]].rename(columns={"loss": "val_loss"})
    if train is None and val is None:
        return None
    if train is None:
        return val.sort_values("round")
    if val is None:
        return train.sort_values("round")
    return pd.merge(train, val, on="round", how="outer").sort_values("round")


def plot_nodes(run_dir: Path, out_dir: Path, dpi: int) -> list[Path]:
    plt, _ = _mpl()
    written: list[Path] = []
    fit_dfs = _collect_node_dfs(run_dir, "fit_metrics.csv")
    eval_dfs = _collect_node_dfs(run_dir, "eval_metrics.csv")

    if not fit_dfs and not eval_dfs:
        _logger.info("  [nodes] no per-node CSVs found — skipping.")
        return []

    node_ids = sorted(set(list(fit_dfs) + list(eval_dfs)))

    # --- overlaid: train loss all nodes ---
    if fit_dfs:
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, (cid, df) in enumerate(sorted(fit_dfs.items())):
            col = next((c for c in ("train_loss", "loss") if c in df.columns), None)
            if col:
                ax.plot(_x_col(df), df[col], label=f"Node {cid}", color=_COLORS[i % len(_COLORS)])
        _style_ax(ax, "Train loss per node", "Round", "Loss")
        fig.tight_layout()
        out = out_dir / "nodes_train_loss.png"
        _savefig(fig, out, dpi)
        written.append(out)

    # --- overlaid: val AUC and F1 all nodes ---
    for col_suffix, label, out_name in [
        ("roc_auc", "AUC-ROC", "nodes_val_auc.png"),
        ("f1", "F1", "nodes_val_f1.png"),
    ]:
        candidates = eval_dfs if eval_dfs else fit_dfs
        cols_exist = any(
            any(c in df.columns for c in (f"val_{col_suffix}", col_suffix))
            for df in candidates.values()
        )
        if not cols_exist:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, cid in enumerate(sorted(candidates)):
            df = candidates[cid]
            col = next((c for c in (f"val_{col_suffix}", col_suffix) if c in df.columns), None)
            if col:
                ax.plot(_x_col(df), df[col], label=f"Node {cid}", color=_COLORS[i % len(_COLORS)])
        _style_ax(ax, f"{label} per node", "Round", label)
        fig.tight_layout()
        out = out_dir / out_name
        _savefig(fig, out, dpi)
        written.append(out)

    # --- overlaid: train loss (left) + val loss (right), all nodes ---
    node_loss = {cid: _node_loss_frame(fit_dfs.get(cid), eval_dfs.get(cid)) for cid in node_ids}
    node_loss = {cid: df for cid, df in node_loss.items() if df is not None}
    if node_loss:
        fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 4.5))
        for i, cid in enumerate(sorted(node_loss)):
            df = node_loss[cid]
            color = _COLORS[i % len(_COLORS)]
            if "train_loss" in df.columns:
                ax_train.plot(df["round"], df["train_loss"], label=f"Node {cid}", color=color)
            if "val_loss" in df.columns:
                ax_val.plot(df["round"], df["val_loss"], label=f"Node {cid}", color=color)
        _style_ax(ax_train, "Train loss — all nodes", "Round", "Loss")
        _style_ax(ax_val, "Val loss — all nodes", "Round", "Loss")
        fig.suptitle("Train vs val loss per node", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = out_dir / "nodes_loss_train_val.png"
        _savefig(fig, out, dpi)
        written.append(out)

    # --- per-node: train + val loss on one axes ---
    for cid, df in sorted(node_loss.items()):
        fig, ax = plt.subplots(figsize=(7, 4))
        if "train_loss" in df.columns:
            ax.plot(df["round"], df["train_loss"], label="Train loss", color=_COLORS[0])
        if "val_loss" in df.columns:
            ax.plot(df["round"], df["val_loss"], label="Val loss", color=_COLORS[1], linestyle="--")
        _style_ax(ax, f"Node {cid} — train vs val loss", "Round", "Loss")
        fig.tight_layout()
        out = out_dir / f"node_{cid}_loss.png"
        _savefig(fig, out, dpi)
        written.append(out)

    # --- per-node detailed dashboard ---
    for cid in node_ids:
        fit_df = fit_dfs.get(cid)
        eval_df = eval_dfs.get(cid)

        panels: list[tuple[pd.DataFrame, str, str, str]] = []  # (df, col, title, ylabel)
        for df, prefix in [(fit_df, "train"), (eval_df, "val")]:
            if df is None:
                continue
            for col_suffix, label in _TRAIN_VAL_PAIRS:
                for candidate in (f"{prefix}_{col_suffix}", col_suffix):
                    if candidate in df.columns:
                        panels.append((df, candidate, f"{prefix.capitalize()} {label}", label))
                        break

        if not panels:
            continue

        ncols = 2
        nrows = (len(panels) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, (df, col, title, ylabel) in enumerate(panels):
            ax = axes_flat[i]
            ax.plot(_x_col(df), df[col], color=_COLORS[i % len(_COLORS)])
            _style_ax(ax, title, "Round", ylabel)
            ax.get_legend().remove() if ax.get_legend() else None
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"Node {cid} — training curves", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = out_dir / f"node_{cid}_curves.png"
        _savefig(fig, out, dpi)
        written.append(out)

    _logger.info("  [nodes] plots written to %s", out_dir)
    return written


# ---------------------------------------------------------------------------
# Test-set diagnostics (ROC curve + confusion matrix)
# ---------------------------------------------------------------------------

def _find_test_predictions(run_dir: Path) -> Path | None:
    """Locate a post-hoc test predictions CSV under ``run_dir/eval/``.

    Written by ``scripts/run_evaluation.py --split test --predictions-out ...``
    (not produced automatically during training), so this is opt-in: only
    experiments that had this evaluation run manually get these plots.
    """
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return None
    # Direct children first (scripts/run_evaluation.py --predictions-out convention);
    # fall back to one level of nesting (--output-dir default: eval/<config_name>/predictions.csv).
    candidates = sorted(eval_dir.glob("*predictions*.csv")) or sorted(eval_dir.glob("*/*predictions*.csv"))
    return candidates[0] if candidates else None


def plot_test_diagnostics(run_dir: Path, out_dir: Path, dpi: int) -> list[Path]:
    """ROC curve + confusion matrix from a saved test-set ``predictions.csv``.

    Expects ``y_true`` and a probability column (``y_prob_malignant`` or
    ``y_prob``) plus optionally ``y_pred`` (falls back to thresholding the
    probability at 0.5 if absent).
    """
    pred_path = _find_test_predictions(run_dir)
    if pred_path is None:
        _logger.info("  [test] no eval/*predictions*.csv found — skipping.")
        return []

    df = _load(pred_path)
    if df is None or "y_true" not in df.columns:
        _logger.info("  [test] %s missing or has no y_true column — skipping.", pred_path)
        return []

    prob_col = next((c for c in ("y_prob_malignant", "y_prob", "y_score") if c in df.columns), None)
    if prob_col is None:
        _logger.info("  [test] %s has no probability column — skipping.", pred_path)
        return []

    from sklearn.metrics import auc, confusion_matrix, roc_curve

    y_true = df["y_true"].to_numpy()
    y_prob = df[prob_col].to_numpy()
    y_pred = df["y_pred"].to_numpy() if "y_pred" in df.columns else (y_prob >= 0.5).astype(int)

    plt, _ = _mpl()
    written: list[Path] = []

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=_COLORS[0], label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style_ax(ax, "Test ROC curve", "False positive rate", "True positive rate")
    fig.tight_layout()
    out = out_dir / "test_roc_curve.png"
    _savefig(fig, out, dpi)
    written.append(out)

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Benign", "Malignant"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title("Test confusion matrix", fontsize=11, fontweight="bold")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = out_dir / "test_confusion_matrix.png"
    _savefig(fig, out, dpi)
    written.append(out)

    if "source_dataset" in df.columns:
        written += _plot_test_macro_diagnostics(df, prob_col, out_dir, dpi)

    _logger.info("  [test] plots written to %s (source: %s)", out_dir, pred_path)
    return written


def _plot_test_macro_diagnostics(
    df: pd.DataFrame, prob_col: str, out_dir: Path, dpi: int
) -> list[Path]:
    """Per-``source_dataset`` AUC/AUC-PR + macro average — the honest metric.

    Pooling datasets with very different malignancy prevalence and reporting
    a single global AUC lets a model win by recognizing *which dataset* an
    image came from (and emitting that dataset's base rate) rather than by
    reading the image — global AUC can't tell the two apart. Computing AUC
    within each dataset and averaging (macro) removes that shortcut, because
    a single dataset's own prevalence is constant and uninformative for
    ranking its own images.
    """
    from sklearn.metrics import auc as _auc, average_precision_score, roc_auc_score, roc_curve

    plt, _ = _mpl()
    written: list[Path] = []
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, ds in enumerate(sorted(df["source_dataset"].unique())):
        sub = df[df["source_dataset"] == ds]
        y = sub["y_true"].to_numpy()
        p = sub[prob_col].to_numpy()
        if len(set(y.tolist())) < 2:
            _logger.info("  [test-macro] %s has a single class present — skipping its AUC.", ds)
            continue
        ds_auc = roc_auc_score(y, p)
        ds_auc_pr = average_precision_score(y, p)
        rows.append(
            {"source_dataset": ds, "n": len(sub), "prevalence": float(y.mean()),
             "auc": ds_auc, "auc_pr": ds_auc_pr}
        )
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=_COLORS[i % len(_COLORS)], label=f"{ds} (AUC={ds_auc:.3f}, n={len(sub)})")

    if not rows:
        plt.close(fig)
        return []

    import pandas as pd

    macro_auc = float(pd.DataFrame(rows)["auc"].mean())
    macro_auc_pr = float(pd.DataFrame(rows)["auc_pr"].mean())

    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style_ax(
        ax, f"Test ROC by dataset (macro AUC = {macro_auc:.3f})",
        "False positive rate", "True positive rate",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = out_dir / "test_roc_by_dataset.png"
    _savefig(fig, out, dpi)
    written.append(out)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "test_metrics_by_dataset.csv"
    summary.to_csv(summary_path, index=False)
    with open(summary_path, "a") as f:
        f.write(f"\n# macro_auc,{macro_auc:.6f}\n# macro_auc_pr,{macro_auc_pr:.6f}\n")
    written.append(summary_path)
    _logger.info(
        "  [test-macro] macro AUC=%.4f (global AUC would overstate this if dataset "
        "prevalence varies) -> %s", macro_auc, summary_path,
    )
    return written


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def plot_run(run_dir: str | Path, out_dir: str | Path | None = None, dpi: int = 120) -> list[Path]:
    """Auto-detect run mode (centralized / federated) and render all plots.

    Raises on missing ``run_dir`` or unrecognized layout. Use :func:`autoplot`
    instead when calling this from inside a training entry point, where a
    plotting failure must never fail an already-completed run.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    out_dir = Path(out_dir).expanduser().resolve() if out_dir else run_dir / "plots"
    _logger.info("Run dir : %s", run_dir)
    _logger.info("Plots -> : %s", out_dir)

    is_centralized = (run_dir / "metrics.csv").is_file()
    is_federated = (run_dir / "server_metrics.csv").is_file() or (run_dir / "clients").is_dir()

    written: list[Path] = []
    if is_centralized:
        _logger.info("[centralized mode detected]")
        written += plot_centralized(run_dir, out_dir, dpi)

    if is_federated:
        _logger.info("[federated mode detected]")
        written += plot_server(run_dir, out_dir, dpi)
        written += plot_nodes(run_dir, out_dir, dpi)

    if not is_centralized and not is_federated:
        raise ValueError(
            f"no recognized CSV files found under {run_dir} "
            "(expected metrics.csv or server_metrics.csv/clients/)"
        )

    written += plot_test_diagnostics(run_dir, out_dir, dpi)

    _logger.info("Done. All plots saved to %s", out_dir)
    return written


def autoplot(run_dir: str | Path, out_dir: str | Path | None = None, dpi: int = 120) -> None:
    """Best-effort ``plot_run`` for training entry points.

    Called automatically at the end of centralized and federated training
    (``scripts/run_centralized.py``, ``federated/server.py``). Never raises —
    including when matplotlib/pandas aren't installed (``ImportError``) — so
    a plotting bug can never fail a training run that has already finished
    and already has all its metrics safely on disk.
    """
    try:
        plot_run(run_dir, out_dir, dpi)
    except Exception:
        _logger.warning("autoplot failed for %s (run artifacts are unaffected)", run_dir, exc_info=True)


__all__ = [
    "autoplot",
    "plot_centralized",
    "plot_nodes",
    "plot_run",
    "plot_server",
    "plot_test_diagnostics",
]
