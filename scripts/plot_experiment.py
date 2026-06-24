"""Generate training/evaluation plots for a completed experiment run.

Supports both centralized and federated runs. Auto-detects the mode based
on which CSV files are present in the run directory.

Centralized outputs  (reads ``metrics.csv``)
--------------------------------------------
  - loss_curves.png       : train + val loss per epoch
  - metric_curves.png     : val AUC, F1, sensitivity, specificity per epoch

Federated outputs
-----------------
  Server-level (reads ``server_metrics.csv``, ``server_federated_metrics.csv``):
  - server_centralized.png : centralized eval metrics per round
  - server_federated.png   : federated (client-averaged) metrics per round

  Per-node  (reads ``clients/cid_<N>/fit_metrics.csv`` and ``eval_metrics.csv``):
  - nodes_train_loss.png   : train loss per round, all nodes overlaid
  - nodes_val_auc.png      : val AUC per round, all nodes overlaid
  - nodes_val_f1.png       : val F1 per round, all nodes overlaid
  - node_<N>_curves.png    : per-node detailed curves (loss + metrics)

Usage::

    python scripts/plot_experiment.py --run-dir runs/exp08_centralized_resnet50
    python scripts/plot_experiment.py --run-dir runs/exp07_fedavg_resnet50 --dpi 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ---------------------------------------------------------------------------
# Metrics to plot (column suffix in the CSV → display label)
# ---------------------------------------------------------------------------
_TRAIN_VAL_PAIRS: list[tuple[str, str]] = [
    ("loss",        "Loss"),
    ("roc_auc",     "AUC-ROC"),
    ("f1",          "F1"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("accuracy",    "Accuracy"),
]

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def _savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Centralized
# ---------------------------------------------------------------------------

def plot_centralized(run_dir: Path, out_dir: Path, dpi: int) -> None:
    df = _load(run_dir / "metrics.csv")
    if df is None:
        print("  [centralized] metrics.csv not found — skipping.")
        return

    x = df["epoch"] if "epoch" in df.columns else df.index

    # --- loss curves ---
    fig, ax = plt.subplots(figsize=(8, 4))
    if "train_loss" in df.columns:
        ax.plot(x, df["train_loss"], label="Train loss", color=_COLORS[0])
    if "val_loss" in df.columns:
        ax.plot(x, df["val_loss"], label="Val loss", color=_COLORS[1], linestyle="--")
    _style_ax(ax, "Loss curves", "Epoch", "Loss")
    fig.tight_layout()
    _savefig(fig, out_dir / "loss_curves.png", dpi)

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
        _savefig(fig, out_dir / "metric_curves.png", dpi)

    print(f"  [centralized] plots written to {out_dir}")


# ---------------------------------------------------------------------------
# Federated — server level
# ---------------------------------------------------------------------------

def plot_server(run_dir: Path, out_dir: Path, dpi: int) -> None:
    for csv_name, out_name, title_prefix in [
        ("server_metrics.csv",           "server_centralized.png", "Server centralized eval"),
        ("server_federated_metrics.csv", "server_federated.png",   "Server federated (client-avg)"),
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
        _savefig(fig, out_dir / out_name, dpi)

    print(f"  [server] plots written to {out_dir}")


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
        cid_str = node_dir.name.replace("cid_", "").replace("node_", "")
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        df = _load(node_dir / csv_filename)
        if df is not None:
            node_dfs[cid] = df
    return node_dfs


def _x_col(df: pd.DataFrame) -> pd.Series:
    for col in ("round", "epoch", "server_round"):
        if col in df.columns:
            return df[col]
    return pd.Series(range(len(df)))


def plot_nodes(run_dir: Path, out_dir: Path, dpi: int) -> None:
    fit_dfs  = _collect_node_dfs(run_dir, "fit_metrics.csv")
    eval_dfs = _collect_node_dfs(run_dir, "eval_metrics.csv")

    if not fit_dfs and not eval_dfs:
        print("  [nodes] no per-node CSVs found — skipping.")
        return

    all_dfs = {**fit_dfs}  # fit has train cols; eval has val/test cols
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
        _savefig(fig, out_dir / "nodes_train_loss.png", dpi)

    # --- overlaid: val AUC and F1 all nodes ---
    for col_suffix, label, out_name in [
        ("roc_auc", "AUC-ROC", "nodes_val_auc.png"),
        ("f1",      "F1",      "nodes_val_f1.png"),
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
            col = next(
                (c for c in (f"val_{col_suffix}", col_suffix) if c in df.columns), None
            )
            if col:
                ax.plot(_x_col(df), df[col], label=f"Node {cid}", color=_COLORS[i % len(_COLORS)])
        _style_ax(ax, f"{label} per node", "Round", label)
        fig.tight_layout()
        _savefig(fig, out_dir / out_name, dpi)

    # --- per-node detailed dashboard ---
    for cid in node_ids:
        fit_df  = fit_dfs.get(cid)
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
        _savefig(fig, out_dir / f"node_{cid}_curves.png", dpi)

    print(f"  [nodes] plots written to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot experiment training curves.")
    p.add_argument(
        "--run-dir", "-r", required=True, type=str,
        help="Path to the experiment run directory (e.g. runs/exp08_centralized_resnet50).",
    )
    p.add_argument(
        "--out-dir", "-o", default=None, type=str,
        help="Output directory for plots. Defaults to <run-dir>/plots/.",
    )
    p.add_argument("--dpi", default=120, type=int, help="Plot resolution (default 120).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "plots"
    print(f"Run dir : {run_dir}")
    print(f"Plots → : {out_dir}")

    is_centralized = (run_dir / "metrics.csv").is_file()
    is_federated   = (run_dir / "server_metrics.csv").is_file() or (run_dir / "clients").is_dir()

    if is_centralized:
        print("\n[centralized mode detected]")
        plot_centralized(run_dir, out_dir, args.dpi)

    if is_federated:
        print("\n[federated mode detected]")
        plot_server(run_dir, out_dir, args.dpi)
        plot_nodes(run_dir, out_dir, args.dpi)

    if not is_centralized and not is_federated:
        print("WARNING: no recognized CSV files found. Expected metrics.csv or server_metrics.csv.")
        return 1

    print(f"\nDone. All plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
