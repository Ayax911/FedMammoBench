"""Confusion matrices + cross-experiment test comparison for the exp50-55 factorial.

exp50-55 is a 3-depth x 2-dropout linear-probing factorial (see memory
mlp_head_linear_probing_exp50_55). run_centralized.py's final test pass does not
call Evaluator.evaluate(..., return_predictions=True), so no per-sample
predictions.csv exists for these runs -- confusion matrices are reconstructed
from the aggregated test_metrics.csv counts instead:

    TP = round(sensitivity * positives)   FN = positives - TP
    TN = round(specificity * negatives)   FP = negatives - TN

This is exact (not an approximation): sensitivity/specificity were computed by
compute_metrics() as TP/positives and TN/negatives in the first place.

Usage::

    python scripts/plot_exp50_55_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sk_auc

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
OUT_DIR = RUNS / "_comparisons" / "exp50_55_mlp_head"

# (exp, name, depth_label, hidden_dims_len, dropout)
EXPERIMENTS = [
    ("exp50", "exp50_mlp6_nodrop_radimagenet", "6 ocultas", 6, 0.0),
    ("exp51", "exp51_mlp5_nodrop_radimagenet", "5 ocultas", 5, 0.0),
    ("exp52", "exp52_mlp4_nodrop_radimagenet", "4 ocultas", 4, 0.0),
    ("exp53", "exp53_mlp6_d02_radimagenet", "6 ocultas", 6, 0.2),
    ("exp54", "exp54_mlp5_d02_radimagenet", "5 ocultas", 5, 0.2),
    ("exp55", "exp55_mlp4_d02_radimagenet", "4 ocultas", 4, 0.2),
]


def _run_dir(name: str) -> Path:
    return RUNS / name / name


def _predictions_path(name: str) -> Path:
    return RUNS / name / "eval" / "mammo_bench" / "predictions.csv"


def _confusion_counts(test_row: pd.Series) -> np.ndarray:
    pos = float(test_row["test_positives"])
    neg = float(test_row["test_negatives"])
    sens = float(test_row["test_sensitivity"])
    spec = float(test_row["test_specificity"])
    tp = round(sens * pos)
    fn = pos - tp
    tn = round(spec * neg)
    fp = neg - tn
    # rows = true class [benign, malignant], cols = predicted [benign, malignant]
    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrix(exp: str, name: str, cm: np.ndarray, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Benigno", "Maligno"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicho", fontsize=9)
    ax.set_ylabel("Real", fontsize=9)
    ax.set_title(f"{exp} — Matriz de confusión (test)", fontsize=10, fontweight="bold")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            pct = 100 * count / total
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                     color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_dir = _run_dir(name) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(exp: str, name: str, y_true: np.ndarray, y_prob: np.ndarray, dpi: int) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Azar")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de falsos positivos", fontsize=9)
    ax.set_ylabel("Tasa de verdaderos positivos", fontsize=9)
    ax.set_title(f"{exp} — Curva ROC (test)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_dir = _run_dir(name) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roc_curve.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    dpi = 120
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, int, float]] = {}

    for exp, name, depth_label, n_hidden, dropout in EXPERIMENTS:
        run_dir = _run_dir(name)
        test_csv = run_dir / "test_metrics.csv"
        if not test_csv.is_file():
            print(f"  [{exp}] test_metrics.csv not found — skipping.")
            continue
        test_df = pd.read_csv(test_csv)
        test_row = test_df.iloc[0]
        cm = _confusion_counts(test_row)
        out_path = plot_confusion_matrix(exp, name, cm, dpi)
        print(f"  saved -> {out_path}")

        pred_path = _predictions_path(name)
        if pred_path.is_file():
            preds = pd.read_csv(pred_path)
            y_true = preds["y_true"].to_numpy()
            y_prob = preds["y_prob_malignant"].to_numpy()
            out_path = plot_roc_curve(exp, name, y_true, y_prob, dpi)
            print(f"  saved -> {out_path}")
            roc_data[exp] = (y_true, y_prob, n_hidden, dropout)
        else:
            print(f"  [{exp}] predictions.csv not found — sin curva ROC individual.")

        rows.append({
            "exp": exp, "depth_label": depth_label, "n_hidden": n_hidden,
            "dropout": dropout,
            "test_auc": float(test_row["test_roc_auc"]),
            "test_f1": float(test_row["test_f1"]),
            "test_accuracy": float(test_row["test_accuracy"]),
            "test_sensitivity": float(test_row["test_sensitivity"]),
            "test_specificity": float(test_row["test_specificity"]),
        })

    if not rows:
        print("No experiments found — nothing to compare.")
        return
    df = pd.DataFrame(rows).sort_values(["dropout", "n_hidden"], ascending=[True, False])

    # --- grouped bar: test AUC and F1 per experiment, grouped by depth, colored by dropout ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    depths = sorted(df["n_hidden"].unique(), reverse=True)
    dropouts = sorted(df["dropout"].unique())
    x = np.arange(len(depths))
    width = 0.35
    colors = {0.0: "#4C72B0", 0.2: "#DD8452"}
    for metric, ax, title in [("test_auc", axes[0], "AUC-ROC (test)"),
                                ("test_f1", axes[1], "F1 (test)")]:
        for i, dp in enumerate(dropouts):
            sub = df[df["dropout"] == dp].set_index("n_hidden").reindex(depths)
            offset = (i - (len(dropouts) - 1) / 2) * width
            bars = ax.bar(x + offset, sub[metric], width, label=f"dropout={dp}",
                           color=colors.get(dp, None))
            for bar, exp_name in zip(bars, sub["exp"]):
                ax.annotate(f"{bar.get_height():.3f}\n{exp_name}",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, [f"{d} ocultas" for d in depths])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0.75, 0.90)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("exp50-55 — Factorial profundidad x dropout (RadImageNet congelado, test set)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = OUT_DIR / "test_auc_f1_comparison.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    # --- heatmap: test AUC across the depth x dropout grid ---
    fig, ax = plt.subplots(figsize=(5, 4))
    grid = df.pivot(index="n_hidden", columns="dropout", values="test_auc").reindex(depths)
    im = ax.imshow(grid.values, cmap="YlGn", vmin=0.80, vmax=0.86, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [f"dropout={c}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{d} ocultas" for d in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.4f}", ha="center", va="center", fontsize=10)
    ax.set_title("Test AUC-ROC — profundidad x dropout", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = OUT_DIR / "test_auc_heatmap.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    # --- overlay: val AUC curves across all 6 runs (overfitting/convergence comparison) ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    linestyles = {0.0: "-", 0.2: "--"}
    cmap = plt.get_cmap("viridis")
    depth_color = {d: cmap(i / (len(depths) - 1 or 1)) for i, d in enumerate(sorted(depths))}
    for exp, name, depth_label, n_hidden, dropout in EXPERIMENTS:
        m = pd.read_csv(_run_dir(name) / "metrics.csv")
        ax.plot(m["epoch"], m["val_roc_auc"], label=f"{exp} ({depth_label}, dp={dropout})",
                 color=depth_color[n_hidden], linestyle=linestyles[dropout], alpha=0.85)
    ax.set_title("Val AUC-ROC por época — los 6 experimentos", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Val AUC-ROC")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = OUT_DIR / "val_auc_overlay.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    # --- overlay: ROC curves for all 6 runs ---
    if roc_data:
        fig, ax = plt.subplots(figsize=(7, 6.5))
        for exp, (y_true, y_prob, n_hidden, dropout) in roc_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=depth_color[n_hidden], linestyle=linestyles[dropout],
                     lw=2, alpha=0.85,
                     label=f"{exp} ({n_hidden} ocultas, dp={dropout}) AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":", label="Azar")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("exp50-55 — Curvas ROC (test) superpuestas", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        out_path = OUT_DIR / "roc_overlay.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out_path}")

    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"  saved -> {OUT_DIR / 'summary.csv'}")
    print(f"\nDone. Comparison plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
