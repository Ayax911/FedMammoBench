#!/usr/bin/env python3
"""Pooled vs within-database test AUC with patient-level bootstrap CIs and paired deltas between runs."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MANIFEST = "manifests/fedmammobench_norm_0_1.csv"
SOURCES = ["cmmd", "kau-bcmd", "cdd-cesm", "inbreast"]
DUPLICATES = {"CM000814", "CM001309", "CM003084", "CM003913", "CM004739"}
RUNS = {
    "exp37 centralizado": "runs/exp37_hpsearch_v1_e7u7fprr/test/predictions.csv",
    "exp31 centralizado": "runs/exp31_antioverfit_no_inputdrop/test/predictions.csv",
    "exp28 centralizado": "runs/exp28_antioverfit_base/test/predictions.csv",
    "exp46 FedProx": "runs/exp46_fedgrid_fedprox_r30/pooled_eval/test/predictions.csv",
}
LOCAL = "cada base sola (exp24-27)"
LOCAL_RUNS = {
    "cdd-cesm": "runs/exp24_bydatabase_cdd-cesm_standard_mlp_bce/test/predictions.csv",
    "cmmd": "runs/exp25_bydatabase_cmmd_standard_mlp_bce/test/predictions.csv",
    "inbreast": "runs/exp26_bydatabase_inbreast_standard_mlp_bce/test/predictions.csv",
    "kau-bcmd": "runs/exp27_bydatabase_kau-bcmd_standard_mlp_bce/test/predictions.csv",
}
METRICS = ["pooled", "intra_base"] + SOURCES
COMPARISONS = (
    [("exp37 centralizado", "exp28 centralizado", m) for m in ("pooled", "intra_base")]
    + [("exp37 centralizado", "exp31 centralizado", m) for m in ("pooled", "intra_base")]
    + [("exp46 FedProx", "exp37 centralizado", m) for m in METRICS]
    + [("exp46 FedProx", LOCAL, m) for m in ["intra_base"] + SOURCES]
)


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def within_database_auc(y, p, src):
    # Only malignant-benign pairs from the same database count, so recognising the database earns nothing.
    num = den = 0.0
    for s in SOURCES:
        m = src == s
        pairs = y[m].sum() * (1 - y[m]).sum()
        if pairs:
            num += roc_auc_score(y[m], p[m]) * pairs
            den += pairs
    return num / den


def metrics(y, p, src, idx):
    out = {"pooled": auc(y[idx], p[idx]), "intra_base": within_database_auc(y[idx], p[idx], src[idx])}
    for s in SOURCES:
        m = src[idx] == s
        out[s] = auc(y[idx][m], p[idx][m])
    return out


def main():
    df = pd.read_csv(MANIFEST)
    te = df[df["split"] == "test"].reset_index(drop=True)
    y = (te["classification"].str.lower() == "malignant").astype(int).to_numpy()
    src = te["source_dataset"].to_numpy()
    patients = pd.factorize(te["patient_id"])[0]

    preds = {}
    for name, path in RUNS.items():
        p = pd.read_csv(path)
        assert len(p) == len(te) and (p["y_true"].to_numpy() == y).all(), f"{path} does not align with the manifest"
        preds[name] = p["y_prob"].to_numpy()
    local = np.full(len(te), np.nan)
    for s, path in LOCAL_RUNS.items():
        p = pd.read_csv(path)
        m = src == s
        assert len(p) == m.sum() and (p["y_true"].to_numpy() == y[m]).all(), f"{path} does not align with the manifest"
        local[m] = p["y_prob"].to_numpy()
    preds[LOCAL] = local

    everything = np.arange(len(te))
    point = {k: metrics(y, v, src, everything) for k, v in preds.items()}
    point[LOCAL]["pooled"] = np.nan  # four independent models: their scores are not on a common scale

    rng = np.random.default_rng(0)
    by_patient = [np.flatnonzero(patients == g) for g in range(patients.max() + 1)]
    boots = {k: [] for k in preds}
    for _ in range(2000):
        idx = np.concatenate([by_patient[i] for i in rng.integers(0, len(by_patient), len(by_patient))])
        for k, v in preds.items():
            boots[k].append(metrics(y, v, src, idx))
    boots = {k: pd.DataFrame(v) for k, v in boots.items()}

    print("AUC de test [IC 95% bootstrap por paciente, 2000 remuestreos]")
    table = {}
    for k in preds:
        table[k] = {}
        for m in METRICS:
            if not np.isnan(point[k][m]):
                lo, hi = np.nanpercentile(boots[k][m], [2.5, 97.5])
                table[k][m] = f"{point[k][m]:.3f} [{lo:.3f}; {hi:.3f}]"
    print(pd.DataFrame(table).T[METRICS].fillna("—").to_string())

    print("\nDiferencias pareadas A − B (mismas imágenes y remuestreos)")
    rows = []
    for a, b, m in COMPARISONS:
        d = boots[a][m] - boots[b][m]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        rows.append({"A": a, "B": b, "métrica": m, "Δ": round(point[a][m] - point[b][m], 4),
                     "IC 95%": f"[{lo:.3f}; {hi:.3f}]", "P(A>B)": round(float(np.nanmean(d > 0)), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nMalignos / total en test:", {s: f"{int(y[src == s].sum())}/{int((src == s).sum())}" for s in SOURCES})
    keep = np.flatnonzero(~te["ID_image"].isin(DUPLICATES).to_numpy())
    print("Sin los 5 duplicados píxel a píxel con train:")
    for k in ("exp37 centralizado", "exp46 FedProx"):
        m = metrics(y, preds[k], src, keep)
        print(f"  {k}: pooled {m['pooled']:.4f} (antes {point[k]['pooled']:.4f}) · "
              f"intra-base {m['intra_base']:.4f} (antes {point[k]['intra_base']:.4f})")


if __name__ == "__main__":
    main()
