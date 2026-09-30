#!/usr/bin/env python3
"""Measure how different the train/val/test splits are: metadata, pixels, deep features and label transfer."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

META_COLS = ["source_dataset", "classification", "view", "laterality", "density", "BIRADS", "abnormality", "molecular_subtype"]
STAT_NAMES = ["mean", "std", "p05", "p50", "p95", "frac_background", "fg_mean", "fg_std"]
SPLITS = ["train", "val", "test"]
SOURCES = ["cmmd", "kau-bcmd", "cdd-cesm", "inbreast"]
PROBE_CS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


class TiffDataset(Dataset):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        fg = arr[arr > 0.02]
        stats = np.array(
            [arr.mean(), arr.std(), *np.percentile(arr, [5, 50, 95]), (arr <= 0.02).mean(),
             fg.mean() if fg.size else 0.0, fg.std() if fg.size else 0.0],
            dtype=np.float32,
        )
        x = torch.from_numpy(arr)[None].repeat(3, 1, 1)
        return (x - self.mean) / self.std, torch.from_numpy(stats)


def extract_features(paths, device, batch_size):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(TiffDataset(paths), batch_size=batch_size, num_workers=4)
    embs, stats = [], []
    # fp32 on purpose: fp16 autocast returns all-NaN features on GTX 16xx cards at batch 64.
    with torch.no_grad():
        for x, s in loader:
            embs.append(model(x.to(device)).cpu().numpy())
            stats.append(s.numpy())
    E = np.concatenate(embs)
    assert np.isfinite(E).all(), "non-finite embeddings"
    return E, np.concatenate(stats)


def per_patient(values, codes):
    out = np.empty(codes.max() + 1, dtype=np.asarray(values).dtype)
    out[codes] = values
    return out


def patient_perms(img_values, codes, n, rng):
    # Split labels are shuffled between whole patients: images of one patient are not independent samples.
    pat = per_patient(img_values, codes)
    return [rng.permutation(pat)[codes] for _ in range(n)]


def perm_p(real, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= real)) / (1 + len(null)))


def chi2_from_codes(cat_codes, grp_codes):
    table = np.zeros((grp_codes.max() + 1, cat_codes.max() + 1))
    np.add.at(table, (grp_codes, cat_codes), 1)
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        return 0.0, 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(chi2), float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def within_source_auc(y, p, src):
    num = den = 0.0
    for s in np.unique(src):
        m = src == s
        pairs = y[m].sum() * (1 - y[m]).sum()
        if pairs:
            num += roc_auc_score(y[m], p[m]) * pairs
            den += pairs
    return float(num / den)


def patient_bootstrap_ci(y, p, groups, rng, n=1000):
    idx_by_patient = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    aucs = []
    for _ in range(n):
        idx = np.concatenate([idx_by_patient[i] for i in rng.integers(0, len(idx_by_patient), len(idx_by_patient))])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return [float(v) for v in np.percentile(aucs, [2.5, 97.5])]


def cv_auc(Z, y, groups, C=0.1, multiclass=False, seed=0):
    # Grouped by patient: split labels are constant within a patient, so ungrouped CV would score patient identity.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(np.unique(y)))) if multiclass else np.zeros(len(y))
    for tr, te in cv.split(Z, y, groups):
        proba = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], y[tr]).predict_proba(Z[te])
        oof[te] = proba if multiclass else proba[:, 1]
    return float(roc_auc_score(y, oof, multi_class="ovr") if multiclass else roc_auc_score(y, oof))


def adversarial(Z, y, groups, n_perm, rng, n_jobs):
    real = cv_auc(Z, y, groups)
    null = Parallel(n_jobs=n_jobs)(delayed(cv_auc)(Z, yp, groups) for yp in patient_perms(y, groups, n_perm, rng))
    null = np.array(null)
    return {"auc": real, "p": perm_p(real, null), "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)), "null_mean": float(null.mean())}


def pca_fit_transform(E, fit_idx, n_comp=256):
    scaler = StandardScaler().fit(E[fit_idx])
    pca = PCA(n_comp, random_state=0).fit(scaler.transform(E[fit_idx]))
    return lambda idx: pca.transform(scaler.transform(E[idx]))


def probe_oof(E, y, groups, tr_idx):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof = {C: np.zeros(len(tr_idx)) for C in PROBE_CS}
    for a, b in cv.split(tr_idx, y[tr_idx], groups[tr_idx]):
        proj = pca_fit_transform(E, tr_idx[a])
        Za, Zb = proj(tr_idx[a]), proj(tr_idx[b])
        for C in PROBE_CS:
            oof[C][b] = LogisticRegression(C=C, max_iter=3000).fit(Za, y[tr_idx[a]]).predict_proba(Zb)[:, 1]
    return oof


def resplit_once(E, y, patient_codes, patient_source, counts, C, seed):
    rng = np.random.default_rng(seed)
    pat_split = np.empty(len(patient_source), dtype=object)
    for src, c in counts.items():
        pats = rng.permutation(np.flatnonzero(patient_source == src))
        pat_split[pats[: c["train"]]] = "train"
        pat_split[pats[c["train"]: c["train"] + c["val"]]] = "val"
        pat_split[pats[c["train"] + c["val"]:]] = "test"
    img_split = pat_split[patient_codes]
    tr = np.flatnonzero(img_split == "train")
    proj = pca_fit_transform(E, tr)
    clf = LogisticRegression(C=C, max_iter=3000).fit(proj(tr), y[tr])
    out = {}
    for s in ("val", "test"):
        idx = np.flatnonzero(img_split == s)
        out[s] = roc_auc_score(y[idx], clf.predict_proba(proj(idx))[:, 1])
    return out


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_tsne(Y, df, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4), facecolor=SURFACE)
    for r, (col, values) in enumerate([("split", SPLITS), ("source_dataset", SOURCES)]):
        for c in range(4):
            ax = axes[r, c]
            if c >= len(values):
                ax.set_visible(False)
                continue
            ax.set_facecolor(SURFACE)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            m = (df[col] == values[c]).to_numpy()
            ax.scatter(Y[~m, 0], Y[~m, 1], s=2, c=AXIS, alpha=0.35, linewidths=0, rasterized=True)
            ax.scatter(Y[m, 0], Y[m, 1], s=3, c=BLUE, alpha=0.75, linewidths=0, rasterized=True)
            ax.set_title(f"{values[c]}   n={m.sum():,}", color=INK, fontsize=11, loc="left")
    fig.text(0.012, 0.975, "Mismo mapa t-SNE de las 8.341 imágenes (embeddings ResNet50-ImageNet congelado); cada panel resalta un grupo",
             color=INK, fontsize=12.5, ha="left", va="top")
    fig.text(0.012, 0.935, "Fila superior: splits oficiales.   Fila inferior: bases de datos (control: grupos que sí difieren).",
             color=INK2, fontsize=10, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_tests(adv, probe, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})

    rows = [
        ("Base de datos (control +)", adv["control_source"], ORANGE),
        ("Vista CC vs MLO (control +)", adv["control_view"], ORANGE),
        ("train vs val+test", adv["train_vs_eval"], BLUE),
        ("train vs val+test · solo cmmd", adv["train_vs_eval_cmmd"], BLUE),
        ("val vs test", adv["val_vs_test"], BLUE),
    ]
    style_axis(a1)
    for i, (label, res, color) in enumerate(rows):
        yv = len(rows) - 1 - i
        if "null_q025" in res:
            a1.add_patch(plt.Rectangle((res["null_q025"], yv - 0.26), res["null_q975"] - res["null_q025"], 0.52,
                                       color=AXIS, alpha=0.55, linewidth=0))
        a1.plot(res["auc"], yv, "o", ms=9, color=color, mec=SURFACE, mew=1.5, zorder=3)
        if color == BLUE:
            a1.text(res["auc"] + 0.018, yv, f"AUC {res['auc']:.3f} · p={res['p']:.2f}", color=INK2, fontsize=9.5, va="center")
    a1.axvline(0.5, color=MUTED, linewidth=1, zorder=1)
    a1.set_yticks(range(len(rows)))
    a1.set_yticklabels([r[0] for r in rows][::-1], color=INK2)
    a1.set_xlim(0.4, 1.02)
    a1.set_ylim(-0.7, len(rows) - 0.3)
    a1.set_xlabel("AUC del clasificador que intenta distinguir los grupos (0,5 = indistinguibles)", color=INK2)
    a1.set_title("¿Un clasificador distingue los splits?", color=INK, fontsize=12.5, loc="left", pad=10)
    a1.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, color=BLUE, label="Split oficial"),
        Line2D([], [], marker="o", ls="", ms=8, color=ORANGE, label="Control positivo (diferencia real)"),
        Patch(color=AXIS, alpha=0.55, label="95% esperado con asignación al azar por paciente"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)

    style_axis(a2)
    prow = [
        ("train · pacientes no vistos (CV)", probe["train_oof"], None),
        ("val oficial", probe["val"], probe["resplit_val"]),
        ("test oficial", probe["test"], probe["resplit_test"]),
    ]
    for i, (label, res, band) in enumerate(prow):
        yv = len(prow) - 1 - i
        if band is not None:
            a2.add_patch(plt.Rectangle((band["q025"], yv - 0.26), band["q975"] - band["q025"], 0.52, color=AXIS, alpha=0.55, linewidth=0))
        a2.plot(res["ci"], [yv, yv], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
        a2.plot(res["auc"], yv, "o", ms=9, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
        a2.text(res["ci"][1] + 0.006, yv, f"{res['auc']:.3f}", color=INK2, fontsize=9.5, va="center")
    lo = min(min(r[1]["ci"][0] for r in prow), probe["resplit_test"]["q025"], probe["resplit_val"]["q025"]) - 0.03
    hi = max(max(r[1]["ci"][1] for r in prow), probe["resplit_test"]["q975"], probe["resplit_val"]["q975"]) + 0.04
    a2.set_xlim(lo, hi)
    a2.set_yticks(range(len(prow)))
    a2.set_yticklabels([r[0] for r in prow][::-1], color=INK2)
    a2.set_ylim(-0.7, len(prow) - 0.3)
    a2.set_xlabel("AUC de malignidad de una sonda lineal entrenada en train", color=INK2)
    a2.set_title("¿Lo aprendido en train sirve igual en val y test?", color=INK, fontsize=12.5, loc="left", pad=10)
    a2.legend(handles=[
        Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=BLUE, label="AUC · IC 95% bootstrap por paciente"),
        Patch(color=AXIS, alpha=0.55, label="95% de 100 re-splits aleatorios por paciente"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=9, labelcolor=INK2)

    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/fedmammobench_norm_0_1.csv")
    ap.add_argument("--image-root", default="/home/akira/snap/steam/preproccesed_julian")
    ap.add_argument("--out-dir", default="runs/centralizado/split_shift_audit")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-perm-adv", type=int, default=200)
    ap.add_argument("--n-resplits", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs", nargs="*", default=["exp37_hpsearch_v1_e7u7fprr", "exp28_antioverfit_base",
                                                  "exp31_antioverfit_no_inputdrop", "exp05_fedmammobench_full_weighted"])
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {}

    df = pd.read_csv(args.manifest)
    df["y"] = (df["classification"].str.strip().str.lower() == "malignant").astype(int)
    codes, _ = pd.factorize(df["patient_id"])
    assert df.groupby("patient_id")["split"].nunique().max() == 1, "patient in more than one split"
    assert df.groupby("patient_id")["source_dataset"].nunique().max() == 1, "patient in more than one source"
    split = df["split"].to_numpy()
    y = df["y"].to_numpy()
    split_codes = pd.Series(pd.Categorical(split, categories=SPLITS)).cat.codes.to_numpy()
    print(f"{len(df)} imágenes, {codes.max() + 1} pacientes")

    # --- 1. Metadata ---------------------------------------------------------------------------
    print("\n=== 1. METADATOS: composición de cada split ===")
    by_split = df.groupby("split").agg(imagenes=("y", "size"), pacientes=("patient_id", "nunique"), pct_maligno=("y", "mean"))
    by_split["img_por_paciente"] = by_split["imagenes"] / by_split["pacientes"]
    by_split = by_split.loc[SPLITS]
    print(by_split.round(3).to_string())
    print("\n% de imágenes por base de datos en cada split:")
    print((pd.crosstab(df["split"], df["source_dataset"], normalize="index") * 100).loc[SPLITS].round(1).to_string())
    print("\n% maligno por base de datos en cada split:")
    print((df.pivot_table(index="source_dataset", columns="split", values="y", aggfunc="mean")[SPLITS] * 100).round(1).to_string())

    perms3 = patient_perms(split_codes, codes, args.n_perm, rng)
    meta = {}
    for col in META_COLS:
        cat_codes = pd.factorize(df[col].fillna("NA").astype(str))[0]
        chi2, v = chi2_from_codes(cat_codes, split_codes)
        null = [chi2_from_codes(cat_codes, g)[0] for g in perms3]
        meta[col] = {"cramers_v": v, "p_perm": perm_p(chi2, null), "n_categories": int(cat_codes.max() + 1)}
    age = df["subject_age"].to_numpy(dtype=float)
    ok = ~np.isnan(age)
    is_eval = split != "train"
    d_age = ks_2samp(age[ok & ~is_eval], age[ok & is_eval]).statistic
    null_age = [ks_2samp(age[ok & (g == 0)], age[ok & (g != 0)]).statistic for g in perms3]
    meta["subject_age"] = {"ks_D_train_vs_eval": float(d_age), "p_perm": perm_p(d_age, null_age),
                           "mean_by_split": {s: float(np.nanmean(age[split == s])) for s in SPLITS}}
    print("\nAsociación split × variable (Cramér's V: 0 = idéntico, >0,1 = diferencia apreciable; p por permutación de pacientes):")
    print(pd.DataFrame(meta).T.drop(columns=["mean_by_split"], errors="ignore").to_string())
    report["metadata"] = {"by_split": by_split.reset_index().to_dict(orient="records"), "tests": meta}

    # --- Features ------------------------------------------------------------------------------
    cache = out / "embeddings.pt"
    paths = [str(Path(args.image_root) / p) for p in df["preprocessed_image_path"]]
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        E, S = blob["E"], blob["S"]
        print(f"\n(embeddings cargados de {cache})")
    else:
        print(f"\nExtrayendo embeddings ResNet50-ImageNet de {len(paths)} imágenes en {args.device}...")
        E, S = extract_features(paths, args.device, args.batch_size)
        torch.save({"E": E, "S": S}, cache)
    print(f"embeddings {E.shape} [{time.time() - t0:.0f}s]")

    # --- 2. Pixel statistics -------------------------------------------------------------------
    print("\n=== 2. ESTADÍSTICAS DE PÍXELES por imagen (KS D: 0 = distribuciones idénticas) ===")
    eval_perms = patient_perms(is_eval.astype(np.int8), codes, args.n_perm, rng)
    ev = np.flatnonzero(is_eval)
    ev_codes = pd.factorize(df["patient_id"].to_numpy()[ev])[0]
    vt_perms = patient_perms((split[ev] == "test").astype(np.int8), ev_codes, args.n_perm, rng)
    src = df["source_dataset"].to_numpy()
    pix = {}
    for j, name in enumerate(STAT_NAMES):
        x = S[:, j]
        d_te = ks_2samp(x[~is_eval], x[is_eval]).statistic
        p_te = perm_p(d_te, [ks_2samp(x[g == 0], x[g == 1]).statistic for g in eval_perms])
        xe = x[ev]
        d_vt = ks_2samp(xe[split[ev] == "val"], xe[split[ev] == "test"]).statistic
        p_vt = perm_p(d_vt, [ks_2samp(xe[g == 0], xe[g == 1]).statistic for g in vt_perms])
        d_src = np.mean([ks_2samp(x[src == a], x[src == b]).statistic for i, a in enumerate(SOURCES) for b in SOURCES[i + 1:]])
        pix[name] = {"D_train_vs_eval": float(d_te), "p_train_vs_eval": p_te, "D_val_vs_test": float(d_vt),
                     "p_val_vs_test": p_vt, "D_medio_entre_bases (referencia)": float(d_src)}
    print(pd.DataFrame(pix).T.round(4).to_string())
    report["pixel_stats"] = pix

    # --- 3. Adversarial validation -------------------------------------------------------------
    print(f"\n=== 3. VALIDACIÓN ADVERSARIAL sobre embeddings (CV agrupada por paciente, {args.n_perm_adv} permutaciones) ===")
    # PCA is fit without any split/label information, so it cannot leak what the classifier is asked to detect.
    Z = PCA(128, random_state=0).fit_transform(StandardScaler().fit_transform(E))
    adv = {}
    adv["train_vs_eval"] = adversarial(Z, is_eval.astype(int), codes, args.n_perm_adv, rng, args.n_jobs)
    adv["val_vs_test"] = adversarial(Z[ev], (split[ev] == "test").astype(int), ev_codes, args.n_perm_adv, rng, args.n_jobs)
    cm = np.flatnonzero(src == "cmmd")
    cm_codes = pd.factorize(df["patient_id"].to_numpy()[cm])[0]
    adv["train_vs_eval_cmmd"] = adversarial(Z[cm], is_eval[cm].astype(int), cm_codes, args.n_perm_adv, rng, args.n_jobs)
    adv["control_source"] = {"auc": cv_auc(Z, pd.factorize(src)[0], codes, multiclass=True)}
    vw = np.flatnonzero(df["view"].isin(["CC", "MLO"]).to_numpy())
    vw_codes = pd.factorize(df["patient_id"].to_numpy()[vw])[0]
    adv["control_view"] = {"auc": cv_auc(Z[vw], (df["view"].to_numpy()[vw] == "MLO").astype(int), vw_codes)}
    adv["reference_malignancy"] = {"auc": cv_auc(Z, y, codes)}
    print(pd.DataFrame(adv).T.round(4).to_string())
    report["adversarial"] = adv
    print(f"[{time.time() - t0:.0f}s]")

    # --- 4. Label transfer (P(y|x)) ------------------------------------------------------------
    print("\n=== 4. TRANSFERENCIA DE LO APRENDIDO: sonda lineal de malignidad entrenada en train ===")
    tr_idx = np.flatnonzero(split == "train")
    va_idx = np.flatnonzero(split == "val")
    te_idx = np.flatnonzero(split == "test")
    oof = probe_oof(E, y, codes, tr_idx)
    proj = pca_fit_transform(E, tr_idx)
    Ztr, Zva, Zte = proj(tr_idx), proj(va_idx), proj(te_idx)
    per_c, fitted = {}, {}
    for C in PROBE_CS:
        clf = LogisticRegression(C=C, max_iter=3000).fit(Ztr, y[tr_idx])
        fitted[C] = (clf.predict_proba(Zva)[:, 1], clf.predict_proba(Zte)[:, 1])
        per_c[C] = {"train_oof": roc_auc_score(y[tr_idx], oof[C]),
                    "val": roc_auc_score(y[va_idx], fitted[C][0]), "test": roc_auc_score(y[te_idx], fitted[C][1])}
    print("AUC por regularización C (misma conclusión debe valer para cualquier C):")
    print(pd.DataFrame(per_c).T.round(4).to_string())
    best_c = max(PROBE_CS, key=lambda c: per_c[c]["train_oof"])
    p_va, p_te = fitted[best_c]
    probe = {
        "C": best_c,
        "train_oof": {"auc": float(per_c[best_c]["train_oof"]), "ci": patient_bootstrap_ci(y[tr_idx], oof[best_c], codes[tr_idx], rng)},
        "val": {"auc": float(per_c[best_c]["val"]), "ci": patient_bootstrap_ci(y[va_idx], p_va, codes[va_idx], rng)},
        "test": {"auc": float(per_c[best_c]["test"]), "ci": patient_bootstrap_ci(y[te_idx], p_te, codes[te_idx], rng)},
        "per_C": {str(k): v for k, v in per_c.items()},
    }
    by_src = {}
    for s in SOURCES:
        a, b, c = src[tr_idx] == s, src[va_idx] == s, src[te_idx] == s
        by_src[s] = {"train_oof": safe_auc(y[tr_idx][a], oof[best_c][a]), "val": safe_auc(y[va_idx][b], p_va[b]),
                     "test": safe_auc(y[te_idx][c], p_te[c]), "n_test": int(c.sum()), "malignos_test": int(y[te_idx][c].sum())}
    probe["by_source"] = by_src
    print(f"\nC elegido (mejor AUC fuera de muestra DENTRO de train): {best_c}")
    for k in ("train_oof", "val", "test"):
        print(f"  {k:10s} AUC {probe[k]['auc']:.4f}  IC95% por paciente [{probe[k]['ci'][0]:.4f}, {probe[k]['ci'][1]:.4f}]")
    print("\nPor base de datos:")
    print(pd.DataFrame(by_src).T.round(4).to_string())

    pat_source = per_patient(src, codes)
    counts = {s: {sp: int(df[(src == s) & (split == sp)]["patient_id"].nunique()) for sp in SPLITS} for s in SOURCES}
    print(f"\nRe-splits aleatorios por paciente ({args.n_resplits}), con la misma cantidad de pacientes por base y split que el oficial...")
    rs = Parallel(n_jobs=min(args.n_jobs, 6))(
        delayed(resplit_once)(E, y, codes, pat_source, counts, best_c, 1000 + i) for i in range(args.n_resplits))
    for s in ("val", "test"):
        dist = np.array([r[s] for r in rs])
        probe[f"resplit_{s}"] = {"q025": float(np.quantile(dist, 0.025)), "q975": float(np.quantile(dist, 0.975)),
                                 "mean": float(dist.mean()), "official_percentile": float((dist < probe[s]["auc"]).mean() * 100)}
        print(f"  {s}: re-splits AUC media {dist.mean():.4f}, 95% [{probe[f'resplit_{s}']['q025']:.4f}, {probe[f'resplit_{s}']['q975']:.4f}]"
              f" -> el split oficial ({probe[s]['auc']:.4f}) cae en el percentil {probe[f'resplit_{s}']['official_percentile']:.0f}")
    report["probe"] = probe
    print(f"[{time.time() - t0:.0f}s]")

    # --- 5. Nearest-neighbour distance to train ------------------------------------------------
    print("\n=== 5. DISTANCIA AL VECINO MÁS CERCANO EN TRAIN (similitud coseno, de otro paciente) ===")
    En = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    Et = En[tr_idx]
    sim_tt = Et @ Et.T
    sim_tt[codes[tr_idx][:, None] == codes[tr_idx][None, :]] = -np.inf
    nn = {"train (otro paciente)": sim_tt.max(1)}
    del sim_tt
    dups = []
    for s, idx in (("val", va_idx), ("test", te_idx)):
        sims = En[idx] @ Et.T
        best = sims.argmax(1)
        nn[s] = sims[np.arange(len(idx)), best]
        for i in np.flatnonzero(nn[s] > 0.99):
            a, b = idx[i], tr_idx[best[i]]
            pa = np.array(Image.open(paths[a]), dtype=np.float32).ravel()
            pb = np.array(Image.open(paths[b]), dtype=np.float32).ravel()
            dups.append({"split": s, "imagen": df["ID_image"][a], "paciente": df["patient_id"][a], "clase": int(y[a]),
                         "imagen_train": df["ID_image"][b], "paciente_train": df["patient_id"][b], "clase_train": int(y[b]),
                         "coseno": round(float(nn[s][i]), 4), "corr_pixeles": round(float(np.corrcoef(pa, pb)[0, 1]), 4)})
    knn = {}
    for k, v in nn.items():
        knn[k] = {"mediana": float(np.median(v)), "p10": float(np.percentile(v, 10)), "casi_duplicados_>0.99": int((v > 0.99).sum())}
        if k != "train (otro paciente)":
            knn[k]["KS_D_vs_train"] = float(ks_2samp(nn["train (otro paciente)"], v).statistic)
    print(pd.DataFrame(knn).T.to_string())
    if dups:
        print("\nCasi-duplicados val/test -> train (corr_pixeles ~1 = misma imagen bajo otro paciente):")
        print(pd.DataFrame(dups).to_string(index=False))
    report["nearest_neighbour"] = {"summary": knn, "near_duplicates": dups}

    # --- 6. Source shortcut --------------------------------------------------------------------
    print("\n=== 6. COMPLEMENTARIO: ¿cuánto del AUC viene de reconocer la base de datos? ===")
    prev = {s: float(y[(split == "train") & (src == s)].mean()) for s in SOURCES}
    source_score = np.array([prev[s] for s in src])
    shortcut = {"prevalencia_train_por_base": prev,
                "auc_usando_solo_la_base": {s: safe_auc(y[idx], source_score[idx]) for s, idx in (("val", va_idx), ("test", te_idx))},
                "modelos_test": {"sonda_lineal_imagenet": {"pooled": probe["test"]["auc"], "intra_base": within_source_auc(y[te_idx], p_te, src[te_idx])}}}
    for exp in args.runs:
        f = Path("runs") / "centralizado" / exp / "test" / "predictions.csv"
        if not f.exists():
            continue
        pr = pd.read_csv(f)
        if len(pr) != len(te_idx) or not (pr["y_true"].to_numpy() == y[te_idx]).all():
            print(f"  {exp}: predictions.csv no alinea con el manifest, se omite")
            continue
        prob = pr["y_prob"].to_numpy()
        shortcut["modelos_test"][exp] = {"pooled": safe_auc(y[te_idx], prob), "intra_base": within_source_auc(y[te_idx], prob, src[te_idx]),
                                         **{f"auc_{s}": safe_auc(y[te_idx][src[te_idx] == s], prob[src[te_idx] == s]) for s in SOURCES}}
    print(f"Un clasificador que SOLO mira de qué base viene la imagen (score = prevalencia de su base): "
          f"AUC val {shortcut['auc_usando_solo_la_base']['val']:.4f}, test {shortcut['auc_usando_solo_la_base']['test']:.4f}")
    print("AUC pooled vs AUC intra-base (solo pares maligno/benigno de la MISMA base; quita el aporte de reconocer la base):")
    print(pd.DataFrame(shortcut["modelos_test"]).T.round(4).to_string())
    report["source_shortcut"] = shortcut

    # --- Figures -------------------------------------------------------------------------------
    print("\nCalculando t-SNE y figuras...")
    Y2 = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(E)))
    plot_tsne(Y2, df, out / "tsne_splits_vs_bases.png")
    plot_tests(adv, probe, out / "pruebas_shift.png")

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(f"\nListo en {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
