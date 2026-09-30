#!/usr/bin/env python3
"""Measure how different each source database (federated node) is: deep embeddings (PCA/UMAP/t-SNE,
adversarial validation, PERMANOVA) and GLCM texture features, grouped by source_dataset.

Companion to split_shift_audit.py (train/val/test shift) and compare_by_database.py (AUC pooled vs
intra-base) -- this one asks the heterogeneity question one level upstream of any trained model: do
the four source databases look like different distributions in embedding/texture space, before a
classifier ever sees them. Reuses the frozen ResNet50-ImageNet embeddings cached by
split_shift_audit.py when available (same manifest, same image_root) instead of paying for a second
forward pass over 8k+ images.
"""

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
from PIL import Image
from scipy.stats import kruskal
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from umap import UMAP

SOURCES = ["cmmd", "kau-bcmd", "cdd-cesm", "inbreast"]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
GLCM_DISTANCES = (1, 3)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_LEVELS = 32  # manifest images are float32 in [0,1] (norm_0_1); 256 levels would make the co-occurrence matrix too sparse at 224x224.

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE = "#fcfcfb"
PALETTE = ["#2a78d6", "#eb6834", "#3fa66a", "#a04fd6"]  # one color per SOURCES entry, same order


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
        x = torch.from_numpy(arr)[None].repeat(3, 1, 1)
        return (x - self.mean) / self.std


def extract_embeddings(paths, device, batch_size):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(TiffDataset(paths), batch_size=batch_size, num_workers=4)
    embs = []
    # fp32 on purpose: fp16 autocast returns all-NaN features on GTX 16xx cards at batch 64 (see split_shift_audit.py).
    with torch.no_grad():
        for x in loader:
            embs.append(model(x.to(device)).cpu().numpy())
    E = np.concatenate(embs)
    assert np.isfinite(E).all(), "non-finite embeddings"
    return E


def glcm_feature_names():
    return [f"{p}_d{d}" for p in GLCM_PROPS for d in GLCM_DISTANCES]


def glcm_features(path):
    arr = np.array(Image.open(path), dtype=np.float32)
    q = np.clip(arr * (GLCM_LEVELS - 1), 0, GLCM_LEVELS - 1).round().astype(np.uint8)
    glcm = graycomatrix(q, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    # Averaged over angles for rotation invariance; kept separate per distance (short- vs longer-range texture).
    return np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in GLCM_PROPS])


def per_patient(values, codes):
    out = np.empty(codes.max() + 1, dtype=np.asarray(values).dtype)
    out[codes] = values
    return out


def patient_perms(img_values, codes, n, rng):
    # Nodes are whole databases: a patient never appears in two sources, so permuting must respect that structure.
    pat = per_patient(img_values, codes)
    return [rng.permutation(pat)[codes] for _ in range(n)]


def perm_p(real, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= real)) / (1 + len(null)))


def kruskal_by_node(x, node_codes, codes, n_perm, rng):
    groups = [x[node_codes == g] for g in range(node_codes.max() + 1)]
    H = float(kruskal(*groups).statistic)
    perms = patient_perms(node_codes, codes, n_perm, rng)
    null = [kruskal(*[x[g == v] for v in range(g.max() + 1)]).statistic for g in perms]
    return H, perm_p(H, null)


def cv_auc_multiclass(Z, y, groups, C=0.1, seed=0):
    # Grouped by patient: a patient's node is constant, so ungrouped CV would score patient identity, not node shift.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(np.unique(y))))
    for tr, te in cv.split(Z, y, groups):
        oof[te] = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], y[tr]).predict_proba(Z[te])
    return float(roc_auc_score(y, oof, multi_class="ovr"))


def adversarial_multiclass(Z, y, groups, n_perm, rng, n_jobs):
    real = cv_auc_multiclass(Z, y, groups)
    null = Parallel(n_jobs=n_jobs)(delayed(cv_auc_multiclass)(Z, yp, groups) for yp in patient_perms(y, groups, n_perm, rng))
    null = np.array(null)
    return {"auc": real, "p": perm_p(real, null), "null_q975": float(np.quantile(null, 0.975)), "null_mean": float(null.mean())}


def permanova(Z, node_codes, codes, n_perm, rng):
    """Pseudo-F for group separation (Anderson 2001); on Euclidean distance this reduces to a
    between/within sum-of-squares ratio, so no NxN distance matrix is needed for N~8k rows."""

    def pseudo_f(g):
        n, k = len(g), g.max() + 1
        ss_total = ((Z - Z.mean(0)) ** 2).sum()
        ss_within = sum(((Z[g == v] - Z[g == v].mean(0)) ** 2).sum() for v in range(k) if (g == v).sum() > 1)
        ss_between = ss_total - ss_within
        return (ss_between / (k - 1)) / (ss_within / (n - k))

    real = pseudo_f(node_codes)
    null = [pseudo_f(g) for g in patient_perms(node_codes, codes, n_perm, rng)]
    return {"pseudo_F": float(real), "p_perm": perm_p(real, null)}


def style_axis_blank(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_projections(proj, node_codes, path):
    fig, axes = plt.subplots(1, len(proj), figsize=(5.4 * len(proj), 5.4), facecolor=SURFACE)
    for ax, (name, Y) in zip(axes, proj.items()):
        style_axis_blank(ax)
        for i, s in enumerate(SOURCES):
            m = node_codes == i
            ax.scatter(Y[m, 0], Y[m, 1], s=4, c=PALETTE[i], alpha=0.6, linewidths=0,
                       label=f"{s} (n={int(m.sum())})", rasterized=True)
        ax.set_title(name, color=INK, fontsize=13, loc="left")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Embeddings ResNet50-ImageNet por base de datos (nodo federado)", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.9, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_glcm(G, names, node_codes, path):
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.4), facecolor=SURFACE)
    for j, ax in enumerate(axes.flat):
        ax.set_facecolor(SURFACE)
        data = [G[node_codes == i, j] for i in range(len(SOURCES))]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        for median in bp["medians"]:
            median.set_color(INK)
        ax.set_xticks(range(1, len(SOURCES) + 1))
        ax.set_xticklabels(SOURCES, rotation=30, ha="right", fontsize=8, color=INK2)
        ax.set_title(names[j], color=INK, fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    fig.suptitle("Propiedades GLCM por base de datos (nodo federado)", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/fedmammobench_norm_0_1.csv")
    ap.add_argument("--image-root", default="/home/akira/snap/steam/preproccesed_julian")
    ap.add_argument("--out-dir", default="runs/centralizado/node_heterogeneity_audit")
    ap.add_argument("--embeddings-cache", default="runs/centralizado/split_shift_audit/embeddings.pt",
                     help="Reused verbatim if it aligns row-for-row with --manifest (see split_shift_audit.py).")
    ap.add_argument("--n-perm", type=int, default=1000, help="Permutaciones para Kruskal-Wallis y PERMANOVA (baratos).")
    ap.add_argument("--n-perm-adv", type=int, default=100, help="Permutaciones para la validación adversarial (cara: 5 folds x LogisticRegression por permutación).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {}

    df = pd.read_csv(args.manifest)
    assert df.groupby("patient_id")["source_dataset"].nunique().max() == 1, "paciente en más de una base"
    codes, _ = pd.factorize(df["patient_id"])
    src = df["source_dataset"].to_numpy()
    node_codes = pd.Categorical(src, categories=SOURCES).codes.astype(np.int64)  # .codes defaults to int8 -> overflows in permanova's (n - k) arithmetic
    assert (node_codes >= 0).all(), f"source_dataset fuera de SOURCES={SOURCES}"
    paths = [str(Path(args.image_root) / p) for p in df["preprocessed_image_path"]]
    print(f"{len(df)} imágenes, {codes.max() + 1} pacientes, nodos: " +
          ", ".join(f"{s}={int((src == s).sum())}" for s in SOURCES))

    # --- Embeddings (reused from split_shift_audit.py's cache when it matches) --------------------
    ext_cache, own_cache = Path(args.embeddings_cache), out / "embeddings.pt"
    if ext_cache.exists() and len(torch.load(ext_cache, weights_only=False)["E"]) == len(df):
        E = torch.load(ext_cache, weights_only=False)["E"]
        print(f"embeddings reusados de {ext_cache} {E.shape}")
    elif own_cache.exists():
        E = torch.load(own_cache, weights_only=False)["E"]
        print(f"embeddings reusados de {own_cache} {E.shape}")
    else:
        print(f"Extrayendo embeddings ResNet50-ImageNet de {len(paths)} imágenes en {args.device}...")
        E = extract_embeddings(paths, args.device, args.batch_size)
        torch.save({"E": E}, own_cache)
    print(f"embeddings {E.shape} [{time.time() - t0:.0f}s]")

    # --- GLCM ----------------------------------------------------------------------------------
    glcm_cache = out / "glcm.npy"
    if glcm_cache.exists() and (G := np.load(glcm_cache)).shape[0] == len(df):
        print(f"GLCM reusado de {glcm_cache} {G.shape}")
    else:
        print(f"Calculando GLCM ({GLCM_LEVELS} niveles, distancias {GLCM_DISTANCES}) de {len(paths)} imágenes...")
        G = np.array(Parallel(n_jobs=args.n_jobs)(delayed(glcm_features)(p) for p in paths), dtype=np.float32)
        np.save(glcm_cache, G)
    glcm_names = glcm_feature_names()
    print(f"GLCM {G.shape} [{time.time() - t0:.0f}s]")

    # --- 1. GLCM: heterogeneidad univariada por nodo --------------------------------------------
    print(f"\n=== 1. GLCM: heterogeneidad por nodo (Kruskal-Wallis + permutación por paciente, n={args.n_perm}) ===")
    glcm_tests = {}
    for j, name in enumerate(glcm_names):
        H, p = kruskal_by_node(G[:, j], node_codes, codes, args.n_perm, rng)
        means = {f"media_{s}": float(G[node_codes == i, j].mean()) for i, s in enumerate(SOURCES)}
        glcm_tests[name] = {"H": H, "p_perm": p, **means}
    print(pd.DataFrame(glcm_tests).T.round(4).to_string())
    report["glcm"] = glcm_tests
    plot_glcm(G, glcm_names, node_codes, out / "glcm_por_nodo.png")

    # --- 2. Embeddings: proyecciones 2D por nodo -------------------------------------------------
    print("\n=== 2. Embeddings: proyecciones PCA / UMAP / t-SNE, coloreadas por nodo ===")
    Es = StandardScaler().fit_transform(E)
    P50 = PCA(50, random_state=0).fit_transform(Es)
    proj = {
        "PCA": PCA(2, random_state=0).fit_transform(Es),
        "UMAP": UMAP(n_components=2, random_state=0).fit_transform(P50),
        "t-SNE": TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(P50),
    }
    plot_projections(proj, node_codes, out / "embeddings_por_nodo.png")
    print(f"[{time.time() - t0:.0f}s]")

    # --- 3. Embeddings: ¿un clasificador adivina el nodo? ----------------------------------------
    print(f"\n=== 3. Embeddings: validación adversarial multiclase (CV agrupada por paciente, {args.n_perm_adv} permutaciones) ===")
    Z128 = PCA(128, random_state=0).fit_transform(Es)  # fit sin ver node_codes -> no puede filtrar lo que el clasificador debe detectar.
    adv = adversarial_multiclass(Z128, node_codes, codes, args.n_perm_adv, rng, args.n_jobs)
    print(f"AUC multiclase (nodo) = {adv['auc']:.4f}  p={adv['p']:.3f}  (95% al azar hasta {adv['null_q975']:.4f}, media {adv['null_mean']:.4f})")
    report["adversarial_node"] = adv
    print(f"[{time.time() - t0:.0f}s]")

    # --- 4. Embeddings: PERMANOVA por nodo -------------------------------------------------------
    print(f"\n=== 4. Embeddings: PERMANOVA por nodo (pseudo-F, {args.n_perm} permutaciones) ===")
    pn = permanova(Z128, node_codes, codes, args.n_perm, rng)
    print(f"pseudo-F = {pn['pseudo_F']:.2f}  p={pn['p_perm']:.3f}")
    report["permanova_node"] = pn

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(f"\nListo en {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
