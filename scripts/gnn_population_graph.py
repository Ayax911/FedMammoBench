"""GNN sobre un grafo de población de mamografías (benigno / maligno), transductivo.

Cada imagen del manifest es un nodo. Sus features son el embedding 2048-d de un ResNet50 congelado
(ImageNet o RadImageNet), reducido con PCA ajustado solo sobre train. Las aristas codifican dos
fuentes de relación que una CNN por imagen no ve:

  * ``patient``  -- todas las vistas (CC/MLO, L/R) de un mismo paciente quedan conectadas.
  * ``breast``   -- solo las vistas de la *misma mama* (paciente + lateralidad): CC <-> MLO, como las
                    lee un radiólogo. 99,2 % de homofilia de etiqueta, vs 70,6 % de ``patient`` (que
                    mezcla mama izquierda y derecha, con diagnósticos distintos) y ~0,55 al azar.
  * ``knn``      -- k vecinos más cercanos (coseno) en espacio de embedding, global.
  * ``pop``      -- kNN restringido a la misma base de datos (grafo de población estilo Parisot et al.
                    2018: la arista exige acuerdo de metadato, así el mensaje no cruza hospitales).
  * ``pop+breast`` -- unión de ``pop`` y ``breast``.

Modelos: MLP (control sin grafo, mismas features), GCN, GraphSAGE, GAT. Control adicional
``mlp/breast-avg``: el MLP con sus probabilidades promediadas por mama a posteriori -- la fusión
multivista más trivial posible, para saber si la GNN aporta algo más que promediar vistas. Entrenamiento full-batch;
las etiquetas de val/test nunca entran a la pérdida (los splits del manifest son disjuntos por
paciente, así que las aristas ``patient`` no cruzan splits). Selección por AUC de val, umbral por
Youden sobre val, reporte en test por imagen, por paciente (max) e intra-base.

    .venv/bin/python -m scripts.gnn_population_graph --image_root /home/akira/snap/steam/preproccesed_julian
"""

from __future__ import annotations

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
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import to_undirected
from torchvision.models import ResNet50_Weights, resnet50

RADIMAGENET_PT = Path(__file__).resolve().parents[2] / "inc-project-models-classification-detection-main/ResNet50.pt"
RADIMAGENET_REMAP = {"backbone.0.": "conv1.", "backbone.1.": "bn1.", "backbone.4.": "layer1.",
                     "backbone.5.": "layer2.", "backbone.6.": "layer3.", "backbone.7.": "layer4."}
MODELS = ["mlp", "gcn", "sage", "gat"]
GRAPHS = ["knn", "pop", "patient", "breast", "pop+breast"]


# ---------------------------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------------------------
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
        x = torch.from_numpy(np.array(img, dtype=np.float32))[None].repeat(3, 1, 1)
        return (x - self.mean) / self.std


def build_backbone(name):
    if name == "imagenet":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = resnet50(weights=None)
        sd = torch.load(RADIMAGENET_PT, map_location="cpu", weights_only=False)
        sd = {next((k.replace(a, b, 1) for a, b in RADIMAGENET_REMAP.items() if k.startswith(a)), k): v
              for k, v in sd.items()}
        missing, _ = model.load_state_dict(sd, strict=False)
        # Solo fc debe faltar; si falta un conv, el remap falló y el embedding sería aleatorio.
        assert set(missing) <= {"fc.weight", "fc.bias"}, missing
    model.fc = nn.Identity()
    return model


def extract(paths, backbone, device, batch_size):
    model = build_backbone(backbone).eval().to(device)
    loader = DataLoader(TiffDataset(paths), batch_size=batch_size, num_workers=4)
    out = []
    # fp32 a propósito: autocast fp16 devuelve NaN en GTX 16xx (ver scripts/split_shift_audit.py).
    with torch.no_grad():
        for x in loader:
            out.append(model(x.to(device)).cpu().numpy())
    E = np.concatenate(out)
    assert np.isfinite(E).all()
    return E


# ---------------------------------------------------------------------------------------------
# Grafos
# ---------------------------------------------------------------------------------------------
def knn_edges(X, k, groups=None):
    src, dst = [], []
    groups = np.zeros(len(X), dtype=int) if groups is None else groups
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        nn_ = NearestNeighbors(n_neighbors=min(k + 1, len(idx)), metric="cosine").fit(X[idx])
        nbr = nn_.kneighbors(X[idx], return_distance=False)[:, 1:]
        src.append(np.repeat(idx, nbr.shape[1]))
        dst.append(idx[nbr.ravel()])
    return np.stack([np.concatenate(src), np.concatenate(dst)])


def patient_edges(patient_codes):
    src, dst = [], []
    for idx in pd.Series(np.arange(len(patient_codes))).groupby(patient_codes).groups.values():
        idx = np.asarray(idx)
        if len(idx) > 1:
            a, b = np.meshgrid(idx, idx)
            m = a != b
            src.append(a[m])
            dst.append(b[m])
    return np.stack([np.concatenate(src), np.concatenate(dst)])


def build_graph(kind, X, db_codes, pat_codes, breast_codes, k):
    parts = []
    if kind == "patient":
        parts.append(patient_edges(pat_codes))
    if kind.endswith("breast"):
        parts.append(patient_edges(breast_codes))
    if kind == "knn":
        parts.append(knn_edges(X, k))
    if kind.startswith("pop"):
        parts.append(knn_edges(X, k, db_codes))
    ei = torch.from_numpy(np.concatenate(parts, axis=1)).long()
    return to_undirected(ei, num_nodes=len(X))  # también deduplica


def homophily(ei, y):
    return float((y[ei[0]] == y[ei[1]]).float().mean())


# ---------------------------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, kind, d_in, hidden=128, dropout=0.5, heads=4):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        if kind == "mlp":
            self.l1, self.l2 = nn.Linear(d_in, hidden), nn.Linear(hidden, 2)
        elif kind == "gcn":
            self.l1, self.l2 = GCNConv(d_in, hidden), GCNConv(hidden, 2)
        elif kind == "sage":
            self.l1, self.l2 = SAGEConv(d_in, hidden), SAGEConv(hidden, 2)
        elif kind == "gat":
            self.l1 = GATConv(d_in, hidden // heads, heads=heads, dropout=dropout)
            self.l2 = GATConv(hidden, 2, heads=1, dropout=dropout)

    def forward(self, x, ei):
        call = (lambda layer, h: layer(h)) if self.kind == "mlp" else (lambda layer, h: layer(h, ei))
        x = F.dropout(x, self.dropout, self.training)
        h = F.elu(call(self.l1, x))
        h = F.dropout(h, self.dropout, self.training)
        return call(self.l2, h)


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def train_one(kind, x, ei, y, masks, class_w, seed, epochs, lr, wd, patience):
    torch.manual_seed(seed)
    model = Net(kind, x.shape[1]).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tr, va = masks["train"], masks["val"]
    yv = y[va].cpu().numpy()
    best, best_state, wait, curve = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[tr], y[tr], weight=class_w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
            vloss = F.cross_entropy(out[va], y[va], weight=class_w).item()
            p = out.softmax(1)[:, 1].cpu().numpy()
        vauc = auc(yv, p[va.cpu().numpy()])
        curve.append((ep, loss.item(), vloss, auc(y[tr].cpu().numpy(), p[tr.cpu().numpy()]), vauc))
        if vauc > best + 1e-4:
            best, wait = vauc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = model(x, ei).softmax(1)[:, 1].cpu().numpy()
    return p, curve


def evaluate(p, df, masks_np):
    y = df["y"].to_numpy()
    va, te = masks_np["val"], masks_np["test"]
    fpr, tpr, thr = roc_curve(y[va], p[va])
    t = float(thr[np.argmax(tpr - fpr)])  # Youden sobre val, nunca sobre test
    pred = p[te] >= t
    yt = y[te]
    tp, tn = int((pred & (yt == 1)).sum()), int((~pred & (yt == 0)).sum())
    sub = df[te].assign(p=p[te])
    pat = sub.groupby("patient_id").agg(y=("y", "max"), p=("p", "max"))
    br = sub.groupby(["patient_id", "laterality"]).agg(y=("y", "max"), p=("p", "mean"))
    res = {
        "val_auc": auc(y[va], p[va]), "test_auc": auc(yt, p[te]), "test_auc_patient": auc(pat.y, pat.p),
        "test_auc_breast": auc(br.y, br.p),
        "test_sens": tp / max(1, (yt == 1).sum()), "test_spec": tn / max(1, (yt == 0).sum()),
        "test_acc": float((pred == yt).mean()), "threshold": t,
    }
    for db, g in sub.groupby("source_dataset"):
        res[f"test_auc_{db}"] = auc(g.y, g.p)
    return res


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def plot_bars(summary, ref_auc, path, backbone):
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.8 / len(GRAPHS)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(GRAPHS)))
    mlp = summary.set_index(["model", "graph"]).test_auc_mean
    ax.axhline(mlp["mlp", "none"], color="gray", ls="--", lw=1.2, label=f"MLP sin grafo ({mlp['mlp', 'none']:.3f})")
    ax.axhline(mlp["mlp", "breast-avg"], color="black", ls="-.", lw=1.2,
               label=f"MLP + promedio por mama ({mlp['mlp', 'breast-avg']:.3f})")
    ax.axhline(ref_auc, color="crimson", ls=":", lw=1.2, label=f"CNN fine-tuned exp28 ({ref_auc:.3f})")
    for j, g in enumerate(GRAPHS):
        s = summary[summary.graph == g].set_index("model").reindex(MODELS[1:])
        xs = np.arange(len(MODELS) - 1) + (j - (len(GRAPHS) - 1) / 2) * w
        ax.bar(xs, s.test_auc_mean, w, yerr=s.test_auc_std, color=colors[j], label=f"grafo: {g}", capsize=2)
    ax.set_xticks(np.arange(len(MODELS) - 1), ["GCN", "GraphSAGE", "GAT"])
    ax.set_ylabel("AUC test (media ± std, 5 semillas)")
    lo = min(summary.test_auc_mean.min(), ref_auc) - 0.05
    ax.set_ylim(lo, min(1.0, max(summary.test_auc_mean.max(), ref_auc) + 0.04))
    ax.set_title(f"GNN sobre grafo de población de mamografías — embeddings ResNet50-{backbone} congelado")
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc(preds, df, te, path):
    y = df["y"].to_numpy()[te]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p[te])
        ax.plot(fpr, tpr, lw=1.8, label=f"{label} (AUC {auc(y, p[te]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 − especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title("Curvas ROC en test (semilla 0)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_homophily(summary, graph_info, bb, path):
    base = summary.set_index(["model", "graph"]).test_auc_mean["mlp", "none"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    marks = {"gcn": "o", "sage": "s", "gat": "^"}
    colors = dict(zip(GRAPHS, plt.cm.viridis(np.linspace(0.15, 0.85, len(GRAPHS)))))
    for _, r in summary[summary.model != "mlp"].iterrows():
        h = graph_info[f"{bb}/{r.graph}"]["label_homophily"]
        ax.scatter(h, r.test_auc_mean - base, marker=marks[r.model], color=colors[r.graph], s=60, edgecolor="k")
    for g, c in colors.items():
        ax.scatter([], [], color=c, label=g, s=40)
    for m, mk in marks.items():
        ax.scatter([], [], marker=mk, color="w", edgecolor="k", label=m.upper())
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set(xlabel="Homofilia de etiqueta del grafo (fracción de aristas benigno–benigno / maligno–maligno)",
           ylabel="Δ AUC test vs MLP sin grafo", title="La GNN ayuda solo si el grafo es homofílico")
    ax.xaxis.label.set_size(8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, c in curves.items():
        c = np.array(c)
        axes[0].plot(c[:, 0], c[:, 1], lw=1.2, label=f"{label} train")
        axes[0].plot(c[:, 0], c[:, 2], lw=1.2, ls="--", label=f"{label} val")
        axes[1].plot(c[:, 0], c[:, 4], lw=1.4, label=label)
    axes[0].set(title="Pérdida (CE ponderada)", xlabel="época")
    axes[1].set(title="AUC de validación", xlabel="época")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_graph(X, ei, df, path, n_patients=10, seed=0):
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    pats = rng.choice(df.loc[df.source_dataset == "cmmd", "patient_id"].unique(), n_patients, replace=False)
    seed_nodes = np.flatnonzero(df.patient_id.isin(pats).to_numpy())
    ei = ei.numpy()
    m = np.isin(ei[0], seed_nodes)
    nodes = np.unique(np.concatenate([seed_nodes, ei[1, m]]))
    keep = np.isin(ei[0], nodes) & np.isin(ei[1], nodes) & (ei[0] < ei[1])
    pos = TSNE(2, perplexity=20, random_state=seed, init="pca").fit_transform(X[nodes])
    loc = {n: i for i, n in enumerate(nodes)}
    same_breast = (df.patient_id + "_" + df.laterality.fillna("NA")).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    for a, b in ei[:, keep].T:
        pa, pb = pos[loc[a]], pos[loc[b]]
        sp = same_breast[a] == same_breast[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="tab:orange" if sp else "0.75",
                lw=1.4 if sp else 0.4, alpha=0.9 if sp else 0.5, zorder=1)
    y = df.y.to_numpy()[nodes]
    for c, lab, col in [(0, "Benigno", "tab:blue"), (1, "Maligno", "tab:red")]:
        ax.scatter(pos[y == c, 0], pos[y == c, 1], s=22, c=col, label=lab, zorder=2, edgecolor="k", lw=0.3)
    ax.plot([], [], color="tab:orange", lw=1.4, label="arista misma mama (CC↔MLO)")
    ax.plot([], [], color="0.75", lw=0.8, label="arista kNN (similitud)")
    ax.set_title(f"Subgrafo pop+breast: {n_patients} pacientes de CMMD y sus vecinos a 1 salto")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/fedmammobench_norm_0_1.csv")
    ap.add_argument("--image_root", default="/home/akira/snap/steam/preproccesed_julian")
    ap.add_argument("--out", default="runs/centralizado/gnn_population_graph")
    ap.add_argument("--backbones", nargs="+", default=["imagenet", "radimagenet"])
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ref_auc", type=float, default=0.8801, help="AUC test de exp28 (CNN fine-tuned)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    df["y"] = (df["classification"].str.strip().str.lower() == "malignant").astype(int)
    masks_np = {s: (df["split"] == s).to_numpy() for s in ["train", "val", "test"]}
    db_codes = pd.factorize(df["source_dataset"])[0]
    pat_codes = pd.factorize(df["patient_id"])[0]
    breast_codes = pd.factorize(df["patient_id"] + "_" + df["laterality"].fillna("NA"))[0]
    dev = torch.device(args.device)
    y = torch.from_numpy(df["y"].to_numpy()).to(dev)
    masks = {s: torch.from_numpy(m).to(dev) for s, m in masks_np.items()}
    n_tr = np.bincount(df["y"][masks_np["train"]])
    class_w = torch.tensor(n_tr.sum() / (2 * n_tr), dtype=torch.float32, device=dev)
    paths = [str(Path(args.image_root) / p) for p in df["preprocessed_image_path"]]
    print(f"{len(df)} imágenes, {df.patient_id.nunique()} pacientes, pesos de clase {class_w.tolist()}")

    rows, report = [], {"args": vars(args), "graphs": {}}
    for bb in args.backbones:
        cache = out / f"emb_{bb}.npy"
        if cache.exists():
            E = np.load(cache)
        else:
            t0 = time.time()
            E = extract(paths, bb, args.device, args.batch_size)
            np.save(cache, E)
            print(f"[{bb}] embeddings {E.shape} en {time.time() - t0:.0f}s")
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        pca = PCA(args.pca, random_state=0).fit(E[masks_np["train"]])
        X = pca.transform(E)
        X = (X - X[masks_np["train"]].mean(0)) / X[masks_np["train"]].std(0)
        x = torch.from_numpy(X).float().to(dev)

        graphs = {g: build_graph(g, X, db_codes, pat_codes, breast_codes, args.k) for g in GRAPHS}
        for g, ei in graphs.items():
            info = {"edges": int(ei.shape[1] // 2), "avg_degree": float(ei.shape[1] / len(df)),
                    "label_homophily": homophily(ei, y.cpu())}
            report["graphs"][f"{bb}/{g}"] = info
            print(f"[{bb}] grafo {g:12s} aristas={info['edges']:6d} grado={info['avg_degree']:.1f} "
                  f"homofilia={info['label_homophily']:.3f}")

        seed0_preds, seed0_curves = {}, {}
        for model in MODELS:
            for g in (["none"] if model == "mlp" else GRAPHS):
                ei = graphs["knn"].to(dev) if model == "mlp" else graphs[g].to(dev)  # el MLP ignora ei
                for seed in range(args.seeds):
                    t0 = time.time()
                    p, curve = train_one(model, x, ei, y, masks, class_w, seed, args.epochs, args.lr, args.wd,
                                         args.patience)
                    r = {"backbone": bb, "model": model, "graph": g, "seed": seed, "epochs_run": len(curve),
                         **evaluate(p, df, masks_np)}
                    rows.append(r)
                    if model == "mlp":
                        pb = pd.Series(p).groupby(breast_codes).transform("mean").to_numpy()
                        rows.append({**r, "graph": "breast-avg", **evaluate(pb, df, masks_np)})
                        if seed == 0:
                            seed0_preds[("mlp", "breast-avg")] = pb
                    if seed == 0:
                        seed0_preds[(model, g)] = p
                        seed0_curves[(model, g)] = curve
                last = [q for q in rows if q["backbone"] == bb and q["model"] == model and q["graph"] == g]
                print(f"[{bb}] {model:4s} {g:12s} test AUC {np.mean([q['test_auc'] for q in last]):.4f} ± "
                      f"{np.std([q['test_auc'] for q in last]):.4f}  ({time.time() - t0:.1f}s/semilla)")

        res = pd.DataFrame([r for r in rows if r["backbone"] == bb])
        summary = (res.groupby(["model", "graph"]).agg(test_auc_mean=("test_auc", "mean"),
                                                         test_auc_std=("test_auc", "std")).reset_index())
        plot_bars(summary, args.ref_auc, out / f"auc_bars_{bb}.png", bb)
        best = summary[summary.model != "mlp"].sort_values("test_auc_mean").iloc[-1]
        plot_roc({"MLP (sin grafo)": seed0_preds[("mlp", "none")],
                  "MLP + promedio por mama": seed0_preds[("mlp", "breast-avg")],
                  "GCN + knn": seed0_preds[("gcn", "knn")],
                  f"{best.model.upper()} + {best.graph}": seed0_preds[(best.model, best.graph)]},
                 df, masks_np["test"], out / f"roc_{bb}.png")
        plot_curves({"MLP": seed0_curves[("mlp", "none")],
                     f"{best.model.upper()}+{best.graph}": seed0_curves[(best.model, best.graph)]},
                    out / f"curves_{bb}.png")
        plot_graph(X, graphs["pop+breast"], df, out / f"graph_{bb}.png")
        plot_homophily(summary, report["graphs"], bb, out / f"homophily_{bb}.png")

    res = pd.DataFrame(rows)
    res.to_csv(out / "results_per_seed.csv", index=False)
    metric_cols = [c for c in res.columns if c.startswith(("test_", "val_"))]
    summ = res.groupby(["backbone", "model", "graph"])[metric_cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index().sort_values(["backbone", "test_auc_mean"], ascending=[True, False])
    summ.to_csv(out / "results_summary.csv", index=False)
    report["summary"] = summ.to_dict(orient="records")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    pd.set_option("display.width", 200)
    print(summ[["backbone", "model", "graph", "test_auc_mean", "test_auc_std", "test_auc_patient_mean", "test_auc_breast_mean",
                "test_sens_mean", "test_spec_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
