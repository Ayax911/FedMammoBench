"""Figuras de presentación para el experimento GNN (scripts/gnn_population_graph.py).

Lee los resultados ya guardados en ``runs/centralizado/gnn_population_graph/`` -- no reentrena nada --
y genera en ``<out>/presentacion/``:

  * ``cnn_explicacion.png``  -- qué hace una CNN, con los mapas de activación REALES de ResNet50
                                sobre una mamografía de test (no una ilustración inventada).
  * ``gnn_explicacion.png``  -- paso de mensajes en un grafo de juguete, en tres paneles.
  * ``pipeline.png``         -- el método de punta a punta con la CC y la MLO reales de una misma mama.
  * ``homofilia_barras.png`` -- homofilia por tipo de grafo frente al azar.
  * ``escalera_auc.png``     -- 5 modelos clave en barras horizontales.
  * ``sens_spec.png``        -- sensibilidad / especificidad al umbral elegido en val.

    .venv/bin/python -m scripts.gnn_presentation_figures --image_root /home/akira/snap/steam/preproccesed_julian
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

NAVY, ORANGE, BLUE, RED, GRAY, INK = "#14213D", "#E07A2E", "#2F6DB5", "#C0392B", "#9AA3AF", "#4A5568"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

# Nombre corto del grafo (como aparece en el código y los CSV) -> qué significa la arista, para que
# ninguna figura muestre un código sin explicar.
GRAPH_LABELS = {
    "knn": "knn = imágenes parecidas (10 vecinos más similares)",
    "pop": "pop = parecidas, solo dentro del mismo hospital",
    "patient": "patient = todas las imágenes de la misma paciente",
    "breast": "breast = misma mama: vista CC ↔ vista MLO",
    "pop+breast": "pop+breast = mezcla de pop y breast",
}
GRAPH_ORDER = ["knn", "pop", "patient", "breast", "pop+breast"]
GRAPH_COLORS = dict(zip(GRAPH_ORDER, ["#3B4A6B", "#5B7DB1", "#2A9D8F", ORANGE, "#A7C957"]))
MODEL_LABELS = {"gcn": "GCN (promedio fijo con vecinos)", "sage": "GraphSAGE (yo y vecinos por separado)",
                "gat": "GAT (atención: elige a quién escuchar)"}

# Una mama maligna de test con exactamente CC + MLO (CMMD, paciente P_000051_CM, mama derecha).
EXAMPLE = {"CC": "norm_0_1/cmmd/cmmd_51.tiff", "MLO": "norm_0_1/cmmd/cmmd_1826.tiff"}


def load(root, rel):
    return np.array(Image.open(Path(root) / rel), dtype=np.float32)


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("->", path)


def arrow(ax, x0, y0, x1, y1, color=ORANGE, lw=3, ms=22, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, **kw))


# ---------------------------------------------------------------------------------------------
def fig_cnn(root, path):
    img = load(root, EXAMPLE["CC"])
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (torch.from_numpy(img)[None, None].repeat(1, 3, 1, 1) - mean) / std
    acts = {}
    with torch.no_grad():
        h = model.relu(model.bn1(model.conv1(x)))
        acts["early"] = h[0]
        h = model.layer2(model.layer1(model.maxpool(h)))
        acts["mid"] = h[0]
        h = model.layer4(model.layer3(h))
        acts["deep"] = h[0]
        vec = model.avgpool(h).flatten().numpy()

    def top(a, n):  # canales más activos -> mapas con algo que ver
        return a[a.mean((1, 2)).argsort(descending=True)[:n]].numpy()

    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(2, 9, width_ratios=[2.2, 0.35, 1, 1, 0.35, 1, 1, 0.35, 1.7], hspace=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, cmap="gray")
    ax.set_title("1. Imagen\n224×224 píxeles", fontsize=13, color=NAVY)
    ax.axis("off")
    for col, key, title in [(2, "early", "2. Primeras capas\nbordes, texturas (112×112)"),
                            (5, "deep", "3. Capas profundas\npatrones complejos (7×7)")]:
        maps = top(acts[key], 4)
        for i in range(4):
            a = fig.add_subplot(gs[i // 2, col + i % 2])
            a.imshow(maps[i], cmap="magma")
            a.axis("off")
        fig.text((fig.add_subplot(gs[0, col]).get_position().x0 + fig.add_subplot(gs[0, col + 1]).get_position().x1) / 2,
                 0.97, title, ha="center", va="top", fontsize=13, color=NAVY)
    ax = fig.add_subplot(gs[:, 8])
    v = vec[:24]
    ax.barh(np.arange(len(v)), v, color=BLUE)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title("4. Vector de 2048 números\n(se muestran 24)", fontsize=13, color=NAVY)
    ax.text(0.5, -0.06, "→ clasificador → p(maligno)", transform=ax.transAxes, ha="center", fontsize=13,
            color=ORANGE, weight="bold")
    for col in (1, 4, 7):
        a = fig.add_subplot(gs[:, col])
        a.axis("off")
        arrow(a, 0.05, 0.5, 0.95, 0.5)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
    # Los ejes auxiliares creados solo para medir posiciones de título quedan vacíos -> se ocultan.
    for a in fig.axes:
        if not a.has_data() and not a.patches and not a.images:
            a.axis("off")
    save(fig, path)


# ---------------------------------------------------------------------------------------------
def fig_gnn(path):
    pos = {0: (0, 0), 1: (-1.3, 0.9), 2: (1.3, 0.9), 3: (-1.4, -0.8), 4: (1.2, -1.0), 5: (-2.6, 1.6), 6: (2.6, -1.7)}
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (4, 6), (1, 3)]
    lab = {0: None, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0}

    def draw(ax, title, mode):
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-2.4, 2.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, color=NAVY, pad=10)
        for a, b in edges:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            hop1 = 0 in (a, b)
            if mode in ("msg", "two") and hop1:
                arrow(ax, x1 if a == 0 else x0, y1 if a == 0 else y0, *pos[0], color=ORANGE, lw=3, ms=20,
                      shrinkA=22, shrinkB=24)
            elif mode == "two" and not hop1:
                src, dst = (b, a) if b in (5, 6) else (a, b)
                arrow(ax, *pos[src], *pos[dst], color="#F2B88A", lw=2.5, ms=16, shrinkA=20, shrinkB=22)
            else:
                ax.plot([x0, x1], [y0, y1], color=GRAY, lw=2, zorder=0)
        for n, (x, y) in pos.items():
            if n == 0:
                fc = "#FFFFFF" if mode != "update" else "#E9A6A0"
                ax.add_patch(Circle((x, y), 0.42, fc=fc, ec=NAVY, lw=3, zorder=3))
                ax.text(x, y, "?" if mode != "update" else "h′", ha="center", va="center", fontsize=20,
                        weight="bold", color=NAVY, zorder=4)
            else:
                ax.add_patch(Circle((x, y), 0.32, fc=RED if lab[n] else BLUE, ec="white", lw=2, zorder=3,
                                    alpha=0.35 if (mode == "msg" and n in (5, 6)) else 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    draw(axes[0], "1. Cada imagen es un nodo\ncon su vector de la CNN", "plain")
    draw(axes[1], "2. Paso de mensajes:\nlos vecinos envían su vector", "msg")
    draw(axes[2], "3. Con 2 capas llega información\nde los vecinos de los vecinos", "two")
    fig.text(0.5, 0.02, "nuevo vector del nodo  =  f( su propio vector ,  promedio de los vectores vecinos )",
             ha="center", fontsize=16, color=NAVY, family="DejaVu Sans Mono",
             bbox=dict(boxstyle="round,pad=0.6", fc="#FBE6D6", ec=ORANGE))
    for c, t in [(RED, "maligno (etiqueta conocida)"), (BLUE, "benigno (etiqueta conocida)"),
                 ("#FFFFFF", "nodo a clasificar")]:
        axes[0].scatter([], [], s=160, c=c, edgecolor=NAVY, label=t)
    axes[0].legend(loc="lower left", fontsize=10, frameon=False, bbox_to_anchor=(-0.05, -0.12))
    fig.subplots_adjust(bottom=0.16)
    save(fig, path)


# ---------------------------------------------------------------------------------------------
def fig_pipeline(root, path):
    cc, mlo = load(root, EXAMPLE["CC"]), load(root, EXAMPLE["MLO"])
    fig = plt.figure(figsize=(17, 5.6))
    gs = fig.add_gridspec(2, 9, width_ratios=[1, 0.32, 1.25, 0.32, 1.1, 0.32, 1.9, 0.32, 1.25], hspace=0.25)
    for r, (im, name) in enumerate([(cc, "CC"), (mlo, "MLO")]):
        a = fig.add_subplot(gs[r, 0])
        a.imshow(im, cmap="gray")
        a.set_title(f"vista {name}", fontsize=12, color=NAVY)
        a.axis("off")

    def box(spec, title, body, dark=False):
        a = fig.add_subplot(spec)
        a.axis("off")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.add_patch(FancyBboxPatch((0.03, 0.12), 0.94, 0.76, boxstyle="round,pad=0.02",
                                   fc=NAVY if dark else "#FFFFFF", ec=NAVY, lw=2))
        a.text(0.5, 0.74, title, ha="center", fontsize=15, weight="bold", color=ORANGE if dark else NAVY)
        a.text(0.5, 0.42, body, ha="center", va="center", fontsize=12.5, color="#F7F5F0" if dark else INK,
               linespacing=1.5)
        return a

    box(gs[:, 2], "ResNet50\ncongelado", "extrae un vector\nde 2048\nnúmeros\npor imagen")
    box(gs[:, 4], "Vector", "PCA → 128\nnúmeros\n=\nfeatures\ndel nodo")
    a = fig.add_subplot(gs[:, 6])
    a.axis("off")
    a.set_xlim(-2.2, 2.2)
    a.set_ylim(-2, 2.2)
    a.set_aspect("equal")
    a.set_title("Grafo", fontsize=15, weight="bold", color=NAVY)
    pts = {"CC": (-0.8, 0.5), "MLO": (0.9, 0.3), "a": (-1.6, -1.1), "b": (-0.2, -1.4), "c": (1.6, -1.0),
           "d": (1.7, 1.5), "e": (-1.7, 1.5)}
    for u, v in [("CC", "a"), ("CC", "b"), ("CC", "e"), ("MLO", "c"), ("MLO", "d"), ("MLO", "b")]:
        a.plot(*zip(pts[u], pts[v]), color=GRAY, lw=1.5, zorder=0)
    a.plot(*zip(pts["CC"], pts["MLO"]), color=ORANGE, lw=5, zorder=1)
    a.text(0.05, 1.05, "misma mama", ha="center", fontsize=11, color=ORANGE, weight="bold")
    for k, (x, y) in pts.items():
        big = k in ("CC", "MLO")
        a.add_patch(Circle((x, y), 0.36 if big else 0.22, fc=RED if big else "#C9D1DC", ec=NAVY, lw=1.5, zorder=2))
        if big:
            a.text(x, y, k, ha="center", va="center", fontsize=9, color="white", weight="bold", zorder=3)
    a.text(0, -1.95, "gris: similitud (kNN)", ha="center", fontsize=10.5, color=INK)
    box(gs[:, 8], "GNN", "2 capas de\npaso de\nmensajes\n→ p(maligno)", dark=True)
    for c in (1, 3, 5, 7):
        a = fig.add_subplot(gs[:, c])
        a.axis("off")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        arrow(a, 0.05, 0.5, 0.95, 0.5)
    save(fig, path)


# ---------------------------------------------------------------------------------------------
def fig_homophily(report, path):
    names = {"knn": "knn\nimágenes\nparecidas", "pop": "pop\nparecidas del\nmismo hospital",
             "patient": "patient\nmisma\npaciente", "pop+breast": "pop+breast\nmezcla de\npop y breast",
             "breast": "breast\nmisma mama\n(CC↔MLO)"}
    h = {g: report["graphs"][f"imagenet/{g}"]["label_homophily"] for g in names}
    order = sorted(h, key=h.get)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar([names[g] for g in order], [h[g] for g in order],
                  color=[ORANGE if g == "breast" else "#B8C2D1" for g in order], width=0.62)
    for b, g in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, h[g] + 0.012, f"{h[g]:.2f}", ha="center", fontsize=15,
                weight="bold", color=NAVY)
    ax.axhline(0.551, color=RED, ls="--", lw=2)
    ax.text(-0.45, 0.56, "azar = 0,55", color=RED, fontsize=13, weight="bold")
    ax.set_ylim(0.4, 1.06)
    ax.set_ylabel("Homofilia\n(aristas que unen el mismo diagnóstico)", fontsize=12)
    ax.set_title("¿Qué tan “honestas” son las aristas de cada grafo?", fontsize=16, color=NAVY, pad=12)
    ax.tick_params(axis="x", labelsize=12)
    save(fig, path)


def fig_ladder(summary, ref_auc, path):
    s = summary[summary.backbone == "imagenet"].set_index(["model", "graph"])
    rows = [("CNN + GNN (grafo por similitud)", s.loc[("gcn", "knn")], "#B8C2D1"),
            ("MLP (sin grafo)", s.loc[("mlp", "none")], "#B8C2D1"),
            ("MLP + promedio por mama", s.loc[("mlp", "breast-avg")], "#7F8FA6"),
            ("GNN (grafo misma mama)", s.loc[("gcn", "breast")], ORANGE)]
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [r[0] for r in rows] + ["CNN fine-tuned completa (referencia)"]
    vals = [r[1].test_auc_mean for r in rows] + [ref_auc]
    errs = [r[1].test_auc_std for r in rows] + [0]
    cols = [r[2] for r in rows] + [NAVY]
    y = np.arange(len(vals))
    ax.barh(y, vals, xerr=errs, color=cols, height=0.62, capsize=4)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.003, yi, f"{v:.3f}".replace(".", ","), va="center", fontsize=15, weight="bold", color=NAVY)
    ax.set_yticks(y, labels, fontsize=13)
    ax.set_xlim(0.75, 0.9)
    ax.set_xlabel("AUC en test (1,0 = perfecto; 0,5 = azar)", fontsize=12)
    ax.set_title("Del peor al mejor: el grafo correcto sube 2,4 puntos de AUC", fontsize=16, color=NAVY, pad=12)
    ax.grid(axis="x", alpha=0.3)
    save(fig, path)


def fig_homophily_scatter(summary, report, path):
    s = summary[summary.backbone == "imagenet"].set_index(["model", "graph"]).test_auc_mean
    base = s["mlp", "none"]
    marks = {"gcn": "o", "sage": "s", "gat": "^"}
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhspan(0, 0.035, color="#E8F3EC", zorder=0)
    ax.axhspan(-0.1, 0, color="#FBEAEA", zorder=0)
    ax.text(0.645, 0.03, "el grafo AYUDA (mejor que el MLP sin grafo)", fontsize=11, color="#2E7D4F", weight="bold")
    ax.text(0.645, -0.095, "el grafo PERJUDICA (peor que el MLP sin grafo)", fontsize=11, color=RED, weight="bold")
    for g in GRAPH_ORDER:
        h = report["graphs"][f"imagenet/{g}"]["label_homophily"]
        for m, mk in marks.items():
            ax.scatter(h, s[m, g] - base, marker=mk, color=GRAPH_COLORS[g], s=150, edgecolor="k", lw=1, zorder=3)
    ax.axhline(0, color="gray", ls="--", lw=1.2)
    ax.set_xlim(0.635, 1.01)
    ax.set_ylim(-0.1, 0.035)
    ax.set_xlabel("Homofilia del grafo = % de aristas que unen imágenes con el mismo diagnóstico", fontsize=12)
    ax.set_ylabel("Δ AUC en test frente al MLP sin grafo (0,833)", fontsize=12)
    ax.set_title("La GNN solo ayuda cuando el grafo es homofílico", fontsize=16, color=NAVY, pad=12)
    hg = [ax.scatter([], [], color=GRAPH_COLORS[g], s=120, edgecolor="k", label=GRAPH_LABELS[g]) for g in GRAPH_ORDER]
    hm = [ax.scatter([], [], marker=mk, color="white", s=120, edgecolor="k", label=MODEL_LABELS[m])
          for m, mk in marks.items()]
    l1 = ax.legend(handles=hg, title="Color = tipo de grafo", fontsize=10.5, title_fontsize=11,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.add_artist(l1)
    ax.legend(handles=hm, title="Forma = tipo de GNN", fontsize=10.5, title_fontsize=11,
              loc="upper left", bbox_to_anchor=(1.01, 0.45), frameon=False)
    ax.grid(alpha=0.3)
    save(fig, path)


def fig_all_bars(summary, ref_auc, path):
    s = summary[summary.backbone == "imagenet"].set_index(["model", "graph"])
    models = ["gcn", "sage", "gat"]
    w = 0.16
    fig, ax = plt.subplots(figsize=(15, 6))
    for j, g in enumerate(GRAPH_ORDER):
        xs = np.arange(3) + (j - 2) * w
        ax.bar(xs, [s.loc[(m, g)].test_auc_mean for m in models], w, yerr=[s.loc[(m, g)].test_auc_std for m in models],
               color=GRAPH_COLORS[g], capsize=3, label=GRAPH_LABELS[g], edgecolor="white")
    for val, style, col, lab in [(s.loc[("mlp", "none")].test_auc_mean, "--", "gray", "MLP sin grafo"),
                                 (s.loc[("mlp", "breast-avg")].test_auc_mean, "-.", "black", "MLP + promedio por mama"),
                                 (ref_auc, ":", RED, "CNN entrenada completa (exp28)")]:
        ax.axhline(val, color=col, ls=style, lw=1.6, label=f"{lab} ({val:.3f})".replace(".", ","))
    ax.set_xticks(np.arange(3), ["GCN\npromedio fijo con vecinos", "GraphSAGE\nyo y vecinos por separado",
                                 "GAT\natención: elige a quién escuchar"], fontsize=12)
    ax.set_ylim(0.72, 0.9)
    ax.set_ylabel("AUC en test (media ± desviación, 5 semillas)", fontsize=12)
    ax.set_title("Las 15 combinaciones GNN × grafo", fontsize=16, color=NAVY, pad=12)
    ax.legend(fontsize=10.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, path)


def fig_sens_spec(per_seed, path):
    r = per_seed[per_seed.backbone == "imagenet"]
    sel = [("mlp", "none", "MLP\n(sin grafo)"), ("gcn", "knn", "GNN\nsimilitud"),
           ("mlp", "breast-avg", "MLP + promedio\npor mama"), ("gcn", "breast", "GNN\nmisma mama")]
    g = r.groupby(["model", "graph"])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(sel))
    for off, col, name, color in [(-0.19, "test_sens", "Sensibilidad (malignos detectados)", RED),
                                  (0.19, "test_spec", "Especificidad (benignos bien descartados)", BLUE)]:
        m = [g.get_group((a, b))[col].mean() for a, b, _ in sel]
        sd = [g.get_group((a, b))[col].std() for a, b, _ in sel]
        ax.bar(x + off, m, 0.36, yerr=sd, color=color, alpha=0.85, capsize=4, label=name)
        for xi, v in zip(x + off, m):
            ax.text(xi, 0.43, f"{v:.0%}", ha="center", fontsize=13, weight="bold", color="white")
    ax.set_xticks(x, [s[2] for s in sel], fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Proporción en test", fontsize=12)
    ax.set_title("Umbral elegido en validación (Youden) — media ± std de 5 semillas", fontsize=14, color=NAVY)
    ax.legend(fontsize=11, loc="upper left", frameon=False, ncol=2)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", default="/home/akira/snap/steam/preproccesed_julian")
    ap.add_argument("--results", default="runs/centralizado/gnn_population_graph")
    ap.add_argument("--ref_auc", type=float, default=0.8801)
    args = ap.parse_args()
    res = Path(args.results)
    out = res / "presentacion"
    out.mkdir(exist_ok=True)
    report = json.loads((res / "report.json").read_text())
    summary = pd.read_csv(res / "results_summary.csv")
    per_seed = pd.read_csv(res / "results_per_seed.csv")
    fig_cnn(args.image_root, out / "cnn_explicacion.png")
    fig_gnn(out / "gnn_explicacion.png")
    fig_pipeline(args.image_root, out / "pipeline.png")
    fig_homophily(report, out / "homofilia_barras.png")
    fig_ladder(summary, args.ref_auc, out / "escalera_auc.png")
    fig_sens_spec(per_seed, out / "sens_spec.png")
    fig_homophily_scatter(summary, report, out / "homofilia_vs_auc.png")
    fig_all_bars(summary, args.ref_auc, out / "todas_las_combinaciones.png")


if __name__ == "__main__":
    main()
