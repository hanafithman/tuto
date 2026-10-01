# -*- coding: utf-8 -*-
"""
Objective 1 - thesis figures
============================
Reads the outputs of `objective1_ablations.py` and writes PNG figures (300 dpi).

  fig1_confusion_macrotrack.png : E6 macro-track confusion matrices (Proposed vs best baseline)
  fig2_confusion_majors.png     : E5 grouped-major confusion matrices (Proposed vs best baseline)
  fig3_per_class_f1.png         : E5 per-class F1, every model x every class
  fig4_metric_heatmaps.png      : Macro-F1 for every model x experiment (E0-E6)
  fig5_significance.png         : difference vs Proposed with paired t-test stars (E0-E6)
  fig6_final_metrics.png        : E5/E6 headline metrics, mean +/- std, majority-class line

Confusion matrices pool the out-of-fold predictions of all 5 folds (every student is
predicted exactly once, by a model that never saw them) and are row-normalised:
cell (i, j) = % of students whose true class is i that were predicted as j.

Usage: python objective1_figures.py [RESULTS_DIR] [OUT_DIR]
       (defaults: current directory, ./figures)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

PROPOSED = "Proposed: MLP + Causal TCN + LSTM"
REFERENCE = "Majority Class (reference)"
EXPERIMENTS = {
    "E0": "E0 Baseline",
    "E1": "E1 Direct track",
    "E2": "E2 No weights",
    "E3": "E3 Grouped",
    "E4": "E4 No Saber Pro",
    "E5": "E5 Final majors",
    "E6": "E6 Final track",
}

# Palette (validated reference instance): sequential blue, diverging blue<->red
# with a neutral gray midpoint, recessive ink for text and axes.
BLUE_STEPS = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
              "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_STEPS)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ef8a7e", "#b8302f"])
HIGHLIGHT = "#2a78d6"
MUTED = "#b9b8b2"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 10,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

SHORT = {
    REFERENCE: "Majority class",
    "Logistic Regression": "Logistic Reg.",
    "Random Forest": "Random Forest",
    "HistGradientBoosting": "HistGradBoost",
    "Static MLP Only": "Static MLP",
    "Standalone LSTM": "LSTM",
    "Hybrid 1: MLP + CNN + LSTM": "H1 MLP+CNN+LSTM",
    "Hybrid 2: MLP + BiLSTM": "H2 MLP+BiLSTM",
    "Hybrid 3: MLP + GRU": "H3 MLP+GRU",
    "Hybrid 4: MLP + Transformer": "H4 MLP+Transf.",
    PROPOSED: "Proposed TCN+LSTM",
}


def short_class(name: str) -> str:
    rep = [("ENGINEERING", "ENG."), ("ENGINEERY", "ENG."), ("AND ", "& "),
           ("TELECOMMUNICATIONS", "TELECOM"), ("OTHER - ", "OTHER: "),
           ("Mechanical, Electrical & Tech", "Mech/Elec/Tech"),
           ("Industrial & Management", "Industrial"), ("Civil & Infrastructure", "Civil"),
           ("Chemical & Process", "Chemical")]
    for a, b in rep:
        name = name.replace(a, b)
    return name.title().replace("Eng.", "Eng.").replace("&", "&")


def load_oof(tag):
    d = np.load(os.path.join(RESULTS, f"objective1_{tag}_oof.npz"), allow_pickle=False)
    return {k: d[k] for k in d.files}


def mean_of(cell):
    return float(str(cell).split("±")[0])


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def best_baseline(results, metric):
    r = results[(results["Model"] != PROPOSED) & (results["Model"] != REFERENCE)]
    return r.loc[r[metric].map(mean_of).idxmax(), "Model"]


def draw_cm(ax, cm_pct, counts, labels, title, annotate_min=0.0, fontsize=8):
    im = ax.imshow(cm_pct, cmap=SEQ, vmin=0, vmax=100)
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels([f"{lab}  (n={c:,})" for lab, c in zip(labels, counts)])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            v = cm_pct[i, j]
            if np.isnan(v) or v < annotate_min:
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fontsize,
                    color="white" if v >= 55 else INK,
                    fontweight="bold" if i == j else "normal")
    return im


def confusion_figure(tag, results, metric, use_macro, fname, suptitle, annotate_min, fontsize,
                     panel_size):
    d = load_oof(tag)
    models = list(d["models"])
    if use_macro:
        y_true = d["y_macro"]
        labels = [short_class(m) for m in d["macros"]]
    else:
        y_true = d["y"]
        labels = [short_class(m) for m in d["labels"]]
    n = len(labels)
    baseline = best_baseline(results, metric)

    fig, axes = plt.subplots(1, 2, figsize=(panel_size * 2 + 1.2, panel_size),
                             constrained_layout=True)
    for ax, model in zip(axes, [PROPOSED, baseline]):
        probs = d["probs"][models.index(model)]
        if use_macro and probs.shape[1] != n:  # aggregate labels -> tracks
            agg = np.zeros((len(probs), n))
            for li, mi in enumerate(d["label_to_macro_idx"]):
                agg[:, mi] += probs[:, li]
            probs = agg
        y_pred = probs.argmax(axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n))
        counts = cm.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = 100.0 * cm / counts[:, None]
        acc = 100.0 * np.trace(cm) / cm.sum()
        f1 = f1_score(y_true, y_pred, average="macro", labels=np.arange(n), zero_division=0)
        im = draw_cm(ax, pct, counts, labels,
                     f"{SHORT[model]}\naccuracy {acc:.1f}%  |  Macro-F1 {f1:.3f}",
                     annotate_min, fontsize)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("% of true class (row-normalised)")
    cb.outline.set_visible(False)
    fig.suptitle(suptitle, x=0.0, ha="left", fontsize=11, fontweight="bold")
    save(fig, fname)


def per_class_f1_figure():
    d = load_oof("E5")
    models = list(d["models"])
    y = d["y"]
    labels = list(d["labels"])
    support = np.bincount(y, minlength=len(labels))
    order = np.argsort(-support)
    mat = np.zeros((len(models), len(labels)))
    for mi in range(len(models)):
        pred = d["probs"][mi].argmax(axis=1)
        mat[mi] = f1_score(y, pred, average=None, labels=np.arange(len(labels)),
                           zero_division=0)
    mat = mat[:, order]
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=max(0.7, mat.max()), aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{short_class(labels[i])} ({support[i]:,})" for i in order],
                       rotation=50, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([SHORT[m] for m in models])
    ax.get_yticklabels()[models.index(PROPOSED)].set_fontweight("bold")
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "0" if v == 0 else f"{v:.2f}".lstrip("0"), ha="center",
                    va="center", fontsize=7, color="white" if v >= 0.45 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("F1 (out-of-fold)")
    cb.outline.set_visible(False)
    ax.set_title("E5 per-class F1 by model - classes sorted by size (students in brackets)",
                 loc="left")
    save(fig, "fig3_per_class_f1.png")


def load_results():
    out = {}
    for tag in EXPERIMENTS:
        path = os.path.join(RESULTS, f"objective1_{tag}_results.csv")
        if os.path.exists(path):
            out[tag] = pd.read_csv(path)
    return out


def metric_heatmaps(res):
    models = list(res["E0"]["Model"])
    panels = [
        ("Macro-Track Macro-F1", "Macro-track Macro-F1 (4 tracks)", list(res)),
        ("Granular Macro-F1", "Major Macro-F1 (21 majors; 16 grouped in E3/E5)",
         [t for t in res if "Granular Macro-F1" in res[t].columns]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [len(panels[0][2]), len(panels[1][2])]})
    for ax, (metric, title, tags) in zip(axes, panels):
        mat = np.array([[mean_of(res[t].set_index("Model").loc[m, metric]) for t in tags]
                        for m in models])
        im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=mat.max() * 1.05, aspect="auto")
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([EXPERIMENTS[t] for t in tags], rotation=35, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([SHORT[m] for m in models])
        ax.get_yticklabels()[models.index(PROPOSED)].set_fontweight("bold")
        ax.set_xticks(np.arange(-.5, len(tags), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(mat.shape[0]):
            best = mat[1:, :].max(axis=0)  # best real model per column (skip reference)
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v >= mat.max() * 0.6 else INK,
                        fontweight="bold" if (i > 0 and np.isclose(v, best[j])) else "normal")
        ax.set_title(title, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.outline.set_visible(False)
    fig.suptitle("Macro-F1 by model and experiment (5-fold mean; best real model per column in bold)",
                 x=0.0, ha="left", fontsize=11, fontweight="bold")
    save(fig, "fig4_metric_heatmaps.png")


def significance_heatmap(res):
    models = [m for m in res["E0"]["Model"] if m != PROPOSED]
    tags = list(res)
    diff = np.zeros((len(models), len(tags)))
    stars = [["" for _ in tags] for _ in models]
    for j, t in enumerate(tags):
        r = res[t].set_index("Model")
        pcol = [c for c in r.columns if c.startswith("p-value")][0]
        metric = pcol.split("(")[1].rstrip(")")
        prop = mean_of(r.loc[PROPOSED, metric])
        for i, m in enumerate(models):
            diff[i, j] = mean_of(r.loc[m, metric]) - prop
            s = str(r.loc[m, pcol])
            stars[i][j] = s[s.find("(") + 1:s.find(")")].replace("n.s.", "")
    # Scale colours to the real models; the majority-class row is far below every model
    # and would otherwise wash out the differences that matter (it saturates instead).
    real = [i for i, m in enumerate(models) if m != REFERENCE]
    lim = np.abs(diff[real]).max()
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    im = ax.imshow(diff, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([EXPERIMENTS[t] for t in tags], rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([SHORT[m] for m in models])
    ax.set_xticks(np.arange(-.5, len(tags), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff[i, j]
            ax.text(j, i, f"{v:+.3f}\n{stars[i][j]}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) >= lim * 0.6 else INK, linespacing=0.9)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Model Macro-F1 minus Proposed Macro-F1\n(majority-class row saturates)")
    cb.outline.set_visible(False)
    ax.set_title("Each model vs the Proposed model on the primary Macro-F1\n"
                 "red = better than Proposed, blue = worse;  paired t-test on 5 folds: "
                 "* p<.05  ** p<.01  *** p<.001", loc="left")
    save(fig, "fig5_significance.png")


def final_bars(res):
    panels = [
        ("E6", "Macro-Track Top-1 Acc (%)", "E6  macro-track accuracy (%)", 80),
        ("E6", "Macro-Track Macro-F1", "E6  macro-track Macro-F1", None),
        ("E5", "Granular Top-3 Acc (%)", "E5  major top-3 accuracy (%)", 85),
        ("E5", "Granular Macro-F1", "E5  major Macro-F1 (16 classes)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (tag, metric, title, target) in zip(axes.ravel(), panels):
        r = res[tag]
        r = r[r["Model"] != REFERENCE]
        mu = r[metric].map(mean_of).values
        sd = r[metric].map(lambda c: float(str(c).split("±")[1])).values
        names = [SHORT[m] for m in r["Model"]]
        colors = [HIGHLIGHT if m == PROPOSED else MUTED for m in r["Model"]]
        y = np.arange(len(names))[::-1]
        ax.barh(y, mu, xerr=sd, color=colors, height=0.68, edgecolor="white", linewidth=1.5,
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        ref = mean_of(res[tag].set_index("Model").loc[REFERENCE, metric])
        ax.axvline(ref, color=INK, linestyle="--", linewidth=1.2)
        ax.text(ref, len(names) - 0.35, f" majority class {ref:.3g}", fontsize=7.5,
                color=INK, va="bottom")
        if target is not None:
            ax.axvline(target, color="#d03b3b", linestyle=":", linewidth=1.4)
            ax.text(target, -0.9, f" target {target}%", fontsize=7.5, color="#d03b3b",
                    va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.get_yticklabels()[names.index(SHORT[PROPOSED])].set_fontweight("bold")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        hi = max(mu.max() + sd.max(), ref, target or 0) * 1.08
        ax.set_xlim(0, hi)
        ax.set_ylim(-1, len(names))
    fig.suptitle("Final leakage-free protocols (5-fold mean ± std). Blue = Proposed; "
                 "dashed = majority-class reference; dotted red = thesis target",
                 x=0.0, ha="left", fontsize=11, fontweight="bold")
    save(fig, "fig6_final_metrics.png")


def main():
    res = load_results()
    confusion_figure("E6", res["E6"], "Macro-Track Macro-F1", True,
                     "fig1_confusion_macrotrack.png",
                     "E6 confusion matrices - 4 macro-tracks (out-of-fold, 12,411 students)",
                     annotate_min=0.0, fontsize=10, panel_size=4.6)
    confusion_figure("E5", res["E5"], "Granular Macro-F1", False,
                     "fig2_confusion_majors.png",
                     "E5 confusion matrices - 16 grouped majors (out-of-fold, 12,411 students)",
                     annotate_min=0.5, fontsize=6.5, panel_size=7.2)
    per_class_f1_figure()
    metric_heatmaps(res)
    significance_heatmap(res)
    final_bars(res)


if __name__ == "__main__":
    main()
