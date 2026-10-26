"""
figures3.py
-----------
Every figure in paper_3, as vector PDF sized for the IEEE journal layout.

    python src/figures3.py --which data
    python src/figures3.py --which pl
    python src/figures3.py --which cd
    python src/figures3.py --which online
    python src/figures3.py --which sens
    python src/figures3.py --which conf
    python src/figures3.py --which unc
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(ROOT), "paper_3", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.1, "axes.linewidth": 0.6, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
})

COL1, COL2 = 3.45, 7.16
C = ["#1b4f9c", "#c1272d", "#2e8b57", "#e08214", "#6a3d9a", "#777777",
     "#17becf"]
SC = ["#4575b4", "#91bfdb", "#fc8d59", "#d73027"]
STAGE = ["S0 break-in", "S1 steady", "S2 severe", "S3 failure"]
DATASETS = ["nasa", "phm", "nuaa"]
DS_NAME = {"nasa": "NASA Milling", "phm": "PHM 2010", "nuaa": "NUAA Ideahouse"}
SEEDS = list(range(10))
VERSION = "j2"
MAIN_FRAC = 0.08
SHORT = {"sup_low": "Supervised", "pseudo": "Pseudo-Label",
         "meanteacher": "Mean Teacher", "fixmatch": "FixMatch",
         "simclr": "SimCLR", "tcl_ft": "TCL f/t", "uctcl": "UC-TCL",
         "rf": "RF", "svm": "SVM"}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def load():
    idx = {}
    # NOTE: the raw directory also holds the label-sweep cells at other
    # budgets.  They must not be mixed in here -- the key is (dataset, method,
    # seed), so without this filter a 2% cell would silently overwrite the 8%
    # cell of the same name and corrupt every main-study number.
    pat = f"{VERSION}__*__f{MAIN_FRAC:.3f}__*.json"
    for f in glob.glob(os.path.join(ROOT, "results", "raw", pat)):
        r = json.load(open(f))
        idx[(r["dataset"], r["method"], r["seed"])] = r
    return idx


def _edges(ds):
    import data_nasa, data_phm, data_nuaa
    return np.asarray({"nasa": data_nasa.WEAR_EDGES, "phm": data_phm.WEAR_EDGES,
                       "nuaa": data_nuaa.WEAR_EDGES}[ds], float)


def sweep(ds, m="uctcl", s=0, frac=0.08):
    p = os.path.join(ROOT, "results", "sweeps",
                     f"{VERSION}__{ds}__{m}__f{frac:.3f}__s{s}.npz")
    return np.load(p) if os.path.exists(p) else None


def f_pipeline():
    """Schematic of the UC-TCL pipeline."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(COL2, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, txt, fc="#eef2f8", ec=C[0], fs=6.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=0.8, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.35,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, ls="-", col="0.35"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color=col,
                                     ls=ls, zorder=1,
                                     shrinkA=0, shrinkB=0))

    box(1, 24, 15, 13, "unlabelled\nwindows\n(training tools)", "#f5f5f5", "0.5")
    box(1, 6, 15, 13, "labelled\npasses\n($<$10% of them)", "#fdf0ef", C[1])
    box(20, 15, 15, 22, "temporal\ncontrastive\npre-training\n"
                        "Eq. (1)\u2013(2)", "#eef2f8", C[0])
    box(39, 15, 14, 22, "encoder\n+ 3 heads\nMC dropout", "#eef2f8", C[0])
    box(57, 22, 18, 15, "MC sweep\n$\\rightarrow$ $u^{\\rm epi}$, "
                        "$\\sigma^2_{\\rm tot}$\nEq. (5)\u2013(6)", "#eef2f8", C[0])
    box(57, 3, 18, 16, "acceptance rule\nisotonic gate $\\rightarrow$\n"
                       "per-class $\\tau_k$\nEq. (7)\u2013(8)", "#eaf4ee", C[2],
        bold=True)
    box(79, 22, 20, 15, "online correction\nanchor + shrink,\n"
                        "PAVA, fusion", "#fff6ea", C[3])
    box(79, 3, 20, 16, "wear stage $\\hat c$\nand wear\nvalue $\\hat y$",
        "#f5f5f5", "0.5")

    arrow(16, 30.5, 20, 28)
    arrow(16, 12.5, 39, 22)
    arrow(35, 26, 39, 26)
    arrow(53, 28, 57, 29)
    arrow(66, 22, 66, 19)
    ax.annotate("", xy=(46, 37.6), xytext=(66, 37.6),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C[2],
                                connectionstyle="arc3,rad=0.32"))
    ax.text(56, 40.0, "$T$ self-training rounds", ha="center", fontsize=6.4,
            color=C[2])
    arrow(75, 29.5, 79, 29.5)
    arrow(89, 22, 89, 19)
    ax.text(89, 20.6, "", ha="center")
    ax.text(2, 0.6, "Every stage above reads only training-tool labels; the "
                    "online correction reads the held-out tool's unlabelled "
                    "stream and no label at all.", fontsize=6, color="0.4")
    save(fig, "fig_pipeline.pdf")


def f_budget():
    """The mechanism scales with scarcity; the end task does not."""
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["budget"]
    keys = list(a["pooled"].keys())
    xs = [100 * float(k) for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.15))

    ax = axes[0]
    for m, col, nm in [("sup_low", "0.35", "supervised"),
                       ("pseudo", C[1], "fixed $\\tau_c$"),
                       ("uctcl", C[0], "UC-TCL (ours)")]:
        ax.plot(xs, [a["pooled"][k][m]["inductive"] for k in keys], "o-",
                color=col, ms=4, label=nm, lw=1.7 if m == "uctcl" else 1.0)
    for ax_ in [axes[0], axes[1], axes[2]]:
        ax_.set_xticks(xs)
        ax_.set_xticklabels([f"{x:.0f}%" for x in xs])
        ax_.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
        ax_.set_xlabel("label budget")
    ax.set_ylabel("pooled macro-F1")
    ax.set_title("(a) end task: no separation")
    ax.legend(loc="lower right", frameon=False, fontsize=6.2)

    ax = axes[1]
    for m, col, nm in [("pseudo", C[1], "fixed $\\tau_c$"),
                       ("uctcl", C[0], "adaptive $\\tau_k$")]:
        for i, ds in enumerate(DATASETS):
            ax.plot(xs, [a["pl"][k][ds][m]["coverage"][0] for k in keys], "o-",
                    color=col, ms=3, alpha=0.6, lw=0.9,
                    label=nm if i == 0 else None)
    ax.set_ylim(-3, 60)
    ax.set_ylabel("pseudo-label coverage (%)")
    ax.set_title("(b) mechanism: coverage")
    ax.legend(loc="center right", frameon=False, fontsize=6.2)

    ax = axes[2]
    for m, col, nm in [("pseudo", C[1], "fixed $\\tau_c$"),
                       ("uctcl", C[0], "adaptive $\\tau_k$")]:
        for i, ds in enumerate(DATASETS):
            ax.plot(xs, [a["pl"][k][ds][m]["evenness"][0] for k in keys], "o-",
                    color=col, ms=3, alpha=0.6, lw=0.9,
                    label=nm if i == 0 else None)
    ax.axhline(1.0, color="0.4", ls=":", lw=0.7)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("class evenness")
    ax.set_title("(c) mechanism: balance")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_budget.pdf")




# --------------------------------------------------------------------------- #
def f_data():
    """Wear trajectories and class imbalance on the three benchmarks."""
    import run_experiments as R
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.3),
                             gridspec_kw=dict(height_ratios=[2.4, 1.0]))
    for j, ds in enumerate(DATASETS):
        S = R.prepare_split(ds, 0, 0.08)
        ax = axes[0, j]
        traj, order, y = S["traj"], S["order"], S["y_raw"]
        edges = _edges(ds)
        for t in np.unique(traj):
            k = traj == t
            o = np.argsort(order[k])
            ax.plot(np.arange(k.sum()) / max(k.sum() - 1, 1),
                    y[k][o], lw=0.8, alpha=0.75,
                    color=C[int(t) % len(C)])
        for e in edges:
            ax.axhline(e, color="0.35", ls="--", lw=0.6)
        pad = 0.015 * (y.max() - y.min())
        for i, e in enumerate(edges):
            ax.text(0.985, e + pad, f"S{i+1}", fontsize=6, color="0.3",
                    ha="right", va="bottom")
        ax.text(0.985, edges[0] - pad, "S0", fontsize=6, color="0.3",
                ha="right", va="top")
        ax.set_title(f"{DS_NAME[ds]}  ({len(np.unique(traj))} tools)")
        ax.set_xlabel("normalised tool life")
        if j == 0:
            ax.set_ylabel("flank wear $VB$ (mm)")
        ax.set_xlim(0, 1)

        ax = axes[1, j]
        bc = np.bincount(S["y_cls"], minlength=4)
        ax.bar(range(4), bc, color=SC, width=0.72, edgecolor="none")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"S{i}" for i in range(4)])
        ax.set_ylabel("windows" if j == 0 else "")
        ax.text(0.97, 0.86, f"IR {bc.max()/max(bc.min(),1):.1f}",
                transform=ax.transAxes, ha="right", fontsize=7,
                bbox=dict(fc="white", ec="0.7", lw=0.4, pad=1.5))
        ax.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=0.6, w_pad=0.9)
    save(fig, "fig_data.pdf")


def f_pl(idx):
    """How the acceptance rule behaves over the self-training rounds."""
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.4), sharex=True)
    for j, ds in enumerate(DATASETS):
        for m, col, lab in [("pseudo", C[1], "fixed $\\tau_c=0.95$"),
                            ("uctcl", C[0], "adaptive $\\tau_k$ (ours)")]:
            covs, evs, pres = [], [], []
            for s in SEEDS:
                pl = idx[(ds, m, s)].get("extra", {}).get("pl")
                if not pl:
                    continue
                n_unl = idx[(ds, m, s)]["n_unlabelled"]
                covs.append([100.0 * d["n"] / n_unl for d in pl])
                pres.append([100.0 * d["precision"] for d in pl])
                e = []
                for d in pl:
                    pc = np.asarray(d["per_class"], float)
                    q = pc[pc > 0] / max(pc.sum(), 1)
                    e.append(float(-(q * np.log(q)).sum() / np.log(4))
                             if pc.sum() > 0 else 0.0)
                evs.append(e)
            covs, evs, pres = map(lambda a: np.asarray(a, float),
                                  (covs, evs, pres))
            r = np.arange(1, covs.shape[1] + 1)
            for ax, v in [(axes[0, j], covs), (axes[1, j], evs)]:
                mu, sd = v.mean(0), v.std(0, ddof=1)
                ax.plot(r, mu, "o-", color=col, ms=3, label=lab)
                ax.fill_between(r, mu - sd, mu + sd, color=col, alpha=0.16,
                                lw=0)
        axes[0, j].set_title(DS_NAME[ds])
        axes[0, j].set_ylim(-3, 63)
        axes[1, j].set_ylim(-0.03, 1.03)
        axes[1, j].set_xlabel("self-training round")
        axes[1, j].set_xticks(r)
    axes[0, 0].set_ylabel("coverage (\\%)")
    axes[1, 0].set_ylabel("class evenness")
    axes[1, 0].axhline(1.0, color="0.4", ls=":", lw=0.7)
    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.tight_layout(h_pad=0.5, w_pad=0.7)
    save(fig, "fig_pl.pdf")


def f_cd():
    """Nemenyi critical-difference diagram for online macro-F1."""
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["ranks"]
    keys = [("macro_f1_online", "macro-F1, online protocol"),
            ("rmse_mm_inductive", "wear RMSE, inductive protocol")]
    nrows = [int(np.ceil(len(a[k]["mean_rank"]) / 2)) for k, _ in keys]
    fig, axes = plt.subplots(2, 1, figsize=(COL1, sum(0.55 + 0.17 * r
                                                      for r in nrows)),
                             gridspec_kw=dict(height_ratios=nrows))
    for ax, (key, ttl) in zip(axes, keys):
        d = a[key]
        mr, cd = d["mean_rank"], d["critical_difference"]
        ms = sorted(mr, key=lambda m: mr[m])   # best (lowest rank) first
        n = len(ms)
        nrow = int(np.ceil(n / 2))
        lo, hi = 1, n
        span = hi - lo
        # x reversed: rank 1 sits at the right-hand end, as is conventional
        ax.set_xlim(hi + 0.34 * span, lo - 0.34 * span)
        y_first, dy = 0.80, 0.17
        ax.set_ylim(y_first - dy * (nrow - 1) - 0.10, 1.34)
        ax.axis("off")
        ax.plot([lo, hi], [1, 1], "k-", lw=0.9)
        for t in range(lo, hi + 1):
            ax.plot([t, t], [1, 1.05], "k-", lw=0.7)
            ax.text(t, 1.08, str(t), ha="center", va="bottom", fontsize=6.5)
        ax.text(0.5, 0.975, f"{ttl}    CD = {cd:.2f}", fontsize=7,
                ha="center", va="top", color="0.2", transform=ax.transAxes)
        for i, m in enumerate(ms):
            best_half = i < nrow            # drawn on the right-hand side
            row = i if best_half else n - 1 - i
            yy = y_first - dy * row
            xend = (lo - 0.32 * span) if best_half else (hi + 0.32 * span)
            ax.plot([mr[m], mr[m]], [1, yy], color="0.45", lw=0.55)
            ax.plot([mr[m], xend], [yy, yy], color="0.45", lw=0.55)
            b = m == "uctcl"
            ax.text(xend, yy, f" {SHORT[m]}" if best_half else f"{SHORT[m]} ",
                    fontsize=6.5, ha="left" if best_half else "right",
                    va="center", fontweight="bold" if b else "normal",
                    color=C[0] if b else "black")
        # maximal cliques of methods that are not significantly different
        groups = []
        for i in range(n):
            j = i
            while j + 1 < n and mr[ms[j + 1]] - mr[ms[i]] < cd:
                j += 1
            if j > i:
                groups.append((mr[ms[i]], mr[ms[j]]))
        drawn = []
        for g in groups:
            if any(g[0] >= h[0] and g[1] <= h[1] for h in drawn):
                continue
            drawn.append(g)
        for k, (x0, x1) in enumerate(drawn):
            ax.plot([x0 - 0.04, x1 + 0.04], [0.955 - 0.045 * k] * 2,
                    color=C[1], lw=2.0, solid_capstyle="butt", alpha=0.85)
    fig.tight_layout(h_pad=0.4)
    save(fig, "fig_cd.pdf")


def f_online(idx):
    """When online adaptation helps, and what it does to one tool."""
    import run_experiments as R
    from tta import postprocess
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["online"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35),
                             gridspec_kw=dict(width_ratios=[1.0, 1.45]))

    # (a) the sign of the online gain is a property of the data set, not of
    #     the method: on NUAA almost every method loses, on NASA and PHM
    #     almost every method gains.
    ax = axes[0]
    order_m = ["sup_low", "pseudo", "meanteacher", "fixmatch", "simclr",
               "tcl_ft", "uctcl"]
    wdt = 0.11
    for j, m in enumerate(order_m):
        xs = np.arange(3) + (j - 3) * wdt
        ys = [a[ds]["gain_by_method"][m] for ds in DATASETS]
        ax.bar(xs, ys, width=wdt * 0.9,
               color=C[0] if m == "uctcl" else "0.72",
               edgecolor="none", zorder=2,
               label="UC-TCL" if m == "uctcl" else
                     ("other six methods" if j == 0 else None))
    for i, ds in enumerate(DATASETS):
        g = a[ds]["mean_gain"]
        ax.plot([i - 3.6 * wdt, i + 3.6 * wdt], [g, g], color=C[1], lw=1.3,
                zorder=4, label="mean over methods" if i == 0 else None)
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([DS_NAME[d].split()[0] for d in DATASETS])
    ax.set_ylabel("$\\Delta$ macro-F1 from online protocol")
    ax.set_title("(a) online adaptation amplifies base-model quality")
    ax.legend(loc="lower left", frameon=False, fontsize=6.2)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ds = "phm"
    z = sweep(ds, "uctcl", 0)
    S = R.prepare_split(ds, 0, 0.08)
    te = S["te"]
    scale, lo = S["y_hi"] - S["y_lo"], S["y_lo"]
    o = np.argsort(S["order"][te])
    raw_mu = z["mu"].astype(np.float64).mean(0)
    _, adj, _ = postprocess(z["probs"].astype(np.float64),
                            z["mu"].astype(np.float64),
                            z["lv"].astype(np.float64), S["traj"][te],
                            S["order"][te], S["edges_norm"], S["vb0"],
                            spread=S["vb0_spread"])
    t = np.arange(len(o))
    ax.plot(t, S["y_raw"][te][o], color="0.25", lw=1.4, label="ground truth")
    ax.plot(t, raw_mu[o] * scale + lo, color=C[1], lw=0.9, alpha=0.8,
            label="inductive prediction")
    ax.plot(t, adj[o] * scale + lo, color=C[0], lw=1.2,
            label="after online correction")
    for e in _edges(ds):
        ax.axhline(e, color="0.5", ls="--", lw=0.55)
    ax.set_xlabel("window index along the held-out tool's life")
    ax.set_ylabel("flank wear $VB$ (mm)")
    ax.set_title("(b) anchoring, shrinkage and monotone smoothing")
    ax.legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout(w_pad=1.1)
    save(fig, "fig_online.pdf")


def f_sens():
    """Sensitivity of the online stack."""
    s = json.load(open(os.path.join(ROOT, "results", "sensitivity.json")))
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.95))
    panels = [("alpha_fuse", "fusion weight $\\alpha$", 0.5),
              ("q", "anchor window $q$", 0.25)]
    for ax, (k, xl, dflt) in zip(axes, panels):
        g = s["online"][k]
        xs = sorted(float(v) for v in g)
        for i, ds in enumerate(DATASETS):
            ys = [g[_fmt(x, g)]["per_dataset"][ds]["macro_f1"][0] for x in xs]
            ax.plot(xs, ys, "o-", ms=3, color=C[i], label=DS_NAME[ds].split()[0])
        ys = [g[_fmt(x, g)]["pooled_macro_f1"] for x in xs]
        ax.plot(xs, ys, "s--", ms=3.4, color="0.2", label="pooled")
        ax.axvline(dflt, color="0.55", ls=":", lw=0.8)
        ax.set_xlabel(xl)
    g = s["mc_passes"]
    ax = axes[2]
    xs = sorted(int(v) for v in g)
    for i, ds in enumerate(DATASETS):
        ax.plot(xs, [g[str(x)]["per_dataset"][ds][0] for x in xs], "o-", ms=3,
                color=C[i])
    ax.plot(xs, [g[str(x)]["pooled"] for x in xs], "s--", ms=3.4, color="0.2")
    ax.axvline(10, color="0.55", ls=":", lw=0.8)
    ax.set_xlabel("Monte-Carlo passes $M$")
    axes[0].set_ylabel("macro-F1, online")
    axes[0].legend(loc="lower left", frameon=False, fontsize=6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_sens.pdf")


def _fmt(x, g):
    for k in g:
        if abs(float(k) - x) < 1e-9:
            return k
    raise KeyError(x)


def f_conf(idx):
    """Row-normalised confusion of UC-TCL under the online protocol."""
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["stages"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.15))
    for j, ds in enumerate(DATASETS):
        cm = np.asarray(a[ds]["online"]["confusion"], float)
        n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax = axes[j]
        im = ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
        for r in range(4):
            for c in range(4):
                if cm[r].sum() == 0:
                    continue
                ax.text(c, r, f"{n[r,c]:.2f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if n[r, c] > 0.55 else "0.15")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([f"S{i}" for i in range(4)])
        ax.set_yticklabels([f"S{i}" for i in range(4)])
        ax.set_xlabel("predicted")
        if j == 0:
            ax.set_ylabel("true stage")
        adj = a[ds]["online"]["adjacent_fraction"]
        ax.set_title(f"{DS_NAME[ds]}\n{100*adj:.0f}\\% of errors adjacent")
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.016, pad=0.015)
    save(fig, "fig_conf.pdf")


def f_unc():
    """Does epistemic uncertainty know when the model is wrong?"""
    import run_experiments as R
    from uncertainty import mc_classification_uncertainty, expected_calibration_error
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))

    # (a) risk-coverage
    ax = axes[0]
    for i, ds in enumerate(DATASETS):
        curves = []
        for s in SEEDS:
            z = sweep(ds, "uctcl", s)
            if z is None:
                continue
            S = R.prepare_split(ds, s, 0.08)
            yt = S["y_cls"][S["te"]]
            p = z["probs"].astype(np.float64)
            p_bar, _, _, epi = mc_classification_uncertainty(p)
            correct = (p_bar.argmax(1) == yt).astype(float)
            o = np.argsort(epi)
            err = 1.0 - np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
            grid = np.linspace(0.05, 1.0, 40)
            curves.append(np.interp(grid, np.arange(1, len(o) + 1) / len(o),
                                    err))
        cu = np.asarray(curves)
        ax.plot(grid, cu.mean(0), color=C[i], label=DS_NAME[ds].split()[0])
        ax.fill_between(grid, cu.mean(0) - cu.std(0), cu.mean(0) + cu.std(0),
                        color=C[i], alpha=0.15, lw=0)
        ax.axhline(cu.mean(0)[-1], color=C[i], ls=":", lw=0.7)
    ax.set_xlabel("coverage (windows retained, least uncertain first)")
    ax.set_ylabel("selective error rate")
    ax.set_title("(a) epistemic uncertainty ranks errors")
    ax.legend(loc="upper left", frameon=False)

    # (b) reliability
    ax = axes[1]
    for i, ds in enumerate(DATASETS):
        cf, ac, allp, ally = [], [], [], []
        for s in SEEDS:
            z = sweep(ds, "uctcl", s)
            if z is None:
                continue
            S = R.prepare_split(ds, s, 0.08)
            yt = S["y_cls"][S["te"]]
            pm = z["probs"].astype(np.float64).mean(0)
            cf.append(pm.max(1)); ac.append((pm.argmax(1) == yt).astype(float))
            allp.append(pm); ally.append(yt)
        cf, ac = np.concatenate(cf), np.concatenate(ac)
        allp, ally = np.concatenate(allp), np.concatenate(ally)
        bins = np.linspace(0.25, 1.0, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = (cf >= lo) & (cf < hi)
            if k.sum() > 30:
                xs.append(cf[k].mean()); ys.append(ac[k].mean())
        ece, _ = expected_calibration_error(allp, ally, n_bins=10)
        ax.plot(xs, ys, "o-", ms=3, color=C[i],
                label=f"{DS_NAME[ds].split()[0]}  ECE {ece:.3f}")
    ax.plot([0.25, 1], [0.25, 1], color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("(b) reliability of the MC-dropout posterior")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(w_pad=1.1)
    save(fig, "fig_unc.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    idx = load()
    todo = ["pipeline", "data", "pl", "cd", "online", "sens", "conf", "unc", "budget"]
    if a.which != "all":
        todo = a.which.split(",")
    for t in todo:
        {"pipeline": f_pipeline, "data": f_data, "pl": lambda: f_pl(idx), "cd": f_cd,
         "online": lambda: f_online(idx), "sens": f_sens,
         "conf": lambda: f_conf(idx), "unc": f_unc,
         "budget": f_budget}[t]()
