"""
figures.py
----------
Generate every figure in the paper as a vector PDF sized for an IEEE
double-column layout.

    python src/figures.py --which data
    python src/figures.py --which pl
    python src/figures.py --which sweep
    python src/figures.py --which unc
    python src/figures.py --which conf
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
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
PAPERFIG = os.path.join(os.path.dirname(ROOT), "paper_1", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PAPERFIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.1,
    "axes.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

COL1, COL2 = 3.4, 7.1           # IEEE single / double column width in inches
C = ["#1b4f9c", "#c1272d", "#2e8b57", "#e08214", "#6a3d9a", "#777777"]
STAGE = ["S0 Initial", "S1 Steady", "S2 Severe", "S3 Failure"]


def save(fig, name):
    for d in (FIG, PAPERFIG):
        fig.savefig(os.path.join(d, name + ".pdf"))
    plt.close(fig)
    print("wrote", name + ".pdf")


def _load(version="v8"):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "raw",
                                           f"{version}__*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


# --------------------------------------------------------------------------- #
# Fig. 2 : data, wear trajectories, stage definition, class imbalance
# --------------------------------------------------------------------------- #
def fig_data():
    import data_nasa
    import pandas as pd

    runs = data_nasa.load_runs(os.path.join(ROOT, "data", "mill.mat"))
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.62))

    # (a) raw multi-sensor window
    r = [x for x in runs if x["case"] == 1 and x["run"] == 8][0]
    seg = r["x"][:, 3000:4024]
    names = ["AC current", "DC current", "Table vib.", "Spindle vib.",
             "Table AE", "Spindle AE"]
    t = np.arange(seg.shape[1]) / data_nasa.FS
    off = 0.0
    for i in range(6):
        s = (seg[i] - seg[i].mean())
        s = s / (np.abs(s).max() + 1e-9)
        ax[0].plot(t, s + off, lw=0.35, color=C[i % len(C)])
        ax[0].text(t[-1] * 1.01, off, names[i], fontsize=5.0, va="center")
        off -= 2.3
    ax[0].set_xlabel("Time (s)")
    ax[0].set_yticks([])
    ax[0].set_xlim(0, t[-1] * 1.45)
    ax[0].set_title("(a) Multi-sensor window, $VB\\!=\\!0.40$ mm", fontsize=7)
    ax[0].grid(False)

    # (b) wear trajectories with the physical stage bands
    e = data_nasa.WEAR_EDGES
    bands = [(0, e[0]), (e[0], e[1]), (e[1], e[2]), (e[2], 1.6)]
    for (lo, hi), col in zip(bands, ["#e8f1fb", "#dcecdc", "#fdebd0", "#fadbd8"]):
        ax[1].axhspan(lo, hi, color=col, zorder=0)
    for k, (lo, hi) in enumerate(bands):
        ax[1].text(0.985, (lo + min(hi, 1.6)) / 2, STAGE[k], fontsize=5.0,
                   color="#555555", va="center", ha="right",
                   transform=ax[1].get_yaxis_transform(which="grid"))
    for c in sorted({x["case"] for x in runs}):
        rs = sorted([x for x in runs if x["case"] == c and np.isfinite(x["VB"])],
                    key=lambda z: z["time"])
        if len(rs) < 3:
            continue
        ax[1].plot([x["time"] for x in rs], [x["VB"] for x in rs],
                   lw=0.7, alpha=0.75, marker="o", ms=1.2)
    ax[1].set_xlabel("Cumulative machining time (s)")
    ax[1].set_ylabel("Flank wear $VB$ (mm)")
    ax[1].set_ylim(0, 1.6)
    ax[1].set_title("(b) 16 tool-life trajectories", fontsize=7)

    # (c) class imbalance on both benchmarks
    _, _, cn, _, _, _ = data_nasa.build_windows(runs, win=512, stride=512)
    import data_phm
    _, _, cp, _, _, _ = data_phm.build_windows(
        os.path.join(ROOT, "data", "phm2010_bundle.csv"), win=64, stride=24)
    bn = np.bincount(cn, minlength=4) / len(cn) * 100
    bp = np.bincount(cp, minlength=4) / len(cp) * 100
    x = np.arange(4)
    ax[2].bar(x - 0.19, bn, 0.36, label=f"NASA (IR {bn.max()/bn.min():.1f})",
              color=C[0])
    ax[2].bar(x + 0.19, bp, 0.36, label=f"PHM 2010 (IR {bp.max()/bp.min():.1f})",
              color=C[1])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["S0", "S1", "S2", "S3"])
    ax[2].set_ylabel("Share of windows (%)")
    ax[2].set_xlabel("Physical wear stage")
    ax[2].legend(frameon=False)
    ax[2].set_title("(c) Stage imbalance", fontsize=7)
    fig.tight_layout(w_pad=1.1)
    save(fig, "fig_data")


# --------------------------------------------------------------------------- #
# Fig. 3 : pseudo-label coverage / precision / balance
# --------------------------------------------------------------------------- #
def fig_pl():
    rows = _load()
    fig, ax = plt.subplots(1, 4, figsize=(COL2, 1.55))
    dss = ["nasa", "phm"]
    lbl = {"nasa": "NASA Milling", "phm": "PHM 2010"}

    def collect(ds, m, key):
        out = []
        for r in rows:
            if (r["dataset"] == ds and r["method"] == m
                    and abs(r["label_frac"] - 0.08) < 1e-9
                    and r.get("extra", {}).get("pl")):
                d = r["extra"]["pl"][-1]
                n_unl = d.get("n_unlabelled") or r["n_unlabelled"]
                if key == "cov":
                    out.append(100.0 * d["n"] / max(1, n_unl))
                elif key == "prec":
                    if np.isfinite(d["precision"]):
                        out.append(100.0 * d["precision"])
                else:
                    pc = np.asarray(d["per_class"], float)
                    if pc.sum() > 0:
                        p = pc[pc > 0] / pc.sum()
                        out.append(float(-(p * np.log(p)).sum() / np.log(4)))
                    else:
                        out.append(0.0)
        return np.asarray(out)

    titles = ["(a) Coverage of the unlabelled pool",
              "(b) Pseudo-label precision",
              "(c) Class balance of accepted labels"]
    ylabs = ["Accepted (%)", "Correct (%)", "Normalised evenness"]
    for j, key in enumerate(["cov", "prec", "even"]):
        x = np.arange(2)
        for i, (m, nm, col) in enumerate([("pseudo", "Fixed threshold $\\tau=0.95$", C[5]),
                                          ("uctcl", "UC-TCL adaptive (ours)", C[0])]):
            mu = [collect(d, m, key).mean() for d in dss]
            sd = [collect(d, m, key).std(ddof=1) for d in dss]
            ax[j].bar(x + (i - 0.5) * 0.36, mu, 0.34, yerr=sd, capsize=2,
                      label=nm, color=col, error_kw=dict(lw=0.6))
        ax[j].set_xticks(x)
        ax[j].set_xticklabels([lbl[d] for d in dss])
        ax[j].set_ylabel(ylabs[j])
        ax[j].set_title(titles[j], fontsize=7)
        if key == "even":
            ax[j].axhline(1.0, ls=":", lw=0.6, color="k")
            ax[j].text(1.35, 1.01, "perfectly balanced", fontsize=5, ha="right")
            ax[j].set_ylim(0, 1.15)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.42)
    ax[0].legend(frameon=False, loc="upper center", fontsize=5.0,
                 handlelength=1.2, borderaxespad=0.1)

    # (d) label efficiency, expressed relative to supervised training at the
    # same budget so the two benchmarks share one axis
    sw = _load("v7")
    a = ax[3]
    for ds, mk, ls in [("nasa", "o", "-"), ("phm", "s", "--")]:
        ref = np.mean([r["rmse_mm"] for r in sw
                       if r["dataset"] == ds and r["method"] == "sup_low"
                       and abs(r["label_frac"] - 0.08) < 1e-9])
        for m, col in [("sup_low", C[5]), ("uctcl", C[0])]:
            xs, ys = [], []
            for f in [0.02, 0.05, 0.08, 0.10, 0.20]:
                v = [r["rmse_mm"] for r in sw if r["dataset"] == ds
                     and r["method"] == m and abs(r["label_frac"] - f) < 1e-9]
                p_ = [r["actual_label_pct"] for r in sw if r["dataset"] == ds
                      and r["method"] == m and abs(r["label_frac"] - f) < 1e-9]
                if len(v) < 2:
                    continue
                xs.append(np.mean(p_)); ys.append(np.mean(v) / ref)
            if xs:
                a.plot(xs, ys, ls, marker=mk, ms=2.4, color=col, lw=1.0,
                       label=f"{'NASA' if ds=='nasa' else 'PHM'}, "
                             f"{'sup.' if m=='sup_low' else 'ours'}")
    a.axhline(1.0, ls=":", lw=0.6, color="k")
    a.axvspan(0, 10, color="#eef4fb", zorder=0)
    a.set_xlabel("Labelled passes (%)")
    a.set_ylabel("Relative wear RMSE")
    a.set_title("(d) Label efficiency", fontsize=7)
    a.legend(frameon=False, fontsize=4.4, ncol=2, handlelength=1.2)

    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_pseudolabels")


# --------------------------------------------------------------------------- #
# Fig. 4 : label-efficiency curves
# --------------------------------------------------------------------------- #
def fig_sweep():
    rows = _load("v7")
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.45))
    meth = [("sup_low", "Supervised (labels only)", C[5], "o"),
            ("fixmatch", "FixMatch", C[3], "s"),
            ("uctcl", "UC-TCL (ours)", C[0], "D")]
    for a, ds, name in [(ax[0], "nasa", "NASA Milling"),
                        (ax[1], "phm", "PHM 2010")]:
        for m, nm, col, mk in meth:
            fr, mu, sd = [], [], []
            for f in [0.02, 0.05, 0.08, 0.10, 0.20]:
                v = [r["rmse_mm"] for r in rows
                     if r["dataset"] == ds and r["method"] == m
                     and abs(r["label_frac"] - f) < 1e-9]
                p = [r["actual_label_pct"] for r in rows
                     if r["dataset"] == ds and r["method"] == m
                     and abs(r["label_frac"] - f) < 1e-9]
                if len(v) < 2:
                    continue
                fr.append(np.mean(p)); mu.append(np.mean(v))
                sd.append(np.std(v, ddof=1) / np.sqrt(len(v)))
            if not fr:
                continue
            a.errorbar(fr, mu, yerr=sd, marker=mk, ms=3, capsize=2, color=col,
                       label=nm, lw=1.1, elinewidth=0.6)
        sf = [r["rmse_mm"] for r in rows
              if r["dataset"] == ds and r["method"] == "sup_full"]
        if sf:
            a.axhline(np.mean(sf), ls="--", lw=0.8, color="k")
            a.text(a.get_xlim()[1], np.mean(sf), " 100% labels", fontsize=5.5,
                   va="bottom", ha="right")
        a.set_xlabel("Labelled passes (%)", fontsize=6.5)
        a.set_ylabel("Wear RMSE (mm)", fontsize=6.5)
        a.set_title(name, fontsize=6.8)
        a.tick_params(labelsize=5.6)
        a.axvspan(0, 10, color="#eef4fb", zorder=0)
    ax[0].legend(frameon=False, loc="upper right", fontsize=4.6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_labelsweep")


# --------------------------------------------------------------------------- #
# Fig. 5 : uncertainty behaviour
# --------------------------------------------------------------------------- #
def fig_unc():
    import torch
    import run_experiments as R
    import methods as M
    from model import mc_predict
    from uncertainty import (mc_classification_uncertainty, normalised_epistemic,
                             expected_calibration_error, adaptive_thresholds,
                             percentile_schedule)
    torch.set_num_threads(4)

    S = R.prepare_split("phm", 0, 0.08)
    cfg = R.build_cfg("phm", method="uctcl")
    m = M._make_model(7, 4, cfg, 0)
    from pretrain import load_pretrained
    load_pretrained(m, "phm", 0, "tcl", cfg["e_con"])
    M.supervised_fit(m, S["X"], S["y_cls"], S["y_reg"], S["lab"], cfg, 0,
                     cfg["e_warm"], K=4)

    probs, mu, lv = mc_predict(m, S["X"][S["unl"]], n_mc=30)
    p_bar, u_tot, u_ale, u_epi = mc_classification_uncertainty(probs)
    u = normalised_epistemic(u_epi, 4)
    pred = p_bar.argmax(1)
    corr = pred == S["y_cls"][S["unl"]]

    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.45))

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(~corr, u)

    # (b) how much of the rarest stage survives each thresholding rule
    counts = np.bincount(pred, minlength=4)
    rare = int(np.argmin(np.where(counts > 0, counts, 10 ** 9)))
    qs = np.linspace(0.02, 1.0, 40)
    gl, cl, pg, pcl = [], [], [], []
    for q in qs:
        g = np.quantile(u, q)
        mg = u <= g
        tau = adaptive_thresholds(u, pred, 4, q)
        mc = u <= tau[pred]
        gl.append(100 * (mg & (pred == rare)).sum() / max(1, (pred == rare).sum()))
        cl.append(100 * (mc & (pred == rare)).sum() / max(1, (pred == rare).sum()))
        pg.append(100 * corr[mg].mean() if mg.sum() else np.nan)
        pcl.append(100 * corr[mc].mean() if mc.sum() else np.nan)
    # class evenness of the accepted set under each rule
    ev_g, ev_c = [], []
    for q in qs:
        for rule, store in ((u <= np.quantile(u, q), ev_g),
                            (u <= adaptive_thresholds(u, pred, 4, q)[pred], ev_c)):
            h = np.bincount(pred[rule], minlength=4).astype(float)
            if h.sum() > 0:
                p = h[h > 0] / h.sum()
                store.append(float(-(p * np.log(p)).sum() / np.log(4)))
            else:
                store.append(0.0)
    ax[0].plot(qs * 100, pg, color=C[5], label="global $\\tau$: precision")
    ax[0].plot(qs * 100, pcl, color=C[0], label="per-class $\\tau$: precision")
    ax[0].set_xlabel("Acceptance quantile $\\rho$ (%)", fontsize=6.5)
    ax[0].set_ylabel("Precision (%)", fontsize=6.5)
    a2 = ax[0].twinx()
    a2.plot(qs * 100, ev_g, color=C[5], ls=":", lw=0.9, label="global: evenness")
    a2.plot(qs * 100, ev_c, color=C[0], ls=":", lw=0.9,
            label="per-class: evenness")
    a2.set_ylabel("Class evenness", fontsize=6.5)
    a2.set_ylim(0, 1.05)
    a2.grid(False)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=4.2)
    ax[0].tick_params(labelsize=5.6); a2.tick_params(labelsize=5.6)
    ax[0].set_title("(a) Precision / balance trade-off", fontsize=6.6)
    cov = cl

    # (c) reliability diagram
    ece, binsr = expected_calibration_error(p_bar, S["y_cls"][S["unl"]], 12)
    cen = [b[0] for b in binsr]
    acc = [b[1] for b in binsr]
    ax[1].plot([0, 1], [0, 1], ls="--", lw=0.7, color="k")
    ax[1].plot(cen, acc, marker="o", ms=2.5, color=C[0])
    ax[1].set_xlabel("Predicted confidence", fontsize=6.5)
    ax[1].set_ylabel("Empirical accuracy", fontsize=6.5)
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    ax[1].tick_params(labelsize=5.6)
    ax[1].set_title(f"(b) Reliability, ECE {ece:.3f}", fontsize=6.6)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_uncertainty")

    with open(os.path.join(ROOT, "results", "uncertainty_diag.json"), "w") as fh:
        json.dump(dict(auroc_error_detection=float(auc), ece=float(ece),
                       mean_u=float(u.mean()), rarest_stage=int(rare),
                       precision_global={f"{q:.2f}": float(p)
                                         for q, p in zip(qs, pg)},
                       precision_perclass={f"{q:.2f}": float(p)
                                           for q, p in zip(qs, pcl)},
                       rare_accepted_global={f"{q:.2f}": float(v)
                                             for q, v in zip(qs, gl)},
                       rare_accepted_perclass={f"{q:.2f}": float(v)
                                               for q, v in zip(qs, cl)}),
                  fh, indent=1)


# --------------------------------------------------------------------------- #
# Fig. 6 : confusion matrices
# --------------------------------------------------------------------------- #
def fig_conf():
    rows = _load()
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.28))
    panels = [("phm", None, "PHM 2010, inductive"),
              ("phm", "tta", "PHM 2010, online")]
    for a, (ds, prot, title) in zip(ax, panels):
        cms = [np.asarray(r[prot]["confusion"] if prot else r["confusion"], float)
               for r in rows
               if r["dataset"] == ds and r["method"] == "uctcl"
               and abs(r["label_frac"] - 0.08) < 1e-9]
        cm = np.sum(cms, axis=0)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(1)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                       fontsize=5.0,
                       color="white" if cmn[i, j] > 0.55 else "#222222")
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(["S0", "S1", "S2", "S3"])
        a.set_yticklabels(["S0", "S1", "S2", "S3"])
        a.set_xlabel("Predicted", fontsize=6.5)
        a.tick_params(labelsize=5.6)
        a.set_title(title, fontsize=6.4)
        a.grid(False)
    ax[0].set_ylabel("True stage", fontsize=6.5)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_confusion")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    fns = dict(data=fig_data, pl=fig_pl, sweep=fig_sweep, unc=fig_unc,
               conf=fig_conf)
    for k, f in fns.items():
        if a.which in ("all", k):
            f()
