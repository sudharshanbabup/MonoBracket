"""
aggregate.py
------------
Collect results/raw/*.json into summary tables, significance tests and the
LaTeX fragments used by the paper.

    python src/aggregate.py --show main
    python src/aggregate.py --latex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")
VERSION = "v8"

PRETTY = {
    "rf": "Random Forest (features)",
    "svm": "SVM-RBF (features)",
    "sup_low": "Supervised (labelled only)",
    "sup_low_fuse": "Supervised + fusion only",
    "pseudo": "Pseudo-Label (fixed $\\tau$)",
    "meanteacher": "Mean Teacher",
    "fixmatch": "FixMatch",
    "simclr": "SimCLR + linear probe",
    "tcl_ft": "TCL + fine-tune (no PL)",
    "uctcl": "\\textbf{UC-TCL (ours)}",
    "sup_full": "Supervised (100\\% labels)",
    "uctcl_noTCL": "w/o contrastive pre-training",
    "uctcl_simclr": "w/o temporal kernel (SimCLR)",
    "uctcl_noMC": "w/o MC-dropout (soft-max conf.)",
    "uctcl_globalTau": "w/o per-class threshold",
    "uctcl_noMono": "w/o monotonicity filter",
    "uctcl_noCB": "w/o class-balanced focal",
    "uctcl_fusion": "+ threshold-consistent fusion",
    "uctcl_agree": "+ hard head-agreement gate",
    "uctcl_relabel": "+ isotonic label correction",
}

MAIN_ORDER = ["rf", "svm", "sup_low", "pseudo", "meanteacher",
              "fixmatch", "simclr", "tcl_ft", "uctcl", "sup_full"]
# methods that produce a model and can therefore be run under the online
# (test-time adaptation) protocol as well
TTA_ORDER = ["sup_low", "pseudo", "meanteacher", "fixmatch", "simclr",
             "tcl_ft", "uctcl", "sup_full"]
PROTOCOLS = {"ind": None, "tta": "tta"}
ABL_ORDER = ["uctcl_noTCL", "uctcl_simclr", "uctcl_noMC", "uctcl_globalTau",
             "uctcl_noMono", "uctcl_noCB", "uctcl_fusion", "uctcl_agree",
             "uctcl_relabel", "uctcl"]
DS_PRETTY = {"nasa": "NASA Milling", "phm": "PHM 2010"}


def load(version=VERSION):
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, f"{version}__*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def group(rows, frac=None):
    g = defaultdict(list)
    for r in rows:
        if frac is not None and abs(r["label_frac"] - frac) > 1e-9:
            continue
        g[(r["dataset"], r["method"], round(r["label_frac"], 3))].append(r)
    return g


def get(rec, key, protocol=None):
    """Read a metric under either protocol; `protocol=None` is inductive."""
    return rec[protocol][key] if protocol else rec[key]


def by_run(rows, frac=0.08, version_methods=None):
    """Index results as {(dataset, seed): {method: record}}."""
    out = defaultdict(dict)
    for r in rows:
        if abs(r["label_frac"] - frac) > 1e-9:
            continue
        out[(r["dataset"], r["seed"])][r["method"]] = r
    return out


def protocol_table(rows, ds, order, frac=0.08, metric="macro_f1"):
    """Mean/sd of one metric under both protocols, one row per method."""
    g = group(rows, frac)
    out = []
    for m in order:
        rs = sorted(g.get((ds, m, round(frac, 3)), []), key=lambda r: r["seed"])
        if not rs:
            continue
        rec = dict(method=m, n=len(rs))
        for pname, pkey in PROTOCOLS.items():
            if pkey and pkey not in rs[0]:
                rec[pname] = (float("nan"), float("nan"))
                continue
            rec[pname] = stat([get(r, metric, pkey) for r in rs])
            rec[pname + "_seeds"] = [get(r, metric, pkey) for r in rs]
        out.append(rec)
    return out


def pooled_vector(rows, method, metric, protocol=None, frac=0.08,
                  datasets=("nasa", "phm"), seeds=range(10)):
    """
    One value per (dataset, seed) run, in a fixed order, so that comparisons
    across methods are correctly paired.  Pooling the two benchmarks doubles
    the sample size of the paired test, which matters at these effect sizes.
    """
    idx = by_run(rows, frac)
    v = []
    for ds in datasets:
        for s in seeds:
            rec = idx.get((ds, s), {}).get(method)
            v.append(np.nan if rec is None else get(rec, metric, protocol))
    return np.asarray(v, dtype=float)


def stat(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def paired_test(a, b):
    """
    Two-sided Wilcoxon signed-rank test on the per-seed differences, with a
    paired t-test as a cross-check.  Returns (wilcoxon p, t p, mean diff).
    """
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.allclose(a, b):
        return float("nan"), float("nan"), float(np.mean(a - b) if len(a) else np.nan)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except Exception:
        w = float("nan")
    t = stats.ttest_rel(a, b).pvalue
    return float(w), float(t), float(np.mean(a - b))


def table(rows, ds, order, frac=0.08, metrics=("macro_f1", "balanced_acc",
                                               "rmse_mm", "ece")):
    g = group(rows, frac)
    out = []
    for m in order:
        rs = g.get((ds, m, round(frac, 3)), [])
        if not rs:
            continue
        rec = dict(method=m, n_seeds=len(rs))
        for k in metrics:
            mu, sd = stat([r[k] for r in rs])
            rec[k] = (mu, sd)
        rec["seeds"] = {k: [r[k] for r in sorted(rs, key=lambda z: z["seed"])]
                        for k in metrics}
        rec["label_pct"] = stat([r["actual_label_pct"] for r in rs])[0]
        out.append(rec)
    return out


def fmt(mu, sd, nd=3, bold=False):
    s = f"{mu:.{nd}f}\\,$\\pm$\\,{sd:.{nd}f}"
    return f"\\textbf{{{s}}}" if bold else s


def latex_main(rows, frac=0.08):
    lines = []
    for ds in ["nasa", "phm"]:
        t = table(rows, ds, MAIN_ORDER, frac)
        if not t:
            continue
        ours = next((r for r in t if r["method"] == "uctcl"), None)
        # best among semi-supervised competitors (exclude the 100 % upper bound)
        comp = [r for r in t if r["method"] not in ("sup_full",)]
        best_f1 = max(r["macro_f1"][0] for r in comp)
        best_rmse = min(r["rmse_mm"][0] for r in comp)
        best_ece = min(r["ece"][0] for r in comp)
        lines.append(f"% ---- {ds} ----")
        for r in t:
            up = r["method"] == "sup_full"
            lines.append(
                f"{PRETTY[r['method']]} & "
                f"{fmt(*r['macro_f1'], bold=(not up and r['macro_f1'][0] >= best_f1 - 1e-9))} & "
                f"{fmt(*r['balanced_acc'], bold=(not up and r['balanced_acc'][0] >= max(x['balanced_acc'][0] for x in comp) - 1e-9))} & "
                f"{fmt(*r['rmse_mm'], bold=(not up and r['rmse_mm'][0] <= best_rmse + 1e-9))} & "
                f"{fmt(*r['ece'], bold=(not up and r['ece'][0] <= best_ece + 1e-9))} \\\\")
    return "\n".join(lines)


def significance(rows, frac=0.08):
    g = group(rows, frac)
    out = {}
    for ds in ["nasa", "phm"]:
        ours = g.get((ds, "uctcl", round(frac, 3)), [])
        if not ours:
            continue
        ours = sorted(ours, key=lambda r: r["seed"])
        out[ds] = {}
        for m in MAIN_ORDER:
            if m == "uctcl":
                continue
            rs = sorted(g.get((ds, m, round(frac, 3)), []), key=lambda r: r["seed"])
            if len(rs) != len(ours):
                continue
            for k, better in [("macro_f1", "max"), ("rmse_mm", "min")]:
                w, t, d = paired_test([r[k] for r in ours], [r[k] for r in rs])
                out[ds][f"{m}:{k}"] = dict(wilcoxon_p=w, ttest_p=t, mean_diff=d)
    return out


def matched_table(rows, ds, order, frac=0.08, seeds=(0, 1, 2)):
    """Ablation view restricted to the seeds every variant was run on."""
    g = group(rows, frac)
    out = []
    for m in order:
        rs = [r for r in g.get((ds, m, round(frac, 3)), []) if r["seed"] in seeds]
        if len(rs) != len(seeds):
            continue
        rs = sorted(rs, key=lambda z: z["seed"])
        rec = dict(method=m, n_seeds=len(rs))
        for k in ("macro_f1", "balanced_acc", "rmse_mm", "ece"):
            rec[k] = stat([r[k] for r in rs])
            rec[k + "_seeds"] = [r[k] for r in rs]
        out.append(rec)
    return out


def rank_analysis(rows, frac=0.08, metric="macro_f1", higher=True):
    """
    Average rank of each method across every (dataset, seed) run.  Ranking is
    immune to the very different metric scales of the two benchmarks and to the
    large split-to-split spread, so it is the fairest single summary of which
    method is most consistently good.
    """
    g = group(rows, frac)
    runs = defaultdict(dict)
    for (ds, m, f), rs in g.items():
        if m not in MAIN_ORDER:
            continue
        for r in rs:
            runs[(ds, r["seed"])][m] = r[metric]
    ranks = defaultdict(list)
    n_runs = 0
    for key_, d in sorted(runs.items()):
        present = [m for m in MAIN_ORDER if m in d]
        if len(present) < len(MAIN_ORDER):
            continue
        n_runs += 1
        order_ = sorted(present, key=lambda m: -d[m] if higher else d[m])
        for i, m in enumerate(order_):
            ranks[m].append(i + 1)
    return {m: (float(np.mean(v)), float(np.std(v, ddof=1)), len(v))
            for m, v in ranks.items()}, n_runs


def pseudo_label_quality(rows, frac=0.08):
    """
    Coverage / precision / class-balance of the pseudo-labels actually accepted
    in the final round, for the two self-training methods that expose it.
    """
    out = {}
    for ds in ["nasa", "phm"]:
        out[ds] = {}
        for m in ["pseudo", "uctcl"]:
            rs = [r for r in rows
                  if r["dataset"] == ds and r["method"] == m
                  and abs(r["label_frac"] - frac) < 1e-9
                  and r.get("extra", {}).get("pl")]
            if not rs:
                continue
            cov, prec, ent, nsel = [], [], [], []
            for r in rs:
                d = r["extra"]["pl"][-1]
                n_unl = d.get("n_unlabelled") or r["n_unlabelled"]
                cov.append(100.0 * d["n"] / max(1, n_unl))
                nsel.append(d["n"])
                if np.isfinite(d["precision"]):
                    prec.append(100.0 * d["precision"])
                pc = np.asarray(d["per_class"], dtype=float)
                if pc.sum() > 0:
                    p = pc / pc.sum()
                    nz = p[p > 0]
                    # normalised Shannon evenness of the pseudo-label histogram
                    ent.append(float(-(nz * np.log(nz)).sum() / np.log(len(pc))))
                else:
                    ent.append(0.0)
            out[ds][m] = dict(coverage=stat(cov), precision=stat(prec),
                              evenness=stat(ent), n_selected=stat(nsel),
                              n_runs=len(rs))
    return out


def label_sweep(rows):
    out = defaultdict(dict)
    for r in rows:
        if r["method"] not in ("sup_low", "fixmatch", "uctcl"):
            continue
        out[(r["dataset"], r["method"])].setdefault(round(r["label_frac"], 3), [])
        out[(r["dataset"], r["method"])][round(r["label_frac"], 3)].append(r)
    res = {}
    for k, d in out.items():
        res[f"{k[0]}|{k[1]}"] = {
            str(f): dict(macro_f1=stat([r["macro_f1"] for r in v]),
                         rmse_mm=stat([r["rmse_mm"] for r in v]),
                         balanced_acc=stat([r["balanced_acc"] for r in v]),
                         label_pct=stat([r["actual_label_pct"] for r in v]),
                         n=len(v))
            for f, v in sorted(d.items())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.08)
    ap.add_argument("--version", default=VERSION)
    a = ap.parse_args()
    rows = load(a.version)
    print(f"{len(rows)} result files\n")

    for ds in ["nasa", "phm"]:
        for name, order in [("MAIN", MAIN_ORDER), ("ABLATION", ABL_ORDER)]:
            t = table(rows, ds, order, a.frac)
            if not t:
                continue
            print(f"=== {DS_PRETTY[ds]} :: {name} (label frac {a.frac:.0%}) ===")
            print(f"{'method':34s} {'n':>2s} {'macroF1':>15s} {'balAcc':>15s} "
                  f"{'RMSE(mm)':>15s} {'ECE':>15s}")
            for r in t:
                print(f"{r['method']:34s} {r['n_seeds']:2d} "
                      f"{r['macro_f1'][0]:.3f}+-{r['macro_f1'][1]:.3f}  "
                      f"{r['balanced_acc'][0]:.3f}+-{r['balanced_acc'][1]:.3f}  "
                      f"{r['rmse_mm'][0]:.4f}+-{r['rmse_mm'][1]:.4f} "
                      f"{r['ece'][0]:.3f}+-{r['ece'][1]:.3f}")
            print()

    sig = significance(rows, a.frac)
    for ds, d in sig.items():
        print(f"--- paired tests vs UC-TCL on {DS_PRETTY[ds]} ---")
        for k, v in d.items():
            print(f"  {k:28s} diff={v['mean_diff']:+.4f} "
                  f"wilcoxon p={v['wilcoxon_p']:.4f} t p={v['ttest_p']:.4f}")
        print()

    print("=== average rank over every (dataset, seed) run ===")
    for metric, higher in [("macro_f1", True), ("rmse_mm", False),
                           ("balanced_acc", True), ("ece", False)]:
        rk, n = rank_analysis(rows, a.frac, metric, higher)
        best = sorted(rk.items(), key=lambda kv: kv[1][0])
        print(f"  {metric} (n={n} runs): " +
              ", ".join(f"{m}={v[0]:.2f}" for m, v in best))
    print()

    plq = pseudo_label_quality(rows, a.frac)
    print("=== pseudo-label quality, final round ===")
    for ds, d in plq.items():
        for m, v in d.items():
            print(f"  {ds:5s} {m:8s} coverage={v['coverage'][0]:5.1f}%"
                  f"+-{v['coverage'][1]:4.1f}  precision={v['precision'][0]:5.1f}%"
                  f"  evenness={v['evenness'][0]:.3f}  n={v['n_selected'][0]:.0f}")
    print()

    os.makedirs(OUT, exist_ok=True)
    rk_all = {f"{mt}": rank_analysis(rows, a.frac, mt, hi)[0]
              for mt, hi in [("macro_f1", True), ("rmse_mm", False),
                             ("balanced_acc", True), ("ece", False)]}
    with open(os.path.join(OUT, "extras.json"), "w") as fh:
        json.dump(dict(ranks=rk_all, pseudo_label_quality=plq,
                       label_sweep=label_sweep(rows),
                       ablation_matched={ds: matched_table(rows, ds, ABL_ORDER,
                                                           a.frac)
                                         for ds in ["nasa", "phm"]}), fh, indent=1)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(dict(
            tables={f"{ds}_{n}": table(rows, ds, o, a.frac)
                    for ds in ["nasa", "phm"]
                    for n, o in [("main", MAIN_ORDER), ("ablation", ABL_ORDER)]},
            significance=sig, frac=a.frac, n_files=len(rows)), fh, indent=1)
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
