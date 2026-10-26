"""
analyses.py
-----------
The journal-grade analyses that go beyond the main comparison table:

  ranks        scale-free pooling across three benchmarks on different metric
               scales, with Friedman + Nemenyi post-hoc
  power        retrospective statistical power, and the number of runs that
               would be needed to resolve the effects we cannot resolve
  pl           pseudo-label coverage / precision / class balance, all datasets
  online       when does online adaptation help?  Relates the measured gain to
               the break-in dispersion of the training tools, which is the
               quantity the shrinkage rule keys on
  deploy       edge-deployment cost: parameters, inference latency, memory,
               and the marginal cost of the online corrections
  stages       per-stage error structure, including adjacent-stage confusion

    python src/analyses.py --which all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
VERSION = "j2"
MAIN_FRAC = 0.08
DATASETS = ["nasa", "phm", "nuaa"]
SEEDS = list(range(10))
LABEL_LIMITED = ["rf", "svm", "sup_low", "pseudo", "meanteacher", "fixmatch",
                 "simclr", "tcl_ft", "uctcl"]
DEEP = ["sup_low", "pseudo", "meanteacher", "fixmatch", "simclr", "tcl_ft",
        "uctcl"]


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


def val(idx, ds, m, s, key, prot=None):
    r = idx[(ds, m, s)]
    return r[prot][key] if prot else r[key]


def runs():
    return [(d, s) for d in DATASETS for s in SEEDS]


# --------------------------------------------------------------------------- #
def a_ranks(idx):
    """
    Average rank with a Friedman test and Nemenyi critical difference.

    The three benchmarks have wear in different ranges (NASA up to 1.5 mm,
    PHM 0.03-0.24 mm, NUAA 0.05-0.31 mm), so averaging raw RMSE across them is
    meaningless.  Ranks are scale-free and are the standard way to compare
    methods over several data sets.
    """
    from scipy import stats
    out = {}
    for key, higher in [("macro_f1", True), ("rmse_mm", False)]:
        for prot, pn in [(None, "inductive"), ("tta", "online")]:
            methods = [m for m in LABEL_LIMITED
                       if not (prot and m in ("rf", "svm"))]
            R = defaultdict(list)
            mat = []
            for ds, s in runs():
                v = {m: val(idx, ds, m, s, key, prot) for m in methods}
                order = sorted(methods, key=lambda m: -v[m] if higher else v[m])
                for i, m in enumerate(order):
                    R[m].append(i + 1)
                mat.append([v[m] for m in methods])
            mat = np.asarray(mat)
            fr = stats.friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
            k, n = len(methods), len(mat)
            # Nemenyi critical difference at alpha = 0.05
            q_alpha = {5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102}.get(k, 3.1)
            cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
            out[f"{key}_{pn}"] = dict(
                mean_rank={m: float(np.mean(R[m])) for m in methods},
                friedman_stat=float(fr.statistic), friedman_p=float(fr.pvalue),
                critical_difference=float(cd), n_runs=int(n), k_methods=int(k))
    return out


def a_power(idx):
    """
    Retrospective power, and the sample size that would be needed.

    For every comparison we could not resolve, we report the observed effect
    size (Cohen's d_z on the paired differences) and the number of runs that a
    two-sided paired test would need for 80% power.  This turns "not
    significant" into a quantitative statement about what it would take.
    """
    from scipy import stats
    ours = np.array([val(idx, d, "uctcl", s, "macro_f1", "tta")
                     for d, s in runs()])
    out = {}
    for m in DEEP:
        if m == "uctcl":
            continue
        b = np.array([val(idx, d, m, s, "macro_f1", "tta") for d, s in runs()])
        diff = ours - b
        dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
        p = stats.wilcoxon(ours, b).pvalue
        if abs(dz) > 1e-6:
            n_req = int(np.ceil(((stats.norm.ppf(0.975)
                                  + stats.norm.ppf(0.80)) / abs(dz)) ** 2))
        else:
            n_req = None
        out[m] = dict(mean_diff=float(diff.mean()), cohen_dz=float(dz),
                      wilcoxon_p=float(p), n_runs=len(diff),
                      n_for_80pct_power=n_req)
    return out


def a_pl(idx):
    """Pseudo-label coverage, precision and class balance on every data set."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for m in ["pseudo", "uctcl"]:
            cov, pre, ev, nsel = [], [], [], []
            for s in SEEDS:
                r = idx[(ds, m, s)]
                pl = r.get("extra", {}).get("pl")
                if not pl:
                    continue
                d = pl[-1]
                n_unl = d.get("n_unlabelled") or r["n_unlabelled"]
                cov.append(100.0 * d["n"] / max(1, n_unl))
                nsel.append(d["n"])
                if np.isfinite(d["precision"]):
                    pre.append(100.0 * d["precision"])
                pc = np.asarray(d["per_class"], float)
                if pc.sum() > 0:
                    q = pc[pc > 0] / pc.sum()
                    ev.append(float(-(q * np.log(q)).sum() / np.log(len(pc))))
                else:
                    ev.append(0.0)

            def ms(v):
                v = np.asarray(v, float)
                return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]

            out[ds][m] = dict(coverage=ms(cov), precision=ms(pre),
                              evenness=ms(ev), n_selected=ms(nsel))
    return out


def a_online(idx):
    """
    When does online adaptation help?

    For each data set we record the gain the online protocol delivers and the
    dispersion of break-in wear across its training tools.  The shrinkage rule
    keys on exactly that dispersion, so the two should move together: where
    tools begin at consistent wear, an apparent offset is model bias worth
    removing; where they do not, it is genuine and correcting it injects error.
    """
    import run_experiments as R
    out = {}
    for ds in DATASETS:
        S = R.prepare_split(ds, 0, 0.08)
        scale = S["y_hi"] - S["y_lo"]
        gains, raws, applied = [], [], []
        for m in DEEP:
            g = np.mean([val(idx, ds, m, s, "macro_f1", "tta")
                         - val(idx, ds, m, s, "macro_f1") for s in SEEDS])
            gains.append((m, float(g)))
        for s in SEEDS:
            t = idx[(ds, "uctcl", s)]["tta"]
            raws += [abs(v) for v in t["raw_offsets_mm"].values()]
            applied += [abs(v) for v in t["offsets_mm"].values()]
        out[ds] = dict(
            spread_norm=float(S["vb0_spread"]),
            spread_mm=float(S["vb0_spread"] * scale),
            median_raw_offset_mm=float(np.median(raws)),
            median_applied_offset_mm=float(np.median(applied)),
            shrinkage_ratio=float(1.0 - np.median(applied) / max(np.median(raws), 1e-9)),
            gain_by_method=dict(gains),
            mean_gain=float(np.mean([g for _, g in gains])),
            uctcl_gain=float(dict(gains)["uctcl"]),
            n_tools=int(len(np.unique(S["traj"]))),
        )
    return out


def a_deploy():
    """
    Edge-deployment cost.

    Reports parameter count, single-window inference latency, the cost of the
    M-pass Monte-Carlo sweep that the uncertainty estimate needs, and the
    marginal cost of the online corrections, which are pure post-processing.
    """
    import torch
    import run_experiments as R
    import methods as M
    from model import count_params
    from tta import postprocess
    torch.set_num_threads(4)

    out = {}
    for ds in DATASETS:
        cfg = R.build_cfg(ds, method="uctcl")
        in_ch = R.DS_CFG[ds]["in_ch"]
        win = R.DS_CFG[ds]["win"]
        m = M._make_model(in_ch, 4, cfg, 0)
        m.eval()
        x1 = torch.randn(1, in_ch, win)
        xb = torch.randn(64, in_ch, win)
        with torch.no_grad():
            for _ in range(5):
                m(x1)
            t0 = time.time()
            for _ in range(50):
                m(x1)
            lat1 = (time.time() - t0) / 50 * 1e3
            t0 = time.time()
            for _ in range(10):
                m(xb)
            lat64 = (time.time() - t0) / 10 * 1e3
        n_par = count_params(m)
        # activation + parameter memory in float32
        mem_par = n_par * 4 / 1024.0
        # cost of the MC sweep relative to a single deterministic pass
        mc = cfg["n_mc"]
        # post-processing cost on a whole tool life
        S = R.prepare_split(ds, 0, 0.08)
        te = S["te"]
        n = len(te)
        rng = np.random.default_rng(0)
        pm = rng.random((mc, n, 4)); pm /= pm.sum(2, keepdims=True)
        mu = rng.normal(size=(mc, n)); lv = np.full((mc, n), -4.0)
        t0 = time.time()
        for _ in range(5):
            postprocess(pm, mu, lv, S["traj"][te], S["order"][te],
                        S["edges_norm"], S["vb0"], spread=S["vb0_spread"])
        post_ms = (time.time() - t0) / 5 * 1e3
        out[ds] = dict(
            params=int(n_par), param_memory_kb=float(mem_par),
            latency_ms_single=float(lat1), latency_ms_batch64=float(lat64),
            throughput_windows_per_s=float(64.0 / (lat64 / 1e3)),
            n_mc=int(mc), mc_sweep_ms_single=float(lat1 * mc),
            online_postprocess_ms_per_tool=float(post_ms),
            windows_per_tool=int(n),
            window_seconds=float(win / {"nasa": 250.0, "phm": 25.0,
                                        "nuaa": 15.15}[ds]),
        )
    return out


def a_budget():
    """
    Label-efficiency: does the advantage of the acceptance rule grow as labels
    become scarcer?

    The per-class quantile is meant to matter most when labels are rare, so the
    2% budget is where a benefit should be largest if one exists.  We run the
    same paired comparison as at 8% and report it whatever it says.

    Also re-measures the pseudo-label mechanism (coverage / precision /
    evenness) at 2%, to separate "the mechanism stops working" from "the
    mechanism works but does not move the end task".
    """
    from scipy import stats
    METH = ["sup_low", "pseudo", "uctcl"]
    raw = {}
    for f in glob.glob(os.path.join(ROOT, "results", "raw", f"{VERSION}__*.json")):
        r = json.load(open(f))
        if r["method"] in METH:
            raw[(r["dataset"], round(r["label_frac"], 3), r["method"],
                 r["seed"])] = r

    fracs = sorted({k[1] for k in raw})
    out = {"fracs": fracs, "per_dataset": {}, "pooled": {}, "pl": {},
           "label_pct": {}}

    for f in fracs:
        have = all((d, f, m, s) in raw for d in DATASETS for m in METH
                   for s in SEEDS)
        if not have:
            continue                       # only report fully-populated budgets
        key = f"{f:.3f}"
        out["per_dataset"][key] = {}
        vecs = {m: {"ind": [], "onl": []} for m in METH}
        for d in DATASETS:
            e = {}
            for m in METH:
                ind = np.array([raw[(d, f, m, s)]["macro_f1"] for s in SEEDS])
                onl = np.array([raw[(d, f, m, s)]["tta"]["macro_f1"]
                                for s in SEEDS])
                vecs[m]["ind"] += list(ind)
                vecs[m]["onl"] += list(onl)
                e[m] = dict(inductive=[float(ind.mean()),
                                       float(ind.std(ddof=1))],
                            online=[float(onl.mean()), float(onl.std(ddof=1))])
            for base in ("sup_low", "pseudo"):
                a = np.array([raw[(d, f, "uctcl", s)]["macro_f1"] for s in SEEDS])
                b = np.array([raw[(d, f, base, s)]["macro_f1"] for s in SEEDS])
                e[f"uctcl_minus_{base}"] = dict(
                    diff=float((a - b).mean()),
                    p=float(stats.wilcoxon(a, b).pvalue))
            out["per_dataset"][key][d] = e
            out["label_pct"].setdefault(key, {})[d] = float(np.mean(
                [raw[(d, f, "uctcl", s)]["actual_label_pct"] for s in SEEDS]))

        pooled = {}
        for m in METH:
            pooled[m] = dict(
                inductive=float(np.mean(vecs[m]["ind"])),
                online=float(np.mean(vecs[m]["onl"])))
        U = np.array(vecs["uctcl"]["ind"])
        for base in ("sup_low", "pseudo"):
            B = np.array(vecs[base]["ind"])
            d_ = U - B
            pooled[f"uctcl_minus_{base}"] = dict(
                diff=float(d_.mean()),
                p=float(stats.wilcoxon(U, B).pvalue),
                dz=float(d_.mean() / d_.std(ddof=1)),
                n=int(len(d_)))
        out["pooled"][key] = pooled

        # mechanism at this budget
        out["pl"][key] = {}
        for d in DATASETS:
            out["pl"][key][d] = {}
            for m in ("pseudo", "uctcl"):
                cov, pre, ev = [], [], []
                for s in SEEDS:
                    pld = raw[(d, f, m, s)].get("extra", {}).get("pl")
                    if not pld:
                        continue
                    z = pld[-1]
                    n_unl = z.get("n_unlabelled") or raw[(d, f, m, s)]["n_unlabelled"]
                    cov.append(100.0 * z["n"] / max(1, n_unl))
                    if np.isfinite(z["precision"]):
                        pre.append(100.0 * z["precision"])
                    pc = np.asarray(z["per_class"], float)
                    if pc.sum() > 0:
                        q = pc[pc > 0] / pc.sum()
                        ev.append(float(-(q * np.log(q)).sum() / np.log(len(pc))))
                ms = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1))]
                out["pl"][key][d][m] = dict(coverage=ms(cov),
                                            precision=ms(pre), evenness=ms(ev))
    return out


def a_stages(idx):
    """Per-stage F1 and how much of the error is between adjacent stages."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for prot, pn in [(None, "inductive"), ("tta", "online")]:
            cm = np.zeros((4, 4))
            f1 = []
            for s in SEEDS:
                r = idx[(ds, "uctcl", s)]
                d = r[prot] if prot else r
                cm += np.asarray(d["confusion"], float)
                f1.append(d["per_class_f1"])
            off = cm.copy()
            np.fill_diagonal(off, 0.0)
            tot = off.sum()
            adj = sum(off[i, j] for i in range(4) for j in range(4)
                      if abs(i - j) == 1)
            out[ds][pn] = dict(
                per_stage_f1=np.mean(f1, axis=0).tolist(),
                errors_total=float(tot),
                adjacent_fraction=float(adj / max(tot, 1)),
                confusion=cm.tolist())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    idx = load()
    print(f"{len(idx)} result cells\n")

    res = {}
    todo = ["ranks", "power", "pl", "online", "deploy", "stages", "budget"]
    if a.which != "all":
        todo = [a.which]

    for name in todo:
        fn = {"ranks": lambda: a_ranks(idx), "power": lambda: a_power(idx),
              "pl": lambda: a_pl(idx), "online": lambda: a_online(idx),
              "deploy": a_deploy, "stages": lambda: a_stages(idx),
              "budget": a_budget}[name]
        res[name] = fn()
        print(f"--- {name} ---")
        print(json.dumps(res[name], indent=1)[:1400])
        print()

    path = os.path.join(OUT, "analyses.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    with open(path, "w") as fh:
        json.dump(old, fh, indent=1)
    print("wrote results/analyses.json")
