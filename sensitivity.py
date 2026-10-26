"""
sensitivity.py
--------------
Hyper-parameter sensitivity.

The online corrections are a pure function of the cached Monte-Carlo sweep, so
their hyper-parameters (fusion weight alpha, monotonicity tolerance kappa,
anchor window q, number of MC passes M) can be swept exhaustively at no
training cost.  Training-side hyper-parameters need a re-fit and are swept on a
reduced grid.

    python src/sensitivity.py --which online
    python src/sensitivity.py --which mc
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEPS = os.path.join(ROOT, "results", "sweeps")
VERSION = "j2"
DATASETS = ["nasa", "phm", "nuaa"]
SEEDS = list(range(10))


def _sweep_path(ds, m, s, frac=0.08):
    return os.path.join(SWEEPS, f"{VERSION}__{ds}__{m}__f{frac:.3f}__s{s}.npz")


def _score(ds, m, s, **kw):
    """Re-evaluate one cached sweep under different online settings."""
    import run_experiments as R
    from tta import postprocess
    from sklearn.metrics import f1_score

    p = _sweep_path(ds, m, s)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    S = R.prepare_split(ds, s, 0.08)
    te = S["te"]
    yt = S["y_cls"][te]
    scale, lo = S["y_hi"] - S["y_lo"], S["y_lo"]
    kw.setdefault("spread", S["vb0_spread"])
    probs, mu, _ = postprocess(z["probs"].astype(np.float64),
                               z["mu"].astype(np.float64),
                               z["lv"].astype(np.float64),
                               S["traj"][te], S["order"][te], S["edges_norm"],
                               S["vb0"], **kw)
    f1 = f1_score(yt, probs.argmax(1), average="macro",
                  labels=R._present(yt), zero_division=0)
    rmse = float(np.sqrt(np.mean((mu * scale + lo - S["y_raw"][te]) ** 2)))
    return float(f1), rmse


def sweep_online(method="uctcl"):
    """alpha, kappa, q and the shrinkage switch, all from cached sweeps."""
    grids = {
        "alpha_fuse": [0.0, 0.25, 0.5, 0.75, 1.0],
        "q": [0.10, 0.15, 0.25, 0.35, 0.50],
        "shrinkage": ["off", "on"],
    }
    out = {}
    for pname, values in grids.items():
        out[pname] = {}
        for v in values:
            per_ds = {}
            for ds in DATASETS:
                f1s, rms = [], []
                for s in SEEDS:
                    kw = {}
                    if pname == "alpha_fuse":
                        kw["alpha"] = v
                    elif pname == "q":
                        kw["q"] = v
                    else:
                        if v == "off":
                            kw["spread"] = 0.0
                    r = _score(ds, method, s, **kw)
                    if r:
                        f1s.append(r[0]); rms.append(r[1])
                if f1s:
                    per_ds[ds] = dict(macro_f1=[float(np.mean(f1s)),
                                                float(np.std(f1s, ddof=1))],
                                      rmse_mm=[float(np.mean(rms)),
                                               float(np.std(rms, ddof=1))])
            # pooled rank-free summary: mean of per-dataset means
            out[pname][str(v)] = dict(
                per_dataset=per_ds,
                pooled_macro_f1=float(np.mean([d["macro_f1"][0]
                                               for d in per_ds.values()])))
    return out


def sweep_components(method="uctcl"):
    """
    Ablation of the three online corrections.

    Each is a switch inside `postprocess`, so the full 2^3 - 1 leave-one-out
    grid is re-evaluated from the cached sweeps at no training cost.
    """
    cfgs = {
        "full": {},
        "no_anchor": dict(use_anchor=False),
        "no_mono": dict(use_mono=False),
        "no_fusion": dict(use_fuse=False),
        "none": dict(use_anchor=False, use_mono=False, use_fuse=False),
    }
    out = {}
    for name, kw in cfgs.items():
        per_ds = {}
        for ds in DATASETS:
            f1s, rms = [], []
            for s in SEEDS:
                r = _score(ds, method, s, **kw)
                if r:
                    f1s.append(r[0]); rms.append(r[1])
            if f1s:
                per_ds[ds] = dict(macro_f1=[float(np.mean(f1s)),
                                            float(np.std(f1s, ddof=1))],
                                  rmse_mm=[float(np.mean(rms)),
                                           float(np.std(rms, ddof=1))])
        out[name] = dict(per_dataset=per_ds,
                         pooled_macro_f1=float(np.mean(
                             [d["macro_f1"][0] for d in per_ds.values()])))
    return out


def sweep_mc(method="uctcl"):
    """
    Number of Monte-Carlo passes.

    The cached sweep holds M = 10 passes; taking the first m of them is exactly
    an M = m sweep, so the whole curve comes free.
    """
    import run_experiments as R
    from tta import postprocess
    from sklearn.metrics import f1_score
    out = {}
    for m_passes in [2, 3, 5, 8, 10]:
        per_ds = {}
        for ds in DATASETS:
            f1s, rms = [], []
            for s in SEEDS:
                p = _sweep_path(ds, method, s)
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                S = R.prepare_split(ds, s, 0.08)
                te = S["te"]; yt = S["y_cls"][te]
                scale, lo = S["y_hi"] - S["y_lo"], S["y_lo"]
                probs, mu, _ = postprocess(
                    z["probs"][:m_passes].astype(np.float64),
                    z["mu"][:m_passes].astype(np.float64),
                    z["lv"][:m_passes].astype(np.float64),
                    S["traj"][te], S["order"][te], S["edges_norm"], S["vb0"],
                    spread=S["vb0_spread"])
                f1s.append(f1_score(yt, probs.argmax(1), average="macro",
                                    labels=R._present(yt), zero_division=0))
                rms.append(float(np.sqrt(np.mean(
                    (mu * scale + lo - S["y_raw"][te]) ** 2))))
            if f1s:
                per_ds[ds] = [float(np.mean(f1s)), float(np.std(f1s, ddof=1))]
        out[str(m_passes)] = dict(
            per_dataset=per_ds,
            pooled=float(np.mean([v[0] for v in per_ds.values()])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    res = {}
    if a.which in ("all", "online"):
        res["online"] = sweep_online()
    if a.which in ("all", "mc"):
        res["mc_passes"] = sweep_mc()
    if a.which in ("all", "components"):
        res["components"] = sweep_components()

    path = os.path.join(ROOT, "results", "sensitivity.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    with open(path, "w") as fh:
        json.dump(old, fh, indent=1)

    for k, v in res.items():
        print(f"--- {k} ---")
        if k == "components":
            for nm, d in v.items():
                print(f"  {nm:>10}  pooled {d['pooled_macro_f1']:.4f}  "
                      + "  ".join(f"{ds} {x['macro_f1'][0]:.3f}"
                                  for ds, x in d["per_dataset"].items()))
        elif k == "mc_passes":
            for m, d in v.items():
                print(f"  M={m:>2}  pooled macro-F1 {d['pooled']:.4f}  "
                      + "  ".join(f"{ds} {x[0]:.3f}" for ds, x in
                                  d["per_dataset"].items()))
        else:
            for pname, g in v.items():
                print(f"  {pname}:")
                for val, d in g.items():
                    print(f"    {val:>6}  pooled {d['pooled_macro_f1']:.4f}  "
                          + "  ".join(f"{ds} {x['macro_f1'][0]:.3f}"
                                      for ds, x in d["per_dataset"].items()))
    print("\nwrote results/sensitivity.json")
