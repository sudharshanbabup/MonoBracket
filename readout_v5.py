"""
readout_v5.py -- label-free monotone-path read-out on the held-out tool.

Pure function of the cached MC sweeps (results/sweeps), so no model is
retrained.  For every deep method and every split it scores four read-outs of
the same sweep:
    ind     per-window arg-max of the MC-mean class probabilities
    onl     the v4 online stack (anchor + isotonic smoothing + fusion)
    path    onl followed by MAP non-decreasing path decoding (whole stream)
    filt    onl probabilities passed through the causal monotone filter
Writes results/readout_v5.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R
import monopath
from tta import postprocess
from sklearn.metrics import f1_score

METHODS = ["sup_low", "pseudo", "meanteacher", "fixmatch", "simclr", "tcl_ft",
           "uctcl", "mbpl", "mbpl_nopath", "mbpl_noanch", "mbpl_cert", "sup_full"]


def f1(yt, pred):
    return float(f1_score(yt, pred, average="macro", labels=R._present(yt),
                          zero_division=0))


def main(frac=0.08):
    out = {}
    for ds in ["nasa", "phm", "nuaa"]:
        for seed in range(10):
            S = R.prepare_split(ds, seed, frac)
            te = S["te"]
            traj, order, yt = S["traj"][te], S["order"][te], S["y_cls"][te]
            for m in METHODS:
                p = os.path.join(R.SWEEPS, R.key(ds, m, seed, frac) + ".npz")
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                sw = (z["probs"].astype(np.float64), z["mu"].astype(np.float64),
                      z["lv"].astype(np.float64))
                probs, mu_s, _ = postprocess(*sw, traj, order, S["edges_norm"],
                                             S["vb0"], alpha=0.5,
                                             spread=S["vb0_spread"])
                path, q = monopath.decode_trajectories(probs, traj, order)
                filt = monopath.filter_trajectories(probs, traj, order)
                p_raw = sw[0].mean(0)
                path_raw, _ = monopath.decode_trajectories(p_raw, traj, order)
                out[f"{ds}|{m}|{seed}"] = dict(
                    ind=f1(yt, p_raw.argmax(1)), onl=f1(yt, probs.argmax(1)),
                    path=f1(yt, path), filt=f1(yt, filt.argmax(1)),
                    path_raw=f1(yt, path_raw),
                    viol_ind=int((np.diff(p_raw.argmax(1)[np.lexsort((order, traj))]) < 0).sum()),
                    viol_onl=int((np.diff(probs.argmax(1)[np.lexsort((order, traj))]) < 0).sum()),
                    n=int(len(te)))
    json.dump(out, open(os.path.join(R.ROOT, "results", "readout_v5.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    o = main()
    for m in METHODS:
        for k in ["ind", "onl", "path", "filt", "path_raw"]:
            v = [o[x][k] for x in o if x.split("|")[1] == m]
            print(f"{m:12s} {k:9s} {np.mean(v):.3f} n={len(v)}", end=" | ")
        print()
    for ds in ["nasa", "phm", "nuaa"]:
        for m in ["uctcl", "mbpl_nopath", "sup_low"]:
            print(ds, m, {k: round(float(np.mean([o[x][k] for x in o if x.startswith(ds + "|" + m + "|")])), 3) for k in ["ind", "onl", "path", "filt", "path_raw", "viol_ind", "viol_onl"]})
