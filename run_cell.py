"""
run_cell.py
-----------
Run exactly one (dataset, method, seed, label-fraction) cell and write its JSON.
Used to fill in individual missing cells of a suite without re-scanning the
whole job list.

    python src/run_cell.py --ds phm --method uctcl --seed 2 --frac 0.08
    python src/run_cell.py --missing ablation --ds phm --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(4)

import run_experiments as R

ABL_CORE = ["uctcl_noTCL", "uctcl_simclr", "uctcl_noMC", "uctcl_globalTau",
            "uctcl_noMono", "uctcl_noCB", "uctcl_relabel"]
# The two optional add-ons expand the accepted pseudo-label pool so much that a
# single NASA cell exceeds the wall-clock budget of this machine; they are
# therefore reported on PHM 2010 only.
ABL_OPTIONAL = ["uctcl_fusion", "uctcl_agree"]
ABL = {"nasa": ABL_CORE, "phm": ABL_CORE + ABL_OPTIONAL}
MAIN = ["rf", "svm", "sup_low", "pseudo", "meanteacher", "fixmatch",
        "simclr", "tcl_ft", "uctcl", "sup_full"]
SWEEP = ["sup_low", "fixmatch", "uctcl"]


def run(ds, method, seed, frac):
    path = os.path.join(R.RAW, R.key(ds, method, seed, frac) + ".json")
    if os.path.exists(path):
        return False
    t = time.time()
    res = R.run_one(ds, method, seed, frac)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{os.path.basename(path)}: F1={res['macro_f1']:.3f} "
          f"RMSE={res['rmse_mm']:.4f} ({time.time() - t:.1f}s)", flush=True)
    return True


def missing(suite, ds, seeds, frac):
    out = []
    if suite == "ablation":
        for s in seeds:
            for m in ABL[ds]:
                out.append((ds, m, s, frac))
    elif suite == "main":
        for s in seeds:
            for m in MAIN:
                out.append((ds, m, s, frac))
    else:
        for s in seeds:
            for f in [0.02, 0.05, 0.10, 0.20]:
                for m in SWEEP:
                    out.append((ds, m, s, f))
    return [c for c in out
            if not os.path.exists(os.path.join(R.RAW, R.key(*c) + ".json"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds")
    ap.add_argument("--method")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--frac", type=float, default=0.08)
    ap.add_argument("--missing")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--limit", type=int, default=1)
    a = ap.parse_args()

    if a.missing:
        cells = missing(a.missing, a.ds, [int(x) for x in a.seeds.split(",")],
                        a.frac)
        print(f"{len(cells)} cells still missing")
        for c in cells[: a.limit]:
            run(*c)
    else:
        run(a.ds, a.method, a.seed, a.frac)


if __name__ == "__main__":
    main()
