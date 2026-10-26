"""
pretrain.py
-----------
Self-supervised pre-training stage, run separately from the downstream study
and cached on disk.

Why it is a separate stage
--------------------------
The contrastive objective uses no labels, so its result depends only on
(dataset, split seed, kernel mode) -- never on the label budget or on the
downstream learning rule.  Caching it therefore
  * gives every pre-trained method *bit-identical* initial weights, which makes
    the TCL-vs-SimCLR ablation a controlled comparison, and
  * avoids repeating the most expensive stage for each label fraction.

Only windows from the training tools are used; the held-out tools are never
touched, so the protocol stays inductive.

The loop is checkpointed after every epoch and stops at a wall-clock deadline,
so it can be resumed across short sessions.

    python src/pretrain.py --ds nasa --seed 0 --mode tcl --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

import methods as M
from losses import temporal_contrastive_loss
from augment import contrastive_pair

ROOT = os.path.dirname(os.path.abspath(__file__))
_A = os.environ.get("UCTCL_ASSETS") or (
    ROOT if os.path.isdir(os.path.join(ROOT, "results", "pretrain"))
    else os.path.join(os.path.dirname(os.path.dirname(ROOT)), "codes_3"))
PRE = os.path.join(_A, "results", "pretrain")


def ckpt_path(ds, seed, mode):
    return os.path.join(PRE, f"{ds}_s{seed}_{mode}.pt")


def pretrain(ds, seed, mode, epochs, deadline_s, fresh=False):
    import run_experiments as R

    S = R.prepare_split(ds, seed, 0.08)          # label budget irrelevant here
    cfg = R.build_cfg(ds)
    cfg["e_con"] = epochs
    in_ch = R.DS_CFG[ds]["in_ch"]

    tr = S["tr"]
    X = S["X"][tr]
    traj = torch.tensor(S["traj"][tr], dtype=torch.long)
    tpos = torch.tensor(S["tpos"][tr], dtype=torch.float32)
    sigma = cfg["sigma_t"] if mode == "tcl" else 1e-4

    model = M._make_model(in_ch, 4, cfg, seed)
    n, bs = len(X), cfg["batch"]
    spe = max(1, int(np.ceil(n / bs)))
    opt, sch = M._opt(model, cfg, epochs, spe)

    path = ckpt_path(ds, seed, mode)
    start_ep, hist = 0, []
    if os.path.exists(path) and not fresh:
        st = torch.load(path, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sch.load_state_dict(st["sch"])
        start_ep, hist = st["epoch"], st["hist"]
        if start_ep >= epochs:
            print(f"{os.path.basename(path)}: already complete "
                  f"({start_ep}/{epochs} epochs, final loss {hist[-1]:.4f})")
            return True

    t0 = time.time()
    ep = start_ep
    while ep < epochs:
        rng = np.random.default_rng(seed * 1000 + ep)
        model.train()
        tot = 0.0
        for b in M._batches(n, bs, rng):
            if len(b) < 4:
                continue
            v1, v2 = contrastive_pair(X[b])
            loss = temporal_contrastive_loss(
                model(v1, mode="proj"), model(v2, mode="proj"),
                traj[b], tpos[b], tau=cfg["tau_con"], sigma=sigma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sch.step()
            tot += float(loss) * len(b)
        hist.append(tot / n)
        ep += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sch=sch.state_dict(), epoch=ep, hist=hist,
                        cfg=cfg, mode=mode, ds=ds, seed=seed), path)
        if time.time() - t0 > deadline_s:
            break

    done = ep >= epochs
    print(f"{os.path.basename(path)}: epoch {ep}/{epochs} loss {hist[-1]:.4f} "
          f"({time.time() - t0:.1f}s) {'COMPLETE' if done else 'partial'}")
    return done


def load_pretrained(model, ds, seed, mode, required_epochs):
    """Load cached weights into `model`; returns the loss history."""
    path = ckpt_path(ds, seed, mode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing pre-training checkpoint {path}")
    st = torch.load(path, map_location="cpu")
    if st["epoch"] < required_epochs:
        raise RuntimeError(f"{path} only has {st['epoch']}/{required_epochs} epochs")
    model.load_state_dict(st["model"])
    return st["hist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nasa")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", default="tcl,simclr")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--deadline", type=float, default=30.0)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for s in [int(x) for x in a.seeds.split(",")]:
        for m in a.modes.split(","):
            if time.time() - t0 > a.deadline:
                print("deadline reached")
                return
            pretrain(a.ds, s, m, a.epochs, a.deadline - (time.time() - t0),
                     fresh=a.fresh)


if __name__ == "__main__":
    main()
