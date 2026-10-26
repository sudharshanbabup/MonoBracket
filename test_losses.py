"""
test_losses.py
--------------
Numerical unit tests for the loss functions.  Run with

    python src/test_losses.py

Every property that the paper claims about a loss is checked here, including
the claim that the temporal contrastive loss reduces exactly to NT-Xent when
the temporal bandwidth goes to zero.
"""

from __future__ import annotations

import math
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from losses import (temporal_contrastive_loss, focal_ce, hetero_nll,
                    sigmoid_rampup)

OK = True


def chk(name, cond):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def nt_xent_reference(z1, z2, tau):
    """Textbook SimCLR NT-Xent, written independently of the implementation."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)])
    return F.cross_entropy(sim, tgt)


def main():
    torch.manual_seed(0)
    B, d = 16, 8

    # ---- temporal contrastive loss ---------------------------------------- #
    z1 = F.normalize(torch.randn(B, d), dim=1)
    z2 = F.normalize(torch.randn(B, d), dim=1)
    traj = torch.randint(0, 3, (B,))
    tpos = torch.rand(B)

    l = temporal_contrastive_loss(z1, z2, traj, tpos, tau=0.1, sigma=0.15)
    chk("loss is finite", torch.isfinite(l))
    chk("loss is non-negative", float(l) >= 0.0)

    # sigma -> 0 : temporal neighbours vanish, only the augmented view remains
    l0 = temporal_contrastive_loss(z1, z2, traj, tpos, tau=0.1, sigma=1e-6)
    ref = nt_xent_reference(z1, z2, 0.1)
    chk(f"sigma->0 reduces to NT-Xent ({float(l0):.6f} vs {float(ref):.6f})",
        torch.allclose(l0, ref, atol=1e-5))

    # perfect representation: the two views identical and all trajectories
    # distinct -> loss equals the theoretical minimum log(2B-1) - 1/tau + ...
    zz = F.normalize(torch.randn(B, d), dim=1)
    tr_u = torch.arange(B)
    lp = temporal_contrastive_loss(zz, zz, tr_u, tpos, tau=0.01, sigma=1e-6)
    lr = temporal_contrastive_loss(z1, z2, tr_u, tpos, tau=0.01, sigma=1e-6)
    chk("aligned views give a lower loss than random views", float(lp) < float(lr))

    # gradient flows and is finite
    z1r = torch.randn(B, d, requires_grad=True)
    lg = temporal_contrastive_loss(F.normalize(z1r, dim=1), z2, traj, tpos,
                                   tau=0.1, sigma=0.15)
    lg.backward()
    chk("gradient is finite (no 0 * -inf NaN)", torch.isfinite(z1r.grad).all())

    # every anchor with a same-trajectory neighbour must get weight on it
    traj_same = torch.zeros(B, dtype=torch.long)
    tpos_close = torch.linspace(0, 0.1, B)
    lc = temporal_contrastive_loss(z1, z2, traj_same, tpos_close, tau=0.1, sigma=1.0)
    lf = temporal_contrastive_loss(z1, z2, traj_same, tpos_close, tau=0.1, sigma=1e-6)
    chk("temporal kernel changes the loss", abs(float(lc) - float(lf)) > 1e-4)

    # permutation invariance of the batch
    perm = torch.randperm(B)
    la = temporal_contrastive_loss(z1, z2, traj, tpos, 0.1, 0.15)
    lb = temporal_contrastive_loss(z1[perm], z2[perm], traj[perm], tpos[perm],
                                   0.1, 0.15)
    chk("invariant to batch permutation", torch.allclose(la, lb, atol=1e-5))

    # ---- focal / class-balanced cross-entropy ----------------------------- #
    logits = torch.randn(64, 4)
    y = torch.randint(0, 4, (64,))
    ce = F.cross_entropy(logits, y)
    fc = focal_ce(logits, y, None, gamma=0.0)
    chk(f"gamma=0, w=1 reduces to cross-entropy ({float(fc):.6f} vs {float(ce):.6f})",
        torch.allclose(fc, ce, atol=1e-6))

    w = torch.tensor([2.0, 1.0, 0.5, 0.5])
    fw = focal_ce(logits, y, w, gamma=0.0)
    manual = (F.cross_entropy(logits, y, reduction="none") * w[y]).mean()
    chk("class weights applied per sample", torch.allclose(fw, manual, atol=1e-6))

    easy = torch.zeros(1, 4); easy[0, 1] = 20.0            # p ~ 1 for class 1
    hard = torch.zeros(1, 4)                               # uniform
    ye = torch.tensor([1])
    fe = focal_ce(easy, ye, None, gamma=2.0)
    fh = focal_ce(hard, ye, None, gamma=2.0)
    chk("focal term down-weights easy examples", float(fe) < 1e-6 < float(fh))
    chk("focal <= CE pointwise",
        float(focal_ce(logits, y, None, 2.0)) <= float(ce) + 1e-6)

    sw = torch.rand(64)
    fs = focal_ce(logits, y, None, 0.0, sample_w=sw)
    chk("sample weights applied",
        torch.allclose(fs, (F.cross_entropy(logits, y, reduction="none") * sw).mean(),
                       atol=1e-6))

    # ---- heteroscedastic NLL ---------------------------------------------- #
    yv = torch.randn(256)
    mu = torch.randn(256)
    lv = torch.zeros(256)                                  # sigma^2 = 1
    nll = hetero_nll(mu, lv, yv)
    mse = 0.5 * ((yv - mu) ** 2).mean()
    chk(f"logvar=0 gives 0.5*MSE ({float(nll):.6f} vs {float(mse):.6f})",
        torch.allclose(nll, mse, atol=1e-6))

    # the NLL is minimised at sigma^2 = (y - mu)^2  (analytic optimum)
    r2 = torch.tensor([4.0])
    yv1, mu1 = torch.tensor([2.0]), torch.tensor([0.0])
    grid = torch.linspace(-3, 5, 4001)
    vals = torch.stack([hetero_nll(mu1, g.view(1), yv1) for g in grid])
    best = grid[vals.argmin()]
    chk(f"NLL minimised at log sigma^2 = log (y-mu)^2 = {math.log(4.0):.4f} "
        f"(found {float(best):.4f})", abs(float(best) - math.log(4.0)) < 0.01)

    # a confident wrong prediction must cost more than an unconfident one
    conf_wrong = hetero_nll(torch.tensor([0.0]), torch.tensor([-4.0]),
                            torch.tensor([3.0]))
    humble = hetero_nll(torch.tensor([0.0]), torch.tensor([2.0]),
                        torch.tensor([3.0]))
    chk("over-confident error penalised more than a humble one",
        float(conf_wrong) > float(humble))

    # ---- ramp-up ----------------------------------------------------------- #
    chk("rampup(0) = exp(-5)", abs(sigmoid_rampup(0, 10) - math.exp(-5)) < 1e-9)
    chk("rampup(L) = 1", abs(sigmoid_rampup(10, 10) - 1.0) < 1e-9)
    chk("rampup is monotone",
        all(sigmoid_rampup(i, 10) <= sigmoid_rampup(i + 1, 10) for i in range(10)))
    chk("rampup saturates past L", abs(sigmoid_rampup(50, 10) - 1.0) < 1e-9)

    print("\nALL LOSS TESTS PASSED" if OK else "\nSOME LOSS TESTS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
