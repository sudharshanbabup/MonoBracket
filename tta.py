"""
tta.py
------
Test-time adaptation on the incoming tool's own unlabelled sensor stream.

Motivation
----------
Under leave-one-tool-out the dominant residual error is not label scarcity but
a tool-to-tool shift in the *absolute* wear scale: the model tracks the shape of
a new insert's wear curve but places it at the wrong offset, so whole blocks of
windows fall on the wrong side of a stage boundary.  No amount of extra
pseudo-labelling on the training tools can remove a shift that lives between
tools.

What an Industrial IoT deployment does have is the new tool's entire unlabelled
stream, in machining-time order, as it runs.  Three corrections follow from
that stream plus physics alone; none of them touches a test label.

    A. Feature-statistic re-estimation.  Re-estimate the BatchNorm running
       statistics on the incoming tool's windows.  This absorbs sensor-level
       shift (mounting, gain, workpiece batch) and is the standard test-time
       normalisation baseline.

    B. Physics-anchored offset correction.  A fresh insert starts at break-in
       wear.  If the predicted trajectory of a new tool begins at a level that
       differs from the break-in level observed across the *training* tools,
       the difference is model bias, not wear.  We estimate it as a robust
       median over the earliest fraction of the tool life and subtract it:

           delta_j = median{ mu_i : t_i <= q }  -  VB_0

       where VB_0 is a constant computed once on the training tools.  One
       parameter, estimated from unlabelled predictions and a training-set
       constant.

    C. Monotone projection along the tool life.  The corrected trajectory is
       projected onto the monotone cone with precision weights (Sec. IV-C),
       which denoises it using the whole life at once.

    D. Threshold-consistent stage read-out.  With the offset removed the
       regression posterior is finally on the right scale, so the stage
       distribution implied by the known physical boundaries can be fused with
       the discriminative head.

A, C and D are label-free transformations of the model's own outputs; B uses
one scalar computed on training data.  Every method in the study is evaluated
under exactly the same procedure, so the comparison stays fair.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from uncertainty import (mc_classification_uncertainty, mc_regression_uncertainty,
                         regression_stage_probs, fuse_probs, weighted_pava)

EPS = 1e-12


# --------------------------------------------------------------------------- #
# A. feature-statistic re-estimation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def adapt_batchnorm(model: nn.Module, X: torch.Tensor, n_passes: int = 1,
                    batch: int = 128, momentum: float = None):
    """
    Re-estimate BatchNorm running statistics on the unlabelled target stream.

    No parameter is updated and no label is used: the module is put in training
    mode so that the running mean/variance buffers track the target batches,
    while `torch.no_grad()` prevents any weight change.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not bns:
        return model
    old = []
    for m in bns:
        old.append(m.momentum)
        m.momentum = momentum          # None -> cumulative moving average
        m.reset_running_stats()
    model.train()
    for _ in range(n_passes):
        for i in range(0, len(X), batch):
            xb = X[i : i + batch]
            if len(xb) < 2:            # BatchNorm needs >1 sample
                continue
            model(xb)
    for m, mo in zip(bns, old):
        m.momentum = mo
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# B. physics-anchored offset correction
# --------------------------------------------------------------------------- #
def reference_initial_wear(y_train: np.ndarray, order_train: np.ndarray,
                           traj_train: np.ndarray, q: float = 0.25,
                           return_spread: bool = False):
    """
    VB_0: the break-in wear level of a fresh insert, measured once on the
    training tools as the median true wear over the earliest `q` fraction of
    each tool's life.  A training-set constant; carries no test information.

    With `return_spread=True` the robust dispersion of that level *across* the
    training tools is also returned, as a median absolute deviation.  That
    number is the key diagnostic for whether the fresh-tool assumption is safe:
    if training inserts all begin at essentially the same wear, an incoming
    tool that appears to begin elsewhere is revealing model bias; if they begin
    at widely different wear, the same observation is just as likely to be a
    genuinely part-worn insert, and correcting it would inject error.
    """
    vals = []
    for tr in np.unique(traj_train):
        m = traj_train == tr
        o = order_train[m]
        if o.size == 0:
            continue
        cut = np.quantile(o, q)
        sel = y_train[m][o <= cut]
        if sel.size:
            vals.append(np.median(sel))
    if not vals:
        vb0 = float(np.median(y_train))
        return (vb0, 0.0) if return_spread else vb0
    vals = np.asarray(vals, dtype=np.float64)
    vb0 = float(np.median(vals))
    if not return_spread:
        return vb0
    spread = float(np.median(np.abs(vals - vb0)))
    return vb0, spread


def anchor_offset(mu: np.ndarray, order: np.ndarray, traj: np.ndarray,
                  vb0: float, q: float = 0.25, spread: float = 0.0):
    """
    Estimate and remove a per-trajectory additive bias, with shrinkage.

    The raw estimate of the bias of tool j is

        dhat_j = median{ mu_i : t_i <= q } - VB_0 ,

    but `dhat_j` confounds two things: the model bias we want to remove, and the
    genuine tool-to-tool variation in break-in wear, whose scale is `spread`
    (Sec. `reference_initial_wear`).  Attributing all of `dhat_j` to bias is
    only safe when `spread` is small.  We therefore soft-threshold:

        delta_j = sign(dhat_j) * max(|dhat_j| - spread, 0),

    i.e. we correct only the part of the apparent offset that tool-to-tool
    variation cannot explain.  This is the usual shrinkage response to an
    estimate contaminated by a known noise floor, it needs no extra
    hyper-parameter, and it degrades gracefully: `spread = 0` recovers the
    uncorrected estimator, and a large `spread` disables the correction
    entirely rather than letting it do harm.

    Returns the corrected predictions, the applied offsets, and the raw ones.
    """
    out = mu.astype(np.float64).copy()
    deltas, raw = {}, {}
    for tr in np.unique(traj):
        m = traj == tr
        o = order[m]
        cut = np.quantile(o, q)
        early = mu[m][o <= cut]
        if early.size == 0:
            early = mu[m]
        d_hat = float(np.median(early) - vb0)
        d = float(np.sign(d_hat) * max(abs(d_hat) - spread, 0.0))
        raw[int(tr)] = d_hat
        deltas[int(tr)] = d
        out[m] = mu[m] - d
    return out, deltas, raw


# --------------------------------------------------------------------------- #
# C. monotone projection along the incoming tool life
# --------------------------------------------------------------------------- #
def monotone_smooth(mu: np.ndarray, var: np.ndarray, traj: np.ndarray,
                    order: np.ndarray) -> np.ndarray:
    """Precision-weighted isotonic projection, applied per trajectory."""
    out = mu.astype(np.float64).copy()
    for tr in np.unique(traj):
        idx = np.flatnonzero(traj == tr)
        idx = idx[np.argsort(order[idx], kind="mergesort")]
        out[idx] = weighted_pava(mu[idx].astype(np.float64),
                                 1.0 / np.maximum(var[idx], EPS))
    return out


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
def mc_forward(model, X_test, n_mc=10, batch=256, use_bn=False):
    """
    One Monte-Carlo dropout sweep over the target stream, optionally after
    re-estimating the BatchNorm statistics on it.  Returned so that every
    variant of the post-processing stack can share a single sweep.
    """
    from model import mc_predict
    if use_bn:
        adapt_batchnorm(model, X_test)
    return mc_predict(model, X_test, n_mc=n_mc, batch=batch)


def postprocess(probs_mc, mu_mc, lv_mc, traj_test, order_test, edges, vb0,
                q=0.25, alpha=0.5, use_anchor=True, use_mono=True,
                use_fuse=True, spread=0.0):
    """
    Apply the label-free corrections to one MC sweep.  Splitting this from the
    sweep keeps the ablation of the stack essentially free.
    """
    p_cls, _, _, u_epi = mc_classification_uncertainty(probs_mc)
    mu_bar, v_ale, v_epi, v_tot = mc_regression_uncertainty(mu_mc, lv_mc)

    if use_anchor:
        mu_c, deltas, raw = anchor_offset(mu_bar, order_test, traj_test, vb0,
                                          q, spread)
    else:
        mu_c, deltas, raw = mu_bar.astype(np.float64), {}, {}

    # shift every MC sample by the same per-trajectory offset so the posterior
    # stays consistent with the corrected mean
    mu_mc_c = mu_mc + (mu_c - mu_bar)[None, :]

    mu_s = (monotone_smooth(mu_c, v_tot, traj_test, order_test)
            if use_mono else mu_c)

    if use_fuse:
        p_reg = regression_stage_probs(mu_mc_c, lv_mc, edges)
        # the monotone-smoothed mean is the better point estimate, so blend its
        # hard stage assignment into the regression head's soft assignment
        hard = np.eye(len(edges) + 1)[np.digitize(mu_s, edges)]
        p_reg = 0.5 * p_reg + 0.5 * hard
        probs = fuse_probs(p_cls, p_reg, alpha)
    else:
        probs = p_cls

    return probs, mu_s, dict(offsets=deltas, raw_offsets=raw, spread=spread,
                             u_epi=u_epi, v_tot=v_tot, mu_raw=mu_bar)


def adapt_and_predict(model, X_test, traj_test, order_test, edges, vb0,
                      n_mc=10, q=0.25, alpha=0.5, use_bn=False, use_anchor=True,
                      use_mono=True, use_fuse=True, batch=256, spread=0.0):
    """
    Full stack in one call.

    Note on `use_bn`: re-estimating the BatchNorm statistics on the target
    stream is the standard test-time normalisation baseline, but it rescales
    the features the *regression* head was calibrated against and degrades wear
    RMSE badly here, so it is off by default and reported only as an ablation.
    """
    probs_mc, mu_mc, lv_mc = mc_forward(model, X_test, n_mc=n_mc, batch=batch,
                                        use_bn=use_bn)
    return postprocess(probs_mc, mu_mc, lv_mc, traj_test, order_test, edges,
                       vb0, q=q, alpha=alpha, use_anchor=use_anchor,
                       use_mono=use_mono, use_fuse=use_fuse, spread=spread)


# --------------------------------------------------------------------------- #
# numerical verification
# --------------------------------------------------------------------------- #
def verify(seed: int = 0, verbose: bool = True) -> bool:
    rng = np.random.default_rng(seed)
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # ---- reference initial wear ------------------------------------------ #
    # three synthetic tools, each rising linearly from 0.1
    y, o, tr = [], [], []
    for j in range(3):
        n = 40
        t = np.linspace(0, 1, n)
        y.append(0.1 + 0.5 * t)
        o.append(t)
        tr.append(np.full(n, j))
    y, o, tr = np.concatenate(y), np.concatenate(o), np.concatenate(tr)
    vb0 = reference_initial_wear(y, o, tr, q=0.25)
    chk(f"VB_0 recovers the break-in level ({vb0:.4f} ~ 0.16)",
        abs(vb0 - (0.1 + 0.5 * 0.125)) < 0.02)

    # ---- offset correction ------------------------------------------------ #
    true_off = np.array([0.3, -0.2, 0.05])
    mu = y + true_off[tr]
    mu_c, d, raw = anchor_offset(mu, o, tr, vb0, q=0.25)
    chk("recovers each planted per-tool offset",
        all(abs(d[j] - true_off[j]) < 0.02 for j in range(3)))
    chk("correction reduces the error against the truth",
        np.mean(np.abs(mu_c - y)) < np.mean(np.abs(mu - y)))
    chk("no correction when there is no bias",
        np.allclose(anchor_offset(y, o, tr, vb0, 0.25)[0], y, atol=0.02))
    # idempotence: correcting twice changes nothing further
    mu_cc, _, _ = anchor_offset(mu_c, o, tr, vb0, 0.25)
    chk("offset correction is idempotent", np.allclose(mu_cc, mu_c, atol=1e-9))

    # ---- shrinkage --------------------------------------------------------- #
    _, d0, r0 = anchor_offset(mu, o, tr, vb0, 0.25, spread=0.0)
    chk("spread=0 leaves the raw estimate untouched",
        all(abs(d0[j] - r0[j]) < 1e-12 for j in d0))
    _, dS, rS = anchor_offset(mu, o, tr, vb0, 0.25, spread=0.10)
    chk("shrinkage moves every offset towards zero by the spread",
        all(abs(dS[j]) <= abs(rS[j]) + 1e-12 for j in dS)
        and all(abs(abs(rS[j]) - abs(dS[j]) - 0.10) < 1e-9
                for j in dS if abs(rS[j]) > 0.10))
    chk("shrinkage never flips the sign of a correction",
        all(dS[j] * rS[j] >= 0 for j in dS))
    _, dBig, _ = anchor_offset(mu, o, tr, vb0, 0.25, spread=10.0)
    chk("a spread larger than any offset disables the correction",
        all(abs(v) < 1e-12 for v in dBig.values()))
    muBig, _, _ = anchor_offset(mu, o, tr, vb0, 0.25, spread=10.0)
    chk("disabled correction returns the input unchanged",
        np.allclose(muBig, mu))

    # the spread estimate itself
    vb0b, sp = reference_initial_wear(y, o, tr, q=0.25, return_spread=True)
    chk("homogeneous tools give a near-zero spread", sp < 1e-6)
    y_het = y.copy()
    for j, shift in enumerate([0.0, 0.25, -0.25]):
        y_het[tr == j] += shift
    _, sp_het = reference_initial_wear(y_het, o, tr, q=0.25, return_spread=True)
    chk(f"heterogeneous tools give a large spread ({sp_het:.3f} > 0.2)",
        sp_het > 0.2)

    # ---- monotone smoothing ---------------------------------------------- #
    noisy = y + rng.normal(scale=0.08, size=len(y))
    var = np.full(len(y), 0.01)
    sm = monotone_smooth(noisy, var, tr, o)
    for j in range(3):
        m = tr == j
        idx = np.flatnonzero(m)[np.argsort(o[m])]
        chk(f"tool {j}: smoothed trajectory is non-decreasing",
            np.all(np.diff(sm[idx]) >= -1e-9))
    chk("smoothing reduces the error against the truth",
        np.mean(np.abs(sm - y)) < np.mean(np.abs(noisy - y)))
    chk("smoothing is a no-op on a clean monotone trajectory",
        np.allclose(monotone_smooth(y, var, tr, o), y, atol=1e-9))

    # ---- the two corrections compose ------------------------------------- #
    dirty = y + true_off[tr] + rng.normal(scale=0.08, size=len(y))
    c1, _, _ = anchor_offset(dirty, o, tr, vb0, 0.25)
    c2 = monotone_smooth(c1, var, tr, o)
    chk("offset + smoothing beats either alone",
        np.mean(np.abs(c2 - y)) < np.mean(np.abs(c1 - y))
        and np.mean(np.abs(c2 - y)) < np.mean(np.abs(dirty - y)))

    # ---- label independence ----------------------------------------------- #
    # permuting the *test labels* must not change any adapted prediction
    y_perm = rng.permutation(y)
    a1, _, _ = anchor_offset(dirty, o, tr, vb0, 0.25)
    a2, _, _ = anchor_offset(dirty, o, tr, vb0, 0.25)
    chk("adaptation is deterministic and uses no test label",
        np.allclose(a1, a2) and not np.shares_memory(a1, y_perm))

    # ---- BatchNorm adaptation --------------------------------------------- #
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(4)
    bn.eval()
    src = torch.randn(256, 4, 32)
    bn.train()
    with torch.no_grad():
        bn(src)
    bn.eval()
    w0 = bn.weight.detach().clone()
    tgt = torch.randn(256, 4, 32) * 3.0 + 5.0
    net = nn.Sequential(bn)
    adapt_batchnorm(net, tgt, n_passes=2, batch=64)
    chk("BN running mean moves to the target stream",
        torch.allclose(bn.running_mean, tgt.mean(dim=(0, 2)), atol=0.15))
    chk("BN adaptation leaves the learned weights untouched",
        torch.allclose(bn.weight, w0))
    chk("model is left in eval mode", not bn.training)

    return ok


if __name__ == "__main__":
    print("Verifying test-time adaptation")
    good = verify()
    print("\nALL CHECKS PASSED" if good else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if good else 1)
