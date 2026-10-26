"""
monopath.py
-----------
Monotone bracketing and monotone-path decoding of wear stages (new in v5).

Physical premise
----------------
Flank wear on one cutting edge never decreases, so along a tool life the stage
sequence c_1 <= c_2 <= ... <= c_n is non-decreasing and is fully described by
at most K-1 change points.  Two consequences are used here.

(1) Monotone bracketing.  A sparse set of inspected passes on a tool bounds the
    stage of every other window of the same tool:

        lo_i = max{ c_a : a labelled, t_a <= t_i }      (0 if none)
        hi_i = min{ c_b : b labelled, t_b >= t_i }      (K-1 if none)

    Whenever lo_i == hi_i the stage of window i is *certified* by physics
    alone, with no network involved.  The same argument brackets the wear
    value between the two neighbouring inspections.

(2) Monotone-path decoding.  For the remaining windows the network's stage
    probabilities are decoded jointly over the whole tool life, restricted to
    non-decreasing paths that respect the brackets.  The MAP path (Viterbi)
    and the path-posterior marginals (forward-backward) are both exact and
    cost O(nK^2).

All functions are pure NumPy.  `verify()` checks them against brute force.
"""
from __future__ import annotations

import itertools
import numpy as np

NEG = -1e12


def _lse(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def monotone_bounds(stage, wear, labelled, traj, order, K):
    """
    Brackets implied by the labelled windows of each trajectory.

    Returns lo, hi (int stage bounds) and ylo, yhi (wear bounds; -inf / +inf
    where no inspection exists on that side).  Labelled windows get
    lo == hi == their own stage.
    """
    n = len(stage)
    lo = np.zeros(n, dtype=np.int64)
    hi = np.full(n, K - 1, dtype=np.int64)
    ylo = np.full(n, -np.inf)
    yhi = np.full(n, np.inf)
    for tr in np.unique(traj):
        idx = np.flatnonzero(traj == tr)
        idx = idx[np.argsort(order[idx], kind="mergesort")]
        L = labelled[idx]
        c = stage[idx]
        y = wear[idx].astype(np.float64)
        l = np.maximum.accumulate(np.where(L, c, 0))
        h = np.minimum.accumulate(np.where(L, c, K - 1)[::-1])[::-1]
        yl = np.maximum.accumulate(np.where(L, y, -np.inf))
        yh = np.minimum.accumulate(np.where(L, y, np.inf)[::-1])[::-1]
        # measured wear is not perfectly monotone on every data set; if two
        # inspections contradict each other the bracket is widened to contain
        # both rather than trusted
        bad = l > h
        l2, h2 = np.where(bad, h, l), np.where(bad, l, h)
        bady = yl > yh
        yl2, yh2 = np.where(bady, yh, yl), np.where(bady, yl, yh)
        lo[idx], hi[idx], ylo[idx], yhi[idx] = l2, h2, yl2, yh2
    lo[labelled] = stage[labelled]
    hi[labelled] = stage[labelled]
    return lo, hi, ylo, yhi


def monotone_path(logp, lo=None, hi=None):
    """
    Exact inference over non-decreasing stage paths.

    logp : (n, K) per-window log-probabilities in machining-time order
    lo, hi : optional (n,) inclusive feasibility brackets

    Returns
    -------
    path : (n,) MAP non-decreasing path
    q    : (n, K) path-posterior marginals (rows sum to one)
    """
    n, K = logp.shape
    e = logp.astype(np.float64).copy()
    if lo is not None:
        ks = np.arange(K)[None, :]
        e[(ks < lo[:, None]) | (ks > hi[:, None])] = NEG
    # ---- Viterbi ---------------------------------------------------------- #
    d = np.empty((n, K))
    bp = np.zeros((n, K), dtype=np.int64)
    d[0] = e[0]
    for i in range(1, n):
        # best predecessor j <= k is a running arg-max
        run = np.maximum.accumulate(d[i - 1])
        arg = np.zeros(K, dtype=np.int64)
        for k in range(1, K):
            arg[k] = k if d[i - 1, k] >= run[k - 1] else arg[k - 1]
        bp[i] = arg
        d[i] = e[i] + run
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(d[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    # ---- forward-backward -------------------------------------------------- #
    a = np.empty((n, K))
    b = np.zeros((n, K))
    a[0] = e[0]
    for i in range(1, n):
        a[i] = e[i] + np.logaddexp.accumulate(a[i - 1])
    for i in range(n - 2, -1, -1):
        s = e[i + 1] + b[i + 1]
        b[i] = np.logaddexp.accumulate(s[::-1])[::-1]
    lq = a + b
    lq -= _lse(lq, axis=1)[:, None]
    return path, np.exp(lq)


def monotone_filter(logp):
    """
    Causal counterpart of `monotone_path`: the stage estimate at window i uses
    windows 1..i only (forward recursion), so it can run while the tool cuts.
    Returns the filtered marginals (n, K).
    """
    n, K = logp.shape
    a = np.empty((n, K))
    a[0] = logp[0]
    for i in range(1, n):
        a[i] = logp[i] + np.logaddexp.accumulate(a[i - 1])
    a -= _lse(a, axis=1)[:, None]
    return np.exp(a)


def decode_trajectories(p, traj, order, lo=None, hi=None, temperature=1.0):
    """Apply `monotone_path` per trajectory; returns path (N,), q (N, K)."""
    N, K = p.shape
    path = np.zeros(N, dtype=np.int64)
    q = np.zeros((N, K))
    logp = np.log(np.clip(p, 1e-12, 1.0)) / temperature
    for tr in np.unique(traj):
        idx = np.flatnonzero(traj == tr)
        idx = idx[np.argsort(order[idx], kind="mergesort")]
        pa, qq = monotone_path(logp[idx],
                               None if lo is None else lo[idx],
                               None if hi is None else hi[idx])
        path[idx], q[idx] = pa, qq
    return path, q


def filter_trajectories(p, traj, order, temperature=1.0):
    N, K = p.shape
    q = np.zeros((N, K))
    logp = np.log(np.clip(p, 1e-12, 1.0)) / temperature
    for tr in np.unique(traj):
        idx = np.flatnonzero(traj == tr)
        idx = idx[np.argsort(order[idx], kind="mergesort")]
        q[idx] = monotone_filter(logp[idx])
    return q


def verify(seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # brute force over every non-decreasing path of length n
    for trial in range(20):
        n, K = 6, 4
        lp = np.log(rng.dirichlet(np.ones(K), size=n))
        lo = np.zeros(n, dtype=int); hi = np.full(n, K - 1)
        if trial % 2:
            j = rng.integers(1, n - 1); c = rng.integers(0, K)
            lo[j:] = np.maximum(lo[j:], c); hi[:j + 1] = np.minimum(hi[:j + 1], c)
        best, bestp, Z, marg = -np.inf, None, 0.0, np.zeros((n, K))
        for pth in itertools.product(range(K), repeat=n):
            pth = np.array(pth)
            if (np.diff(pth) < 0).any() or (pth < lo).any() or (pth > hi).any():
                continue
            s = lp[np.arange(n), pth].sum()
            if s > best:
                best, bestp = s, pth
            w = np.exp(s); Z += w
            marg[np.arange(n), pth] += w
        path, q = monotone_path(lp, lo, hi)
        if not (np.isclose(lp[np.arange(n), path].sum(), best)
                and np.allclose(q, marg / Z, atol=1e-8)
                and (np.diff(path) >= 0).all()
                and (path >= lo).all() and (path <= hi).all()):
            ok = False
    chk("Viterbi and marginals match brute force on 20 random problems", ok)

    # bracketing
    stage = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    wear = np.linspace(0.0, 0.9, 10)
    lab = np.zeros(10, bool); lab[[2, 4, 8]] = True
    lo, hi, ylo, yhi = monotone_bounds(stage, wear, lab, np.zeros(10, int),
                                       np.arange(10.0), 4)
    chk("window between two equal-stage inspections is certified",
        lo[3] == hi[3] == 1)
    chk("brackets always contain the truth",
        (lo <= stage).all() and (stage <= hi).all()
        and (ylo <= wear + 1e-12).all() and (wear <= yhi + 1e-12).all())
    chk("no inspection before -> lower bound 0", lo[0] == 0 and hi[0] == 1)
    chk("no inspection after -> upper bound K-1", hi[9] == 3 and lo[9] == 3)

    # the filter at the last window equals the smoother at the last window
    lp = np.log(rng.dirichlet(np.ones(4), size=30))
    _, q = monotone_path(lp)
    f = monotone_filter(lp)
    chk("filter == smoother at the final window", np.allclose(f[-1], q[-1]))
    chk("filtered rows are distributions", np.allclose(f.sum(1), 1.0))
    return ok


if __name__ == "__main__":
    print("monopath checks:")
    raise SystemExit(0 if verify() else 1)
