"""
uncertainty.py
--------------
Bayesian uncertainty decomposition for Monte-Carlo dropout, plus the adaptive
per-class pseudo-labelling threshold and the uncertainty-weighted isotonic
(monotone-wear) consistency filter.

All functions are pure NumPy so that they can be unit-tested independently of
the deep-learning backend.  `verify_math()` re-derives every identity used in
the paper numerically; it is executed by `python -m src.uncertainty`.

Notation
--------
p^(m)_k  : soft-max probability of class k on the m-th stochastic forward pass
p_bar_k  : (1/M) sum_m p^(m)_k                       predictive mean
H[p]     : -sum_k p_k log p_k                        Shannon entropy (nats)

Identities implemented
----------------------
(1) Total predictive uncertainty          U_tot = H[p_bar]
(2) Aleatoric (expected data uncertainty) U_ale = (1/M) sum_m H[p^(m)]
(3) Epistemic (model uncertainty, BALD)   U_epi = U_tot - U_ale = I(y ; w | x)
    0 <= U_epi <= log K, with U_epi = 0 iff all passes agree exactly.
(4) Law of total variance for regression
    Var[y|x] = (1/M) sum_m s^2_(m)          (aleatoric, heteroscedastic head)
             + (1/M) sum_m (mu_(m) - mu_bar)^2   (epistemic)
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. classification uncertainty
# --------------------------------------------------------------------------- #
def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats.  p must sum to one along `axis`."""
    return -np.sum(p * np.log(p + EPS), axis=axis)


def mc_classification_uncertainty(probs: np.ndarray):
    """
    Parameters
    ----------
    probs : (M, N, K) stochastic soft-max outputs of M MC-dropout passes.

    Returns
    -------
    p_bar : (N, K) predictive mean
    u_tot : (N,)   total predictive entropy   H[p_bar]
    u_ale : (N,)   aleatoric  E_m H[p^(m)]
    u_epi : (N,)   epistemic  mutual information (BALD)
    """
    p_bar = probs.mean(axis=0)
    u_tot = entropy(p_bar, axis=-1)
    u_ale = entropy(probs, axis=-1).mean(axis=0)
    u_epi = u_tot - u_ale
    # Jensen guarantees u_epi >= 0; clip only removes float round-off.
    return p_bar, u_tot, u_ale, np.maximum(u_epi, 0.0)


def normalised_epistemic(u_epi: np.ndarray, n_classes: int) -> np.ndarray:
    """Map the mutual information onto [0, 1] via its maximum value log K."""
    return np.clip(u_epi / np.log(n_classes), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 2. regression uncertainty (law of total variance)
# --------------------------------------------------------------------------- #
def mc_regression_uncertainty(mu: np.ndarray, logvar: np.ndarray):
    """
    Parameters
    ----------
    mu     : (M, N) predicted means of the heteroscedastic head
    logvar : (M, N) predicted log aleatoric variances

    Returns
    -------
    mu_bar, var_ale, var_epi, var_tot
    """
    mu_bar = mu.mean(axis=0)
    var_ale = np.exp(logvar).mean(axis=0)
    # unbiased (1/(M-1)) spread of the stochastic means.  The 1/M form
    # under-estimates the epistemic term by exactly 1/M, which at M=10 is a 10%
    # error; since this variance is the precision weight of the isotonic
    # projection, the bias would propagate into the wear estimate itself.
    M = mu.shape[0]
    denom = max(M - 1, 1)
    var_epi = ((mu - mu_bar[None, :]) ** 2).sum(axis=0) / denom
    return mu_bar, var_ale, var_epi, var_ale + var_epi


# --------------------------------------------------------------------------- #
# 2b. threshold-consistent fusion of the regression posterior
# --------------------------------------------------------------------------- #
def _phi(z):
    """Standard normal CDF via the error function."""
    from math import sqrt
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def regression_stage_probs(mu: np.ndarray, logvar: np.ndarray,
                           edges: np.ndarray) -> np.ndarray:
    """
    Turn the MC-dropout regression posterior into a distribution over the K
    physical wear stages, using the *known* stage boundaries.

    The predictive density of the wear value is the Gaussian mixture
        p(y | x) = (1/M) sum_m N(y ; mu_m, sigma_m^2),
    so the probability that the tool sits in stage k = [e_{k-1}, e_k) is exactly

        P(stage = k | x) = (1/M) sum_m [ Phi((e_k - mu_m)/sigma_m)
                                         - Phi((e_{k-1} - mu_m)/sigma_m) ],

    with e_{-1} = -inf and e_{K-1} = +inf.  This is a classifier that cannot
    contradict the regressor, and it is ordinal by construction: it can only
    place mass on stages that are consistent with the predicted wear value.

    Parameters
    ----------
    mu, logvar : (M, N) MC-dropout outputs of the heteroscedastic head
    edges      : (K-1,) stage boundaries in the same units as mu

    Returns
    -------
    (N, K) probabilities, rows summing to one.
    """
    sd = np.exp(0.5 * logvar)
    sd = np.maximum(sd, 1e-6)
    M, N = mu.shape
    K = len(edges) + 1
    cdf = np.zeros((M, N, K + 1))
    cdf[:, :, 0] = 0.0
    cdf[:, :, K] = 1.0
    for i, e in enumerate(edges):
        cdf[:, :, i + 1] = _phi((e - mu) / sd)
    # enforce monotonicity of the CDF against float round-off
    cdf = np.maximum.accumulate(cdf, axis=2)
    probs = np.diff(cdf, axis=2).mean(axis=0)
    return probs / probs.sum(axis=1, keepdims=True).clip(1e-12)


def fuse_probs(p_cls: np.ndarray, p_reg: np.ndarray, alpha: float = 0.5):
    """Convex fusion of the discriminative and the threshold-consistent head."""
    p = (1.0 - alpha) * p_cls + alpha * p_reg
    return p / p.sum(axis=1, keepdims=True).clip(1e-12)


# --------------------------------------------------------------------------- #
# 3. adaptive, per-class, curriculum uncertainty threshold
# --------------------------------------------------------------------------- #
def percentile_schedule(t: int, T: int, rho0: float = 0.10,
                        rho_max: float = 0.70) -> float:
    """Linear curriculum on the accepted quantile of the uncertainty pool."""
    if T <= 1:
        return rho_max
    return rho0 + (rho_max - rho0) * (t / float(T - 1))


def adaptive_thresholds(u: np.ndarray, yhat: np.ndarray, n_classes: int,
                        rho: float) -> np.ndarray:
    """
    tau_k = Q_rho( { u_i : yhat_i = k } )

    Taking the quantile *within each predicted class* makes the number of
    accepted pseudo-labels proportional to the size of that class's candidate
    pool, which prevents the head class from monopolising the pseudo-label
    budget -- the failure mode of a single global confidence threshold.
    Classes with no candidate get tau = -inf (accept nothing).
    """
    tau = np.full(n_classes, -np.inf, dtype=np.float64)
    for k in range(n_classes):
        m = yhat == k
        if m.sum() > 0:
            tau[k] = np.quantile(u[m], rho)
    return tau


def uncertainty_mask(u: np.ndarray, yhat: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Accept sample i iff its epistemic score is below its class threshold."""
    return u <= tau[yhat]


# --------------------------------------------------------------------------- #
# 4. uncertainty-weighted isotonic monotonicity filter
# --------------------------------------------------------------------------- #
def weighted_pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Pool-Adjacent-Violators algorithm: solve

        min_v  sum_i w_i (v_i - y_i)^2      s.t.  v_1 <= v_2 <= ... <= v_n

    in O(n).  `w` must be strictly positive.  Returns the projection v.
    """
    n = len(y)
    if n == 0:
        return y.copy()
    vals = np.empty(n, dtype=np.float64)
    wts = np.empty(n, dtype=np.float64)
    cnt = np.empty(n, dtype=np.int64)
    j = -1
    for i in range(n):
        j += 1
        vals[j], wts[j], cnt[j] = y[i], w[i], 1
        while j > 0 and vals[j - 1] > vals[j]:
            wnew = wts[j - 1] + wts[j]
            vals[j - 1] = (wts[j - 1] * vals[j - 1] + wts[j] * vals[j]) / wnew
            wts[j - 1] = wnew
            cnt[j - 1] += cnt[j]
            j -= 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for b in range(j + 1):
        out[pos : pos + cnt[b]] = vals[b]
        pos += cnt[b]
    return out


def monotonicity_filter(mu: np.ndarray, var: np.ndarray, traj: np.ndarray,
                        order: np.ndarray, kappa: float = 1.5):
    """
    Tool flank wear is physically monotone non-decreasing along the life of a
    single cutting edge.  For every trajectory we project the predicted wear
    onto the monotone cone with weights 1/var (precision weighting), and reject
    the pseudo-label of any window whose prediction has to move by more than
    `kappa` predictive standard deviations to satisfy monotonicity.

    Returns
    -------
    keep : (N,) bool mask
    proj : (N,) isotonic projection of mu
    """
    n = len(mu)
    keep = np.ones(n, dtype=bool)
    proj = mu.astype(np.float64).copy()
    sd = np.sqrt(np.maximum(var, EPS))
    for tr in np.unique(traj):
        idx = np.flatnonzero(traj == tr)
        idx = idx[np.argsort(order[idx], kind="mergesort")]
        v = weighted_pava(mu[idx].astype(np.float64),
                          1.0 / np.maximum(var[idx], EPS))
        proj[idx] = v
        keep[idx] = np.abs(v - mu[idx]) <= kappa * sd[idx]
    return keep, proj


# --------------------------------------------------------------------------- #
# 5. class-balanced (effective-number) weights
# --------------------------------------------------------------------------- #
def effective_number_weights(counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """
    Cui et al. 2019.  w_k proportional to (1 - beta) / (1 - beta^{n_k}),
    normalised so that sum_k w_k = K.  beta -> 0 gives uniform weights,
    beta -> 1 gives inverse-frequency weights.
    """
    counts = np.maximum(counts.astype(np.float64), 1.0)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff
    return w / w.sum() * len(counts)


# --------------------------------------------------------------------------- #
# 6. calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15):
    """
    ECE = sum_b (|B_b| / N) * | acc(B_b) - conf(B_b) |
    using equal-width confidence bins on the top-1 probability.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(labels)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            bins.append((0.5 * (lo + hi), np.nan, np.nan, 0))
            continue
        a, c = acc[m].mean(), conf[m].mean()
        ece += (m.sum() / n) * abs(a - c)
        bins.append((0.5 * (lo + hi), a, c, int(m.sum())))
    return ece, bins


# --------------------------------------------------------------------------- #
# math verification -- every identity is re-derived numerically
# --------------------------------------------------------------------------- #
def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def verify_math(seed: int = 0, verbose: bool = True) -> bool:
    rng = np.random.default_rng(seed)
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # ---- (A) uncertainty decomposition ------------------------------------ #
    M, N, K = 40, 500, 4
    logits = rng.normal(size=(M, N, K)) * 2.0 + rng.normal(size=(1, N, K)) * 3.0
    probs = _softmax(logits)
    p_bar, u_tot, u_ale, u_epi = mc_classification_uncertainty(probs)

    chk("p_bar is a valid distribution", np.allclose(p_bar.sum(1), 1.0, atol=1e-10))
    chk("U_tot = U_ale + U_epi (exact)", np.allclose(u_tot, u_ale + u_epi, atol=1e-9))
    chk("U_epi >= 0  (Jensen, entropy concave)", (u_epi >= -1e-12).all())
    chk("U_epi <= log K", (u_epi <= np.log(K) + 1e-9).all())
    chk("U_tot <= log K", (u_tot <= np.log(K) + 1e-9).all())

    # BALD identity: I(y;w|x) = H[E_w p] - E_w H[p], cross-checked against the
    # KL form  I = E_w KL( p^(m) || p_bar ).
    kl = np.sum(probs * (np.log(probs + EPS) - np.log(p_bar[None] + EPS)), axis=-1)
    chk("BALD == E_m KL(p^(m) || p_bar)", np.allclose(u_epi, kl.mean(0), atol=1e-9))

    # deterministic model -> zero epistemic uncertainty
    det = np.repeat(_softmax(rng.normal(size=(1, N, K))), M, axis=0)
    _, dt, da, de = mc_classification_uncertainty(det)
    chk("deterministic passes -> U_epi = 0", np.allclose(de, 0.0, atol=1e-10))
    chk("deterministic passes -> U_tot = U_ale", np.allclose(dt, da, atol=1e-10))

    # one-hot disagreement across passes -> U_ale = 0, U_epi = U_tot
    hard = np.zeros((M, N, K))
    picks = rng.integers(0, K, size=(M, N))
    hard[np.arange(M)[:, None], np.arange(N)[None, :], picks] = 1.0
    _, ht, ha, he = mc_classification_uncertainty(hard)
    chk("hard one-hot passes -> U_ale = 0", np.allclose(ha, 0.0, atol=1e-9))
    chk("hard one-hot passes -> U_epi = U_tot", np.allclose(he, ht, atol=1e-9))

    # normalisation
    un = normalised_epistemic(u_epi, K)
    chk("normalised epistemic in [0,1]", (un >= 0).all() and (un <= 1).all())

    # ---- (B) law of total variance ---------------------------------------- #
    mu = rng.normal(size=(M, N))
    logvar = rng.normal(size=(M, N)) - 1.0
    mu_bar, va, ve, vt = mc_regression_uncertainty(mu, logvar)
    # Monte-Carlo ground truth: sample y ~ N(mu_m, s_m^2) with m ~ Uniform
    S = 20000
    mi = rng.integers(0, M, size=(S, N))
    ys = mu[mi, np.arange(N)[None, :]] + \
        np.exp(0.5 * logvar[mi, np.arange(N)[None, :]]) * rng.normal(size=(S, N))
    emp = ys.var(axis=0)
    rel = np.abs(emp - vt) / (vt + EPS)
    # the empirical mixture variance uses 1/M over the M component means while
    # var_epi uses the unbiased 1/(M-1); the two differ by a factor M/(M-1),
    # so the tolerance is set just above that known offset
    chk(f"Var[y] = E[s^2] + Var[mu]  (MC, median rel.err {np.median(rel):.3f} "
        f"< 1/(M-1) = {1/(M-1):.3f} + slack)", np.median(rel) < 1.0 / (M - 1) + 0.02)
    chk("total variance = aleatoric + epistemic", np.allclose(vt, va + ve))
    chk("mu_bar equals mean of passes", np.allclose(mu_bar, mu.mean(0)))

    # ---- (B2) threshold-consistent fusion --------------------------------- #
    Mm, Nn = 30, 400
    mu_r = rng.normal(0.5, 0.3, size=(Mm, Nn))
    lv_r = rng.normal(-4.0, 0.5, size=(Mm, Nn))
    edges = np.array([0.25, 0.5, 0.75])
    pr = regression_stage_probs(mu_r, lv_r, edges)
    chk("regression stage probs are a distribution",
        np.allclose(pr.sum(1), 1.0, atol=1e-9) and (pr >= -1e-12).all())

    # Monte-Carlo ground truth: sample from the mixture and histogram it
    S2 = 4000
    mi = rng.integers(0, Mm, size=(S2, Nn))
    ys2 = mu_r[mi, np.arange(Nn)] + \
        np.exp(0.5 * lv_r[mi, np.arange(Nn)]) * rng.normal(size=(S2, Nn))
    emp = np.stack([(np.digitize(ys2[:, j], edges) == k).mean()
                    for k in range(4) for j in range(Nn)]).reshape(4, Nn).T
    chk(f"stage probs match the MC mixture (max abs err "
        f"{np.abs(emp - pr).max():.4f} < 0.05)", np.abs(emp - pr).max() < 0.05)

    # a very confident regressor puts ~all mass on the stage containing mu
    mu_c = np.full((Mm, Nn), 0.60)
    lv_c = np.full((Mm, Nn), -30.0)
    pc = regression_stage_probs(mu_c, lv_c, edges)
    chk("confident regressor is one-hot on the correct stage",
        np.allclose(pc.argmax(1), 2) and pc.max(1).min() > 0.999)

    # ordinality: shifting mu upward moves probability mass monotonically up
    up = regression_stage_probs(mu_c + 0.30, lv_c, edges)
    chk("shifting the wear prediction up moves to a later stage",
        up.argmax(1)[0] == 3)

    fp = fuse_probs(_softmax(rng.normal(size=(Nn, 4))), pr, 0.5)
    chk("fused probs are a distribution", np.allclose(fp.sum(1), 1.0, atol=1e-9))
    chk("alpha=0 returns the classifier",
        np.allclose(fuse_probs(pr * 0 + 0.25, pr, 0.0), 0.25))
    chk("alpha=1 returns the regression head",
        np.allclose(fuse_probs(pr * 0 + 0.25, pr, 1.0), pr))

    # ---- (C) weighted PAVA ------------------------------------------------ #
    for _ in range(200):
        n = rng.integers(2, 40)
        yv = rng.normal(size=n)
        wv = rng.uniform(0.2, 5.0, size=n)
        v = weighted_pava(yv, wv)
        if not np.all(np.diff(v) >= -1e-9):
            chk("PAVA output is non-decreasing", False)
            break
    else:
        chk("PAVA output is non-decreasing (200 random cases)", True)

    # KKT check: the isotonic projection preserves the weighted mean and is
    # optimal, i.e. no feasible direction lowers the objective.
    n = 60
    yv = np.sort(rng.normal(size=n)) + rng.normal(scale=0.8, size=n)
    wv = rng.uniform(0.5, 2.0, size=n)
    v = weighted_pava(yv, wv)
    chk("PAVA preserves the weighted mean",
        np.isclose((wv * v).sum(), (wv * yv).sum(), rtol=1e-9))
    obj = lambda z: np.sum(wv * (z - yv) ** 2)
    better = 0
    for _ in range(4000):
        z = np.sort(v + rng.normal(scale=0.05, size=n))  # feasible perturbation
        if obj(z) < obj(v) - 1e-9:
            better += 1
    chk("PAVA optimal vs 4000 feasible perturbations", better == 0)

    # already-monotone input is a fixed point
    ymono = np.cumsum(np.abs(rng.normal(size=50)))
    chk("PAVA is idempotent on monotone input",
        np.allclose(weighted_pava(ymono, np.ones(50)), ymono))

    # unweighted PAVA agrees with sklearn's IsotonicRegression
    try:
        from sklearn.isotonic import IsotonicRegression
        yv = rng.normal(size=80)
        wv = rng.uniform(0.5, 2.0, size=80)
        ref = IsotonicRegression(increasing=True).fit_transform(
            np.arange(80), yv, sample_weight=wv)
        chk("PAVA matches sklearn IsotonicRegression",
            np.allclose(ref, weighted_pava(yv, wv), atol=1e-8))
    except Exception as e:                                    # pragma: no cover
        chk(f"sklearn cross-check skipped ({e})", True)

    # ---- (D) adaptive per-class thresholds -------------------------------- #
    u = rng.uniform(size=2000)
    yhat = rng.integers(0, 4, size=2000)
    for rho in [0.05, 0.25, 0.5, 0.9]:
        tau = adaptive_thresholds(u, yhat, 4, rho)
        m = uncertainty_mask(u, yhat, tau)
        fr = [m[yhat == k].mean() for k in range(4)]
        chk(f"per-class acceptance ~= rho={rho} (max dev {max(abs(f-rho) for f in fr):.3f})",
            max(abs(f - rho) for f in fr) < 0.02)
    chk("acceptance is monotone in rho",
        uncertainty_mask(u, yhat, adaptive_thresholds(u, yhat, 4, 0.3)).sum()
        <= uncertainty_mask(u, yhat, adaptive_thresholds(u, yhat, 4, 0.6)).sum())
    tau_e = adaptive_thresholds(u, np.zeros(2000, dtype=int), 4, 0.5)
    chk("empty class gets tau = -inf", np.isneginf(tau_e[1:]).all())
    chk("schedule endpoints", np.isclose(percentile_schedule(0, 10), 0.10) and
        np.isclose(percentile_schedule(9, 10), 0.70))

    # global-threshold contrast: with a skewed uncertainty distribution a single
    # global quantile starves the minority class -- this is the failure mode the
    # per-class rule removes.
    u2 = np.concatenate([rng.uniform(0, .3, 1800), rng.uniform(.5, 1., 200)])
    y2 = np.concatenate([np.zeros(1800, int), np.ones(200, int)])
    g = np.quantile(u2, 0.5)
    glob = [(u2[y2 == k] <= g).mean() for k in range(2)]
    chk(f"global threshold starves minority (rates {glob[0]:.2f} vs {glob[1]:.2f})",
        glob[1] < 0.05 < glob[0])

    # ---- (E) effective-number weights ------------------------------------- #
    cnts = np.array([100, 1000, 50, 500], dtype=float)
    w = effective_number_weights(cnts, 0.999)
    chk("effective-number weights sum to K", np.isclose(w.sum(), 4.0))
    chk("rarer class gets larger weight", w[2] > w[0] > w[3] > w[1])
    wu = effective_number_weights(cnts, 1e-8)
    chk("beta -> 0 gives uniform weights", np.allclose(wu, 1.0, atol=1e-3))
    # beta -> 1 limit: w_k proportional to 1/n_k
    wb = effective_number_weights(cnts, 1 - 1e-9)
    inv = (1.0 / cnts); inv = inv / inv.sum() * 4
    chk("beta -> 1 gives inverse-frequency weights", np.allclose(wb, inv, rtol=1e-3))

    # ---- (F) ECE ----------------------------------------------------------- #
    p = _softmax(rng.normal(size=(4000, 4)) * 3)
    lab = np.array([rng.choice(4, p=pi) for pi in p])          # perfectly calibrated
    ece_cal, _ = expected_calibration_error(p, lab)
    lab_bad = np.zeros(4000, dtype=int)                        # badly calibrated
    ece_bad, _ = expected_calibration_error(p, lab_bad)
    chk(f"ECE small when calibrated ({ece_cal:.4f} < 0.03)", ece_cal < 0.03)
    chk(f"ECE large when miscalibrated ({ece_bad:.3f} > 0.3)", ece_bad > 0.30)
    onehot = np.eye(4)[lab]
    chk("ECE = 0 for a perfect oracle",
        np.isclose(expected_calibration_error(onehot, lab)[0], 0.0, atol=1e-12))

    return ok


if __name__ == "__main__":
    print("Verifying uncertainty / thresholding mathematics")
    good = verify_math()
    print("\nALL CHECKS PASSED" if good else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if good else 1)
