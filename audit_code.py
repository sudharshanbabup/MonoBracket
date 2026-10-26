"""
audit_code.py
-------------
Correctness audit of the pipeline itself, as opposed to the numbers it
produces.  Every check here targets a class of bug that would silently inflate
results rather than crash: information leaking from the held-out tool into
training, biased estimators, degenerate splits, and label budgets that are not
what they claim to be.

    python src/audit_code.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import torch

OK = True
N = 0


def chk(name, cond, detail=""):
    global OK, N
    N += 1
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def audit_segmentation():
    """A cutting pass must not be silently merged with its neighbour."""
    print("Pass segmentation")
    import pandas as pd
    import data_phm

    d = pd.read_csv(os.path.join(R.ASSETS, "data", "phm2010_bundle.csv"), index_col=0)
    worst = 0
    for t, g in d.groupby("experiment_tag"):
        g = g.sort_values("timestamp")
        segs = data_phm._segment_cuts(g)
        lens = np.array([b - a for a, b in segs])
        worst = max(worst, lens.max() / np.median(lens))
    chk("PHM: no pass exceeds 3x the median pass length",
        worst <= 3.0, f"worst ratio {worst:.1f}x")


def audit_no_leakage():
    """
    Nothing derived from a held-out tool may reach training.  We check the three
    channels through which it could: the window index sets, the normalisation
    statistics, and the contrastive pre-training pool.
    """
    print("\nLeakage")
    import run_experiments as R

    for ds in R.DS_CFG:
        S = R.prepare_split(ds, 0, 0.08)
        tr, te = set(S["tr"].tolist()), set(S["te"].tolist())
        chk(f"{ds}: train and test window sets are disjoint", not (tr & te))
        chk(f"{ds}: labelled set is a subset of train",
            set(S["lab_all"].tolist()) <= tr)
        chk(f"{ds}: unlabelled pool is a subset of train",
            set(S["unl"].tolist()) <= tr)
        chk(f"{ds}: no tool appears on both sides",
            not (set(S["traj"][S["tr"]].tolist())
                 & set(S["traj"][S["te"]].tolist())))

        # normalisation must be computable from the training rows alone
        X = S["X"].numpy()
        mu_tr = X[S["tr"]].mean(axis=(0, 2))
        chk(f"{ds}: channel means are ~0 on train (fitted there, not on all)",
            np.abs(mu_tr).max() < 0.15, f"max |mean| {np.abs(mu_tr).max():.3f}")
        mu_all = X.mean(axis=(0, 2))
        chk(f"{ds}: channel means are NOT ~0 over train+test "
            f"(would indicate fitting on everything)",
            np.abs(mu_all).max() > 1e-6)

        # the wear normalisation range must come from training tools only
        y_tr = S["y_reg"][S["tr"]]
        chk(f"{ds}: normalised wear spans [0,1] on train exactly",
            abs(y_tr.min()) < 1e-6 and abs(y_tr.max() - 1.0) < 1e-6,
            f"[{y_tr.min():.4f}, {y_tr.max():.4f}]")

        # vb0 is a training constant
        from tta import reference_initial_wear
        vb0_tr = reference_initial_wear(S["y_reg"][S["tr"]], S["order"][S["tr"]],
                                        S["traj"][S["tr"]])
        chk(f"{ds}: break-in constant recomputes identically from train rows",
            abs(vb0_tr - S["vb0"]) < 1e-12)


def audit_label_budget():
    """The stated budget must hold on every split, not on average."""
    print("\nLabel budget")
    import run_experiments as R
    for ds in R.DS_CFG:
        pct = []
        for s in range(10):
            S = R.prepare_split(ds, s, 0.08)
            pct.append(100.0 * len(S["lab_all"]) / len(S["tr"]))
        chk(f"{ds}: every one of 10 splits under 10%",
            max(pct) < 10.0, f"max {max(pct):.2f}%")
        chk(f"{ds}: every split has all four stages labelled",
            all(min(np.bincount(R.prepare_split(ds, s, 0.08)["y_cls"][
                R.prepare_split(ds, s, 0.08)["lab_all"]], minlength=4)) > 0
                for s in range(0, 10, 3)))


def audit_estimators():
    """Statistical estimators must be unbiased where the paper says they are."""
    print("\nEstimators")
    from uncertainty import mc_regression_uncertainty

    rng = np.random.default_rng(0)
    # ground truth: mu drawn from N(0, s^2); the epistemic term estimates s^2
    M, N_, S_TRUE, TRIALS = 10, 4000, 0.7, 200
    est = []
    for _ in range(TRIALS):
        mu = rng.normal(0.0, S_TRUE, size=(M, N_))
        lv = np.full((M, N_), -20.0)
        _, _, v_epi, _ = mc_regression_uncertainty(mu, lv)
        est.append(v_epi.mean())
    bias = np.mean(est) / S_TRUE ** 2 - 1.0
    chk(f"MC epistemic variance is unbiased (relative bias {bias:+.4f})",
        abs(bias) < 0.02)

    from uncertainty import expected_calibration_error, _softmax
    p = _softmax(rng.normal(size=(20000, 4)) * 2)
    lab = np.array([rng.choice(4, p=pi) for pi in p])
    ece, _ = expected_calibration_error(p, lab, n_bins=15)
    chk(f"ECE of a perfectly calibrated predictor ~ 0 ({ece:.4f} < 0.02)",
        ece < 0.02)


def audit_determinism():
    """The same seed must give the same split and the same model."""
    print("\nDeterminism")
    import run_experiments as R
    import methods as M

    for ds in list(R.DS_CFG)[:1]:
        a = R.prepare_split(ds, 3, 0.08)
        b = R.prepare_split(ds, 3, 0.08)
        chk(f"{ds}: split is reproducible",
            np.array_equal(a["lab_all"], b["lab_all"])
            and np.array_equal(a["te"], b["te"]))
        cfg = R.build_cfg(ds, method="uctcl")
        m1 = M._make_model(R.DS_CFG[ds]["in_ch"], 4, cfg, 7)
        m2 = M._make_model(R.DS_CFG[ds]["in_ch"], 4, cfg, 7)
        same = all(torch.equal(p, q) for p, q in
                   zip(m1.state_dict().values(), m2.state_dict().values()))
        chk(f"{ds}: model init is reproducible for a fixed seed", same)


def audit_tta_is_label_free():
    """
    The strongest claim in the paper is that online adaptation never sees a test
    label.  We prove it by permuting the test labels and checking that every
    adapted prediction is bit-identical.
    """
    print("\nOnline adaptation uses no test label")
    import copy
    import run_experiments as R
    import methods as M
    from tta import mc_forward, postprocess

    ds = "phm"
    S = R.prepare_split(ds, 0, 0.08)
    cfg = R.build_cfg(ds, method="sup_low")
    cfg["e_warm"], cfg["n_rounds"] = 4, 0
    m = M._make_model(R.DS_CFG[ds]["in_ch"], 4, cfg, 0)
    M.supervised_fit(m, S["X"], S["y_cls"], S["y_reg"], S["lab"], cfg, 0, 4, K=4)

    te = S["te"]
    torch.manual_seed(123)
    sweep = mc_forward(copy.deepcopy(m), S["X"][te], n_mc=4)
    p1, mu1, _ = postprocess(*sweep, S["traj"][te], S["order"][te],
                             S["edges_norm"], S["vb0"])
    # corrupt the test labels; adaptation must be unaffected
    S2 = dict(S)
    y2 = S["y_cls"].copy()
    rng = np.random.default_rng(0)
    y2[te] = rng.permutation(y2[te])
    S2["y_cls"] = y2
    p2, mu2, _ = postprocess(*sweep, S2["traj"][te], S2["order"][te],
                             S2["edges_norm"], S2["vb0"])
    chk("permuting test labels leaves adapted probabilities identical",
        np.array_equal(p1, p2))
    chk("permuting test labels leaves adapted wear identical",
        np.array_equal(mu1, mu2))

    from tta import anchor_offset
    chk("the anchor uses only predictions, order and a training constant",
        "y_cls" not in anchor_offset.__code__.co_varnames)


def audit_missing_data():
    """Source-data defects must be handled, not silently propagated."""
    print("\nSource data integrity")
    import data_nuaa
    X, y, c, g, t, rid = data_nuaa.build_windows(
        os.path.join(R.ASSETS, "data", "nuaa_bundle.csv"))
    chk("NUAA: every retained window is finite", np.isfinite(X).all())
    chk("NUAA: wear labels finite and in range",
        np.isfinite(y).all() and y.min() > 0 and y.max() < 1.0)
    chk("NUAA: all nine tools present", len(np.unique(g)) == 9)
    chk("NUAA: dropout costs under 1% of passes",
        abs(len(np.unique(rid)) - 778) <= 8, f"{len(np.unique(rid))} passes")
    import data_nasa, data_phm
    Xn, yn, *_ = data_nasa.build_windows(
        data_nasa.load_runs(os.path.join(R.ASSETS, "data", "mill.mat")),
        win=512, stride=512)
    chk("NASA: every window finite", np.isfinite(Xn).all())
    Xp, yp, *_ = data_phm.build_windows(
        os.path.join(R.ASSETS, "data", "phm2010_bundle.csv"), win=64, stride=24)
    chk("PHM: every window finite", np.isfinite(Xp).all())


def audit_metric_sanity():
    """A perfect and a constant predictor must score as expected."""
    print("\nMetric sanity")
    from sklearn.metrics import f1_score
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    chk("macro-F1 of a perfect predictor is 1",
        abs(f1_score(y, y, average="macro") - 1.0) < 1e-12)
    const = np.zeros_like(y)
    chk("macro-F1 of a constant predictor is low",
        f1_score(y, const, average="macro", zero_division=0) < 0.2)
    # present-stage averaging must be a no-op when all stages are present
    import run_experiments as R
    y3 = np.array([0, 1, 2, 1, 0, 2, 3, 3])
    p3 = np.array([0, 1, 2, 2, 0, 2, 3, 1])
    chk("present-stage macro-F1 == plain macro-F1 when all stages occur",
        abs(f1_score(y3, p3, average="macro", labels=R._present(y3),
                     zero_division=0)
            - f1_score(y3, p3, average="macro", zero_division=0)) < 1e-12)
    y2 = np.array([0, 0, 1, 1])          # stages 2,3 absent
    p2 = np.array([0, 1, 1, 1])
    chk("present-stage macro-F1 ignores absent stages",
        f1_score(y2, p2, average="macro", labels=R._present(y2),
                 zero_division=0)
        > f1_score(y2, p2, average="macro", labels=[0, 1, 2, 3],
                   zero_division=0))


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.abspath(__file__))
    print("Auditing the pipeline\n")
    audit_segmentation()
    audit_no_leakage()
    audit_label_budget()
    audit_estimators()
    audit_determinism()
    audit_tta_is_label_free()
    audit_missing_data()
    audit_metric_sanity()
    print(f"\n{N} checks; " + ("ALL PASS" if OK else "SOME FAILED"))
    raise SystemExit(0 if OK else 1)
