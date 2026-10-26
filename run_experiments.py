"""
run_experiments.py
------------------
Resumable experiment driver.  Every (dataset, method, label-fraction, seed)
cell is written to results/raw/<key>.json as soon as it finishes, so the study
can be interrupted and restarted without losing work.

Usage
-----
    python src/run_experiments.py --suite main
    python src/run_experiments.py --suite ablation
    python src/run_experiments.py --suite labelsweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

import data_nasa
import data_phm
import data_nuaa
from splits import group_holdout, sample_labels
from model import mc_predict, deterministic_predict, count_params
from uncertainty import (mc_classification_uncertainty, normalised_epistemic,
                         mc_regression_uncertainty, expected_calibration_error)
import methods as M

ROOT = os.path.dirname(os.path.abspath(__file__))
# Data and cached encoders are large; when they are not present beside this
# code they are read from the sibling codes_3 folder (set UCTCL_ASSETS to
# override).
ASSETS = os.environ.get("UCTCL_ASSETS") or (
    ROOT if os.path.isdir(os.path.join(ROOT, "data"))
    else os.path.join(os.path.dirname(os.path.dirname(ROOT)), "codes_3"))
RAW = os.path.join(ROOT, "results", "raw")
SWEEPS = os.path.join(ROOT, "results", "sweeps")
os.makedirs(RAW, exist_ok=True)
os.makedirs(SWEEPS, exist_ok=True)

# --------------------------------------------------------------------------- #
# dataset configuration
# --------------------------------------------------------------------------- #
DS_CFG = {
    "nasa": dict(win=512, stride=512, in_ch=6, n_test_groups=4,
                 width=24, depth=3, pool=4, stem_stride=2),
    "phm":  dict(win=64, stride=24, in_ch=7, n_test_groups=1,
                 width=24, depth=3, pool=2, stem_stride=1),
    # stem_stride=2 leaves 8 temporal positions after the encoder, matching the
    # effective resolution of the other two benchmarks (NASA 4, PHM 8)
    "nuaa": dict(win=128, stride=48, in_ch=8, n_test_groups=1,
                 width=24, depth=3, pool=2, stem_stride=2),
}
# datasets whose held-out fold is one whole tool, cycled by seed
LOTO = {"phm": 3, "nuaa": 9}

_CACHE = {}


def get_dataset(name):
    if name in _CACHE:
        return _CACHE[name]
    if name == "nasa":
        runs = data_nasa.load_runs(os.path.join(ASSETS, "data", "mill.mat"))
        X, y, c, g, tp, rid = data_nasa.build_windows(
            runs, win=DS_CFG["nasa"]["win"], stride=DS_CFG["nasa"]["stride"])
    elif name == "phm":
        X, y, c, g, tp, rid = data_phm.build_windows(
            os.path.join(ASSETS, "data", "phm2010_bundle.csv"),
            win=DS_CFG["phm"]["win"], stride=DS_CFG["phm"]["stride"])
    else:
        X, y, c, g, tp, rid = data_nuaa.build_windows(
            os.path.join(ASSETS, "data", "nuaa_bundle.csv"),
            win=DS_CFG["nuaa"]["win"], stride=DS_CFG["nuaa"]["stride"])
    _CACHE[name] = (X, y, c, g, tp, rid)
    return _CACHE[name]


def prepare_split(name, seed, label_frac):
    X, y, c, g, tp, rid = get_dataset(name)
    dc = DS_CFG[name]
    if name in LOTO:
        held = sorted(np.unique(g))[seed % LOTO[name]]
        te = np.flatnonzero(g == held)
        tr = np.flatnonzero(g != held)
    else:
        tr, te, _ = group_holdout(g, dc["n_test_groups"], seed, stages=c)

    # channel standardisation fitted on the training tools only
    mu = X[tr].mean(axis=(0, 2), keepdims=True)
    sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-8
    Xn = ((X - mu) / sd).astype(np.float32)

    # wear normalised to [0, 1] using the training range only
    y_lo, y_hi = float(y[tr].min()), float(y[tr].max())
    y_n = ((y - y_lo) / max(1e-8, y_hi - y_lo)).astype(np.float32)

    # machining-time position normalised per trajectory (for the TCL kernel)
    tpos = np.zeros_like(tp, dtype=np.float32)
    for gg in np.unique(g):
        m = g == gg
        lo, hi = tp[m].min(), tp[m].max()
        tpos[m] = (tp[m] - lo) / max(1e-8, hi - lo)

    # physical stage boundaries expressed in the normalised wear scale, so the
    # threshold-consistent head can be evaluated in the model's own units
    edges_mm = np.asarray(
        {"nasa": data_nasa.WEAR_EDGES, "phm": data_phm.WEAR_EDGES,
         "nuaa": data_nuaa.WEAR_EDGES}[name], dtype=np.float64)
    edges_norm = (edges_mm - y_lo) / max(1e-8, y_hi - y_lo)

    # break-in wear level of a fresh insert, measured on the TRAINING tools
    # only; a single scalar used by the test-time anchor correction
    from tta import reference_initial_wear
    vb0, vb0_spread = reference_initial_wear(y_n[tr], tp[tr], g[tr], q=0.25,
                                             return_spread=True)

    lab_local = sample_labels(rid[tr], c[tr], label_frac, seed)
    lab_all = tr[lab_local]
    unl_idx = tr[~lab_local]

    # Protocol note.  With a sub-10 % budget a held-out validation slice would
    # contain only a handful of windows drawn from two or three cutting passes;
    # in a pilot study on development seeds (5-7, never used for any reported
    # number) checkpoint selection on such a slice was dominated by its own
    # sampling noise and was worse than no selection at all for every method.
    # We therefore spend the whole budget on fitting and give every method the
    # same fixed number of epochs, chosen once on those development seeds.
    lab_idx = lab_all
    val_idx = np.array([], dtype=np.int64)
    return dict(X=torch.tensor(Xn), y_reg=y_n, y_cls=c, groups=g, tpos=tpos,
                traj=g, order=tp, tr=tr, te=te, lab=lab_idx, unl=unl_idx,
                val=val_idx, lab_all=lab_all,
                y_raw=y, y_lo=y_lo, y_hi=y_hi, rid=rid, dc=dc,
                edges_mm=edges_mm, edges_norm=edges_norm, vb0=vb0,
                vb0_spread=vb0_spread)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def _present(yt):
    """
    Stages actually visited by the held-out tool.

    Macro-F1 averages per-stage F1, so a stage absent from a fold would
    contribute a hard zero and penalise every method by an amount that depends
    only on which tool was held out.  We therefore average over the stages the
    fold contains.  On NASA Milling and PHM 2010 every fold contains all four
    by construction, so this is a no-op there; on NUAA, whose nine tools cover
    different parts of the wear range, it is what makes folds comparable.
    """
    return sorted(int(k) for k in np.unique(yt))


def evaluate(model, S, n_mc=20, fuse=True, alpha=0.5):
    from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix)
    from uncertainty import regression_stage_probs, fuse_probs
    te = S["te"]
    Xte = S["X"][te]
    probs, mu, lv = mc_predict(model, Xte, n_mc=n_mc)
    p_cls, u_tot, u_ale, u_epi = mc_classification_uncertainty(probs)
    mu_bar, v_ale, v_epi, v_tot = mc_regression_uncertainty(mu, lv)

    if fuse:
        p_reg = regression_stage_probs(mu, lv, S["edges_norm"])
        p_bar = fuse_probs(p_cls, p_reg, alpha)
    else:
        p_bar = p_cls

    yt = S["y_cls"][te]
    pred = p_bar.argmax(1)
    scale = S["y_hi"] - S["y_lo"]
    y_mm = S["y_raw"][te]
    pred_mm = mu_bar * scale + S["y_lo"]

    lab_present = _present(yt)
    ece, _ = expected_calibration_error(p_bar, yt)
    rmse = float(np.sqrt(np.mean((pred_mm - y_mm) ** 2)))
    mae = float(np.mean(np.abs(pred_mm - y_mm)))
    ss_res = float(np.sum((pred_mm - y_mm) ** 2))
    ss_tot = float(np.sum((y_mm - y_mm.mean()) ** 2))
    r2 = 1.0 - ss_res / max(1e-12, ss_tot)

    return dict(
        macro_f1=float(f1_score(yt, pred, average="macro",
                                labels=lab_present, zero_division=0)),
        weighted_f1=float(f1_score(yt, pred, average="weighted", zero_division=0)),
        accuracy=float(accuracy_score(yt, pred)),
        balanced_acc=float(balanced_accuracy_score(yt, pred)),
        kappa=float(cohen_kappa_score(yt, pred)),
        n_stages_present=len(lab_present),
        per_class_f1=f1_score(yt, pred, average=None,
                              labels=list(range(4)), zero_division=0).tolist(),
        confusion=confusion_matrix(yt, pred, labels=list(range(4))).tolist(),
        ece=float(ece),
        rmse_mm=rmse, mae_mm=mae, r2=r2,
        mean_epistemic=float(normalised_epistemic(u_epi, 4).mean()),
        mean_aleatoric=float(u_ale.mean()),
        n_test=int(len(te)),
    )


def _cls_metrics(yt, pred, probs, pred_mm, y_mm):
    from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix)
    lab_present = _present(yt)
    ece, _ = expected_calibration_error(probs, yt)
    ss_res = float(np.sum((pred_mm - y_mm) ** 2))
    ss_tot = float(np.sum((y_mm - y_mm.mean()) ** 2))
    return dict(
        macro_f1=float(f1_score(yt, pred, average="macro",
                                labels=lab_present, zero_division=0)),
        weighted_f1=float(f1_score(yt, pred, average="weighted", zero_division=0)),
        accuracy=float(accuracy_score(yt, pred)),
        balanced_acc=float(balanced_accuracy_score(yt, pred)),
        kappa=float(cohen_kappa_score(yt, pred)),
        per_class_f1=f1_score(yt, pred, average=None, labels=list(range(4)),
                              zero_division=0).tolist(),
        confusion=confusion_matrix(yt, pred, labels=list(range(4))).tolist(),
        ece=float(ece),
        rmse_mm=float(np.sqrt(np.mean((pred_mm - y_mm) ** 2))),
        mae_mm=float(np.mean(np.abs(pred_mm - y_mm))),
        r2=1.0 - ss_res / max(1e-12, ss_tot))


def evaluate_tta_all(model, S, n_mc=10, alpha=0.5, cache_key=None):
    """
    Online protocol: the model may read the held-out tool's unlabelled stream
    (never its labels) before predicting.

    All variants of the correction stack share a single Monte-Carlo sweep, so
    the ablation is almost free; only the BatchNorm variant needs its own sweep
    because it changes the forward pass itself.
    """
    import copy as _copy
    from tta import mc_forward, postprocess

    te = S["te"]
    Xte, traj, order = S["X"][te], S["traj"][te], S["order"][te]
    yt, y_mm = S["y_cls"][te], S["y_raw"][te]
    scale, lo = S["y_hi"] - S["y_lo"], S["y_lo"]

    sweep = mc_forward(_copy.deepcopy(model), Xte, n_mc=n_mc, use_bn=False)
    # Cache the Monte-Carlo sweep.  Everything in the online protocol is a pure
    # function of this array, so any later change to the correction stack can be
    # re-evaluated without retraining a single model.
    if cache_key is not None:
        np.savez_compressed(os.path.join(SWEEPS, cache_key + ".npz"),
                            probs=sweep[0].astype(np.float32),
                            mu=sweep[1].astype(np.float32),
                            lv=sweep[2].astype(np.float32))
    variants = {
        "tta":           dict(),
        "tta_no_anchor": dict(use_anchor=False),
        "tta_no_mono":   dict(use_mono=False),
        "tta_no_fuse":   dict(use_fuse=False),
    }
    out = {}
    for name, kw in variants.items():
        probs, mu_s, info = postprocess(*sweep, traj, order, S["edges_norm"],
                                        S["vb0"], alpha=alpha,
                                        spread=S["vb0_spread"], **kw)
        out[name] = _cls_metrics(yt, probs.argmax(1), probs,
                                 mu_s * scale + lo, y_mm)
        if name == "tta":
            out[name]["offsets_mm"] = {int(k): float(v * scale)
                                       for k, v in info["offsets"].items()}
            out[name]["raw_offsets_mm"] = {int(k): float(v * scale)
                                           for k, v in info["raw_offsets"].items()}
            out[name]["spread_mm"] = float(S["vb0_spread"] * scale)
    # standard test-time BatchNorm re-estimation, for contrast
    sweep_bn = mc_forward(_copy.deepcopy(model), Xte, n_mc=n_mc, use_bn=True)
    probs, mu_s, _ = postprocess(*sweep_bn, traj, order, S["edges_norm"],
                                 S["vb0"], alpha=alpha, spread=S["vb0_spread"])
    out["tta_bn"] = _cls_metrics(yt, probs.argmax(1), probs,
                                 mu_s * scale + lo, y_mm)
    return out


def eval_classical(kind, S):
    """RandomForest / SVM on the 17-descriptor-per-channel feature bank."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix)

    Xa = S["X"].numpy()
    fs = {6: data_nasa.FS, 7: data_phm.FS, 8: data_nuaa.FS}[Xa.shape[1]]
    F = data_nasa.handcrafted_features(Xa, fs=fs)
    lab, te = S["lab_all"], S["te"]
    yt = S["y_cls"][te]
    scale = S["y_hi"] - S["y_lo"]

    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=0, n_jobs=4)
        reg = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=4)
    else:
        clf = make_pipeline(StandardScaler(),
                            SVC(C=10.0, gamma="scale", class_weight="balanced",
                                probability=True, random_state=0))
        reg = make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale"))

    clf.fit(F[lab], S["y_cls"][lab])
    reg.fit(F[lab], S["y_reg"][lab])
    pred = clf.predict(F[te])
    prob = clf.predict_proba(F[te])
    pred_mm = reg.predict(F[te]) * scale + S["y_lo"]
    y_mm = S["y_raw"][te]
    ece, _ = expected_calibration_error(prob, yt)
    ss_res = float(np.sum((pred_mm - y_mm) ** 2))
    ss_tot = float(np.sum((y_mm - y_mm.mean()) ** 2))
    return dict(
        macro_f1=float(f1_score(yt, pred, average="macro",
                                labels=_present(yt), zero_division=0)),
        weighted_f1=float(f1_score(yt, pred, average="weighted", zero_division=0)),
        accuracy=float(accuracy_score(yt, pred)),
        balanced_acc=float(balanced_accuracy_score(yt, pred)),
        kappa=float(cohen_kappa_score(yt, pred)),
        n_stages_present=len(_present(yt)),
        per_class_f1=f1_score(yt, pred, average=None, labels=list(range(4)),
                              zero_division=0).tolist(),
        confusion=confusion_matrix(yt, pred, labels=list(range(4))).tolist(),
        ece=float(ece),
        rmse_mm=float(np.sqrt(np.mean((pred_mm - y_mm) ** 2))),
        mae_mm=float(np.mean(np.abs(pred_mm - y_mm))),
        r2=1.0 - ss_res / max(1e-12, ss_tot),
        mean_epistemic=float("nan"), mean_aleatoric=float("nan"),
        n_test=int(len(te)),
    )


VERSION = "j2"          # bump whenever the preprocessing or protocol changes


def key(ds, method, seed, frac):
    return f"{VERSION}__{ds}__{method}__f{frac:.3f}__s{seed}"



# --------------------------------------------------------------------------- #
# method dispatch
# --------------------------------------------------------------------------- #
ABLATIONS = {
    "uctcl":          {},
    "uctcl_fusion":   dict(use_fusion=True),
    "uctcl_agree":    dict(use_fusion=True, require_agreement=True),
    "uctcl_relabel":  dict(mono_relabel=True),
    "uctcl_noTCL":    dict(use_tcl=False),
    "uctcl_simclr":   dict(temporal_weight=False),
    "uctcl_noMC":     dict(use_mc=False),
    "uctcl_globalTau": dict(per_class_tau=False),
    "uctcl_noMono":   dict(use_mono=False),
    "uctcl_noCB":     dict(class_balanced=False),
    # ---- v5 ----
    "mbpl":        dict(mono_bracket=True, path_decode=True),   # proposed
    "mbpl_nopath": dict(mono_bracket=True),                     # brackets, no joint decoding
    "mbpl_noanch": dict(path_decode=True),                      # decoding, no brackets
    "mbpl_cert":   dict(mono_bracket=True, cert_only=True),     # certified windows only
}


def build_cfg(ds, overrides=None, method=None):
    cfg = dict(M.DEFAULT)
    dc = DS_CFG[ds]
    cfg.update(width=dc["width"], depth=dc["depth"], pool=dc["pool"],
               stem_stride=dc["stem_stride"])
    # threshold-consistent fusion is a component of the proposed method, so the
    # published baselines are evaluated with their own discriminative head.
    # `sup_low_fuse` isolates how much of our gain fusion alone can explain.
    if method is not None and not (method.startswith(("uctcl", "mbpl"))
                                   or method.endswith("_fuse")):
        cfg["use_fusion"] = False
        cfg["require_agreement"] = False
    if overrides:
        cfg.update(overrides)
    return cfg


def run_one(ds, method, seed, label_frac):
    S = prepare_split(ds, seed, label_frac)
    K, in_ch = 4, DS_CFG[ds]["in_ch"]
    t0 = time.time()

    if method in ("rf", "svm"):
        met = eval_classical(method, S)
        extra, nparams = {}, 0
    else:
        cfg = build_cfg(ds, ABLATIONS.get(method), method=method)
        args = (S["X"], S["y_cls"], S["y_reg"])

        from pretrain import load_pretrained

        def pre_loader(model, mode, _ds=ds, _seed=seed, _n=cfg["e_con"]):
            return load_pretrained(model, _ds, _seed, mode, _n)

        V = S["val"]
        if method in ("sup_low", "sup_low_fuse"):
            m, extra = M.run_sup(*args, S["lab"], cfg, seed, K, in_ch, V)
        elif method == "sup_full":
            # the upper bound sees ~15x more samples per epoch, so the epoch
            # count is reduced to keep the number of gradient steps comparable
            trn = np.setdiff1d(S["tr"], V)
            cfg = dict(cfg)
            cfg["e_warm"], cfg["n_rounds"], cfg["e_round"] = 30, 0, 0
            m, extra = M.run_sup(*args, trn, cfg, seed, K, in_ch, V)
        elif method == "pseudo":
            m, extra = M.run_pseudo_fixed(*args, S["lab"], S["unl"], cfg, seed,
                                          K, in_ch, V)
        elif method == "fixmatch":
            m, extra = M.run_fixmatch(*args, S["lab"], S["unl"], cfg, seed,
                                      K, in_ch, V)
        elif method == "meanteacher":
            m, extra = M.run_meanteacher(*args, S["lab"], S["unl"], cfg, seed,
                                         K, in_ch, V)
        elif method == "simclr":
            m, extra = M.run_contrastive_then_head(
                *args, S["lab"], cfg, seed, K, in_ch, False, S["traj"], S["tpos"],
                True, pre_loader, V)
        elif method == "tcl_ft":
            m, extra = M.run_contrastive_then_head(
                *args, S["lab"], cfg, seed, K, in_ch, True, S["traj"], S["tpos"],
                False, pre_loader, V)
        elif method.startswith(("uctcl", "mbpl")):
            m, extra = M.run_uctcl(*args, S["lab"], S["unl"], cfg, seed, K, in_ch,
                                   S["traj"], S["tpos"], S["order"], pre_loader,
                                   S["edges_norm"], V)
        else:
            raise ValueError(method)
        met = evaluate(m, S, n_mc=cfg["n_mc"], fuse=cfg["use_fusion"],
                       alpha=cfg["alpha_fuse"])
        # Same trained model, second protocol: adaptation on the target stream.
        # Ablated variants of the stack quantify what each correction buys.
        met.update(evaluate_tta_all(m, S, n_mc=cfg["n_mc"],
                                    cache_key=key(ds, method, seed, label_frac)))
        nparams = count_params(m)

    met.update(
        dataset=ds, method=method, seed=int(seed), label_frac=float(label_frac),
        n_labelled=int(len(S["lab_all"])), n_fit=int(len(S["lab"])),
        n_val=int(len(S["val"])), n_unlabelled=int(len(S["unl"])),
        n_train=int(len(S["tr"])),
        actual_label_pct=100.0 * len(S["lab_all"]) / max(1, len(S["tr"])),
        labelled_class_counts=np.bincount(S["y_cls"][S["lab_all"]],
                                          minlength=4).tolist(),
        train_class_counts=np.bincount(S["y_cls"][S["tr"]], minlength=4).tolist(),
        test_class_counts=np.bincount(S["y_cls"][S["te"]], minlength=4).tolist(),
        params=int(nparams), seconds=round(time.time() - t0, 1), extra=extra,
    )
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    ap.add_argument("--datasets", default="nasa,phm,nuaa")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--frac", type=float, default=0.08)
    ap.add_argument("--budget", type=float, default=1e9, help="seconds")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    dss = args.datasets.split(",")

    if args.suite == "main":
        jobs = [(d, m, s, args.frac)
                for d in dss
                for s in seeds
                for m in ["rf", "svm", "sup_low", "pseudo",
                          "meanteacher", "fixmatch", "simclr", "tcl_ft",
                          "uctcl", "sup_full"]]
    elif args.suite == "ablation":
        jobs = [(d, m, s, args.frac)
                for d in dss for s in seeds
                for m in ["uctcl_noTCL", "uctcl_simclr", "uctcl_noMC",
                          "uctcl_globalTau", "uctcl_noMono", "uctcl_noCB",
                          "uctcl_fusion", "uctcl_agree", "uctcl_relabel"]]
    elif args.suite == "v5":
        jobs = [(d, m, s, args.frac)
                for d in dss for s in seeds
                for m in ["mbpl", "mbpl_nopath", "mbpl_noanch", "mbpl_cert"]]
    elif args.suite == "v5sweep":
        jobs = [(d, m, s, f)
                for f in [0.02] for d in dss for s in seeds
                for m in ["mbpl_nopath"]]
    elif args.suite == "labelsweep":
        # budget-major order: the scarcest-label point is the most
        # informative, so it completes first if the run is interrupted
        jobs = [(d, m, s, f)
                for f in [0.02, 0.05, 0.20]
                for d in dss for s in seeds
                for m in ["sup_low", "pseudo", "uctcl"]]
    else:
        raise SystemExit("unknown suite")

    t_start = time.time()
    done = skipped = 0
    for ds, m, s, f in jobs:
        k = key(ds, m, s, f)
        path = os.path.join(RAW, k + ".json")
        if os.path.exists(path):
            skipped += 1
            continue
        if time.time() - t_start > args.budget:
            print(f"budget reached; {done} done, {skipped} skipped", flush=True)
            return
        try:
            res = run_one(ds, m, s, f)
            with open(path, "w") as fh:
                json.dump(res, fh, indent=1)
            print(f"{k}: F1={res['macro_f1']:.3f} RMSE={res['rmse_mm']:.4f} "
                  f"({res['seconds']}s)", flush=True)
            done += 1
        except Exception as e:                                # pragma: no cover
            import traceback
            print(f"FAIL {k}: {e}", flush=True)
            traceback.print_exc()
    print(f"suite {args.suite} complete: {done} new, {skipped} cached", flush=True)


if __name__ == "__main__":
    main()
