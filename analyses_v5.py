"""
analyses_v5.py -- every number quoted in the version_5 paper.

Reads results/raw (300 v4 cells + 150 v5 cells) and writes
results/facts_v5.json.  Nothing here trains a model except --example, which
re-runs one cell to dump per-window trajectories for the figures.

    python src/analyses_v5.py            # tables + statistics
    python src/analyses_v5.py --example  # trajectory dumps for the figures
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R
import monopath
from splits import sample_labels

DS = ["nasa", "phm", "nuaa"]
PROPOSED = "mbpl_nopath"
LIMITED = ["rf", "svm", "sup_low", "pseudo", "meanteacher", "fixmatch",
           "simclr", "tcl_ft", "uctcl", PROPOSED]
DEEP = LIMITED[2:]
RES = os.path.join(R.ROOT, "results")


def load(frac):
    out = {}
    for f in glob.glob(os.path.join(RES, "raw", f"j2__*__f{frac:.3f}__s*.json")):
        d = json.load(open(f))
        out[(d["dataset"], d["method"], d["seed"])] = d
    return out


def val(r, metric, online):
    if r is None:
        return np.nan
    if online:
        return r["tta"][metric] if "tta" in r else np.nan
    return r[metric]


def ms(x):
    x = np.asarray(x, float)
    return [float(np.nanmean(x)), float(np.nanstd(x, ddof=1))]


def evenness(counts):
    c = np.asarray(counts, float)
    if c.sum() == 0:
        return 0.0
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(c)))


def main():
    A = load(0.08)
    F = dict(n_cells=len(A))
    # ---------------- main table ---------------- #
    main_t = {}
    for d in DS:
        main_t[d] = {}
        for m in LIMITED + ["sup_full", "mbpl", "mbpl_noanch", "mbpl_cert"]:
            rs = [A.get((d, m, s)) for s in range(10)]
            assert all(r is not None for r in rs), (d, m)
            main_t[d][m] = dict(
                f1_ind=ms([val(r, "macro_f1", 0) for r in rs]),
                f1_onl=ms([val(r, "macro_f1", 1) for r in rs]),
                rmse_ind=ms([val(r, "rmse_mm", 0) for r in rs]),
                rmse_onl=ms([val(r, "rmse_mm", 1) for r in rs]),
                mae_ind=ms([val(r, "mae_mm", 0) for r in rs]),
                mae_onl=ms([val(r, "mae_mm", 1) for r in rs]))
        main_t[d]["label_pct"] = ms([A[(d, PROPOSED, s)]["actual_label_pct"] for s in range(10)])
    F["main"] = main_t

    def vec(m, metric, online, dss=DS):
        return np.array([val(A.get((d, m, s)), metric, online) for d in dss for s in range(10)])

    # ---------------- pooled means ---------------- #
    F["pooled"] = {m: dict(f1_ind=float(vec(m, "macro_f1", 0).mean()),
                           f1_onl=float(np.nanmean(vec(m, "macro_f1", 1))))
                   for m in LIMITED + ["sup_full", "mbpl", "mbpl_noanch", "mbpl_cert"]}

    # ---------------- ranks ---------------- #
    ranks = {}
    for name, metric, online, meths, sign in [
            ("f1_ind", "macro_f1", 0, LIMITED, -1), ("f1_onl", "macro_f1", 1, DEEP, -1),
            ("rmse_ind", "rmse_mm", 0, LIMITED, 1), ("rmse_onl", "rmse_mm", 1, DEEP, 1)]:
        M = np.stack([vec(m, metric, online) for m in meths], 1)
        rk = np.vstack([rankdata(sign * row) for row in M])
        st, p = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
        ranks[name] = dict(mean_rank=dict(zip(meths, rk.mean(0).tolist())),
                           friedman_stat=float(st), friedman_p=float(p),
                           n_runs=int(M.shape[0]), k=len(meths))
    F["ranks"] = ranks

    # ---------------- paired tests, proposed vs every baseline ------------- #
    tests = {}
    for b in [m for m in LIMITED if m != PROPOSED] + ["mbpl", "mbpl_noanch", "mbpl_cert"]:
        tests[b] = {}
        for name, metric, online in [("f1_ind", "macro_f1", 0), ("f1_onl", "macro_f1", 1),
                                     ("rmse_ind", "rmse_mm", 0), ("rmse_onl", "rmse_mm", 1)]:
            if online and b in ("rf", "svm"):
                continue
            a, c = vec(PROPOSED, metric, online), vec(b, metric, online)
            # RMSE lives on different scales per benchmark -> relative change
            dlt = (a - c) / c if metric == "rmse_mm" else a - c
            better = (dlt < 0) if metric == "rmse_mm" else (dlt > 0)
            tests[b][name] = dict(mean_delta=float(dlt.mean()),
                                  wins=int(better.sum()), n=int(len(dlt)),
                                  p=float(wilcoxon(dlt).pvalue),
                                  dz=float(dlt.mean() / dlt.std(ddof=1)))
        per = {}
        for d in DS:
            a, c = vec(PROPOSED, "rmse_mm", 1, [d]), vec(b, "rmse_mm", 1, [d])
            if np.isnan(c).any():
                continue
            per[d] = dict(rel=float(((a - c) / c).mean()), p=float(wilcoxon(a - c).pvalue),
                          wins=int((a < c).sum()))
        tests[b]["rmse_onl_per_ds"] = per
    F["tests"] = tests

    # ---------------- pseudo-label mechanics (final round) ----------------- #
    pl = {}
    for d in DS:
        pl[d] = {}
        for m in ["pseudo", "uctcl", PROPOSED, "mbpl_cert"]:
            cov, prec, ev, cert, certp, tmae, nmae = [], [], [], [], [], [], []
            for s in range(10):
                r = A[(d, m, s)]
                g = r["extra"]["pl"][-1]
                nu = r["n_unlabelled"]
                cov.append(100.0 * g["n"] / nu)
                prec.append(100.0 * g["precision"] if g["n"] else np.nan)
                ev.append(evenness(g["per_class"]))
                if "n_certified" in g:
                    S = R.prepare_split(d, s, 0.08)
                    sc = S["y_hi"] - S["y_lo"]
                    cert.append(100.0 * g["n_certified"] / nu)
                    certp.append(100.0 * g["cert_precision"])
                    tmae.append(g["wear_target_mae"] * sc)
                    nmae.append(g["wear_net_mae"] * sc)
            e = dict(coverage=ms(cov), precision=ms(prec), evenness=ms(ev))
            if cert:
                e.update(certified=ms(cert), cert_precision=ms(certp),
                         target_mae_mm=ms(tmae), net_mae_mm=ms(nmae))
            pl[d][m] = e
    F["pl"] = pl

    # ---------------- label budget 2 % ---------------- #
    B = load(0.02)
    bud = {}
    for d in DS:
        bud[d] = {}
        for m in ["sup_low", "pseudo", "uctcl", PROPOSED]:
            rs = [B.get((d, m, s)) for s in range(10)]
            assert all(r is not None for r in rs), (d, m)
            bud[d][m] = dict(f1_ind=ms([val(r, "macro_f1", 0) for r in rs]),
                             f1_onl=ms([val(r, "macro_f1", 1) for r in rs]),
                             rmse_ind=ms([val(r, "rmse_mm", 0) for r in rs]),
                             rmse_onl=ms([val(r, "rmse_mm", 1) for r in rs]))
        bud[d]["label_pct"] = ms([B[(d, PROPOSED, s)]["actual_label_pct"] for s in range(10)])
        g = [B[(d, PROPOSED, s)] for s in range(10)]
        bud[d]["certified"] = ms([100.0 * r["extra"]["pl"][-1]["n_certified"] / r["n_unlabelled"] for r in g])
    bt = {}
    for b in ["sup_low", "pseudo", "uctcl"]:
        for name, metric, online in [("f1_ind", "macro_f1", 0), ("f1_onl", "macro_f1", 1),
                                     ("rmse_ind", "rmse_mm", 0), ("rmse_onl", "rmse_mm", 1)]:
            a = np.array([val(B[(d, PROPOSED, s)], metric, online) for d in DS for s in range(10)])
            c = np.array([val(B[(d, b, s)], metric, online) for d in DS for s in range(10)])
            dlt = (a - c) / c if metric == "rmse_mm" else a - c
            bt.setdefault(b, {})[name] = dict(mean_delta=float(dlt.mean()), p=float(wilcoxon(dlt).pvalue))
    F["budget2"] = dict(per_ds=bud, tests=bt,
                        pooled={m: dict(f1_ind=float(np.mean([val(B[(d, m, s)], "macro_f1", 0) for d in DS for s in range(10)])),
                                        f1_onl=float(np.mean([val(B[(d, m, s)], "macro_f1", 1) for d in DS for s in range(10)])))
                                for m in ["sup_low", "pseudo", "uctcl", PROPOSED]})

    # ---------------- certified share vs inspection budget (no training) --- #
    fr = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    cs = {}
    for d in DS:
        cs[d] = dict(frac=fr, label_pct=[], certified=[], certified_sd=[], width=[], wear_width_mm=[])
        for f in fr:
            cc, ww, lp, yw = [], [], [], []
            for s in range(10):
                S = R.prepare_split(d, s, f)
                tr = S["tr"]
                lab = np.zeros(len(S["y_cls"]), bool); lab[S["lab"]] = True
                lo, hi, ylo, yhi = monopath.monotone_bounds(
                    S["y_cls"][tr], S["y_raw"][tr], lab[tr], S["traj"][tr], S["order"][tr], 4)
                u = ~lab[tr]
                cc.append(100.0 * (lo[u] == hi[u]).mean())
                ww.append(float((hi[u] - lo[u] + 1).mean()))
                lp.append(100.0 * lab[tr].mean())
                yl = np.where(np.isfinite(ylo[u]), ylo[u], S["y_raw"][tr].min())
                yh = np.where(np.isfinite(yhi[u]), yhi[u], S["y_raw"][tr].max())
                yw.append(float((yh - yl).mean()))
                assert (lo[u] <= S["y_cls"][tr][u]).mean() > 0.99
            cs[d]["label_pct"].append(float(np.mean(lp)))
            cs[d]["certified"].append(float(np.mean(cc)))
            cs[d]["certified_sd"].append(float(np.std(cc, ddof=1)))
            cs[d]["width"].append(float(np.mean(ww)))
            cs[d]["wear_width_mm"].append(float(np.mean(yw)))
    F["cert_curve"] = cs

    # ---------------- data set facts ---------------- #
    dsf = {}
    for d in DS:
        X, y, c, g, tp, rid = R.get_dataset(d)
        viol = 0
        for t in np.unique(g):
            i = np.flatnonzero(g == t); i = i[np.argsort(tp[i], kind="mergesort")]
            viol += int((np.diff(c[i]) < 0).sum())
        dsf[d] = dict(n_windows=int(len(y)), n_tools=int(len(np.unique(g))),
                      n_passes=int(len(np.unique(rid))), channels=int(X.shape[1]),
                      win=int(X.shape[2]), wear_min=float(y.min()), wear_max=float(y.max()),
                      class_counts=np.bincount(c, minlength=4).tolist(),
                      stage_order_violations=viol,
                      imbalance=float(np.bincount(c, minlength=4).max() / np.bincount(c, minlength=4).min()))
    F["datasets"] = dsf

    # ---------------- test-time path read-out (negative result) ------------ #
    p = os.path.join(RES, "readout_v5.json")
    if os.path.exists(p):
        ro = json.load(open(p))
        F["readout"] = {m: {d: {k: float(np.mean([ro[f"{d}|{m}|{s}"][k] for s in range(10)]))
                                for k in ["onl", "path", "filt"]} for d in DS}
                        for m in ["uctcl", PROPOSED]}
        for m in ["uctcl", PROPOSED]:
            F["readout"][m]["pooled"] = {k: float(np.mean([ro[f"{d}|{m}|{s}"][k] for d in DS for s in range(10)]))
                                         for k in ["onl", "path", "filt"]}
    F["deploy"] = json.load(open(os.path.join(RES, "analyses.json")))["deploy"]
    json.dump(F, open(os.path.join(RES, "facts_v5.json"), "w"), indent=1)
    print("wrote facts_v5.json")
    return F


def example():
    """Per-window dumps for the trajectory figures (one PHM and one NUAA split)."""
    import methods as M
    from pretrain import load_pretrained
    from tta import postprocess
    out = {}
    for ds, seed in [("phm", 0), ("nuaa", 0)]:
        S = R.prepare_split(ds, seed, 0.08)
        sc, lo0 = S["y_hi"] - S["y_lo"], S["y_lo"]
        # (a) bracket on one training tool
        tr = S["tr"]
        lab = np.zeros(len(S["y_cls"]), bool); lab[S["lab"]] = True
        lo, hi, ylo, yhi = monopath.monotone_bounds(
            S["y_cls"][tr], S["y_raw"][tr], lab[tr], S["traj"][tr], S["order"][tr], 4)
        tool = np.unique(S["traj"][tr])[0]
        i = np.flatnonzero(S["traj"][tr] == tool)
        i = i[np.argsort(S["order"][tr][i], kind="mergesort")]
        e = dict(tool=int(tool), t=S["tpos"][tr][i].tolist(), wear=S["y_raw"][tr][i].tolist(),
                 lab=lab[tr][i].tolist(),
                 ylo=np.where(np.isfinite(ylo[i]), ylo[i], np.nan).tolist(),
                 yhi=np.where(np.isfinite(yhi[i]), yhi[i], np.nan).tolist(),
                 lo=lo[i].tolist(), hi=hi[i].tolist(), stage=S["y_cls"][tr][i].tolist(),
                 edges_mm=S["edges_mm"].tolist())
        # (b) held-out tool tracking from the cached sweeps
        te = S["te"]
        o = np.argsort(S["order"][te], kind="mergesort")
        h = dict(t=S["tpos"][te][o].tolist(), wear=S["y_raw"][te][o].tolist())
        for m in ["sup_low", "uctcl", PROPOSED]:
            z = np.load(os.path.join(R.SWEEPS, R.key(ds, m, seed, 0.08) + ".npz"))
            sw = (z["probs"].astype(float), z["mu"].astype(float), z["lv"].astype(float))
            _, mu_s, _ = postprocess(*sw, S["traj"][te], S["order"][te], S["edges_norm"],
                                     S["vb0"], alpha=0.5, spread=S["vb0_spread"])
            h[m + "_onl"] = (mu_s[o] * sc + lo0).tolist()
            h[m + "_ind"] = (sw[1].mean(0)[o] * sc + lo0).tolist()
        out[ds] = dict(train=e, test=h)
    json.dump(out, open(os.path.join(RES, "example_v5.json"), "w"))
    print("wrote example_v5.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--example", action="store_true")
    a = ap.parse_args()
    example() if a.example else main()
