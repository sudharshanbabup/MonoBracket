"""
verify_paper.py
---------------
Independent audit of the manuscript.  Re-derives, straight from
results/raw/*.json, every quantitative claim that appears in paper_1/main.tex
and in the generated tables, and reports any mismatch.

    python src/verify_paper.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate import (load, table, matched_table, group, paired_test,
                       pooled_vector, pseudo_label_quality, stat,
                       MAIN_ORDER, TTA_ORDER)

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(ROOT), "paper_1", "main.tex")
V_MAIN, V_ABL = "v8", "v7"

OK = True
CHECKS = 0


def chk(name, cond, detail=""):
    global OK, CHECKS
    CHECKS += 1
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


def main():
    rows = load(V_MAIN)
    abl = load(V_ABL)
    tex = open(PAPER).read()
    print(f"{len(rows)} main + {len(abl)} ablation result files\n")

    g = group(rows, 0.08)

    # ---------------- grid coverage --------------------------------------- #
    print("Experiment grid")
    for ds in ["nasa", "phm"]:
        for m in MAIN_ORDER:
            n = len(g.get((ds, m, 0.08), []))
            chk(f"{ds}/{m}: 10 seeds", n == 10, f"got {n}")
    chk("both protocols recorded for every deep method",
        all("tta" in r for r in rows if r["method"] in TTA_ORDER))

    # ---------------- protocol integrity ---------------------------------- #
    print("\nProtocol")
    for ds in ["nasa", "phm"]:
        pcts = [r["actual_label_pct"] for r in g[(ds, "uctcl", 0.08)]]
        chk(f"{ds}: every split under the 10% label budget",
            max(pcts) < 10.0, f"max {max(pcts):.2f}%")
        chk(f"{ds}: all 4 stages in every test fold",
            all(min(r["test_class_counts"]) > 0 for r in g[(ds, "uctcl", 0.08)]))
        chk(f"{ds}: labelled set covers all 4 stages",
            all(min(r["labelled_class_counts"]) > 0
                for r in g[(ds, "uctcl", 0.08)]))
    pars = {r["params"] for r in rows if r["params"] > 0 and r["dataset"] == "nasa"}
    chk("all deep methods share one encoder size", len(pars) == 1, f"{pars}")

    # every online variant must be present wherever the protocol applies
    need = ["tta", "tta_no_anchor", "tta_no_mono", "tta_no_fuse", "tta_bn"]
    chk("all online variants recorded for every deep method",
        all(all(k in r for k in need) for r in rows
            if r["method"] in TTA_ORDER))

    # ---------------- headline numbers ------------------------------------- #
    print("\nHeadline numbers")
    def cell(ds, m, key, prot=None):
        rs = g[(ds, m, 0.08)]
        return stat([(r[prot][key] if prot else r[key]) for r in rs])[0]

    chk("NASA UC-TCL inductive RMSE = 0.173 mm",
        close(cell("nasa", "uctcl", "rmse_mm"), 0.173),
        f"{cell('nasa','uctcl','rmse_mm'):.4f}")
    chk("NASA UC-TCL online macro-F1 = 0.438",
        close(cell("nasa", "uctcl", "macro_f1", "tta"), 0.438),
        f"{cell('nasa','uctcl','macro_f1','tta'):.4f}")
    chk("PHM UC-TCL online macro-F1 = 0.587",
        close(cell("phm", "uctcl", "macro_f1", "tta"), 0.587),
        f"{cell('phm','uctcl','macro_f1','tta'):.4f}")
    chk("PHM fixed-threshold online macro-F1 = 0.591 (level with ours)",
        close(cell("phm", "pseudo", "macro_f1", "tta"), 0.591),
        f"{cell('phm','pseudo','macro_f1','tta'):.4f}")

    lim = [m for m in MAIN_ORDER if m != "sup_full"]
    chk("NASA: UC-TCL best inductive RMSE among label-limited",
        min(cell("nasa", m, "rmse_mm") for m in lim)
        == cell("nasa", "uctcl", "rmse_mm"))
    lt = [m for m in TTA_ORDER if m != "sup_full"]
    chk("NASA: UC-TCL best online macro-F1 among label-limited",
        max(cell("nasa", m, "macro_f1", "tta") for m in lt)
        == cell("nasa", "uctcl", "macro_f1", "tta"))

    base = min(cell("nasa", m, "rmse_mm") for m in lim if m != "uctcl")
    red_b = 100 * (1 - cell("nasa", "uctcl", "rmse_mm") / base)
    red_s = 100 * (1 - cell("nasa", "uctcl", "rmse_mm")
                   / cell("nasa", "sup_low", "rmse_mm"))
    chk("NASA RMSE 7% below the strongest baseline", abs(red_b - 7) < 1.0,
        f"{red_b:.1f}%")
    chk("NASA RMSE 11% below supervised-only", abs(red_s - 11) < 1.0, f"{red_s:.1f}%")

    chk("NASA label budget 8.5 %",
        close(np.mean([r["actual_label_pct"] for r in g[("nasa", "uctcl", 0.08)]]),
              8.5, 0.06))
    chk("PHM label budget 6.5 %",
        close(np.mean([r["actual_label_pct"] for r in g[("phm", "uctcl", 0.08)]]),
              6.5, 0.06))

    # ---------------- pooled results --------------------------------------- #
    print("\nPooled over 20 runs")
    pi = np.nanmean(pooled_vector(rows, "uctcl", "macro_f1"))
    pt = np.nanmean(pooled_vector(rows, "uctcl", "macro_f1", "tta"))
    chk("UC-TCL pooled inductive macro-F1 = 0.476", close(pi, 0.476), f"{pi:.4f}")
    chk("UC-TCL pooled online macro-F1 = 0.513", close(pt, 0.513), f"{pt:.4f}")
    chk("UC-TCL has the best pooled online macro-F1 among label-limited",
        max(np.nanmean(pooled_vector(rows, m, "macro_f1", "tta"))
            for m in TTA_ORDER if m != "sup_full") == pt)

    # ---------------- significance ----------------------------------------- #
    print("\nSignificance (pooled, n = 20)")
    ours_t = pooled_vector(rows, "uctcl", "macro_f1", "tta")
    ours_i = pooled_vector(rows, "uctcl", "macro_f1")
    expect = {"fixmatch": (0.050, 0.048), "simclr": (0.062, 0.011),
              "tcl_ft": (0.034, 0.017), "meanteacher": (0.289, 0.0)}
    for m, (d_exp, p_exp) in expect.items():
        p, _, d = paired_test(ours_t, pooled_vector(rows, m, "macro_f1", "tta"))
        chk(f"online vs {m}: +{d_exp:.3f} at p={p_exp:.3f}",
            close(d, d_exp, 0.0006) and (p < 0.05) and close(p, p_exp, 0.0015),
            f"d={d:+.4f} p={p:.4f}")
    for m in ["sup_low", "pseudo"]:
        p, _, d = paired_test(ours_t, pooled_vector(rows, m, "macro_f1", "tta"))
        chk(f"online vs {m}: reported as NOT significant", p >= 0.05,
            f"p={p:.3f}")
    # inductive: only fixmatch / meanteacher significant
    for m in ["sup_low", "pseudo", "simclr", "tcl_ft"]:
        p, _, _ = paired_test(ours_i, pooled_vector(rows, m, "macro_f1"))
        chk(f"inductive vs {m}: reported as NOT significant", p >= 0.05,
            f"p={p:.3f}")

    print("\nOnline-protocol gain")
    for m, d_exp, p_exp, sig in [("uctcl", 0.037, 0.015, True),
                                 ("fixmatch", 0.055, 0.040, True),
                                 ("sup_low", 0.027, 0.090, False)]:
        p, _, d = paired_test(pooled_vector(rows, m, "macro_f1", "tta"),
                              pooled_vector(rows, m, "macro_f1"))
        chk(f"{m}: +{d_exp:.3f}, p={p_exp:.3f}",
            close(d, d_exp, 0.0006) and close(p, p_exp, 0.0015)
            and ((p < 0.05) == sig), f"d={d:+.4f} p={p:.4f}")
    gu = (pooled_vector(rows, "uctcl", "macro_f1", "tta")
          - pooled_vector(rows, "uctcl", "macro_f1"))
    gs = (pooled_vector(rows, "sup_low", "macro_f1", "tta")
          - pooled_vector(rows, "sup_low", "macro_f1"))
    p, _, d = paired_test(gu, gs)
    chk("interaction vs supervised: +0.010, reported as NOT significant",
        close(d, 0.010, 0.0006) and p >= 0.05, f"d={d:+.4f} p={p:.4f}")

    # ---------------- pseudo-label mechanism -------------------------------- #
    print("\nPseudo-label mechanism")
    q = pseudo_label_quality(rows, 0.08)
    chk("fixed threshold 5.5% (PHM) / 12.8% (NASA)",
        close(q["phm"]["pseudo"]["coverage"][0], 5.5, 0.05)
        and close(q["nasa"]["pseudo"]["coverage"][0], 12.8, 0.05))
    chk("ours 50.1% on both",
        close(q["phm"]["uctcl"]["coverage"][0], 50.1, 0.05)
        and close(q["nasa"]["uctcl"]["coverage"][0], 50.1, 0.05))
    chk("ours evenness 0.87-0.89",
        0.885 <= q["nasa"]["uctcl"]["evenness"][0] <= 0.895
        and 0.865 <= q["phm"]["uctcl"]["evenness"][0] <= 0.875)
    chk("fixed-threshold evenness 0.17 / 0.43",
        close(q["phm"]["pseudo"]["evenness"][0], 0.17, 0.006)
        and close(q["nasa"]["pseudo"]["evenness"][0], 0.43, 0.006))
    chk("PHM precision 88.3 vs 95.2, cost 6.9 points",
        close(q["phm"]["uctcl"]["precision"][0], 88.3, 0.06)
        and close(q["phm"]["pseudo"]["precision"][0], 95.2, 0.06))
    gap = q["nasa"]["pseudo"]["precision"][0] - q["nasa"]["uctcl"]["precision"][0]
    chk("NASA precision cost 1.6 points", close(gap, 1.6, 0.06), f"{gap:.2f}")

    # ---------------- adaptation ------------------------------------------- #
    print("\nOnline adaptation")
    offs = [abs(v) for ds in ["nasa", "phm"] for r in g[(ds, "uctcl", 0.08)]
            for v in r["tta"]["offsets_mm"].values()]
    chk("median |offset| = 0.077 mm", close(np.median(offs), 0.077, 0.0006),
        f"{np.median(offs):.4f}")
    chk("50 held-out tools contributed an offset", len(offs) == 50, f"{len(offs)}")
    anc = stat([r["tta"]["macro_f1"] for r in g[("phm", "uctcl", 0.08)]])[0] - \
        stat([r["tta_no_anchor"]["macro_f1"] for r in g[("phm", "uctcl", 0.08)]])[0]
    chk("PHM anchor worth +0.034 macro-F1", close(anc, 0.034, 0.0006), f"{anc:+.4f}")
    anc_n = stat([r["tta"]["macro_f1"] for r in g[("nasa", "uctcl", 0.08)]])[0] - \
        stat([r["tta_no_anchor"]["macro_f1"] for r in g[("nasa", "uctcl", 0.08)]])[0]
    chk("NASA anchor is -0.009 (reported as marginally harmful)",
        close(anc_n, -0.009, 0.0006), f"{anc_n:+.4f}")
    bn = stat([r["tta_bn"]["macro_f1"] for r in g[("phm", "uctcl", 0.08)]])[0] - \
        stat([r["tta"]["macro_f1"] for r in g[("phm", "uctcl", 0.08)]])[0]
    chk("BatchNorm re-estimation costs -0.205 on PHM", close(bn, -0.205, 0.0015),
        f"{bn:+.4f}")

    # ---------------- component ablation (v7) ------------------------------- #
    print("\nComponent ablation")
    ab = {ds: {r["method"]: r for r in matched_table(
        abl, ds, ["uctcl_noMC", "uctcl_noCB", "uctcl_noTCL", "uctcl_globalTau",
                  "uctcl"], 0.08)} for ds in ["nasa", "phm"]}
    for ds, m, val in [("nasa", "uctcl_noMC", -0.056), ("phm", "uctcl_noMC", -0.052),
                       ("nasa", "uctcl_noCB", -0.042), ("phm", "uctcl_noCB", -0.063),
                       ("nasa", "uctcl_noTCL", -0.061), ("phm", "uctcl_noTCL", -0.013),
                       ("phm", "uctcl_globalTau", +0.044)]:
        d = ab[ds][m]["macro_f1"][0] - ab[ds]["uctcl"]["macro_f1"][0]
        chk(f"{ds} {m}: dF1 = {val:+.3f}", close(d, val, 0.001), f"{d:+.4f}")

    # ---------------- physical priors --------------------------------------- #
    print("\nPhysical priors and data")
    import data_nasa, data_phm
    import pandas as pd
    runs = data_nasa.load_runs(os.path.join(ROOT, "data", "mill.mat"))
    tot = viol = 0
    for c in sorted({r["case"] for r in runs}):
        rr = sorted([r for r in runs if r["case"] == c and np.isfinite(r["VB"])],
                    key=lambda z: z["time"])
        dd = np.diff([r["VB"] for r in rr])
        tot += len(dd); viol += int((dd < -1e-9).sum())
    chk("NASA wear monotone in 98.5% of consecutive measurements",
        close(100 * (1 - viol / tot), 98.5, 0.05), f"{100*(1-viol/tot):.2f}%")
    df = pd.read_csv(os.path.join(ROOT, "data", "phm2010_bundle.csv"), index_col=0)
    viol = 0
    for _, gg in df.groupby("experiment_tag"):
        wv = gg.sort_values("timestamp").tool_wear.values
        cw = wv[np.concatenate([[0], np.flatnonzero(np.diff(wv) != 0) + 1])]
        viol += int((np.diff(cw) < -1e-9).sum())
    chk("PHM wear monotone in 100%", viol == 0)
    _, _, cn, _, _, _ = data_nasa.build_windows(runs, win=512, stride=512)
    _, _, cp, _, _, _ = data_phm.build_windows(
        os.path.join(ROOT, "data", "phm2010_bundle.csv"), win=64, stride=24)
    bn_, bp_ = np.bincount(cn, minlength=4), np.bincount(cp, minlength=4)
    chk("NASA IR 2.7 / PHM IR 5.2",
        close(bn_.max() / bn_.min(), 2.7, 0.05)
        and close(bp_.max() / bp_.min(), 5.2, 0.05))
    chk("NASA 1752 / PHM 2511 windows", len(cn) == 1752 and len(cp) == 2511)
    chk("NASA wear range starts between 0 and 0.08 mm",
        min(r["VB"] for r in runs if np.isfinite(r["VB"])) == 0.0)

    dpath = os.path.join(ROOT, "results", "uncertainty_diag.json")
    if os.path.exists(dpath):
        dj = json.load(open(dpath))
        chk("error-detection AUROC 0.628",
            close(dj["auroc_error_detection"], 0.628, 0.0006))
        chk("warm-up ECE 0.195", close(dj["ece"], 0.195, 0.0006))

    # ---------------- manuscript hygiene ------------------------------------ #
    print("\nManuscript")
    chk("no unresolved TODO/XXX", not re.search(r"\bTODO\b|\bXXX\b", tex))
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    for tf in glob.glob(os.path.join(os.path.dirname(PAPER), "tables", "*.tex")):
        labels |= set(re.findall(r"\\label\{([^}]+)\}", open(tf).read()))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    chk("every \\ref resolves", refs <= labels, f"missing {sorted(refs - labels)}")
    cited = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited |= {x.strip() for x in grp.split(",")}
    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    chk("every \\cite has a bibitem", cited <= defined,
        f"missing {sorted(cited - defined)}")
    chk("no uncited bibitem", defined <= cited, f"unused {sorted(defined - cited)}")
    # --- prose scan: every number quoted in the running text must match the
    # value re-derived from results/raw.  Table numbers are generated, but
    # prose numbers are typed, so this is where staleness hides.
    prose = [
        ("anchor gain on PHM", "worth $+0.034$", 
         stat([r["tta"]["macro_f1"] for r in g[("phm", "uctcl", 0.08)]])[0]
         - stat([r["tta_no_anchor"]["macro_f1"]
                 for r in g[("phm", "uctcl", 0.08)]])[0], 0.034),
        ("anchor loss on NASA", "($-0.009$)",
         stat([r["tta"]["macro_f1"] for r in g[("nasa", "uctcl", 0.08)]])[0]
         - stat([r["tta_no_anchor"]["macro_f1"]
                 for r in g[("nasa", "uctcl", 0.08)]])[0], -0.009),
        ("pooled inductive", "from $0.476$", pi, 0.476),
        ("pooled online", "to $0.513$", pt, 0.513),
        ("NASA online F1", "NASA ($0.438$)",
         cell("nasa", "uctcl", "macro_f1", "tta"), 0.438),
        ("PHM online F1", "($0.587$ against", 
         cell("phm", "uctcl", "macro_f1", "tta"), 0.587),
        ("PHM online F1 of pseudo", "$0.591$)",
         cell("phm", "pseudo", "macro_f1", "tta"), 0.591),
        ("median offset", "$0.077$\\,mm", np.median(offs), 0.077),
    ]
    for name, snippet, actual, quoted in prose:
        chk(f"prose: {name} quoted as {quoted:+.3f}",
            snippet in tex and close(actual, quoted, 0.0006),
            f"text={'found' if snippet in tex else 'MISSING'} data={actual:+.4f}")

    chk("75 numerical checks claimed", "75\nautomated" in tex and "All 75 pass" in tex)

    print(f"\n{CHECKS} checks; " + ("ALL PASS" if OK else "SOME FAILED"))
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
