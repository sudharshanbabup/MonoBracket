"""
make_tables2.py
---------------
LaTeX tables for the two-protocol study (inductive and online / test-time
adapted).  Writes paper_1/tables/*.tex directly from results/raw.

    python src/make_tables2.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate import (load, group, stat, paired_test, pooled_vector, by_run,
                       matched_table, pseudo_label_quality, label_sweep,
                       PRETTY, MAIN_ORDER, TTA_ORDER, ABL_ORDER)
from make_tables import float_table, w, bold_if

ROOT = os.path.dirname(os.path.abspath(__file__))
V_MAIN = "v8"          # 10 seeds, dual protocol
V_ABL = "v7"           # component ablations (training path identical to v8)
SEEDS = list(range(10))


def _fmt(mu, sd, nd):
    return f"{mu:.{nd}f}\\,$\\pm$\\,{sd:.{nd}f}"


# --------------------------------------------------------------------------- #
# Table I: both protocols, both benchmarks
# --------------------------------------------------------------------------- #
def main_table(rows):
    g = group(rows, 0.08)
    rec = {}
    for ds in ["nasa", "phm"]:
        for m in MAIN_ORDER:
            rs = sorted(g[(ds, m, 0.08)], key=lambda r: r["seed"])
            rec[(ds, m)] = rs

    def cell(ds, m, key, prot, nd):
        rs = rec[(ds, m)]
        if prot and prot not in rs[0]:
            return "--"
        v = [(r[prot][key] if prot else r[key]) for r in rs]
        return _fmt(*stat(v), nd)

    def best(ds, key, prot, lower):
        vals = {}
        for m in MAIN_ORDER:
            if m == "sup_full":
                continue
            rs = rec[(ds, m)]
            if prot and prot not in rs[0]:
                continue
            vals[m] = stat([(r[prot][key] if prot else r[key]) for r in rs])[0]
        return (min if lower else max)(vals.values())

    lines = []
    for m in MAIN_ORDER:
        if m == "sup_full":
            lines.append("\\hline")
        up = m == "sup_full"
        cells = []
        for ds, nd in [("nasa", 3), ("phm", 4)]:
            for key, prot, lower in [("macro_f1", None, False),
                                     ("macro_f1", "tta", False),
                                     ("rmse_mm", None, True),
                                     ("rmse_mm", "tta", True)]:
                c = cell(ds, m, key, prot, 3 if key == "macro_f1" else nd)
                if c != "--" and not up:
                    b = best(ds, key, prot, lower)
                    v = stat([(r[prot][key] if prot else r[key])
                              for r in rec[(ds, m)]])[0]
                    c = bold_if(c, (v <= b + 1e-12) if lower else (v >= b - 1e-12))
                cells.append(c)
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")

    float_table(
        "tab_main.tex",
        "Both benchmarks at a sub-10\\% label budget, mean\\,$\\pm$\\,s.d.\\ over "
        "10 held-out tool splits. \\emph{Ind.} is the inductive protocol; "
        "\\emph{Onl.} additionally lets the model read the held-out tool's "
        "unlabelled stream (Sec.~\\ref{sec:tta}), and is applied identically to "
        "every method. Bold marks the best label-limited entry in each column.",
        "tab:main", "lcccccccc",
        "\\textbf{Method} & \\multicolumn{4}{c}{\\textbf{NASA Milling}} & "
        "\\multicolumn{4}{c}{\\textbf{PHM 2010}} \\\\\n"
        " & \\multicolumn{2}{c}{Macro-F1} & \\multicolumn{2}{c}{RMSE (mm)}"
        " & \\multicolumn{2}{c}{Macro-F1} & \\multicolumn{2}{c}{RMSE (mm)} \\\\\n"
        " & Ind. & Onl. & Ind. & Onl. & Ind. & Onl. & Ind. & Onl. \\\\",
        "\n".join(lines), size="\\scriptsize", star=True)


# --------------------------------------------------------------------------- #
# Table II: pooled significance, n = 20 runs
# --------------------------------------------------------------------------- #
def sig_table(rows):
    ours_i = pooled_vector(rows, "uctcl", "macro_f1", None)
    ours_t = pooled_vector(rows, "uctcl", "macro_f1", "tta")
    lines = []
    for m in ["sup_low", "pseudo", "fixmatch", "simclr", "tcl_ft", "meanteacher"]:
        bi = pooled_vector(rows, m, "macro_f1", None)
        bt = pooled_vector(rows, m, "macro_f1", "tta")
        wi, _, di = paired_test(ours_i, bi)
        wt, _, dt = paired_test(ours_t, bt)
        star_i = "$^{*}$" if wi < 0.05 else ""
        star_t = "$^{*}$" if wt < 0.05 else ""
        short = {"sup_low": "Supervised (labels only)",
                 "pseudo": "Pseudo-Label (fixed $\\tau$)",
                 "fixmatch": "FixMatch", "simclr": "SimCLR probe",
                 "tcl_ft": "TCL fine-tune", "meanteacher": "Mean Teacher"}[m]
        lines.append(f"vs.\\ {short} & {di:+.3f} & {wi:.3f}{star_i} & "
                     f"{dt:+.3f} & {wt:.3f}{star_t} \\\\")
    # the gain the online protocol itself buys each method
    lines.append("\\hline")
    for m in ["sup_low", "uctcl"]:
        a = pooled_vector(rows, m, "macro_f1", None)
        b = pooled_vector(rows, m, "macro_f1", "tta")
        p, _, dd = paired_test(b, a)
        st = "$^{*}$" if p < 0.05 else ""
        nm = {"sup_low": "supervised", "uctcl": "\\textbf{UC-TCL}"}[m]
        lines.append(f"\\emph{{online gain}}, {nm} & \\multicolumn{{2}}{{c}}{{--}} & "
                     f"{dd:+.3f} & {p:.3f}{st} \\\\")

    float_table(
        "tab_significance.tex",
        "Paired Wilcoxon signed-rank tests on macro-F1, pooled over all 20 "
        "(data set, seed) runs; $\\Delta$ is the mean difference and positive "
        "favours UC-TCL. $^{*}$ marks $p<0.05$. The lower block reports what the "
        "online protocol alone buys each method.",
        "tab:sig", "lcccc",
        "\\textbf{Comparison} & \\multicolumn{2}{c}{\\textbf{Inductive}} & "
        "\\multicolumn{2}{c}{\\textbf{Online}} \\\\\n"
        " & $\\Delta$F1 & $p$ & $\\Delta$F1 & $p$ \\\\",
        "\n".join(lines), size="\\scriptsize")


# --------------------------------------------------------------------------- #
# Table III: component ablation + adaptation-stack ablation
# --------------------------------------------------------------------------- #
def ablation_table(rows_main, rows_abl):
    lines = ["\\multicolumn{5}{l}{\\emph{(a) training components}} \\\\"]
    tabs, full = {}, {}
    for ds in ["nasa", "phm"]:
        t = matched_table(rows_abl, ds, ABL_ORDER, 0.08, seeds=(0, 1, 2))
        tabs[ds] = {r["method"]: r for r in t}
        full[ds] = tabs[ds].get("uctcl")
    core = ["uctcl_noTCL", "uctcl_simclr", "uctcl_noMC", "uctcl_globalTau",
            "uctcl_noMono", "uctcl_noCB"]
    for m in core:
        cells = []
        for ds, nd in [("nasa", 3), ("phm", 4)]:
            r = tabs[ds].get(m)
            if r is None or full[ds] is None:
                cells += ["--", "--"]
            else:
                cells.append(f"{r['macro_f1'][0] - full[ds]['macro_f1'][0]:+.3f}")
                cells.append(f"{r['rmse_mm'][0] - full[ds]['rmse_mm'][0]:+.{nd}f}")
        lines.append(f"\\quad {PRETTY[m]} & " + " & ".join(cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\multicolumn{5}{l}{\\emph{(b) online adaptation stack}} \\\\")
    g = group(rows_main, 0.08)
    base = {ds: sorted(g[(ds, "uctcl", 0.08)], key=lambda r: r["seed"])
            for ds in ["nasa", "phm"]}
    names = [("tta_no_anchor", "w/o physics anchor"),
             ("tta_no_mono", "w/o monotone projection"),
             ("tta_no_fuse", "w/o threshold read-out"),
             ("tta_bn", "+ BatchNorm re-estimation")]
    for key, nm in names:
        cells = []
        for ds, nd in [("nasa", 3), ("phm", 4)]:
            rs = base[ds]
            df = stat([r[key]["macro_f1"] for r in rs])[0] - \
                stat([r["tta"]["macro_f1"] for r in rs])[0]
            dr = stat([r[key]["rmse_mm"] for r in rs])[0] - \
                stat([r["tta"]["rmse_mm"] for r in rs])[0]
            cells += [f"{df:+.3f}", f"{dr:+.{nd}f}"]
        lines.append(f"\\quad {nm} & " + " & ".join(cells) + " \\\\")
    lines.append("\\hline")
    cells = []
    for ds, nd in [("nasa", 3), ("phm", 4)]:
        rs = base[ds]
        cells.append(f"\\textbf{{{stat([r['tta']['macro_f1'] for r in rs])[0]:.3f}}}")
        cells.append(f"\\textbf{{{stat([r['tta']['rmse_mm'] for r in rs])[0]:.{nd}f}}}")
    lines.append("\\textbf{UC-TCL + online (absolute)} & " + " & ".join(cells) + " \\\\")

    float_table(
        "tab_ablation.tex",
        "Ablations reported as the change from the full method. Negative "
        "$\\Delta$F1 and positive $\\Delta$RMSE mean the removed component was "
        "helping. Block (a) removes a training component; block (b) removes one "
        "correction from the test-time stack. The two optional add-ons of (a) "
        "were evaluated on PHM 2010 only.",
        "tab:abl", "lcccc",
        "\\textbf{Variant} & \\multicolumn{2}{c}{\\textbf{NASA Milling}} & "
        "\\multicolumn{2}{c}{\\textbf{PHM 2010}} \\\\\n"
        " & $\\Delta$F1 & $\\Delta$RMSE & $\\Delta$F1 & $\\Delta$RMSE \\\\",
        "\n".join(lines), size="\\scriptsize")


# --------------------------------------------------------------------------- #
def facts(rows_main, rows_abl):
    out = {"n_runs": 20, "seeds": SEEDS}
    g = group(rows_main, 0.08)
    for ds in ["nasa", "phm"]:
        out[ds] = {}
        for m in MAIN_ORDER:
            rs = sorted(g[(ds, m, 0.08)], key=lambda r: r["seed"])
            e = dict(n=len(rs),
                     ind_f1=stat([r["macro_f1"] for r in rs]),
                     ind_rmse=stat([r["rmse_mm"] for r in rs]),
                     ind_bacc=stat([r["balanced_acc"] for r in rs]),
                     ind_ece=stat([r["ece"] for r in rs]))
            if "tta" in rs[0]:
                e.update(tta_f1=stat([r["tta"]["macro_f1"] for r in rs]),
                         tta_rmse=stat([r["tta"]["rmse_mm"] for r in rs]),
                         tta_bacc=stat([r["tta"]["balanced_acc"] for r in rs]),
                         tta_ece=stat([r["tta"]["ece"] for r in rs]))
            out[ds][m] = e
        out[ds]["label_pct"] = stat([r["actual_label_pct"]
                                     for r in g[(ds, "uctcl", 0.08)]])
    pooled = {}
    for m in TTA_ORDER:
        pooled[m] = dict(
            ind=float(np.nanmean(pooled_vector(rows_main, m, "macro_f1"))),
            tta=float(np.nanmean(pooled_vector(rows_main, m, "macro_f1", "tta"))))
    out["pooled_macro_f1"] = pooled
    out["pl"] = pseudo_label_quality(rows_main, 0.08)
    offs = [abs(v) for ds in ["nasa", "phm"]
            for r in g[(ds, "uctcl", 0.08)]
            for v in r["tta"]["offsets_mm"].values()]
    out["median_abs_offset_mm"] = float(np.median(offs))
    out["n_offsets"] = len(offs)
    with open(os.path.join(ROOT, "results", "paper_facts.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote results/paper_facts.json")


if __name__ == "__main__":
    rows_main = load(V_MAIN)
    rows_abl = load(V_ABL)
    print(f"{len(rows_main)} main rows, {len(rows_abl)} ablation rows\n")
    main_table(rows_main)
    sig_table(rows_main)
    ablation_table(rows_main, rows_abl)
    facts(rows_main, rows_abl)
