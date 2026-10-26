"""
make_tables.py
--------------
Emit the LaTeX table bodies used by the paper, straight from results/raw, so
that no number in the manuscript is typed by hand.

    python src/make_tables.py
writes paper_1/tables/*.tex and prints a plain-text preview.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate import (load, table, matched_table, stat, group, paired_test,
                       rank_analysis, pseudo_label_quality, label_sweep,
                       PRETTY, MAIN_ORDER, ABL_ORDER, VERSION)

ROOT = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(ROOT), "paper_1", "tables")
os.makedirs(TAB, exist_ok=True)


def w(name, text):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write(text.rstrip() + "\n")
    print(f"--- {name} ---\n{text}\n")


def float_table(name, caption, label, colspec, header, body, width=None,
                size="\\footnotesize", star=False):
    """
    Write a complete table float.  Emitting the whole float (rather than a body
    fragment that is \\input inside a tabular) avoids the TeX file-boundary
    lookahead after the final row separator.
    """
    env = "table*" if star else "table"
    txt = (f"\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n"
           f"\\label{{{label}}}\n\\centering\n{size}\n"
           f"\\begin{{tabular}}{{{colspec}}}\n\\hline\n{header}\n\\hline\n"
           f"{body}\n\\hline\n\\end{{tabular}}\n\\end{{{env}}}")
    w(name, txt)


def pm(mu, sd, nd=3):
    return f"{mu:.{nd}f}\\,$\\pm$\\,{sd:.{nd}f}"


def bold_if(s, cond):
    return f"\\textbf{{{s}}}" if cond else s


def main_tables(rows):
    """Both benchmarks side by side in one full-width float."""
    t = {ds: {r["method"]: r for r in table(rows, ds, MAIN_ORDER, 0.08)}
         for ds in ["nasa", "phm"]}
    best = {}
    for ds in ["nasa", "phm"]:
        comp = [r for m, r in t[ds].items() if m != "sup_full"]
        best[ds] = dict(f1=max(r["macro_f1"][0] for r in comp),
                        ba=max(r["balanced_acc"][0] for r in comp),
                        rm=min(r["rmse_mm"][0] for r in comp))
    lines = []
    for m in MAIN_ORDER:
        up = m == "sup_full"
        if up:
            lines.append("\\hline")
        cells = []
        for ds, nd_r in [("nasa", 3), ("phm", 4)]:
            r = t[ds].get(m)
            if r is None:
                cells += ["--"] * 4
                continue
            cells.append(bold_if(pm(*r["macro_f1"], nd=3),
                                 not up and r["macro_f1"][0] >= best[ds]["f1"] - 1e-9))
            cells.append(bold_if(f"{r['balanced_acc'][0]:.3f}",
                                 not up and r["balanced_acc"][0] >= best[ds]["ba"] - 1e-9))
            cells.append(bold_if(f"{r['rmse_mm'][0]:.{nd_r}f}",
                                 not up and r["rmse_mm"][0] <= best[ds]["rm"] + 1e-9))
            cells.append(f"{r['ece'][0]:.3f}")
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    float_table("tab_main.tex",
                "Both benchmarks at a sub-10\\% label budget "
                "($8.4\\%$ of cutting passes on NASA Milling, $6.0\\%$ on "
                "PHM 2010), averaged over 5 held-out tool splits. Macro-F1 is "
                "given as mean\\,$\\pm$\\,s.d.; the remaining columns are means. "
                "Bold marks the best label-limited method.",
                "tab:main", "lcccccccc",
                "\\textbf{Method} & \\multicolumn{4}{c}{\\textbf{NASA Milling}} & "
                "\\multicolumn{4}{c}{\\textbf{PHM 2010}} \\\\\n"
                " & Macro-F1 & Bal.\\ acc. & RMSE & ECE"
                " & Macro-F1 & Bal.\\ acc. & RMSE & ECE \\\\",
                "\n".join(lines), size="\\footnotesize", star=True)


def ablation_table(rows):
    """
    One row per component, both benchmarks side by side, reported as the change
    relative to the full method on exactly the splits the variant was run on.
    """
    tabs, full = {}, {}
    for ds in ["nasa", "phm"]:
        t = matched_table(rows, ds, ABL_ORDER, 0.08, seeds=(0, 1, 2))
        tabs[ds] = {r["method"]: r for r in t}
        full[ds] = tabs[ds].get("uctcl")
    lines = []
    for m in ABL_ORDER:
        if m == "uctcl":
            continue
        cells = []
        for ds, nd_r in [("nasa", 3), ("phm", 4)]:
            r = tabs[ds].get(m)
            if r is None or full[ds] is None:
                cells += ["--", "--"]
            else:
                cells.append(f"{r['macro_f1'][0] - full[ds]['macro_f1'][0]:+.3f}")
                cells.append(f"{r['rmse_mm'][0] - full[ds]['rmse_mm'][0]:+.{nd_r}f}")
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append(
        "\\textbf{UC-TCL (full)} & "
        f"\\textbf{{{full['nasa']['macro_f1'][0]:.3f}}} & "
        f"\\textbf{{{full['nasa']['rmse_mm'][0]:.3f}}} & "
        f"\\textbf{{{full['phm']['macro_f1'][0]:.3f}}} & "
        f"\\textbf{{{full['phm']['rmse_mm'][0]:.4f}}} \\\\")
    float_table("tab_ablation.tex",
                "Ablations at an 8\\% budget over the 3 splits every variant was "
                "run on, reported as the change from the full method (last row, "
                "absolute values). Negative $\\Delta$F1 and positive "
                "$\\Delta$RMSE mean the removed component was helping. The two "
                "optional add-ons were evaluated on PHM 2010 only.",
                "tab:abl", "lcccc",
                "\\textbf{Variant} & \\multicolumn{2}{c}{\\textbf{NASA Milling}} & "
                "\\multicolumn{2}{c}{\\textbf{PHM 2010}} \\\\\n"
                " & $\\Delta$F1 & $\\Delta$RMSE & $\\Delta$F1 & $\\Delta$RMSE \\\\",
                "\n".join(lines), size="\\scriptsize")


def pl_table(rows):
    q = pseudo_label_quality(rows, 0.08)
    lines = []
    for ds, name in [("nasa", "NASA Milling"), ("phm", "PHM 2010")]:
        for m, nm in [("pseudo", "Fixed threshold $\\tau_c=0.95$"),
                      ("uctcl", "\\textbf{Adaptive $\\tau_k$ (ours)}")]:
            v = q[ds][m]
            lines.append(
                f"{name if m == 'pseudo' else ''} & {nm} & "
                f"{v['coverage'][0]:.1f}\\,$\\pm$\\,{v['coverage'][1]:.1f} & "
                f"{v['precision'][0]:.1f}\\,$\\pm$\\,{v['precision'][1]:.1f} & "
                f"{v['evenness'][0]:.2f}\\,$\\pm$\\,{v['evenness'][1]:.2f} \\\\")
        lines.append("\\hline")
    body = "\n".join(lines[:-1])
    float_table("tab_pseudolabels.tex",
                "Pseudo-labels accepted in the final round: coverage of the "
                "unlabelled pool, precision against withheld ground truth, and "
                "normalised class evenness ($1$ = uniform over the four stages).",
                "tab:pl", "llccc",
                "\\textbf{Data set} & \\textbf{Rule} & \\textbf{Coverage (\\%)} & "
                "\\textbf{Precision (\\%)} & \\textbf{Evenness} \\\\", body,
                star=True)


def rank_table(rows):
    metrics = [("rmse_mm", False, "Wear RMSE"),
               ("macro_f1", True, "Macro-F1"),
               ("balanced_acc", True, "Balanced acc."),
               ("ece", False, "ECE")]
    rk = {}
    n = 0
    for mt, hi, _ in metrics:
        rk[mt], n = rank_analysis(rows, 0.08, mt, hi)
    lines = []
    for m in MAIN_ORDER:
        if m not in rk["rmse_mm"]:
            continue
        cells = []
        for mt, _, _ in metrics:
            best = min(v[0] for k, v in rk[mt].items() if k != "sup_full")
            val = rk[mt][m][0]
            cells.append(bold_if(f"{val:.1f}",
                                 m != "sup_full" and val <= best + 1e-9))
        if m == "sup_full":
            lines.append("\\hline")
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    float_table("tab_ranks.tex",
                f"Average rank over all {n} (data set, seed) runs; lower is "
                "better. Bold marks the best label-limited method.",
                "tab:ranks", "lcccc",
                "\\textbf{Method} & \\textbf{RMSE} & \\textbf{Macro-F1} & "
                "\\textbf{Bal.\\ acc.} & \\textbf{ECE} \\\\",
                "\n".join(lines))
    return n


def sig_table(rows):
    g = group(rows, 0.08)
    lines = []
    for ds, name, nd_r in [("nasa", "NASA Milling", 3), ("phm", "PHM 2010", 4)]:
        ours = sorted(g[(ds, "uctcl", 0.08)], key=lambda r: r["seed"])
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for m in ["sup_low", "pseudo", "fixmatch", "meanteacher", "simclr",
                  "tcl_ft"]:
            rs = sorted(g[(ds, m, 0.08)], key=lambda r: r["seed"])
            if len(rs) != len(ours):
                continue
            wf, tf, df = paired_test([r["macro_f1"] for r in ours],
                                     [r["macro_f1"] for r in rs])
            wr, tr, dr = paired_test([r["rmse_mm"] for r in ours],
                                     [r["rmse_mm"] for r in rs])
            lines.append(
                f"\\quad vs.\\ {PRETTY[m]} & {df:+.3f} & {wf:.2f} & "
                f"{dr:+.{nd_r}f} & {wr:.2f} \\\\")
        lines.append("\\hline")
    body = "\n".join(lines[:-1])
    float_table("tab_significance.tex",
                "Paired comparison of UC-TCL against each competitor. Positive "
                "$\\Delta$F1 and negative $\\Delta$RMSE favour UC-TCL; $p$ is the "
                "two-sided Wilcoxon signed-rank value over 5 paired splits.",
                "tab:sig", "lcccc",
                "\\textbf{Comparison} & $\\Delta$\\textbf{F1} & $p$ & "
                "$\\Delta$\\textbf{RMSE} & $p$ \\\\", body)


def sweep_table(rows):
    sw = label_sweep(rows)
    lines = []
    for ds, name, nd_r in [("nasa", "NASA Milling", 3), ("phm", "PHM 2010", 4)]:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{name}}}}} \\\\")
        fracs = ["0.02", "0.05", "0.08", "0.1", "0.2"]
        for m in ["sup_low", "fixmatch", "uctcl"]:
            k = f"{ds}|{m}"
            if k not in sw:
                continue
            cells = []
            for f in fracs:
                d = sw[k].get(f)
                cells.append("--" if d is None
                             else f"{d['rmse_mm'][0]:.{nd_r}f}")
            lines.append(f"\\quad {PRETTY[m]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    body = "\n".join(lines[:-1])
    float_table("tab_sweep.tex",
                "Wear RMSE (mm) against the labelled-pass budget, 3 splits.",
                "tab:sweep", "lccccc",
                "\\textbf{Method} & \\textbf{2\\%} & \\textbf{5\\%} & "
                "\\textbf{8\\%} & \\textbf{10\\%} & \\textbf{20\\%} \\\\", body)


def facts(rows):
    """Numbers quoted in the prose, dumped so they can be re-checked."""
    out = {}
    for ds in ["nasa", "phm"]:
        t = table(rows, ds, MAIN_ORDER, 0.08)
        d = {r["method"]: r for r in t}
        out[ds] = {m: dict(macro_f1=d[m]["macro_f1"], rmse=d[m]["rmse_mm"],
                           bacc=d[m]["balanced_acc"], ece=d[m]["ece"])
                   for m in d}
        out[ds]["label_pct"] = d["uctcl"]["label_pct"]
        out[ds]["n_train"] = stat([r["n_train"] for r in group(rows, 0.08)[(ds, "uctcl", 0.08)]])[0]
        out[ds]["n_lab"] = stat([r["n_labelled"] for r in group(rows, 0.08)[(ds, "uctcl", 0.08)]])[0]
        best_base = max((v["macro_f1"][0], k) for k, v in out[ds].items()
                        if k in MAIN_ORDER and k not in ("uctcl", "sup_full"))
        out[ds]["best_baseline_f1"] = best_base
        best_rmse = min((v["rmse"][0], k) for k, v in out[ds].items()
                        if k in MAIN_ORDER and k not in ("uctcl", "sup_full"))
        out[ds]["best_baseline_rmse"] = best_rmse
    out["pl"] = pseudo_label_quality(rows, 0.08)
    out["ranks"] = {m: rank_analysis(rows, 0.08, m, h)[0]
                    for m, h in [("rmse_mm", False), ("macro_f1", True),
                                 ("balanced_acc", True), ("ece", False)]}
    with open(os.path.join(ROOT, "results", "paper_facts.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote results/paper_facts.json")
    return out


if __name__ == "__main__":
    rows = load(VERSION)
    print(f"{len(rows)} result files\n")
    main_tables(rows)
    ablation_table(rows)
    pl_table(rows)
    n = rank_table(rows)
    print(f"rank table computed over n={n} (dataset, seed) runs")
    sig_table(rows)
    sweep_table(rows)
    facts(rows)
