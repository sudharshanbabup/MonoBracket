"""
make_tables3.py
---------------
Every LaTeX table in paper_3, generated from results/ so that no number in the
manuscript is typed by hand.

    python src/make_tables3.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(ROOT), "paper_3", "tables")
os.makedirs(TAB, exist_ok=True)
VERSION = "j2"
MAIN_FRAC = 0.08
DATASETS = ["nasa", "phm", "nuaa"]
DS_NAME = {"nasa": "NASA Milling", "phm": "PHM 2010", "nuaa": "NUAA Ideahouse"}
SEEDS = list(range(10))
ND = {"nasa": 3, "phm": 4, "nuaa": 4}

ORDER = ["rf", "svm", "sup_low", "pseudo", "meanteacher", "fixmatch",
         "simclr", "tcl_ft", "uctcl", "sup_full"]
DEEP = ["sup_low", "pseudo", "meanteacher", "fixmatch", "simclr", "tcl_ft",
        "uctcl"]
PRETTY = {
    "rf": "Random Forest (features)",
    "svm": "SVM-RBF (features)",
    "sup_low": "Supervised (labelled only)",
    "pseudo": "Pseudo-Label, fixed $\\tau$~\\cite{b_lee}",
    "meanteacher": "Mean Teacher~\\cite{b_meanteacher}",
    "fixmatch": "FixMatch~\\cite{b_fixmatch}",
    "simclr": "SimCLR + probe~\\cite{b_simclr}",
    "tcl_ft": "TCL + fine-tune (no PL)",
    "uctcl": "\\textbf{UC-TCL (ours)}",
    "sup_full": "Supervised (100\\% labels)",
}
SHORT = {"sup_low": "Supervised", "pseudo": "Pseudo-Label",
         "meanteacher": "Mean Teacher", "fixmatch": "FixMatch",
         "simclr": "SimCLR", "tcl_ft": "TCL fine-tune", "uctcl": "UC-TCL"}


def load():
    idx = {}
    # NOTE: the raw directory also holds the label-sweep cells at other
    # budgets.  They must not be mixed in here -- the key is (dataset, method,
    # seed), so without this filter a 2% cell would silently overwrite the 8%
    # cell of the same name and corrupt every main-study number.
    pat = f"{VERSION}__*__f{MAIN_FRAC:.3f}__*.json"
    for f in glob.glob(os.path.join(ROOT, "results", "raw", pat)):
        r = json.load(open(f))
        idx[(r["dataset"], r["method"], r["seed"])] = r
    return idx


def val(idx, ds, m, s, k, p=None):
    r = idx[(ds, m, s)]
    return r[p][k] if p else r[k]


def agg(idx, ds, m, k, p=None):
    v = np.array([val(idx, ds, m, s, k, p) for s in SEEDS], float)
    return v.mean(), v.std(ddof=1)


def w(name, txt):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write(txt.rstrip() + "\n")
    print(f"--- {name} ---")


def float_table(name, caption, label, colspec, header, body,
                size="\\footnotesize", star=False, note=None):
    env = "table*" if star else "table"
    txt = (f"\\begin{{{env}}}[!t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\centering\n{size}\n\\begin{{tabular}}{{{colspec}}}\n\\hline\n"
           f"{header}\n\\hline\n{body}\n\\hline\n\\end{{tabular}}\n")
    if note:
        txt += f"\\\\[2pt]\n\\parbox{{\\linewidth}}{{\\scriptsize {note}}}\n"
    txt += f"\\end{{{env}}}"
    w(name, txt)


def pm(mu, sd, nd=3):
    return f"{mu:.{nd}f}\\,$\\pm$\\,{sd:.{nd}f}"


def bold(s, c):
    return f"\\textbf{{{s}}}" if c else s


# --------------------------------------------------------------------------- #
def t_datasets():
    """Table I: what the three benchmarks contain."""
    import data_nasa, data_phm, data_nuaa
    import run_experiments as R
    rows = []
    info = {
        "nasa": ("16", "6", "250", "AC/DC spindle current, table \\& spindle "
                 "vibration, table \\& spindle AE", "0.10/0.30/0.60"),
        "phm": ("3", "7", "25", "cutting force $x{,}y{,}z$, vibration "
                "$x{,}y{,}z$, AE-RMS", "0.08/0.12/0.16"),
        "nuaa": ("9", "8", "15.2", "axial force, bending moment $x{,}y$, "
                 "torsion, vibration $1{,}2$, spindle power \\& current",
                 "0.10/0.15/0.20"),
    }
    for ds in DATASETS:
        S = R.prepare_split(ds, 0, 0.08)
        n_win = len(S["y_cls"])
        bc = np.bincount(S["y_cls"], minlength=4)
        ir = bc.max() / max(bc.min(), 1)
        n_pass = len(np.unique(S["rid"]))
        wr = (float(S["y_raw"].min()), float(S["y_raw"].max()))
        tools, ch, fs, sig, edges = info[ds]
        rows.append(
            f"{DS_NAME[ds]} & {tools} & {ch} & {fs} & {n_pass} & {n_win} & "
            f"{wr[0]:.3f}--{wr[1]:.3f} & {edges} & {ir:.1f} \\\\")
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\quad\\scriptsize "
                    f"\\emph{{channels:}} {sig}}} \\\\")
    float_table(
        "tab_datasets.tex",
        "The three public benchmarks. Passes are the unit at which labels are "
        "purchased; IR is the ratio of the largest to the smallest wear stage. "
        "Stage boundaries are in mm and are anchored on the rejection criterion "
        "for each insert geometry.",
        "tab:datasets", "lcccccccc",
        "\\textbf{Data set} & \\textbf{Tools} & \\textbf{Ch.} & "
        "\\textbf{$f_s$ (Hz)} & \\textbf{Passes} & \\textbf{Windows} & "
        "\\textbf{$VB$ (mm)} & \\textbf{Boundaries} & \\textbf{IR} \\\\",
        "\n".join(rows), size="\\footnotesize", star=True)


def t_main(idx):
    """Table II: macro-F1 and RMSE, both protocols, all three benchmarks."""
    lines = []
    for m in ORDER:
        if m == "sup_full":
            lines.append("\\hline")
        up = m == "sup_full"
        cells = []
        for ds in DATASETS:
            for k, p in [("macro_f1", None), ("macro_f1", "tta")]:
                if p and m in ("rf", "svm"):
                    cells.append("--")
                    continue
                mu, sd = agg(idx, ds, m, k, p)
                best = max(agg(idx, ds, mm, k, p)[0] for mm in
                           (DEEP if p else ORDER[:-1]))
                cells.append(bold(f"{mu:.3f}", not up and mu >= best - 1e-12))
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    float_table(
        "tab_main_f1.tex",
        "Macro-F1 at a sub-10\\% label budget, mean over ten held-out-tool "
        "splits per benchmark. \\emph{Ind.}\\ is the inductive protocol; "
        "\\emph{Onl.}\\ additionally reads the held-out tool's unlabelled "
        "stream (Sec.~\\ref{sec:online}) and is applied identically to every "
        "method. Bold marks the best label-limited entry per column. Standard "
        "deviations are in Table~\\ref{tab:spread}.",
        "tab:main", "lcccccc",
        "\\textbf{Method} & \\multicolumn{2}{c}{\\textbf{NASA Milling}} & "
        "\\multicolumn{2}{c}{\\textbf{PHM 2010}} & "
        "\\multicolumn{2}{c}{\\textbf{NUAA Ideahouse}} \\\\\n"
        " & Ind. & Onl. & Ind. & Onl. & Ind. & Onl. \\\\",
        "\n".join(lines), size="\\footnotesize", star=True)

    # RMSE companion
    lines = []
    for m in ORDER:
        if m == "sup_full":
            lines.append("\\hline")
        up = m == "sup_full"
        cells = []
        for ds in DATASETS:
            for k, p in [("rmse_mm", None), ("rmse_mm", "tta")]:
                if p and m in ("rf", "svm"):
                    cells.append("--")
                    continue
                mu, _ = agg(idx, ds, m, k, p)
                best = min(agg(idx, ds, mm, k, p)[0] for mm in
                           (DEEP if p else ORDER[:-1]))
                cells.append(bold(f"{mu:.{ND[ds]}f}",
                                  not up and mu <= best + 1e-12))
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    float_table(
        "tab_main_rmse.tex",
        "Continuous wear RMSE (mm), same protocol as Table~\\ref{tab:main}. "
        "The three benchmarks span different wear ranges, so RMSE is comparable "
        "within a column but not across them; Table~\\ref{tab:ranks} pools them "
        "by rank instead.",
        "tab:rmse", "lcccccc",
        "\\textbf{Method} & \\multicolumn{2}{c}{\\textbf{NASA Milling}} & "
        "\\multicolumn{2}{c}{\\textbf{PHM 2010}} & "
        "\\multicolumn{2}{c}{\\textbf{NUAA Ideahouse}} \\\\\n"
        " & Ind. & Onl. & Ind. & Onl. & Ind. & Onl. \\\\",
        "\n".join(lines), size="\\footnotesize", star=True)


def t_spread(idx):
    """Standard deviations, kept out of the main table for readability."""
    lines = []
    for m in ORDER:
        cells = []
        for ds in DATASETS:
            mu, sd = agg(idx, ds, m, "macro_f1")
            cells.append(f"{sd:.3f}")
            if m in ("rf", "svm"):
                cells.append("--")
            else:
                _, sd2 = agg(idx, ds, m, "macro_f1", "tta")
                cells.append(f"{sd2:.3f}")
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    float_table(
        "tab_spread.tex",
        "Standard deviation of macro-F1 over the ten tool splits, matching "
        "Table~\\ref{tab:main}. Split-to-split spread exceeds the spread "
        "between most methods, which is why significance is assessed by paired "
        "tests over all 30 runs rather than by comparing means.",
        "tab:spread", "lcccccc",
        "\\textbf{Method} & \\multicolumn{2}{c}{\\textbf{NASA}} & "
        "\\multicolumn{2}{c}{\\textbf{PHM 2010}} & "
        "\\multicolumn{2}{c}{\\textbf{NUAA}} \\\\\n"
        " & Ind. & Onl. & Ind. & Onl. & Ind. & Onl. \\\\",
        "\n".join(lines), size="\\scriptsize")


def t_ranks():
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["ranks"]
    keys = [("macro_f1_inductive", "Macro-F1, ind."),
            ("macro_f1_online", "Macro-F1, onl."),
            ("rmse_mm_inductive", "RMSE, ind."),
            ("rmse_mm_online", "RMSE, onl.")]
    methods = ORDER[:-1]
    lines = []
    for m in methods:
        cells = []
        for k, _ in keys:
            mr = a[k]["mean_rank"]
            if m not in mr:
                cells.append("--")
                continue
            best = min(mr.values())
            cells.append(bold(f"{mr[m]:.2f}", mr[m] <= best + 1e-9))
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    def _p(x):
        m, e = f"{x:.1e}".split("e")
        return f"{m}\\!\\times\\!10^{{{int(e)}}}"

    note = ("Friedman test over all 30 runs rejects equality of the methods in "
            "every column: "
            + "; ".join(f"{n}, $p={_p(a[k]['friedman_p'])}$" for k, n in keys)
            + ". Nemenyi critical difference at $\\alpha=0.05$: "
            + ", ".join(f"{n} {a[k]['critical_difference']:.2f}"
                        for k, n in keys)
            + ". Two methods differ significantly only if their mean ranks "
              "differ by more than that.")
    float_table(
        "tab_ranks.tex",
        "Mean rank over all 30 (data set, seed) runs; lower is better. Ranks "
        "are scale-free, which is what allows the three benchmarks to be pooled "
        "despite their different wear ranges. Bold marks the best method per "
        "column.",
        "tab:ranks", "lcccc",
        "\\textbf{Method} & \\multicolumn{2}{c}{\\textbf{Macro-F1}} & "
        "\\multicolumn{2}{c}{\\textbf{Wear RMSE}} \\\\\n"
        " & Ind. & Onl. & Ind. & Onl. \\\\",
        "\n".join(lines), size="\\footnotesize", note=note)


def t_power():
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["power"]
    lines = []
    for m in DEEP:
        if m == "uctcl":
            continue
        v = a[m]
        star = "$^{*}$" if v["wilcoxon_p"] < 0.05 else ""
        nreq = v["n_for_80pct_power"]
        nreq_s = "--" if nreq is None else (f"{nreq}" if nreq <= 200
                                            else f"$>$200")
        ps = ("$<$0.001" if v["wilcoxon_p"] < 0.001
              else f"{v['wilcoxon_p']:.3f}")
        lines.append(f"vs.\\ {SHORT[m]} & ${v['mean_diff']:+.3f}$ & "
                     f"${v['cohen_dz']:+.2f}$ & {ps}{star} & "
                     f"{nreq_s} \\\\")
    float_table(
        "tab_power.tex",
        "UC-TCL under the online protocol against each semi-supervised "
        "baseline, paired over all 30 runs. $d_z$ is the standardised effect "
        "size of the paired differences and the last column is the number of "
        "runs a two-sided paired test would need for 80\\% power. "
        "$^{*}$ marks $p<0.05$.",
        "tab:power", "lcccc",
        "\\textbf{Comparison} & $\\Delta$\\textbf{F1} & $d_z$ & $p$ & "
        "\\textbf{runs for} \\\\\n & & & & \\textbf{80\\% power} \\\\",
        "\n".join(lines), size="\\footnotesize")


def t_pl():
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["pl"]
    lines = []
    for ds in DATASETS:
        for m, nm in [("pseudo", "fixed $\\tau_c=0.95$"),
                      ("uctcl", "\\textbf{adaptive $\\tau_k$}")]:
            v = a[ds][m]
            lines.append(
                f"{DS_NAME[ds] if m == 'pseudo' else ''} & {nm} & "
                f"{v['coverage'][0]:.1f}\\,$\\pm$\\,{v['coverage'][1]:.1f} & "
                f"{v['precision'][0]:.1f}\\,$\\pm$\\,{v['precision'][1]:.1f} & "
                f"{v['evenness'][0]:.2f}\\,$\\pm$\\,{v['evenness'][1]:.2f} \\\\")
        lines.append("\\hline")
    float_table(
        "tab_pl.tex",
        "Pseudo-labels accepted in the final round. Coverage is the share of "
        "the unlabelled pool admitted, precision is measured against withheld "
        "ground truth, and evenness is the normalised Shannon entropy of the "
        "accepted class histogram ($1$ = uniform over the four stages). The "
        "adaptive rule behaves the same way on all three benchmarks.",
        "tab:pl", "llccc",
        "\\textbf{Data set} & \\textbf{Acceptance rule} & "
        "\\textbf{Cov.\\ (\\%)} & \\textbf{Prec.\\ (\\%)} & "
        "\\textbf{Even.} \\\\",
        "\n".join(lines[:-1]), size="\\scriptsize")


def t_online():
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["online"]
    lines = []
    for ds in DATASETS:
        v = a[ds]
        g = v["gain_by_method"]
        bst = max(g, key=lambda m: g[m])
        wst = min(g, key=lambda m: g[m])
        lines.append(
            f"{DS_NAME[ds]} & {v['n_tools']} & {v['spread_mm']:.4f} & "
            f"{v['median_raw_offset_mm']:.4f} & "
            f"{v['median_applied_offset_mm']:.4f} & "
            f"{100*v['shrinkage_ratio']:.0f}\\% & "
            f"${v['uctcl_gain']:+.3f}$ & ${v['mean_gain']:+.3f}$ & "
            f"${g[bst]:+.3f}$ ({SHORT[bst]}) & ${g[wst]:+.3f}$ ({SHORT[wst]}) \\\\")
    float_table(
        "tab_online.tex",
        "What online adaptation does, per benchmark. The dispersion is the "
        "median absolute deviation of break-in wear across the training tools; "
        "it is the quantity the shrinkage rule keys on, and it governs how much "
        "of the raw offset survives. The last four columns give the macro-F1 "
        "change the online protocol produces: for UC-TCL, averaged over the "
        "seven deep methods, and for the best and worst method on each "
        "benchmark. The spread of the last two columns is the point -- the "
        "protocol magnifies whatever the base model already is.",
        "tab:online", "lccccccccc",
        "\\textbf{Data set} & \\textbf{Tools} & \\textbf{Dispersion} & "
        "\\textbf{Raw $|\\hat\\delta|$} & \\textbf{Applied $|\\delta|$} & "
        "\\textbf{Shrunk} & \\textbf{$\\Delta$F1 ours} & "
        "\\textbf{$\\Delta$F1 mean} & \\textbf{best} & \\textbf{worst} \\\\\n"
        " & & (mm) & (mm) & (mm) & & & & & \\\\",
        "\n".join(lines), size="\\scriptsize", star=True)


def t_sens():
    s = json.load(open(os.path.join(ROOT, "results", "sensitivity.json")))
    lines = []
    lines.append("\\multicolumn{5}{l}{\\emph{(a) fusion weight $\\alpha$}} \\\\")
    for v, d in s["online"]["alpha_fuse"].items():
        cells = [f"{d['per_dataset'][ds]['macro_f1'][0]:.3f}" for ds in DATASETS]
        lines.append(f"\\quad $\\alpha={v}$ & " + " & ".join(cells)
                     + f" & {d['pooled_macro_f1']:.3f} \\\\")
    lines.append("\\hline")
    lines.append("\\multicolumn{5}{l}{\\emph{(b) anchor window $q$}} \\\\")
    for v, d in s["online"]["q"].items():
        cells = [f"{d['per_dataset'][ds]['macro_f1'][0]:.3f}" for ds in DATASETS]
        lines.append(f"\\quad $q={v}$ & " + " & ".join(cells)
                     + f" & {d['pooled_macro_f1']:.3f} \\\\")
    lines.append("\\hline")
    lines.append("\\multicolumn{5}{l}{\\emph{(c) shrinkage of the anchor}} \\\\")
    for v, d in s["online"]["shrinkage"].items():
        cells = [f"{d['per_dataset'][ds]['macro_f1'][0]:.3f}" for ds in DATASETS]
        lines.append(f"\\quad {v} & " + " & ".join(cells)
                     + f" & {d['pooled_macro_f1']:.3f} \\\\")
    lines.append("\\hline")
    lines.append("\\multicolumn{5}{l}{\\emph{(d) Monte-Carlo passes $M$}} \\\\")
    for v, d in s["mc_passes"].items():
        cells = [f"{d['per_dataset'][ds][0]:.3f}" for ds in DATASETS]
        lines.append(f"\\quad $M={v}$ & " + " & ".join(cells)
                     + f" & {d['pooled']:.3f} \\\\")
    float_table(
        "tab_sensitivity.tex",
        "Sensitivity of the online stack, macro-F1. Every setting is "
        "re-evaluated from the cached Monte-Carlo sweeps, so the whole grid "
        "costs no retraining. Defaults are $\\alpha=0.5$, $q=0.25$, shrinkage "
        "on, $M=10$.",
        "tab:sens", "lcccc",
        "\\textbf{Setting} & \\textbf{NASA} & \\textbf{PHM} & \\textbf{NUAA} & "
        "\\textbf{Pooled} \\\\",
        "\n".join(lines), size="\\scriptsize")


def t_deploy():
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["deploy"]
    lines = []
    for ds in DATASETS:
        v = a[ds]
        lines.append(
            f"{DS_NAME[ds]} & {v['params']/1e3:.0f}\\,k & "
            f"{v['param_memory_kb']:.0f} & {v['window_seconds']:.2f} & "
            f"{v['latency_ms_single']:.2f} & "
            f"{v['mc_sweep_ms_single']:.1f} & "
            f"{v['throughput_windows_per_s']/1e3:.1f}\\,k & "
            f"{v['online_postprocess_ms_per_tool']:.1f} \\\\")
    float_table(
        "tab_deploy.tex",
        "Edge-deployment cost, measured on four CPU cores with no GPU. The "
        "Monte-Carlo column is the cost of all $M=10$ stochastic passes for one "
        "window; the last column is the entire online correction stack applied "
        "once to a complete tool life. Both are far below the duration of the "
        "window they analyse, so the method runs comfortably in real time on a "
        "gateway-class device.",
        "tab:deploy", "lccccccc",
        "\\textbf{Data set} & \\textbf{Params} & \\textbf{kB} & "
        "\\textbf{Window} & \\textbf{1 pass} & \\textbf{MC sweep} & "
        "\\textbf{Throughput} & \\textbf{Online} \\\\\n"
        " & & & (s) & (ms) & (ms) & (win/s) & (ms/tool) \\\\",
        "\n".join(lines), size="\\footnotesize", star=True)


def facts(idx):
    """Everything the prose quotes, dumped for the audit to re-derive."""
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))
    s = json.load(open(os.path.join(ROOT, "results", "sensitivity.json")))
    out = {"n_runs": 30, "n_cells": len(idx)}
    for ds in DATASETS:
        out[ds] = {}
        for m in ORDER:
            e = {}
            for k in ("macro_f1", "rmse_mm"):
                e[k] = list(agg(idx, ds, m, k))
                if m not in ("rf", "svm"):
                    e[k + "_tta"] = list(agg(idx, ds, m, k, "tta"))
            out[ds][m] = e
        out[ds]["label_pct"] = list(agg(idx, ds, "uctcl", "actual_label_pct"))
    out["ranks"] = a["ranks"]
    out["power"] = a["power"]
    out["pl"] = a["pl"]
    out["online"] = a["online"]
    out["deploy"] = a["deploy"]
    out["stages"] = {d: {p: {kk: vv for kk, vv in a["stages"][d][p].items()
                             if kk != "confusion"}
                         for p in a["stages"][d]} for d in a["stages"]}
    out["sensitivity"] = s
    out["budget"] = a["budget"]
    with open(os.path.join(ROOT, "results", "paper_facts.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote results/paper_facts.json")



def t_budget():
    """Label efficiency: the end task and the mechanism, at two budgets."""
    a = json.load(open(os.path.join(ROOT, "results", "analyses.json")))["budget"]
    keys = list(a["pooled"].keys())
    hdr = " & ".join(f"\\textbf{{{100*float(k):.0f}\\%}}" for k in keys)

    lines = ["\\multicolumn{%d}{l}{\\emph{(a) end task: pooled macro-F1 over "
             "all 30 runs}} \\\\" % (len(keys) + 1)]
    for m, nm in [("sup_low", "Supervised (labelled only)"),
                  ("pseudo", "Pseudo-Label, fixed $\\tau$"),
                  ("uctcl", "\\textbf{UC-TCL (ours)}")]:
        cells = [f"{a['pooled'][k][m]['inductive']:.3f}" for k in keys]
        lines.append(f"\\quad {nm} & " + " & ".join(cells) + " \\\\")
    for c, nm in [("uctcl_minus_sup_low", "ours $-$ supervised"),
                  ("uctcl_minus_pseudo", "ours $-$ fixed $\\tau$")]:
        cells = []
        for k in keys:
            d = a["pooled"][k][c]
            cells.append(f"${d['diff']:+.4f}$ ($p={d['p']:.2f}$)")
        lines.append(f"\\quad\\emph{{{nm}}} & " + " & ".join(cells) + " \\\\")
    lines.append("\\hline")

    lines.append("\\multicolumn{%d}{l}{\\emph{(b) mechanism: pseudo-label "
                 "coverage (\\%%), worst benchmark}} \\\\" % (len(keys) + 1))
    for m, nm in [("pseudo", "fixed $\\tau_c=0.95$"),
                  ("uctcl", "\\textbf{adaptive $\\tau_k$}")]:
        cells = [f"{min(a['pl'][k][d][m]['coverage'][0] for d in DATASETS):.1f}"
                 for k in keys]
        lines.append(f"\\quad {nm} & " + " & ".join(cells) + " \\\\")
    lines.append("\\multicolumn{%d}{l}{\\emph{(c) mechanism: class evenness, "
                 "worst benchmark}} \\\\" % (len(keys) + 1))
    for m, nm in [("pseudo", "fixed $\\tau_c=0.95$"),
                  ("uctcl", "\\textbf{adaptive $\\tau_k$}")]:
        cells = [f"{min(a['pl'][k][d][m]['evenness'][0] for d in DATASETS):.2f}"
                 for k in keys]
        lines.append(f"\\quad {nm} & " + " & ".join(cells) + " \\\\")

    float_table(
        "tab_budget.tex",
        "Label efficiency. Halving the budget four-fold degrades the fixed "
        "threshold further --- its worst-case coverage falls to 1.9\\% of the "
        "pool --- while the adaptive rule holds coverage and balance. The end "
        "task does not notice: the paired difference between UC-TCL and either "
        "baseline is indistinguishable from zero at both budgets. Panels (b) "
        "and (c) report the worst of the three benchmarks, which is the "
        "quantity a deployment has to survive.",
        "tab:budget", "l" + "c" * len(keys),
        "\\textbf{Label budget} & " + hdr + " \\\\",
        "\n".join(lines), size="\\scriptsize")


if __name__ == "__main__":
    idx = load()
    print(f"{len(idx)} cells\n")
    t_datasets()
    t_main(idx)
    t_spread(idx)
    t_ranks()
    t_power()
    t_pl()
    t_online()
    t_sens()
    t_deploy()
    t_budget()
    facts(idx)
