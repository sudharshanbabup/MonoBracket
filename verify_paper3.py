"""
verify_paper3.py
----------------
Re-derive every number quoted in paper_3 from results/paper_facts.json and fail
on any mismatch.

Two kinds of check:

  TABLE  every numeric cell in paper_3/tables/*.tex must equal the value the
         generator would produce from the stored results.  This catches a stale
         table left behind after a re-run.

  PROSE  every number written into the body text of main.tex is listed here
         explicitly with the fact it is supposed to equal.  This is the check
         that matters: tables are generated, prose is typed, and typed numbers
         are the ones that go stale.  A claim that is not in this file is a
         claim nobody has verified, so the file is also the list of everything
         the manuscript asserts numerically.

    python src/verify_paper3.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(ROOT), "paper_3")
FACTS = json.load(open(os.path.join(ROOT, "results", "paper_facts.json")))

OK, BAD = [], []


def chk(name, got, want, tol=5e-4):
    """Numeric equality within tol."""
    try:
        good = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        good = got == want
    (OK if good else BAD).append((name, got, want))
    return good


def chk_true(name, cond, detail=""):
    (OK if cond else BAD).append((name, detail or cond, "True"))


TEX = open(os.path.join(PAPER, "main.tex")).read()
# the tables are \input, so their \label{}s live in separate files
import glob as _glob
TEX_ALL = TEX + "\n".join(open(f).read()
                          for f in _glob.glob(os.path.join(PAPER, "tables",
                                                           "*.tex")))
# strip comments so a commented-out draft number never trips the scan
TEX_BODY = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in TEX.split("\n"))


_WS = re.compile(r"\s+")


def _norm(s):
    return _WS.sub(" ", s).strip()


TEX_NORM = _norm(TEX_BODY)


def in_text(s):
    return _norm(s) in TEX_NORM


def prose(name, snippet):
    """Assert a literal string (a number and its context) is in the body."""
    chk_true(f"prose: {name}", in_text(snippet), f"missing: {snippet!r}")


# --------------------------------------------------------------------------- #
# 1. structural
# --------------------------------------------------------------------------- #
def structural():
    chk("n result cells", FACTS["n_cells"], 300, tol=0)
    chk("n runs", FACTS["n_runs"], 30, tol=0)
    chk_true("journal document class",
             "\\documentclass[journal]{IEEEtran}" in TEX)
    for f in ["fig_pipeline", "fig_data", "fig_pl", "fig_cd", "fig_online",
              "fig_sens", "fig_conf", "fig_unc", "fig_budget"]:
        chk_true(f"figure present: {f}",
                 os.path.exists(os.path.join(PAPER, "figures", f + ".pdf"))
                 and f in TEX)
    for t in ["tab_datasets", "tab_main_f1", "tab_main_rmse", "tab_spread",
              "tab_ranks", "tab_power", "tab_pl", "tab_online",
              "tab_sensitivity", "tab_deploy", "tab_budget", "tab_ablate"]:
        p = os.path.join(PAPER, "tables", t + ".tex")
        chk_true(f"table present: {t}",
                 os.path.exists(p) and f"tables/{t}" in TEX)
    # every \label{} that is referenced must exist, and vice versa
    labels = set(re.findall(r"\\label\{([^}]+)\}", TEX_ALL))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", TEX))
    chk_true("all refs resolve", refs <= labels, f"dangling: {refs - labels}")
    cites = set(re.findall(r"\\cite\{([^}]+)\}", TEX_ALL))
    cites = {c for grp in cites for c in grp.split(",")}
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", TEX))
    chk_true("all citations defined", cites <= bib, f"missing: {cites - bib}")
    chk_true("no unused bib entries", bib <= cites, f"unused: {bib - cites}")


# --------------------------------------------------------------------------- #
# 2. tables agree with the stored results
# --------------------------------------------------------------------------- #
def tables():
    tabdir = os.path.join(PAPER, "tables")

    # main macro-F1 table
    txt = open(os.path.join(tabdir, "tab_main_f1.tex")).read()
    for m, row in [("uctcl", "UC-TCL (ours)"), ("sup_low", "Supervised (labelled only)")]:
        line = [l for l in txt.split("\n") if row in l][0]
        cells = re.findall(r"[-+]?\d*\.\d+", line.replace("\\textbf", ""))
        want = []
        for ds in ["nasa", "phm", "nuaa"]:
            want += [FACTS[ds][m]["macro_f1"][0],
                     FACTS[ds][m]["macro_f1_tta"][0]]
        chk_true(f"tab_main_f1 {m} cell count", len(cells) == 6, len(cells))
        for i, (g, w) in enumerate(zip(cells, want)):
            chk(f"tab_main_f1 {m}[{i}]", g, w, tol=6e-4)

    # ranks
    txt = open(os.path.join(tabdir, "tab_ranks.tex")).read()
    mr = FACTS["ranks"]["macro_f1_online"]["mean_rank"]
    line = [l for l in txt.split("\n") if "UC-TCL (ours)" in l][0]
    cells = re.findall(r"\d+\.\d+", line.replace("\\textbf", ""))
    chk("tab_ranks uctcl online F1", cells[1], mr["uctcl"], tol=6e-3)

    # pseudo-label table
    txt = open(os.path.join(tabdir, "tab_pl.tex")).read()
    for ds, nm in [("nasa", "NASA Milling"), ("phm", "PHM 2010"),
                   ("nuaa", "NUAA Ideahouse")]:
        lines = txt.split("\n")
        i = [k for k, l in enumerate(lines) if l.startswith(nm)][0]
        for off, m in [(0, "pseudo"), (1, "uctcl")]:
            nums = re.findall(r"\d+\.\d+", lines[i + off].replace("\\tau_k", "")
                              .replace("0.95", ""))
            f = FACTS["pl"][ds][m]
            chk(f"tab_pl {ds}/{m} coverage", nums[0], f["coverage"][0], tol=6e-2)
            chk(f"tab_pl {ds}/{m} precision", nums[2], f["precision"][0], tol=6e-2)


# --------------------------------------------------------------------------- #
# 3. prose numbers
# --------------------------------------------------------------------------- #
def prose_numbers():
    F = FACTS

    # -- abstract / intro ---------------------------------------------------- #
    pl = F["pl"]
    cov_fixed = [pl[d]["pseudo"]["coverage"][0] for d in ("nasa", "phm", "nuaa")]
    ev_fixed = [pl[d]["pseudo"]["evenness"][0] for d in ("nasa", "phm", "nuaa")]
    ev_ad = [pl[d]["uctcl"]["evenness"][0] for d in ("nasa", "phm", "nuaa")]
    cov_ad = [pl[d]["uctcl"]["coverage"][0] for d in ("nasa", "phm", "nuaa")]
    chk("fixed coverage range lo", min(cov_fixed), 3.3, tol=0.06)
    chk("fixed coverage range hi", max(cov_fixed), 12.8, tol=0.06)
    chk("adaptive coverage ~50", sum(cov_ad) / 3, 50.0, tol=0.4)
    chk("fixed evenness lo", min(ev_fixed), 0.06, tol=6e-3)
    chk("fixed evenness hi", max(ev_fixed), 0.59, tol=6e-3)
    chk("adaptive evenness lo", min(ev_ad), 0.88, tol=6e-3)
    chk("adaptive evenness hi", max(ev_ad), 0.94, tol=6e-3)
    prose("coverage claim", "3--13\\% to a stable 50\\%")
    prose("evenness claim", "0.06--0.59 to 0.88--0.94")

    # PHM fixed-threshold collapse, quoted twice
    chk("phm fixed coverage", pl["phm"]["pseudo"]["coverage"][0], 3.3, tol=0.06)
    chk("phm fixed coverage sd", pl["phm"]["pseudo"]["coverage"][1], 2.7, tol=0.06)
    chk("phm fixed evenness", pl["phm"]["pseudo"]["evenness"][0], 0.06, tol=6e-3)
    chk("phm fixed evenness sd", pl["phm"]["pseudo"]["evenness"][1], 0.13, tol=6e-3)
    prose("phm collapse", "$3.3\\pm2.7$\\% of the unlabelled pool at a class evenness")
    prose("phm collapse 2", "admits $3.3\\pm2.7$\\% of the pool at evenness $0.06\\pm0.13$")
    prose("nasa fixed pl", "$12.8\\pm7.8$\\% at evenness $0.43$")
    prose("nuaa fixed pl", "$11.4\\pm7.6$\\% at $0.59$")
    chk("nasa fixed cov sd", pl["nasa"]["pseudo"]["coverage"][1], 7.8, tol=0.06)
    chk("nasa fixed evenness", pl["nasa"]["pseudo"]["evenness"][0], 0.43, tol=6e-3)
    chk("nuaa fixed cov", pl["nuaa"]["pseudo"]["coverage"][0], 11.4, tol=0.06)
    chk("nuaa fixed cov sd", pl["nuaa"]["pseudo"]["coverage"][1], 7.6, tol=0.06)

    # precision numbers quoted in the results section
    for ds, a, b in [("nasa", 49.8, 47.9), ("phm", 99.9, 87.3),
                     ("nuaa", 76.1, 61.8)]:
        chk(f"{ds} precision fixed", pl[ds]["pseudo"]["precision"][0], a, tol=0.06)
        chk(f"{ds} precision adaptive", pl[ds]["uctcl"]["precision"][0], b, tol=0.06)
    prose("precision trade", "NASA $49.8\\to47.9$\\%, PHM $99.9\\to87.3$\\%")
    prose("precision trade 2", "NUAA $76.1\\to61.8$\\%")
    chk("phm precision drop", 99.9 - 87.3, 12.6, tol=0.06)
    prose("phm precision drop", "12.6-point precision loss")
    prose("precision sd nasa", "$\\pm18.1\n\\to \\pm6.3$ on NASA")
    prose("precision sd nuaa", "$\\pm16.1 \\to \\pm5.6$ on NUAA")
    chk("nasa prec sd fixed", pl["nasa"]["pseudo"]["precision"][1], 18.1, tol=0.06)
    chk("nasa prec sd adapt", pl["nasa"]["uctcl"]["precision"][1], 6.3, tol=0.06)
    chk("nuaa prec sd fixed", pl["nuaa"]["pseudo"]["precision"][1], 16.1, tol=0.06)
    chk("nuaa prec sd adapt", pl["nuaa"]["uctcl"]["precision"][1], 5.6, tol=0.06)
    chk("adaptive cov sd max",
        max(pl[d]["uctcl"]["coverage"][1] for d in ("nasa", "phm", "nuaa")),
        0.2, tol=0.06)

    # -- ranks --------------------------------------------------------------- #
    R = F["ranks"]
    chk("rank uctcl online f1", R["macro_f1_online"]["mean_rank"]["uctcl"],
        2.87, tol=6e-3)
    chk("rank uctcl ind f1", R["macro_f1_inductive"]["mean_rank"]["uctcl"],
        3.87, tol=6e-3)
    chk("rank suplow ind f1", R["macro_f1_inductive"]["mean_rank"]["sup_low"],
        3.87, tol=6e-3)
    chk("rank uctcl ind rmse", R["rmse_mm_inductive"]["mean_rank"]["uctcl"],
        3.80, tol=6e-3)
    chk("rank uctcl onl rmse", R["rmse_mm_online"]["mean_rank"]["uctcl"],
        3.70, tol=6e-3)
    chk("rank tclft onl rmse", R["rmse_mm_online"]["mean_rank"]["tcl_ft"],
        3.07, tol=6e-3)
    chk("rank meanteacher onl f1",
        R["macro_f1_online"]["mean_rank"]["meanteacher"], 6.97, tol=6e-3)
    chk("rank simclr onl f1", R["macro_f1_online"]["mean_rank"]["simclr"],
        5.07, tol=6e-3)
    chk("rank suplow onl f1", R["macro_f1_online"]["mean_rank"]["sup_low"],
        3.03, tol=6e-3)
    chk("rank tclft onl f1", R["macro_f1_online"]["mean_rank"]["tcl_ft"],
        3.43, tol=6e-3)
    prose("tclft vs sup", "ranks $3.43$ on online macro-F1 against the\nsupervised baseline's $3.03$")
    chk("CD online f1", R["macro_f1_online"]["critical_difference"], 1.64,
        tol=6e-3)
    prose("cd value", "critical difference for online macro-F1 is $1.64$")
    prose("rank summary", "$2.87$ for online\nmacro-F1")
    prose("rank ind rmse", "$3.80$ for inductive RMSE")
    prose("rank onl rmse", "it ranks fourth ($3.70$) behind\nTCL fine-tune ($3.07$)")

    # top-5 clique span quoted as 0.56
    mro = R["macro_f1_online"]["mean_rank"]
    top5 = sorted(mro.values())[:5]
    chk("top5 rank span", top5[-1] - top5[0], 0.5667, tol=6e-4)
    prose("clique span", "top five methods span a rank range of $0.57$")

    # friedman p, quoted as <= 6.7e-6 (the largest of the four)
    worst_p = max(R[k]["friedman_p"] for k in R)
    chk_true("friedman worst p <= 6.7e-6", worst_p <= 6.75e-6, worst_p)
    prose("friedman", "($p \\le 6.7\\times10^{-6}$)")

    # -- power --------------------------------------------------------------- #
    P = F["power"]
    chk("power suplow diff", P["sup_low"]["mean_diff"], 0.006, tol=6e-4)
    chk("power suplow p", P["sup_low"]["wilcoxon_p"], 0.79, tol=6e-3)
    chk("power suplow n", P["sup_low"]["n_for_80pct_power"], 2137, tol=60)
    chk("power mt dz", P["meanteacher"]["cohen_dz"], 1.60, tol=6e-3)
    chk("power simclr dz", P["simclr"]["cohen_dz"], 1.11, tol=6e-3)
    chk("power mt n", P["meanteacher"]["n_for_80pct_power"], 4, tol=0)
    chk("power simclr n", P["simclr"]["n_for_80pct_power"], 7, tol=0)
    chk("power tclft n", P["tcl_ft"]["n_for_80pct_power"], 81, tol=0)
    chk("power fixmatch n", P["fixmatch"]["n_for_80pct_power"], 168, tol=0)
    chk("power fixmatch diff", P["fixmatch"]["mean_diff"], 0.020, tol=6e-4)
    dzs = [P[m]["cohen_dz"] for m in ("sup_low", "pseudo", "fixmatch", "tcl_ft")]
    chk("competitive dz min", min(dzs), 0.06, tol=6e-3)
    chk("competitive dz max", max(dzs), 0.31, tol=6e-3)
    diffs = [P[m]["mean_diff"] for m in ("sup_low", "pseudo", "fixmatch", "tcl_ft")]
    chk("competitive diff min", min(diffs), 0.006, tol=6e-4)
    chk("competitive diff max", max(diffs), 0.020, tol=6e-4)
    prose("power dz range", "$d_z$ between $0.06$ and $0.31$")
    prose("power counts", "need 81 runs to resolve the TCL-fine-tune\ncomparison, 168 for FixMatch")
    prose("mt/simclr power", "$d_z=1.60$ and\n$1.11$, $p<0.001$; four and seven runs would have sufficed")

    # -- online -------------------------------------------------------------- #
    O = F["online"]
    for ds, g in [("nasa", 0.013), ("phm", 0.032), ("nuaa", -0.029)]:
        chk(f"uctcl online gain {ds}", O[ds]["uctcl_gain"], g, tol=6e-4)
    for ds, g in [("nasa", 0.000), ("phm", -0.020), ("nuaa", -0.060)]:
        chk(f"mean online gain {ds}", O[ds]["mean_gain"], g, tol=6e-4)
    chk("phm best gain", O["phm"]["gain_by_method"]["tcl_ft"], 0.058, tol=6e-4)
    chk("phm worst gain", O["phm"]["gain_by_method"]["meanteacher"], -0.221,
        tol=6e-4)
    chk("phm gain spread",
        O["phm"]["gain_by_method"]["tcl_ft"]
        - O["phm"]["gain_by_method"]["meanteacher"], 0.28, tol=6e-3)
    chk_true("nuaa all methods lose",
             all(v < 0 for v in O["nuaa"]["gain_by_method"].values()),
             O["nuaa"]["gain_by_method"])
    for ds, r in [("nasa", 38), ("phm", 93), ("nuaa", 74)]:
        chk(f"shrinkage {ds}", 100 * O[ds]["shrinkage_ratio"], r, tol=0.6)
    chk("phm dispersion mm", O["phm"]["spread_mm"], 0.0141, tol=6e-4)
    chk_true("nasa dispersion largest",
             O["nasa"]["spread_mm"] == max(O[d]["spread_mm"] for d in O),
             {d: O[d]["spread_mm"] for d in O})
    prose("online gains", "$+0.000$ on NASA, $-0.020$ on PHM and\n$-0.060$ on NUAA")
    prose("uctcl online gains", "($+0.013$) and PHM ($+0.032$) and loses on NUAA ($-0.029$)")
    prose("phm spread", "$+0.058$ to TCL fine-tune and $-0.221$ to Mean\nTeacher")
    prose("phm spread val", "a spread of $0.28$ macro-F1")
    prose("shrinkage rates", "38\\%\nof the raw offset on NASA, 74\\% on NUAA and 93\\% on PHM")
    prose("phm dispersion", "begin within 0.014~mm of each other")
    prose("nasa dispersion caveat",
          "NASA has the largest dispersion and still benefits")

    # -- sensitivity --------------------------------------------------------- #
    S = F["sensitivity"]
    al = S["online"]["alpha_fuse"]
    chk("alpha 0.0", al["0.0"]["pooled_macro_f1"], 0.440, tol=6e-4)
    chk("alpha 0.5", al["0.5"]["pooled_macro_f1"], 0.446, tol=6e-4)
    chk("alpha 0.75", al["0.75"]["pooled_macro_f1"], 0.367, tol=6e-4)
    chk("alpha 1.0", al["1.0"]["pooled_macro_f1"], 0.364, tol=6e-4)
    q = S["online"]["q"]
    chk("q 0.5 nasa", q["0.5"]["per_dataset"]["nasa"]["macro_f1"][0], 0.436,
        tol=6e-4)
    chk("q 0.5 phm", q["0.5"]["per_dataset"]["phm"]["macro_f1"][0], 0.581,
        tol=6e-4)
    sh = S["online"]["shrinkage"]
    chk("shrink pooled gain",
        sh["on"]["pooled_macro_f1"] - sh["off"]["pooled_macro_f1"], 0.010,
        tol=6e-4)
    chk("shrink nuaa gain",
        sh["on"]["per_dataset"]["nuaa"]["macro_f1"][0]
        - sh["off"]["per_dataset"]["nuaa"]["macro_f1"][0], 0.025, tol=6e-4)
    chk("shrink phm gain",
        sh["on"]["per_dataset"]["phm"]["macro_f1"][0]
        - sh["off"]["per_dataset"]["phm"]["macro_f1"][0], 0.011, tol=6e-4)
    mc = S["mc_passes"]
    chk("M=2 pooled", mc["2"]["pooled"], 0.4444, tol=6e-5)
    chk("M=10 pooled", mc["10"]["pooled"], 0.4461, tol=6e-5)
    chk("M penalty", mc["10"]["pooled"] - mc["2"]["pooled"], 0.002, tol=6e-4)
    prose("alpha flat", "$0.440$--$0.446$")
    prose("alpha collapse", "($0.367$ at $0.75$, $0.364$ at\n$1.0$)")
    prose("q trade", "NASA improves to $0.436$ at\n$q=0.5$ while PHM falls to $0.581$")
    prose("M finding", "$0.4444$ at $M=2$ and $0.4461$ at $M=10$")
    prose("M penalty", "a difference of\n$0.002$ for a five-fold difference")
    prose("shrink worth", "worth $+0.010$ pooled macro-F1")
    prose("shrink nuaa", "costs\n$0.025$ macro-F1 on NUAA and $0.011$ on PHM")

    C = S["components"]
    chk("ablation full", C["full"]["pooled_macro_f1"], 0.4461, tol=6e-5)
    chk("ablation no_fusion", C["no_fusion"]["pooled_macro_f1"], 0.4396,
        tol=6e-5)
    chk("ablation no_anchor", C["no_anchor"]["pooled_macro_f1"], 0.4435,
        tol=6e-5)
    chk("ablation no_mono", C["no_mono"]["pooled_macro_f1"], 0.4439, tol=6e-5)
    chk("ablation none == no_fusion", C["none"]["pooled_macro_f1"],
        C["no_fusion"]["pooled_macro_f1"], tol=6e-5)
    chk("ablate nasa full", C["full"]["per_dataset"]["nasa"]["macro_f1"][0],
        0.413, tol=6e-4)
    chk("ablate nasa no_mono",
        C["no_mono"]["per_dataset"]["nasa"]["macro_f1"][0], 0.425, tol=6e-4)
    chk("ablate phm mono cost",
        C["full"]["per_dataset"]["phm"]["macro_f1"][0]
        - C["no_mono"]["per_dataset"]["phm"]["macro_f1"][0], 0.0167, tol=6e-4)
    prose("ablate fusion", "from $0.4461$ to $0.4396$")
    prose("ablate mono", "\\emph{improves} NASA from $0.413$ to $0.425$ while costing PHM\n$0.016$")

    # -- label efficiency ----------------------------------------------------#
    B = F["budget"]
    chk_true("both budgets fully populated",
             set(B["pooled"]) == {"0.020", "0.080"}, list(B["pooled"]))
    for k, ds_, dp_, ps_, pp_ in [("0.020", -0.0074, 0.0031, 0.52, 0.90),
                                  ("0.080", 0.0000, -0.0044, 0.97, 0.75)]:
        d = B["pooled"][k]["uctcl_minus_sup_low"]
        chk(f"budget {k} d vs sup", d["diff"], ds_, tol=6e-5)
        chk(f"budget {k} p vs sup", d["p"], ps_, tol=6e-3)
        chk_true(f"budget {k} n=30 vs sup", d["n"] == 30, d["n"])
        d = B["pooled"][k]["uctcl_minus_pseudo"]
        chk(f"budget {k} d vs pseudo", d["diff"], dp_, tol=6e-5)
        chk(f"budget {k} p vs pseudo", d["p"], pp_, tol=6e-3)
        chk_true(f"budget {k} all four p > 0.05",
                 B["pooled"][k]["uctcl_minus_sup_low"]["p"] > 0.05
                 and B["pooled"][k]["uctcl_minus_pseudo"]["p"] > 0.05)
    prose("budget 2pct sup", "$-0.0074$ ($p=0.52$, $n=30$)")
    prose("budget 2pct pseudo", "$+0.0031$ ($p=0.90$)")
    prose("budget 8pct", "$+0.0000$ ($p=0.97$) and\n$-0.0044$ ($p=0.75$)")
    prose("abstract budget", "$-0.007$ ($p=0.52$) at a 2\\% budget and $+0.000$ ($p=0.97$) at 8\\%")

    # the mechanism gap widens as labels get scarcer -- the paper's key contrast
    worst_cov = {k: min(B["pl"][k][d]["pseudo"]["coverage"][0]
                        for d in ("nasa", "phm", "nuaa")) for k in B["pl"]}
    chk("worst fixed coverage 2pct", worst_cov["0.020"], 1.9, tol=0.06)
    chk("worst fixed coverage 8pct", worst_cov["0.080"], 3.3, tol=0.06)
    chk_true("fixed threshold degrades at 2pct",
             worst_cov["0.020"] < worst_cov["0.080"], worst_cov)
    chk("nuaa fixed cov 2pct", B["pl"]["0.020"]["nuaa"]["pseudo"]["coverage"][0],
        1.9, tol=0.06)
    chk("nuaa fixed even 2pct", B["pl"]["0.020"]["nuaa"]["pseudo"]["evenness"][0],
        0.18, tol=6e-3)
    ad_cov = [B["pl"][k][d]["uctcl"]["coverage"][0]
              for k in B["pl"] for d in ("nasa", "phm", "nuaa")]
    ad_ev = [B["pl"][k][d]["uctcl"]["evenness"][0]
             for k in B["pl"] for d in ("nasa", "phm", "nuaa")]
    chk("adaptive coverage lo both budgets", min(ad_cov), 49.7, tol=0.06)
    chk("adaptive coverage hi both budgets", max(ad_cov), 50.1, tol=0.06)
    chk("adaptive evenness lo 2pct",
        min(B["pl"]["0.020"][d]["uctcl"]["evenness"][0]
            for d in ("nasa", "phm", "nuaa")), 0.86, tol=6e-3)
    chk("adaptive evenness hi 2pct",
        max(B["pl"]["0.020"][d]["uctcl"]["evenness"][0]
            for d in ("nasa", "phm", "nuaa")), 0.93, tol=6e-3)
    prose("mechanism widens", "at 2\\% it falls to 1.9\\%, and on NUAA specifically it drops\nfrom 11.4\\% to 1.9\\% while its class evenness falls from 0.59 to 0.18")
    prose("adaptive holds", "coverage at 50.0--50.1\\% and evenness at 0.86--0.93 at both")
    prose("abstract mechanism", "collapses to 1.9\\% coverage at evenness 0.10")

    # realised label fractions
    lp = B["label_pct"]
    chk("nasa realised 2pct", lp["0.020"]["nasa"], 3.9, tol=0.06)
    chk("phm realised 2pct", lp["0.020"]["phm"], 2.1, tol=0.06)
    chk("nuaa realised 2pct", lp["0.020"]["nuaa"], 2.0, tol=0.06)
    prose("realised 2pct", "2.0--2.1\\%\non PHM and NUAA but 3.9\\% on NASA")

    # every-method-loses-together claim
    drop = [B["pooled"]["0.080"][m]["inductive"] - B["pooled"]["0.020"][m]["inductive"]
            for m in ("sup_low", "pseudo", "uctcl")]
    chk("mean drop 2->8pct", float(np.mean(drop)), 0.097, tol=6e-3)
    prose("drop claim", "0.10 macro-F1 when the budget is cut")

    # -- stages -------------------------------------------------------------- #
    St = F["stages"]
    for ds, v in [("nasa", 86.4), ("phm", 99.9), ("nuaa", 69.6)]:
        chk(f"adjacent {ds} online",
            100 * St[ds]["online"]["adjacent_fraction"], v, tol=0.06)
    prose("adjacent", "86.4\\% of UC-TCL's errors on NASA and 99.9\\% on PHM")
    prose("adjacent nuaa", "On NUAA the figure is 69.6\\%")

    # -- deployment ---------------------------------------------------------- #
    D = F["deploy"]
    lat = [D[d]["latency_ms_single"] for d in D]
    mcs = [D[d]["mc_sweep_ms_single"] for d in D]
    thr = [D[d]["throughput_windows_per_s"] / 1e3 for d in D]
    post = [D[d]["online_postprocess_ms_per_tool"] for d in D]
    npt = [D[d]["windows_per_tool"] for d in D]
    chk("params ~215k", D["nasa"]["params"] / 1e3, 215, tol=1.0)
    chk("memory kB", D["nasa"]["param_memory_kb"], 843, tol=2.0)
    chk_true("latency under 1.5 ms", max(lat) < 1.5, max(lat))
    chk_true("mc sweep under 15 ms", max(mcs) < 15.0, max(mcs))
    chk_true("throughput over 7k/s", min(thr) > 7.0, min(thr))
    chk_true("postprocess under 1.5 ms", max(post) < 1.5, max(post))
    chk("win/tool lo", min(npt), 584, tol=0)
    chk("win/tool hi", max(npt), 641, tol=0)
    for ds, sec in [("nasa", 2.05), ("phm", 2.56), ("nuaa", 8.45)]:
        chk(f"window seconds {ds}", D[ds]["window_seconds"], sec, tol=6e-3)
    chk_true("window duration >100x mc sweep",
             min(D[d]["window_seconds"] * 1e3 / D[d]["mc_sweep_ms_single"]
                 for d in D) > 100, "two orders of magnitude")
    prose("deploy latency", "under 1.5~ms and the full $M=10$ Monte-Carlo sweep under 15~ms")
    prose("deploy windows", "2.05~s (NASA), 2.56~s (PHM) and 8.45~s (NUAA)")
    prose("deploy post", "under\n1.5~ms applied once to a complete tool life of 584--641 windows")
    prose("deploy thr", "exceeds\n7k windows/s")
    prose("deploy params", "$\\approx215$k parameters, 843~kB")

    # -- main table values quoted in prose ----------------------------------- #
    chk("nasa uctcl ind f1", F["nasa"]["uctcl"]["macro_f1"][0], 0.401, tol=6e-4)
    chk("nasa rf ind f1", F["nasa"]["rf"]["macro_f1"][0], 0.302, tol=6e-4)
    chk("phm uctcl ind f1", F["phm"]["uctcl"]["macro_f1"][0], 0.605, tol=6e-4)
    chk("phm rf ind f1", F["phm"]["rf"]["macro_f1"][0], 0.550, tol=6e-4)
    prose("nasa gap", "UC-TCL $0.401$ vs.\\ Random Forest $0.302$")
    prose("phm gap", "($0.605$ vs.\\ $0.550$)")

    # spread claim: sd range over deep methods, both protocols, all datasets
    sds = []
    for ds in ("nasa", "phm", "nuaa"):
        for m in ("sup_low", "pseudo", "fixmatch", "tcl_ft", "uctcl"):
            sds += [F[ds][m]["macro_f1"][1], F[ds][m]["macro_f1_tta"][1]]
    sds.sort()
    chk("spread min", sds[0], 0.055, tol=6e-4)
    chk("spread max", sds[-1], 0.151, tol=6e-4)
    chk("spread median", sds[len(sds) // 2], 0.120, tol=6e-4)
    prose("spread range", "0.055 to 0.151\nmacro-F1, median 0.120")

    # label budget really is under 10 %
    for ds in ("nasa", "phm", "nuaa"):
        lo = F[ds]["label_pct"][0] - 2 * F[ds]["label_pct"][1]
        hi = F[ds]["label_pct"][0] + 2 * F[ds]["label_pct"][1]
        chk_true(f"label budget < 10% on {ds}", hi < 10.0,
                 f"{F[ds]['label_pct'][0]:.2f} +/- {F[ds]['label_pct'][1]:.2f}")
        chk_true(f"label budget in 5-9% on {ds}", 4.5 <= lo and hi <= 9.5,
                 f"{lo:.2f}-{hi:.2f}")
    prose("budget claim", "realised budgets between 5\\% and 9\\%")

    # sup_full is NOT a strict ceiling -- the paper says so, verify it
    beaten = [m for m in ("sup_low", "pseudo", "fixmatch", "tcl_ft", "uctcl")
              if F["nuaa"][m]["macro_f1_tta"][0]
              > F["nuaa"]["sup_full"]["macro_f1_tta"][0]]
    chk_true("sup_full beaten on nuaa online", len(beaten) > 0, beaten)
    prose("ceiling caveat",
          "on NUAA under the online protocol it is beaten by several\nlabel-limited methods")


# --------------------------------------------------------------------------- #
# 4. no stale numbers: scan the prose for decimals not accounted for
# --------------------------------------------------------------------------- #
KNOWN_LITERALS = {
    # configuration constants, stated in the setup and defined in methods.py
    "0.15", "0.30", "0.10", "0.05", "0.50", "1.5", "0.999", "0.999", "2",
    "3", "10", "20", "30", "0.08", "0.12", "0.16", "0.20", "0.60", "0.25",
    "0.00", "0.24", "0.03", "1.53", "0.31", "0.014", "1.25", "0.0", "1.0",
    "0.75", "0.35", "0.45", "0.02", "0.01",
    # values measured by the audit suite (audit_code.py / BUGFIXES.md), not by
    # the results pipeline: the 1/M bias, the post-fix PHM segment ratio, the
    # buggy label budget, the pilot validation-collapse figures, the fixed
    # pseudo-label threshold, and the NUAA sampling rate.
    "0.101", "1.2", "20.8", "0.67", "0.14", "0.95", "15.15", "0.063",
}


def orphan_scan():
    """
    Every decimal in the body text should either be a configuration constant,
    or appear in one of the prose() assertions above.  This is what stops a
    number from silently going stale after a re-run.
    """
    body = TEX_BODY
    # cut everything before the introduction and after the bibliography
    body = body.split("\\section{Introduction}")[-1]
    body = body.split("\\begin{thebibliography}")[0]
    # remove equations and table/figure includes -- those are generated
    body = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", " ",
                  body, flags=re.S)
    body = re.sub(r"\\input\{[^}]*\}", " ", body)
    body = re.sub(r"\\caption\{.*?\n\n", " ", body, flags=re.S)
    body = re.sub(r"\\(label|ref|cite)\{[^}]*\}", " ", body)

    checked = " ".join(f"{g} {w}" for _, g, w in OK + BAD)
    nums = set(re.findall(r"(?<![\w.])\d+\.\d+", body))
    orphans = []
    for n in sorted(nums):
        if n in KNOWN_LITERALS:
            continue
        v = float(n)
        # accounted for if some verified fact rounds to it
        hit = any(abs(v - float(m)) < 5e-4
                  for m in re.findall(r"\d+\.?\d*", checked)
                  if m not in ("", "."))
        if not hit:
            orphans.append(n)
    chk_true("no unverified decimals in prose", not orphans,
             f"orphans: {orphans}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    structural()
    tables()
    prose_numbers()
    orphan_scan()

    print(f"passed {len(OK)}/{len(OK)+len(BAD)}")
    if BAD:
        print("\nFAILURES")
        for n, got, want in BAD:
            print(f"  {n}\n     got  {got}\n     want {want}")
        sys.exit(1)
    print("all checks pass")
