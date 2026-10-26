"""make_tables_v5.py -- writes every table of the version_5 paper from
results/facts_v5.json.  No number in a table is typed by hand.

    python src/make_tables_v5.py [results_dir] [tables_dir]
"""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else (os.path.join(here, "results") if os.path.isdir(os.path.join(here, "results")) else os.path.join(here, "..", "results"))
out = sys.argv[2] if len(sys.argv) > 2 else (os.path.join(here, "tables") if os.path.isdir(os.path.join(here, "tables")) else os.path.join(here, "..", "..", "paper", "tables"))
os.makedirs(out, exist_ok=True)
F = json.load(open(os.path.join(res, "facts_v5.json")))
DS = ["nasa", "phm", "nuaa"]
P = "mbpl_nopath"
NAME = {"rf": "Random forest", "svm": "SVM (RBF)", "sup_low": "Supervised",
        "pseudo": "Pseudo-Label",
        "meanteacher": r"Mean Teacher~\cite{tarvainen2017mean}",
        "fixmatch": r"FixMatch~\cite{sohn2020fixmatch}",
        "simclr": r"SimCLR + probe~\cite{chen2020simple}",
        "tcl_ft": "TCL + fine-tune", "uctcl": "UC-TCL (no bracket)",
        P: r"\textbf{MonoBracket (ours)}"}
LIM = ["rf", "svm", "sup_low", "pseudo", "meanteacher", "fixmatch", "simclr", "tcl_ft", "uctcl", P]


def w(name, s):
    open(os.path.join(out, name), "w").write(s)
    print("wrote", name)


# ---------------------------------------------------------------- main ----
cols = [(d, k) for k in ["f1_ind", "f1_onl"] for d in DS] + \
       [(d, k) for k in ["rmse_ind", "rmse_onl"] for d in DS]
def cell(m, d, k):
    e = F["main"][d][m]
    if k.endswith("onl") and m in ("rf", "svm"):
        return None
    return e[k][0]
best = {}
for d, k in cols:
    v = [cell(m, d, k) for m in LIM if cell(m, d, k) is not None]
    best[(d, k)] = max(v) if k.startswith("f1") else min(v)
def fmt(v, d, k):
    if v is None:
        return "--"
    if k.startswith("f1"):
        return f"{v:.3f}"
    return f"{v:.3f}" if d == "nasa" else f"{v:.4f}"
rows = []
for m in LIM + ["sup_full"]:
    r = [NAME.get(m, "Supervised, 100\\% labels")]
    for d, k in cols:
        v = cell(m, d, k)
        s = fmt(v, d, k)
        if m in LIM and v is not None and fmt(best[(d, k)], d, k) == s:
            s = r"\textbf{" + s + "}"
        r.append(s)
    if m in LIM:
        rk = F["ranks"]
        r += [f"{rk['f1_ind']['mean_rank'][m]:.2f}",
              f"{rk['f1_onl']['mean_rank'][m]:.2f}" if m in rk["f1_onl"]["mean_rank"] else "--",
              f"{rk['rmse_ind']['mean_rank'][m]:.2f}",
              f"{rk['rmse_onl']['mean_rank'][m]:.2f}" if m in rk["rmse_onl"]["mean_rank"] else "--"]
        for j, key in zip(range(-4, 0), ["f1_ind", "f1_onl", "rmse_ind", "rmse_onl"]):
            mr = rk[key]["mean_rank"]
            if m in mr and abs(mr[m] - min(mr.values())) < 1e-9:
                r[j] = r"\textbf{" + r[j] + "}"
    else:
        r += ["--"] * 4
    if m == "sup_full":
        rows.append(r"\midrule")
    rows.append(" & ".join(r) + r" \\")
w("tab_main.tex", r"""\begin{table*}[t]
\caption{Macro-F1 and wear RMSE at an 8\% label budget (mean of ten held-out-tool splits per benchmark; N = NASA Milling, P = PHM 2010, U = NUAA). Ind.: inductive protocol. Onl.: label-free online protocol. Rank: mean rank over the 30 paired runs (lower is better). Bold: best among label-limited methods.}
\label{tab:main}
\centering
\footnotesize
\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{l ccc ccc ccc ccc cccc}
\toprule
 & \multicolumn{3}{c}{Macro-F1, Ind. $\uparrow$} & \multicolumn{3}{c}{Macro-F1, Onl. $\uparrow$} & \multicolumn{3}{c}{RMSE (mm), Ind. $\downarrow$} & \multicolumn{3}{c}{RMSE (mm), Onl. $\downarrow$} & \multicolumn{4}{c}{Mean rank $\downarrow$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-17}
Method & N & P & U & N & P & U & N & P & U & N & P & U & F1\,Ind. & F1\,Onl. & RMSE\,Ind. & RMSE\,Onl. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------- pseudo-labels --
L = {"nasa": "NASA", "phm": "PHM 2010", "nuaa": "NUAA"}
rows = []
for d in DS:
    for m, nm in [("pseudo", r"Fixed $\tau_c$"), ("uctcl", "Uncertainty only"), (P, r"\textbf{MonoBracket}")]:
        e = F["pl"][d][m]
        cert = f"{e['certified'][0]:.1f}" if "certified" in e else "--"
        mae = (f"{e['target_mae_mm'][0]*1000:.1f}" if "target_mae_mm" in e else "--")
        rows.append(f"{L[d] if m == 'pseudo' else ''} & {nm} & {cert} & {e['coverage'][0]:.1f} & "
                    f"{e['precision'][0]:.1f} & {e['evenness'][0]:.2f} & {mae} \\\\")
    if d != "nuaa":
        rows.append(r"\midrule")
w("tab_pl.tex", r"""\begin{table}[t]
\caption{Pseudo-labels admitted in the final round at the 8\% budget (mean of ten splits). Cert.: share of unlabelled windows certified by bracketing. Cov.: share admitted. Prec.: stage precision against the withheld truth. Even.: normalised entropy of the admitted class histogram. MAE: error of the wear target ($\mu$m).}
\label{tab:pl}
\centering
\footnotesize
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{llccccc}
\toprule
Data & Selection rule & Cert. & Cov. & Prec. & Even. & MAE \\
 & & (\%) & (\%) & (\%) & & ($\mu$m) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
# network wear error before the bracket, for the text
print({d: round(F["pl"][d][P]["net_mae_mm"][0] * 1000, 1) for d in DS})

# -------------------------------------------------- ablation + 2 % budget --
rows = []
for m, nm in [("uctcl", "No bracket (UC-TCL)"), ("mbpl_cert", "Certified windows only"),
              ("mbpl_noanch", "Path decoding, no bracket"), ("mbpl", "Bracket + path decoding"),
              (P, r"\textbf{Bracket (MonoBracket)}")]:
    rows.append(f"{nm} & {F['pooled'][m]['f1_ind']:.3f} & {F['pooled'][m]['f1_onl']:.3f} & "
                + " & ".join(f"{F['main'][d][m]['rmse_onl'][0]*1000:.1f}" for d in DS) + r" \\")
rows.append(r"\midrule")
rows.append(r"\multicolumn{6}{l}{\emph{2\% label budget}} \\")
B = F["budget2"]
for m, nm in [("sup_low", "Supervised"), ("pseudo", "Pseudo-Label"), ("uctcl", "UC-TCL (no bracket)"),
              (P, r"\textbf{MonoBracket}")]:
    rows.append(f"{nm} & {B['pooled'][m]['f1_ind']:.3f} & {B['pooled'][m]['f1_onl']:.3f} & "
                + " & ".join(f"{B['per_ds'][d][m]['rmse_onl'][0]*1000:.1f}" for d in DS) + r" \\")
w("tab_ablation.tex", r"""\begin{table}[t]
\caption{Ablation at the 8\% budget and comparison at a 2\% budget. Macro-F1 is pooled over the 30 runs; RMSE is online, in $\mu$m, per benchmark.}
\label{tab:ablation}
\centering
\footnotesize
\setlength{\tabcolsep}{3.6pt}
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{2}{c}{Macro-F1 $\uparrow$} & \multicolumn{3}{c}{RMSE Onl. ($\mu$m) $\downarrow$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
Variant & Ind. & Onl. & N & P & U \\
\midrule
\multicolumn{6}{l}{\emph{8\% label budget}} \\
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ----------------------------------------------------------------- data ---
E = {"nasa": "0.10 / 0.30 / 0.60", "phm": "0.08 / 0.12 / 0.16", "nuaa": "0.10 / 0.15 / 0.20"}
R = {"nasa": "250", "phm": "25", "nuaa": "15.15"}
REF = {"nasa": "NASA Milling", "phm": "PHM 2010",
       "nuaa": "NUAA Ideahouse"}
rows = []
for d in DS:
    e = F["datasets"][d]
    rows.append(f"{REF[d]} & {e['n_tools']} & {e['n_passes']} & {e['n_windows']} & {e['channels']} & "
                f"{E[d]} & {e['imbalance']:.1f} \\\\")
w("tab_datasets.tex", r"""\begin{table}[t]
\caption{Benchmarks. Edges: stage boundaries on flank wear (mm). IR: ratio of the largest to the smallest stage.}
\label{tab:datasets}
\centering
\footnotesize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lcccccc}
\toprule
Benchmark & Tools & Passes & Windows & Ch. & Edges (mm) & IR \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
