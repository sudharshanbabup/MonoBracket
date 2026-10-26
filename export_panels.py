"""Exports the four panels of Fig. 2 as JSON for matlab_fig_1.m.
Every value comes from codes/results/facts_v5.json and example_v5.json."""
import json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else (os.path.join(here, "results") if os.path.isdir(os.path.join(here, "results")) else os.path.join(here, "..", "codes", "results"))
F = json.load(open(os.path.join(res, "facts_v5.json")))
E = json.load(open(os.path.join(res, "example_v5.json")))
out = os.path.join(here, "data"); os.makedirs(out, exist_ok=True)
BLUE, VERM, GREEN, GREY = [0, .447, .698], [.835, .369, 0], [0, .620, .451], [.25, .25, .25]
json.dump(dict(proposed=BLUE, uctcl=VERM, supervised=GREEN, truth=GREY),
          open(os.path.join(here, "palette.json"), "w"), indent=1)

def r(v, n=5): return [None if (x is None or not np.isfinite(x)) else round(float(x), n) for x in v]

# (a) certified share vs inspection budget
c = F["cert_curve"]
ser = []
for d, lab in [("phm", "PHM 2010"), ("nuaa", "NUAA"), ("nasa", "NASA")]:
    x, y = np.array(c[d]["label_pct"]), np.array(c[d]["certified"])
    _, i = np.unique(np.round(x, 3), return_index=True)      # NASA: 1 % and 2 % coincide
    ser.append(dict(label=lab, x=r(x[i], 3), y=r(y[i], 3)))
json.dump(dict(name="fig2a_certified", layout="quad", xlabel="Labelled windows (%)",
               ylabel="Certified windows (%)", xscale="log", xlim=[0.9, 40],
               xticks=[1, 2, 4, 8, 16, 32], xticklabels=["1", "2", "4", "8", "16", "32"],
               ylim=[0, 145], yticks=[0, 25, 50, 75, 100], legend="northwest", lock_ylim=True, series=ser),
          open(os.path.join(out, "fig2a_certified.json"), "w"), indent=1)

# (b) bracket on one NUAA training tool
t = E["nuaa"]["train"]
x = np.array(t["t"]); w = np.array(t["wear"]); lab = np.array(t["lab"])
ylo = np.array([np.nan if v is None else v for v in t["ylo"]], float)
yhi = np.array([np.nan if v is None else v for v in t["yhi"]], float)
ylo = np.where(np.isfinite(ylo), ylo, w.min()); yhi = np.where(np.isfinite(yhi), yhi, w.max())
ser = [dict(kind="band", x=r(x), lo=r(ylo), hi=r(yhi), color=[0.72, 0.84, 0.94], alpha=1.0),
       dict(label="True wear", x=r(x), y=r(w), marker="none", line="-", color=GREY),
       dict(label="Inspection", x=r(x[lab]), y=r(w[lab]), marker="o", line="none", color=VERM)]
for e in t["edges_mm"]:
    ser.append(dict(kind="hline", y=e, line=":"))
json.dump(dict(name="fig2b_bracket", layout="quad", xlabel="Normalised tool life",
               ylabel="Flank wear (mm)", xlim=[0, 1], legend="northwest", series=ser),
          open(os.path.join(out, "fig2b_bracket.json"), "w"), indent=1)

# (c), (d) wear tracking on the held-out tool, online protocol
for tag, d, name in [("c", "phm", "PHM 2010"), ("d", "nuaa", "NUAA")]:
    h = E[d]["test"]
    ser = [dict(label="True wear", x=r(h["t"]), y=r(h["wear"]), marker="none", line="-", color=GREY),
           dict(label="UC-TCL", x=r(h["t"]), y=r(h["uctcl_onl"]), marker="none", line="--", color=VERM),
           dict(label="MonoBracket", x=r(h["t"]), y=r(h[ "mbpl_nopath_onl"]), marker="none", line="-", color=BLUE, proposed=True)]
    json.dump(dict(name=f"fig2{tag}_track_{d}", layout="quad", xlabel="Normalised tool life",
                   ylabel="Flank wear (mm)", xlim=[0, 1], legend="northwest", series=ser),
              open(os.path.join(out, f"fig2{tag}_track_{d}.json"), "w"), indent=1)
print("panels written to", out)
