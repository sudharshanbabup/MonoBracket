"""
data_nuaa.py
------------
Loader / preprocessor for the NUAA Ideahouse milling tool-wear data set, as
redistributed in the UniWear tidy bundle
(md5 f042426ef6c33eee...; `data/nuaa_bundle.csv`).

Nine tungsten-carbide end mills (W1 ... W9) are run to failure under a full
3 x 3 factorial of process conditions:

    feed per tooth      0.045, 0.050, 0.055 mm
    spindle speed       1750, 1800, 1850 rpm
    axial depth of cut  2.5, 3.0, 3.5 mm

Eight synchronised channels are recorded: axial force, two bending moments,
torsion, two vibration axes, spindle power and spindle current.  Flank wear of
each of the four blades is measured and the tidy bundle carries their mean as
`tool_wear` (mm), interpolated onto every sample.

Why this data set matters here
------------------------------
It is the only one of the three with (a) nine independent tools, giving a
nine-fold leave-one-tool-out protocol instead of three, (b) a designed spread of
cutting conditions, so cross-tool generalisation is confounded with process
change exactly as it is in a real shop, and (c) both spindle current and power,
which the problem statement names as motor-current signals.

Wear-stage definition.  The tools span 0.049 - 0.313 mm, and the accepted
rejection criterion for this insert geometry is VB ~ 0.30 mm:

    S0  Initial (break-in)  :          VB < 0.10 mm
    S1  Steady              : 0.10 <= VB < 0.15 mm
    S2  Severe degradation  : 0.15 <= VB < 0.20 mm
    S3  Failure             : 0.20 <= VB

Because wear is interpolated per sample rather than held constant per pass, a
"pass" is reconstructed as a fixed-duration block of the stream; labels are
purchased per block, matching the metrology practice of stopping the machine at
intervals.

Note on evaluation.  Unlike the other two benchmarks, the nine tools do not all
span all four stages -- W3 and W9 never reach failure, W5 never leaves severe
degradation.  Macro-F1 on a held-out tool is therefore averaged over the stages
that tool actually visits, which is the standard convention for leave-one-group-
out with heterogeneous groups.  On NASA Milling and PHM 2010, where every fold
contains all four stages by construction, the rule is a no-op.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

CHANNELS = [
    "axial_force", "bending_moment_x", "bending_moment_y", "torsion",
    "vibration1", "vibration2", "spindle_power", "spindle_current",
]
CHANNEL_PRETTY = [
    "Axial force", "Bending moment X", "Bending moment Y", "Torsion",
    "Vibration 1", "Vibration 2", "Spindle power", "Spindle current",
]
FS = 15.15                       # Hz (dt = 0.066 s, uniform across all nine tools)
WEAR_EDGES = (0.10, 0.15, 0.20)  # mm
N_CLASSES = 4
STAGE_NAMES = ["Initial wear", "Steady wear", "Severe degradation", "Failure"]
TOOLS = [f"W{i}" for i in range(1, 10)]

# rows per reconstructed cutting pass; at 15.15 Hz, 300 samples is ~20 s of
# cutting, the granularity at which a metrologist could stop and measure
PASS_ROWS = 300


def wear_stage(vb: float) -> int:
    e0, e1, e2 = WEAR_EDGES
    if vb < e0:
        return 0
    if vb < e1:
        return 1
    if vb < e2:
        return 2
    return 3


def _passes(n: int, pass_rows: int = PASS_ROWS):
    """Contiguous equal-length blocks standing in for cutting passes."""
    k = max(1, n // pass_rows)
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def build_windows(csv_path: str, win: int = 128, stride: int = 48,
                  pass_rows: int = PASS_ROWS):
    """
    Same 6-tuple contract as the other loaders.

    Returns
    -------
    X    : (N, 8, win) float32
    y    : (N,)  float32  mean flank wear in mm
    c    : (N,)  int64    wear stage
    grp  : (N,)  int64    tool id (0 = W1 ... 8 = W9)
    tpos : (N,)  float32  cumulative machining time (s) at the window start
    rid  : (N,)  int64    unique pass id, the unit at which labels are purchased
    """
    df = pd.read_csv(csv_path, index_col=0,
                     usecols=lambda c: c in set(
                         CHANNELS + ["timestamp", "tool_wear", "experiment_tag"]
                     ) or c == "Unnamed: 0")
    X, y, c, grp, tpos, rid = [], [], [], [], [], []
    pid = 0
    for gi, tag in enumerate(TOOLS):
        sub = df[df["experiment_tag"] == tag]
        if sub.empty:
            continue
        sub = sub.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        arr = sub[CHANNELS].to_numpy(dtype=np.float64).T          # (C, n)
        wear = sub["tool_wear"].to_numpy(dtype=np.float64)
        ts = sub["timestamp"].to_numpy(dtype=np.float64)
        for a, b in _passes(arr.shape[1], pass_rows):
            seg = arr[:, a:b]
            if seg.shape[1] < win:
                continue
            # The published bundle contains one contiguous 149-sample dropout
            # (~10 s) in tool W2, affecting the two bending moments and torsion.
            # We discard the passes it touches rather than impute across it;
            # this costs 0.3% of the passes and invents no data.
            if not np.isfinite(seg).all():
                continue
            # one wear value per pass: the median over the pass, which is what a
            # single measurement at that point in the tool life would report
            vb = float(np.median(wear[a:b]))
            for s in range(0, seg.shape[1] - win + 1, stride):
                X.append(seg[:, s : s + win])
                y.append(vb)
                c.append(wear_stage(vb))
                grp.append(gi)
                tpos.append(ts[a + s])
                rid.append(pid)
            pid += 1
    X = np.asarray(X, dtype=np.float32)
    return (
        X,
        np.asarray(y, dtype=np.float32),
        np.asarray(c, dtype=np.int64),
        np.asarray(grp, dtype=np.int64),
        np.asarray(tpos, dtype=np.float32),
        np.asarray(rid, dtype=np.int64),
    )


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X, y, c, g, t, rid = build_windows(os.path.join(here, "data", "nuaa_bundle.csv"))
    print("X", X.shape)
    bc = np.bincount(c, minlength=4)
    print("class counts", bc, "IR=%.2f" % (bc.max() / max(1, bc.min())))
    print("tools", np.bincount(g), "passes", len(np.unique(rid)))
    print("wear range %.4f .. %.4f mm" % (y.min(), y.max()))
    for gi in range(9):
        m = g == gi
        print(f"  W{gi+1}: {m.sum():5d} windows, "
              f"stages {np.bincount(c[m], minlength=4)}, "
              f"wear {y[m].min():.3f}-{y[m].max():.3f}")
