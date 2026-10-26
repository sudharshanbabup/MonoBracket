"""
data_phm.py
-----------
Loader / preprocessor for the PHM 2010 Data Challenge tool-wear bundle
(experiments c1, c4, c6) as redistributed in the UniWear tidy dataset.

Columns : force_x, force_y, force_z, vibration_x, vibration_y, vibration_z,
          acoustic_emission_rms, timestamp, tool_wear, experiment_tag
Wear    : average flank wear of the three flutes, in mm.

Wear-stage definition for the 6 mm ball-nose tungsten-carbide cutter used in
the challenge (commonly cited rejection criterion VB ~ 0.16 mm):
    S0  Initial (break-in) wear   :          VB < 0.08 mm
    S1  Steady (uniform) wear     : 0.08 <= VB < 0.12 mm
    S2  Severe degradation        : 0.12 <= VB < 0.16 mm
    S3  Failure / tool rejection  : 0.16 <= VB
These edges lie inside the wear range spanned by every one of the three tools,
so all four stages are populated in each leave-one-tool-out fold.

Splitting is done at the *experiment* (tool) level: c1 / c4 / c6 are three
physically distinct cutters, so leave-one-tool-out is the natural protocol.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

CHANNELS = [
    "force_x", "force_y", "force_z",
    "vibration_x", "vibration_y", "vibration_z",
    "acoustic_emission_rms",
]
CHANNEL_PRETTY = [
    "Cutting force X", "Cutting force Y", "Cutting force Z",
    "Vibration X", "Vibration Y", "Vibration Z", "Acoustic emission RMS",
]
FS = 25.0                        # Hz (effective rate of the tidy bundle)
WEAR_EDGES = (0.08, 0.12, 0.16)  # mm
N_CLASSES = 4
STAGE_NAMES = ["Initial wear", "Steady wear", "Severe degradation", "Failure"]


def wear_stage(vb: float) -> int:
    e0, e1, e2 = WEAR_EDGES
    if vb < e0:
        return 0
    if vb < e1:
        return 1
    if vb < e2:
        return 2
    return 3


def _segment_cuts(df: pd.DataFrame, split_factor: float = 2.0):
    """
    Split one experiment into individual cutting passes.

    The tidy bundle stores one constant `tool_wear` value per pass, so a change
    of value normally marks a new cut.  That rule silently merges consecutive
    passes whose measured wear happened to be identical -- on PHM 2010 this
    produces segments up to 44x the median length.  Because label budgets are
    purchased per pass, one such segment can hand the learner several hundred
    windows at once and quietly break the stated budget.

    We therefore split any segment longer than `split_factor` x the median into
    the nearest whole number of median-length chunks.  Acquisition is uniform in
    time, so equal chunks are the correct reconstruction of equal passes.
    """
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    w = df["tool_wear"].to_numpy()
    brk = np.flatnonzero(np.diff(w) != 0) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(w)]])
    raw = [(int(a), int(b)) for a, b in zip(starts, ends)]

    med = float(np.median([b - a for a, b in raw]))
    out = []
    for a, b in raw:
        n = b - a
        if med > 0 and n > split_factor * med:
            k = max(2, int(round(n / med)))
            edges = np.linspace(a, b, k + 1).astype(int)
            out.extend((int(u), int(v)) for u, v in zip(edges[:-1], edges[1:])
                       if v > u)
        else:
            out.append((a, b))
    return out


def build_windows(csv_path: str, win: int = 64, stride: int = 16):
    """
    Returns the same 6-tuple contract as data_nasa.build_windows.
    grp = experiment id (0=c1, 1=c4, 2=c6).
    """
    df = pd.read_csv(csv_path, index_col=0)
    tags = ["c1", "c4", "c6"]
    X, y, c, grp, tpos, rid = [], [], [], [], [], []
    cut_uid = 0
    for gi, tag in enumerate(tags):
        sub = df[df["experiment_tag"] == tag]
        sub = sub.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        arr = sub[CHANNELS].to_numpy(dtype=np.float64).T      # (C, n)
        wear = sub["tool_wear"].to_numpy(dtype=np.float64)
        ts = sub["timestamp"].to_numpy(dtype=np.float64)
        for a, b in _segment_cuts(sub):
            if b - a < win:
                continue
            seg = arr[:, a:b]
            vb = float(wear[a])
            for s in range(0, seg.shape[1] - win + 1, stride):
                X.append(seg[:, s : s + win])
                y.append(vb)
                c.append(wear_stage(vb))
                grp.append(gi)
                tpos.append(ts[a])
                rid.append(cut_uid)
            cut_uid += 1
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
    X, y, c, g, t, rid = build_windows(os.path.join(here, "data", "phm2010_bundle.csv"))
    print("X", X.shape)
    print("class counts", np.bincount(c, minlength=4))
    print("groups", np.bincount(g))
    print("cuts", len(np.unique(rid)), "wear range", y.min(), y.max())
