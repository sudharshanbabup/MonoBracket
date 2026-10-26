"""
data_nasa.py
------------
Loader / preprocessor for the NASA (UC Berkeley) Milling Data Set.

Source file : mill.mat  (MATLAB struct array, 167 runs)
Fields      : case, run, VB, time, DOC, feed, material,
              smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle
Sampling    : 250 Hz, 9000 samples (~36 s) per run, 6 sensor channels.

Physical wear-stage definition (ISO 3685 flank-wear criterion VB_B = 0.30 mm):
    S0  Initial (break-in) wear   :          VB < 0.10 mm
    S1  Steady (uniform) wear     : 0.10 <= VB < 0.30 mm   (ends at VB_B)
    S2  Severe degradation        : 0.30 <= VB < 0.60 mm   (= 2 x VB_B)
    S3  Failure / tool rejection  : 0.60 <= VB

Splitting is done at the *case* (tool) level so that no window from a tool
appears in both train and test -> no temporal / tool leakage.
"""

from __future__ import annotations

import os
import numpy as np
import scipy.io as sio
from scipy import signal as sps

CHANNELS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
CHANNEL_PRETTY = [
    "AC spindle motor current",
    "DC spindle motor current",
    "Table vibration",
    "Spindle vibration",
    "Table acoustic emission",
    "Spindle acoustic emission",
]
FS = 250.0                       # Hz
WEAR_EDGES = (0.10, 0.30, 0.60)  # mm, ISO-3685 anchored (VB_B = 0.30 mm)
N_CLASSES = 4
STAGE_NAMES = ["Initial wear", "Steady wear", "Severe degradation", "Failure"]


# --------------------------------------------------------------------------- #
# raw loading
# --------------------------------------------------------------------------- #
def load_runs(mat_path: str):
    """Return a list of dicts, one per run, with raw signals + metadata."""
    mill = sio.loadmat(mat_path, squeeze_me=False)["mill"]
    runs = []
    for i in range(mill.shape[1]):
        r = mill[0, i]

        def scal(name):
            v = np.ravel(r[name])
            return float(v[0]) if v.size else np.nan

        sig = np.stack([np.ravel(r[c]).astype(np.float64) for c in CHANNELS], axis=0)
        if sig.shape[1] != 9000:          # a couple of runs are short/long
            continue
        runs.append(
            dict(
                case=int(scal("case")),
                run=int(scal("run")),
                VB=scal("VB"),
                time=scal("time"),
                DOC=scal("DOC"),
                feed=scal("feed"),
                material=int(scal("material")),
                x=sig,
            )
        )
    return runs


def wear_stage(vb: float) -> int:
    """Map continuous flank wear VB (mm) to a discrete wear stage."""
    e0, e1, e2 = WEAR_EDGES
    if vb < e0:
        return 0
    if vb < e1:
        return 1
    if vb < e2:
        return 2
    return 3


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def _valid_region(x: np.ndarray, frac: float = 0.70):
    """
    Keep the central `frac` of the record, which removes cutter entry / exit
    transients present at the start and end of every acquisition.
    """
    n = x.shape[1]
    lo = int(n * (1.0 - frac) / 2.0)
    hi = lo + int(n * frac)
    return x[:, lo:hi]


def build_windows(runs, win: int = 1024, stride: int = 256, frac: float = 0.70):
    """
    Slice every run into overlapping multivariate windows.

    Returns
    -------
    X      : (N, C, win) float32     raw windows
    y      : (N,)        float32     continuous wear VB in mm
    c      : (N,)        int64       wear stage
    grp    : (N,)        int64       case id (tool id) -- grouping key for CV
    traj   : (N,)        int64       trajectory id  = case id
    tpos   : (N,)        float32     cumulative machining time of the run (s)
    """
    X, y, c, grp, tpos, rid = [], [], [], [], [], []
    for k, r in enumerate(runs):
        if not np.isfinite(r["VB"]):
            continue                      # 21 runs have no wear measurement
        xs = _valid_region(r["x"], frac)
        n = xs.shape[1]
        for s in range(0, n - win + 1, stride):
            X.append(xs[:, s : s + win])
            y.append(r["VB"])
            c.append(wear_stage(r["VB"]))
            grp.append(r["case"])
            tpos.append(r["time"])
            rid.append(k)
    X = np.asarray(X, dtype=np.float32)
    return (
        X,
        np.asarray(y, dtype=np.float32),
        np.asarray(c, dtype=np.int64),
        np.asarray(grp, dtype=np.int64),
        np.asarray(tpos, dtype=np.float32),
        np.asarray(rid, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def fit_channel_stats(X: np.ndarray):
    """Per-channel mean / std computed on the training split only."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_stats(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# hand-crafted features (for the classical RF / SVM baselines)
# --------------------------------------------------------------------------- #
def handcrafted_features(X: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    17 time- and frequency-domain descriptors per channel, the standard
    condition-monitoring feature bank.
    """
    N, C, T = X.shape
    eps = 1e-12
    feats = []

    absx = np.abs(X)
    rms = np.sqrt(np.mean(X ** 2, axis=2) + eps)
    mean = X.mean(axis=2)
    std = X.std(axis=2) + eps
    peak = absx.max(axis=2)
    p2p = X.max(axis=2) - X.min(axis=2)
    mean_abs = absx.mean(axis=2) + eps
    sra = (np.mean(np.sqrt(absx), axis=2) ** 2) + eps
    skew = np.mean(((X - mean[:, :, None]) / std[:, :, None]) ** 3, axis=2)
    kurt = np.mean(((X - mean[:, :, None]) / std[:, :, None]) ** 4, axis=2)
    crest = peak / (rms + eps)
    shape = rms / mean_abs
    impulse = peak / mean_abs
    clearance = peak / sra
    # zero-crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(X - mean[:, :, None]), axis=2)) > 0, axis=2)

    # spectral descriptors
    w = np.hanning(T)[None, None, :]
    P = np.abs(np.fft.rfft(X * w, axis=2)) ** 2
    f = np.fft.rfftfreq(T, d=1.0 / fs)[None, None, :]
    Ps = P.sum(axis=2) + eps
    f_centroid = (P * f).sum(axis=2) / Ps
    f_spread = np.sqrt(((f - f_centroid[:, :, None]) ** 2 * P).sum(axis=2) / Ps)
    Pn = P / Ps[:, :, None]
    spec_ent = -(Pn * np.log(Pn + eps)).sum(axis=2) / np.log(P.shape[2])
    # band energy ratios: 4 octave-ish bands up to Nyquist (125 Hz)
    edges = [0, 15, 40, 80, 125]
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f[0, 0] >= a) & (f[0, 0] < b)
        bands.append(P[:, :, m].sum(axis=2) / Ps)

    for arr in [rms, mean, std, peak, p2p, skew, kurt, crest, shape,
                impulse, clearance, zcr, f_centroid, f_spread, spec_ent] + bands:
        feats.append(arr)

    F = np.concatenate([a.reshape(N, C) for a in feats], axis=1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def feature_names():
    base = ["rms", "mean", "std", "peak", "p2p", "skew", "kurt", "crest",
            "shape", "impulse", "clearance", "zcr", "f_centroid", "f_spread",
            "spec_entropy", "band0_15", "band15_40", "band40_80", "band80_125"]
    return [f"{b}_{c}" for b in base for c in CHANNELS]


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = load_runs(os.path.join(here, "data", "mill.mat"))
    print(f"runs loaded: {len(runs)}")
    X, y, c, g, t, rid = build_windows(runs)
    print("X", X.shape, "y", y.shape)
    print("class counts", np.bincount(c, minlength=4))
    print("cases", np.unique(g))
    F = handcrafted_features(X[:64])
    print("features", F.shape, len(feature_names()))
