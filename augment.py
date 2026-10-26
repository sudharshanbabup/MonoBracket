"""
augment.py
----------
Physically-motivated stochastic augmentations for multi-sensor machining
windows.  Every transform is written so that it *cannot* change the wear stage
of the window: it perturbs the acquisition chain (gain, noise, sensor loss,
narrow-band interference, small time shift), never the degradation content.

    T1 amplitude scaling      : gain drift of the charge amplifier / DAQ
    T2 additive Gaussian noise: sensor and quantisation noise
    T3 channel dropout        : loss of one IIoT sensor node
    T4 random time shift/crop : lack of cut-entry synchronisation
    T5 band-stop masking      : masking of a spindle-harmonic band
    T6 random polarity flip   : accelerometer mounting orientation
"""

from __future__ import annotations

import torch


def _rand(b, device, lo, hi):
    return torch.rand(b, device=device) * (hi - lo) + lo


def amplitude_scale(x, lo=0.8, hi=1.25):
    B, C, _ = x.shape
    s = _rand(B * C, x.device, lo, hi).view(B, C, 1)
    return x * s


def add_noise(x, snr_db_lo=12.0, snr_db_hi=30.0):
    B, C, T = x.shape
    p_sig = x.pow(2).mean(dim=2, keepdim=True) + 1e-12
    snr = _rand(B * C, x.device, snr_db_lo, snr_db_hi).view(B, C, 1)
    p_noise = p_sig / torch.pow(10.0, snr / 10.0)
    return x + torch.randn_like(x) * p_noise.sqrt()


def channel_dropout(x, p=0.25):
    B, C, _ = x.shape
    m = (torch.rand(B, C, 1, device=x.device) > p).float()
    # never drop every channel
    empty = m.sum(dim=1, keepdim=True) == 0
    m = torch.where(empty.expand_as(m), torch.ones_like(m), m)
    return x * m


def time_shift(x, max_frac=0.15):
    B, C, T = x.shape
    out = torch.empty_like(x)
    sh = torch.randint(-int(T * max_frac), int(T * max_frac) + 1, (B,))
    for i in range(B):
        out[i] = torch.roll(x[i], shifts=int(sh[i]), dims=-1)
    return out


def band_mask(x, max_frac=0.12):
    """Zero a random contiguous band of the rFFT spectrum, then transform back."""
    B, C, T = x.shape
    X = torch.fft.rfft(x, dim=2)
    F = X.shape[2]
    w = max(1, int(F * max_frac))
    start = torch.randint(0, max(1, F - w), (B,))
    mask = torch.ones(B, 1, F, device=x.device, dtype=X.real.dtype)
    for i in range(B):
        mask[i, 0, start[i] : start[i] + w] = 0.0
    return torch.fft.irfft(X * mask, n=T, dim=2)


def polarity_flip(x, p=0.5):
    B, C, _ = x.shape
    s = torch.where(torch.rand(B, C, 1, device=x.device) < p, -1.0, 1.0)
    return x * s


AUGS = [amplitude_scale, add_noise, channel_dropout, time_shift, band_mask,
        polarity_flip]


def weak_augment(x):
    """Weak view: gain + light noise only (used as the FixMatch weak branch)."""
    return add_noise(amplitude_scale(x, 0.95, 1.05), 25.0, 40.0)


def strong_augment(x, n_ops: int = 3, generator=None):
    """Strong view: n_ops transforms sampled without replacement."""
    idx = torch.randperm(len(AUGS), generator=generator)[:n_ops]
    for i in idx.tolist():
        x = AUGS[i](x)
    return x


def contrastive_pair(x):
    return strong_augment(x), strong_augment(x)
