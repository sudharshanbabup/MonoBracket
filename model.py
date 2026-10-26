"""
model.py
--------
Backbone and heads for UC-TCL.

Encoder : dilated temporal convolutional network (TCN) with squeeze-excitation
          channel attention.  Dropout is applied *inside* every residual block
          so that Monte-Carlo dropout approximates a deep Gaussian process
          posterior over the whole representation, not only over the classifier.

Heads   : - projection head g (contrastive space, L2-normalised)
          - classification head h  -> K logits (wear stage)
          - regression head r      -> (mu, log sigma^2) heteroscedastic wear
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(4, ch // r))
        self.fc2 = nn.Linear(max(4, ch // r), ch)

    def forward(self, x):                     # x: (B, C, T)
        s = x.mean(dim=2)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.unsqueeze(2)


class TCNBlock(nn.Module):
    """Residual dilated-convolution block with dropout on both branches."""

    def __init__(self, cin, cout, k=7, dilation=1, p_drop=0.15):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SqueezeExcite(cout)
        self.do = nn.Dropout(p_drop)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.do(F.gelu(self.b1(self.c1(x))))
        h = self.do(F.gelu(self.b2(self.c2(h))))
        h = self.se(h)
        return F.gelu(h + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, in_ch: int, width: int = 48, depth: int = 4,
                 p_drop: float = 0.15, feat_dim: int = 128, pool: int = 4,
                 stem_stride: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 9, stride=stem_stride, padding=4),
            nn.BatchNorm1d(width), nn.GELU())
        blocks, cin = [], width
        for d in range(depth):
            cout = width * (2 ** min(d, 2))
            blocks.append(TCNBlock(cin, cout, k=7, dilation=2 ** d, p_drop=p_drop))
            blocks.append(nn.MaxPool1d(pool))
            cin = cout
        self.body = nn.Sequential(*blocks)
        self.out_ch = cin
        # global statistics pooling (mean + std) doubles the descriptor
        self.fc = nn.Linear(2 * cin, feat_dim)
        self.do = nn.Dropout(p_drop)
        self.feat_dim = feat_dim

    def forward(self, x):
        h = self.body(self.stem(x))
        m = h.mean(dim=2)
        s = h.std(dim=2)
        z = torch.cat([m, s], dim=1)
        return self.fc(self.do(z))


class ProjectionHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ClassifierHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 96, p_drop: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, hidden),
                                 nn.GELU(), nn.Dropout(p_drop),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class RegressorHead(nn.Module):
    """Heteroscedastic head: returns mu and log sigma^2 (clamped for stability)."""

    def __init__(self, dim: int, hidden: int = 96, p_drop: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, hidden),
                                 nn.GELU(), nn.Dropout(p_drop),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        o = self.net(x)
        mu = o[:, 0]
        logvar = torch.clamp(o[:, 1], min=-8.0, max=4.0)
        return mu, logvar


class UCTCL(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 4, width: int = 48,
                 depth: int = 4, feat_dim: int = 128, p_enc: float = 0.15,
                 p_head: float = 0.30, pool: int = 4, stem_stride: int = 2):
        super().__init__()
        self.enc = Encoder(in_ch, width, depth, p_enc, feat_dim, pool, stem_stride)
        self.proj = ProjectionHead(feat_dim)
        self.cls = ClassifierHead(feat_dim, n_classes, p_drop=p_head)
        self.reg = RegressorHead(feat_dim, p_drop=p_head)
        self.n_classes = n_classes

    def forward(self, x, mode: str = "all"):
        f = self.enc(x)
        if mode == "proj":
            return self.proj(f)
        if mode == "feat":
            return f
        mu, logvar = self.reg(f)
        return self.cls(f), mu, logvar, f


# --------------------------------------------------------------------------- #
# MC-dropout inference
# --------------------------------------------------------------------------- #
def enable_mc_dropout(model: nn.Module):
    """eval() everything (freezes BatchNorm running stats) then re-enable dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_predict(model: nn.Module, X: torch.Tensor, n_mc: int = 20,
               batch: int = 256, device: str = "cpu"):
    """
    Returns
    -------
    probs  : (M, N, K)
    mu     : (M, N)
    logvar : (M, N)
    """
    enable_mc_dropout(model)
    P, MU, LV = [], [], []
    for _ in range(n_mc):
        ps, ms, ls = [], [], []
        for i in range(0, len(X), batch):
            xb = X[i : i + batch].to(device)
            logit, mu, lv, _ = model(xb)
            ps.append(torch.softmax(logit, dim=1).cpu())
            ms.append(mu.cpu())
            ls.append(lv.cpu())
        P.append(torch.cat(ps))
        MU.append(torch.cat(ms))
        LV.append(torch.cat(ls))
    model.eval()
    return (torch.stack(P).numpy(), torch.stack(MU).numpy(), torch.stack(LV).numpy())


@torch.no_grad()
def deterministic_predict(model: nn.Module, X: torch.Tensor, batch: int = 256,
                          device: str = "cpu"):
    model.eval()
    ps, ms = [], []
    for i in range(0, len(X), batch):
        xb = X[i : i + batch].to(device)
        logit, mu, lv, _ = model(xb)
        ps.append(torch.softmax(logit, dim=1).cpu())
        ms.append(mu.cpu())
    return torch.cat(ps).numpy(), torch.cat(ms).numpy()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
