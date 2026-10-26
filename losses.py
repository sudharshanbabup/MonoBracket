"""
losses.py
---------
Objective functions used by UC-TCL.

1. Temporal contrastive loss (TCL)  -- a temporally-weighted generalisation of
   NT-Xent / SupCon.  Positives of an anchor are (a) its augmented view and
   (b) windows drawn from the same tool trajectory whose machining times are
   close, weighted by a Gaussian temporal kernel.  This injects the physical
   prior that flank wear evolves smoothly in time, which plain SimCLR (which
   treats every other window as a negative) actively destroys.

2. Class-balanced focal loss for the few labelled samples.

3. Heteroscedastic Gaussian negative log-likelihood for continuous wear.

4. Certainty-weighted pseudo-label loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. temporal contrastive loss
# --------------------------------------------------------------------------- #
def temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                              traj: torch.Tensor, tpos: torch.Tensor,
                              tau: float = 0.1, sigma: float = 1.0,
                              w_aug: float = 1.0) -> torch.Tensor:
    """
    Parameters
    ----------
    z1, z2 : (B, d) L2-normalised projections of two augmented views
    traj   : (B,)   trajectory (tool) id of each anchor
    tpos   : (B,)   normalised machining time of each anchor in [0, 1]
    tau    : temperature
    sigma  : bandwidth of the temporal kernel (in normalised-time units)

    Loss
    ----
        L = -(1/2B) sum_i sum_{p in P(i)} w~_{ip} log
                exp(s_ip / tau) / sum_{a != i} exp(s_ia / tau)

    with w~ the row-normalised positive weights.  Setting sigma -> 0 recovers
    NT-Xent exactly (only the augmented view survives as a positive).
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2B, d)
    sim = (z @ z.t()) / tau                            # (2B, 2B)

    eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))          # drop self-similarity
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # the diagonal is -inf and its weight is exactly zero; replacing it by a
    # finite value keeps the product 0 * (-inf) from producing NaN.
    log_prob = log_prob.masked_fill(eye, 0.0)

    tr = torch.cat([traj, traj])
    tp = torch.cat([tpos, tpos])
    dt = (tp[:, None] - tp[None, :]).abs()
    same_traj = (tr[:, None] == tr[None, :])

    # temporal-neighbour weights
    W = torch.exp(-(dt ** 2) / (2.0 * sigma ** 2)) * same_traj.float()
    W = W.masked_fill(eye, 0.0)

    # the augmented view of the same window is always a positive with weight w_aug
    idx = torch.arange(B, device=z.device)
    W[idx, idx + B] = w_aug
    W[idx + B, idx] = w_aug

    Wn = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)
    loss = -(Wn * log_prob).sum(dim=1).mean()
    return loss


# --------------------------------------------------------------------------- #
# 2. class-balanced focal loss
# --------------------------------------------------------------------------- #
def focal_ce(logits: torch.Tensor, target: torch.Tensor,
             class_w: torch.Tensor | None = None, gamma: float = 1.5,
             reduction: str = "mean", sample_w: torch.Tensor | None = None):
    """
    CB-focal:  L_i = w_{y_i} (1 - p_{i,y_i})^gamma  (-log p_{i,y_i})
    gamma = 0 and class_w = 1 reduces to plain cross-entropy.
    """
    logp = F.log_softmax(logits, dim=1)
    logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
    pt = logpt.exp()
    loss = -((1.0 - pt) ** gamma) * logpt
    if class_w is not None:
        loss = loss * class_w[target]
    if sample_w is not None:
        loss = loss * sample_w
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def soft_focal_ce(logits: torch.Tensor, target_prob: torch.Tensor,
                  class_w: torch.Tensor | None = None, gamma: float = 1.5,
                  sample_w: torch.Tensor | None = None):
    """Soft-target variant used by the Mean-Teacher / consistency baselines."""
    logp = F.log_softmax(logits, dim=1)
    loss = -(target_prob * logp).sum(dim=1)
    if gamma > 0:
        pt = (target_prob * logp.exp()).sum(dim=1).clamp(1e-6, 1.0)
        loss = loss * (1.0 - pt) ** gamma
    if class_w is not None:
        loss = loss * (target_prob * class_w[None, :]).sum(dim=1)
    if sample_w is not None:
        loss = loss * sample_w
    return loss.mean()


# --------------------------------------------------------------------------- #
# 3. heteroscedastic Gaussian NLL
# --------------------------------------------------------------------------- #
def hetero_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor,
               sample_w: torch.Tensor | None = None):
    """
    -log N(y | mu, sigma^2) up to the constant 0.5 log 2pi:
        0.5 * [ (y - mu)^2 / sigma^2 + log sigma^2 ]
    """
    loss = 0.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar)
    if sample_w is not None:
        loss = loss * sample_w
    return loss.mean()


# --------------------------------------------------------------------------- #
# 4. ramp-up schedule
# --------------------------------------------------------------------------- #
def sigmoid_rampup(t: int, length: int) -> float:
    """Laine & Aila ramp-up: exp(-5 (1 - t/length)^2), saturating at 1."""
    if length <= 0:
        return 1.0
    x = min(max(t / float(length), 0.0), 1.0)
    import math
    return float(math.exp(-5.0 * (1.0 - x) ** 2))
