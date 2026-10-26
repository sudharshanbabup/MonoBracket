"""
methods.py
----------
Every training procedure benchmarked in the paper, all sharing the same
encoder, optimiser, schedule and gradient budget so that the comparison is
about the *learning rule*, not about capacity or compute.

Deep methods
------------
sup_low       supervised on the labelled subset only            (lower bound)
sup_full      supervised on 100 % of the training labels        (upper bound)
pseudo        Lee-2013 self-training, fixed global conf. threshold
fixmatch      weak/strong consistency, fixed global conf. threshold
meanteacher   EMA teacher + MSE consistency
simclr        SimCLR pre-training + linear probe on the labels
tcl_ft        temporal contrastive pre-training + fine-tune (no pseudo-labels)
uctcl         OURS: TCL + MC-dropout + adaptive per-class uncertainty PL

Ablations of `uctcl` are selected through the `cfg` dictionary.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F

from model import UCTCL, mc_predict, deterministic_predict, enable_mc_dropout
from losses import (temporal_contrastive_loss, focal_ce, soft_focal_ce,
                    hetero_nll, sigmoid_rampup)
import monopath
from augment import weak_augment, strong_augment, contrastive_pair
from uncertainty import (mc_classification_uncertainty, normalised_epistemic,
                         mc_regression_uncertainty, adaptive_thresholds,
                         uncertainty_mask, percentile_schedule,
                         monotonicity_filter, effective_number_weights,
                         regression_stage_probs, fuse_probs)

DEFAULT = dict(
    lr=2e-3, wd=1e-4, batch=128, ubatch=128,
    e_con=20, e_warm=30, n_rounds=3, e_round=10,
    tau_con=0.10, sigma_t=0.15, gamma=1.5, beta_cb=0.999,
    n_mc=10, rho0=0.05, rho_max=0.50, kappa=1.5,
    lam_u=1.0, lam_reg=0.5, lam_con_ft=0.0,
    conf_thr=0.95, ema=0.99, cons_w=10.0,
    use_tcl=True, use_mc=True, per_class_tau=True, use_mono=True,
    class_balanced=True, temporal_weight=True,
    use_fusion=False, require_agreement=False, alpha_fuse=0.5,
    mono_relabel=False,
    # --- v5: monotone bracketing + monotone-path decoding (monopath.py) ---
    mono_bracket=False,   # use labelled passes as physical brackets
    path_decode=False,    # label = MAP non-decreasing stage path
    cert_only=False,      # ablation: accept certified windows only
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _batches(n, bs, rng, shuffle=True):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def _make_model(in_ch, n_classes, cfg, seed):
    torch.manual_seed(seed)
    return UCTCL(in_ch, n_classes, width=cfg["width"], depth=cfg["depth"],
                 pool=cfg["pool"], stem_stride=cfg["stem_stride"],
                 p_enc=0.15, p_head=0.30)


def _opt(model, cfg, epochs, steps_per_epoch):
    o = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    s = torch.optim.lr_scheduler.OneCycleLR(
        o, max_lr=cfg["lr"], total_steps=max(20, epochs * steps_per_epoch),
        pct_start=0.25)
    return o, s


def _class_weights(y, K, cfg, device):
    if not cfg["class_balanced"]:
        return torch.ones(K, device=device)
    cnt = np.bincount(y, minlength=K)
    return torch.tensor(effective_number_weights(cnt, cfg["beta_cb"]),
                        dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# stage 1: contrastive pre-training
# --------------------------------------------------------------------------- #
def contrastive_pretrain(model, X, traj, tpos, cfg, seed, temporal=True):
    rng = np.random.default_rng(seed)
    n, bs = len(X), cfg["batch"]
    spe = max(1, int(np.ceil(n / bs)))
    opt, sch = _opt(model, cfg, cfg["e_con"], spe)
    tp = torch.tensor(tpos, dtype=torch.float32)
    tr = torch.tensor(traj, dtype=torch.long)
    sigma = cfg["sigma_t"] if temporal else 1e-4
    model.train()
    hist = []
    for ep in range(cfg["e_con"]):
        tot = 0.0
        for b in _batches(n, bs, rng):
            if len(b) < 4:
                continue
            xb = X[b]
            v1, v2 = contrastive_pair(xb)
            z1 = model(v1, mode="proj")
            z2 = model(v2, mode="proj")
            loss = temporal_contrastive_loss(z1, z2, tr[b], tp[b],
                                             tau=cfg["tau_con"], sigma=sigma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sch.step()
            tot += float(loss) * len(b)
        hist.append(tot / n)
    return hist


# --------------------------------------------------------------------------- #
# stage 2: supervised training on an index set
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _val_score(model, X, y_cls, y_reg, val_idx, K):
    """
    Model-selection criterion evaluated on the held-out slice of the *labelled*
    budget: macro-F1 penalised by the mean absolute wear error.  Higher is
    better.  Every deep method is selected with this identical rule, so no
    method is advantaged by the choice of stopping epoch.
    """
    from sklearn.metrics import f1_score
    model.eval()
    logit, mu, lv, _ = model(X[val_idx])
    pred = logit.argmax(1).numpy()
    f1 = f1_score(y_cls[val_idx], pred, average="macro", zero_division=0)
    mae = float(np.abs(mu.numpy() - y_reg[val_idx]).mean())
    return float(f1) - 0.5 * mae


def supervised_fit(model, X, y_cls, y_reg, idx, cfg, seed, epochs,
                   sample_w=None, soft_reg=None, freeze_encoder=False,
                   K=4, opt_sch=None, val_idx=None, eval_every=4):
    rng = np.random.default_rng(seed)
    dev = "cpu"
    cw = _class_weights(y_cls[idx], K, cfg, dev)
    bs = cfg["batch"]
    spe = max(1, int(np.ceil(len(idx) / bs)))
    if opt_sch is None:
        params = ([p for p in model.cls.parameters()] +
                  [p for p in model.reg.parameters()]) if freeze_encoder \
            else list(model.parameters())
        o = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
        s = torch.optim.lr_scheduler.OneCycleLR(
            o, max_lr=cfg["lr"], total_steps=max(20, epochs * spe), pct_start=0.25)
    else:
        o, s = opt_sch

    yt = torch.tensor(y_cls, dtype=torch.long)
    yr = torch.tensor(y_reg, dtype=torch.float32)
    sw = torch.tensor(sample_w, dtype=torch.float32) if sample_w is not None else None
    sr = torch.tensor(soft_reg, dtype=torch.float32) if soft_reg is not None else None

    best_score, best_state = -1e18, None
    for ep in range(epochs):
        model.train()
        if freeze_encoder:
            model.enc.eval()
        for b in _batches(len(idx), bs, rng):
            j = idx[b]
            xb = weak_augment(X[j])
            logit, mu, lv, _ = model(xb)
            w = sw[j] if sw is not None else None
            l_cls = focal_ce(logit, yt[j], cw, cfg["gamma"], sample_w=w)
            tgt = sr[j] if sr is not None else yr[j]
            l_reg = hetero_nll(mu, lv, tgt, sample_w=w)
            loss = l_cls + cfg["lam_reg"] * l_reg
            o.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            o.step(); s.step()
        if val_idx is not None and len(val_idx) > 0 and \
                (ep % eval_every == 0 or ep == epochs - 1):
            sc = _val_score(model, X, y_cls, y_reg, val_idx, K)
            if sc > best_score:
                best_score = sc
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def run_sup(X, y_cls, y_reg, lab_idx, cfg, seed, K, in_ch, val_idx=None):
    m = _make_model(in_ch, K, cfg, seed)
    ep = cfg["e_warm"] + cfg["n_rounds"] * cfg["e_round"]
    supervised_fit(m, X, y_cls, y_reg, lab_idx, cfg, seed, ep, K=K, val_idx=val_idx)
    return m, {}


def run_pseudo_fixed(X, y_cls, y_reg, lab_idx, unl_idx, cfg, seed, K, in_ch,
                     val_idx=None):
    """Lee (2013): retrain on samples whose max soft-max probability > conf_thr."""
    m = _make_model(in_ch, K, cfg, seed)
    supervised_fit(m, X, y_cls, y_reg, lab_idx, cfg, seed, cfg["e_warm"], K=K,
                   val_idx=val_idx)
    diag = []
    yc = y_cls.copy(); yrg = y_reg.copy()
    for t in range(cfg["n_rounds"]):
        probs, mu = deterministic_predict(m, X[unl_idx])
        conf = probs.max(1); pred = probs.argmax(1)
        keep = conf >= cfg["conf_thr"]
        sel = unl_idx[keep]
        yc[sel] = pred[keep]; yrg[sel] = mu[keep]
        idx = np.concatenate([lab_idx, sel])
        w = np.ones(len(y_cls), dtype=np.float32)
        diag.append(_pl_diag(sel, pred[keep], y_cls, K))
        supervised_fit(m, X, yc, yrg, idx, cfg, seed + t, cfg["e_round"],
                       sample_w=w, K=K, val_idx=val_idx)
    return m, dict(pl=diag)


def run_fixmatch(X, y_cls, y_reg, lab_idx, unl_idx, cfg, seed, K, in_ch,
                 val_idx=None):
    m = _make_model(in_ch, K, cfg, seed)
    rng = np.random.default_rng(seed)
    dev = "cpu"
    cw = _class_weights(y_cls[lab_idx], K, cfg, dev)
    ep = cfg["e_warm"] + cfg["n_rounds"] * cfg["e_round"]
    spe = max(1, int(np.ceil(len(lab_idx) / cfg["batch"])))
    o, s = _opt(m, cfg, ep, spe)
    yt = torch.tensor(y_cls, dtype=torch.long)
    yr = torch.tensor(y_reg, dtype=torch.float32)
    acc_n, acc_c = 0, 0
    for e in range(ep):
        m.train()
        lam = cfg["lam_u"] * sigmoid_rampup(e, max(1, ep // 3))
        for b in _batches(len(lab_idx), cfg["batch"], rng):
            j = lab_idx[b]
            logit, mu, lv, _ = m(weak_augment(X[j]))
            loss = focal_ce(logit, yt[j], cw, cfg["gamma"]) + \
                cfg["lam_reg"] * hetero_nll(mu, lv, yr[j])
            u = rng.choice(unl_idx, size=min(cfg["ubatch"], len(unl_idx)),
                           replace=False)
            xu = X[u]
            with torch.no_grad():
                m.eval()
                pw = torch.softmax(m(weak_augment(xu))[0], dim=1)
                m.train()
            conf, ph = pw.max(1)
            msk = (conf >= cfg["conf_thr"]).float()
            acc_n += len(u); acc_c += int(msk.sum())
            if msk.sum() > 0:
                lo_s = m(strong_augment(xu))[0]
                lu = focal_ce(lo_s, ph, cw, cfg["gamma"], reduction="none")
                loss = loss + lam * (lu * msk).sum() / msk.sum().clamp_min(1.0)
            o.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            o.step(); s.step()
    return m, dict(mask_rate=acc_c / max(1, acc_n))


def run_meanteacher(X, y_cls, y_reg, lab_idx, unl_idx, cfg, seed, K, in_ch,
                    val_idx=None):
    m = _make_model(in_ch, K, cfg, seed)
    teacher = copy.deepcopy(m)
    for p in teacher.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(seed)
    cw = _class_weights(y_cls[lab_idx], K, cfg, "cpu")
    ep = cfg["e_warm"] + cfg["n_rounds"] * cfg["e_round"]
    spe = max(1, int(np.ceil(len(lab_idx) / cfg["batch"])))
    o, s = _opt(m, cfg, ep, spe)
    yt = torch.tensor(y_cls, dtype=torch.long)
    yr = torch.tensor(y_reg, dtype=torch.float32)
    for e in range(ep):
        m.train()
        lam = cfg["cons_w"] * sigmoid_rampup(e, max(1, ep // 3))
        for b in _batches(len(lab_idx), cfg["batch"], rng):
            j = lab_idx[b]
            logit, mu, lv, _ = m(weak_augment(X[j]))
            loss = focal_ce(logit, yt[j], cw, cfg["gamma"]) + \
                cfg["lam_reg"] * hetero_nll(mu, lv, yr[j])
            u = rng.choice(unl_idx, size=min(cfg["ubatch"], len(unl_idx)),
                           replace=False)
            xu = X[u]
            with torch.no_grad():
                teacher.train()
                pt = torch.softmax(teacher(weak_augment(xu))[0], dim=1)
            ps = torch.softmax(m(strong_augment(xu))[0], dim=1)
            loss = loss + lam * F.mse_loss(ps, pt)
            o.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            o.step(); s.step()
            with torch.no_grad():
                for tp_, sp in zip(teacher.parameters(), m.parameters()):
                    tp_.mul_(cfg["ema"]).add_(sp, alpha=1 - cfg["ema"])
                for tb, sb in zip(teacher.buffers(), m.buffers()):
                    tb.copy_(sb)
    return teacher, {}


def run_contrastive_then_head(X, y_cls, y_reg, lab_idx, cfg, seed, K, in_ch,
                              temporal, traj, tpos, freeze, pre_loader=None,
                              val_idx=None):
    m = _make_model(in_ch, K, cfg, seed)
    hist = []
    if pre_loader is not None:
        hist = pre_loader(m, "tcl" if temporal else "simclr")
    else:
        hist = contrastive_pretrain(m, X, traj, tpos, cfg, seed, temporal=temporal)
    ep = cfg["e_warm"] + cfg["n_rounds"] * cfg["e_round"]
    supervised_fit(m, X, y_cls, y_reg, lab_idx, cfg, seed, ep,
                   freeze_encoder=freeze, K=K, val_idx=val_idx)
    return m, dict(con_loss=hist)


# --------------------------------------------------------------------------- #
# OURS
# --------------------------------------------------------------------------- #
def _pl_diag(sel_idx, pl, y_true, K):
    """Quality diagnostics of one pseudo-label round."""
    if len(sel_idx) == 0:
        return dict(n=0, precision=float("nan"), per_class=[0] * K,
                    per_class_prec=[float("nan")] * K)
    corr = (pl == y_true[sel_idx])
    pc = np.bincount(pl, minlength=K).tolist()
    pcp = []
    for k in range(K):
        mk = pl == k
        pcp.append(float(corr[mk].mean()) if mk.sum() else float("nan"))
    return dict(n=int(len(sel_idx)), precision=float(corr.mean()),
                per_class=pc, per_class_prec=pcp)


def run_uctcl(X, y_cls, y_reg, lab_idx, unl_idx, cfg, seed, K, in_ch,
              traj, tpos, order, pre_loader=None, edges=None, val_idx=None):
    """
    Uncertainty-Calibrated Temporal Contrastive Learning.

    1. temporal contrastive pre-training on labelled + unlabelled windows
    2. supervised warm-up on the small labelled set
    3. n_rounds of: MC-dropout inference -> epistemic score -> adaptive
       per-class threshold -> isotonic monotonicity filter -> retrain
    """
    m = _make_model(in_ch, K, cfg, seed)
    con_hist = []
    if cfg["use_tcl"]:
        mode = "tcl" if cfg["temporal_weight"] else "simclr"
        if pre_loader is not None:
            con_hist = pre_loader(m, mode)
        else:
            con_hist = contrastive_pretrain(m, X, traj, tpos, cfg, seed,
                                            temporal=cfg["temporal_weight"])
    supervised_fit(m, X, y_cls, y_reg, lab_idx, cfg, seed, cfg["e_warm"], K=K,
                   val_idx=val_idx)

    # ---- v5: physical brackets from the inspected passes (computed once) --- #
    use_mb = cfg["mono_bracket"]
    trn_idx = np.concatenate([lab_idx, unl_idx])
    is_lab = np.zeros(len(y_cls), dtype=bool); is_lab[lab_idx] = True
    if use_mb:
        lo_t, hi_t, ylo_t, yhi_t = monopath.monotone_bounds(
            y_cls[trn_idx], y_reg[trn_idx], is_lab[trn_idx],
            traj[trn_idx], order[trn_idx], K)
        nl = len(lab_idx)
        lo_u, hi_u = lo_t[nl:], hi_t[nl:]
        ylo_u, yhi_u = ylo_t[nl:], yhi_t[nl:]
        cert = lo_u == hi_u
    else:
        cert = np.zeros(len(unl_idx), dtype=bool)

    diag = []
    for t in range(cfg["n_rounds"]):
        if cfg["use_mc"]:
            probs, mu, lv = mc_predict(m, X[unl_idx], n_mc=cfg["n_mc"])
            p_cls, u_tot, u_ale, u_epi = mc_classification_uncertainty(probs)
            mu_bar, v_ale, v_epi, v_tot = mc_regression_uncertainty(mu, lv)
            p_reg = regression_stage_probs(mu, lv, edges)
        else:
            # deterministic ablation: 1 - max soft-max probability as the score
            p_cls, mu_bar = deterministic_predict(m, X[unl_idx])
            p_reg = regression_stage_probs(mu_bar[None, :],
                                           np.full((1, len(mu_bar)), -6.0), edges)
            u_epi = 1.0 - p_cls.max(1)
            v_tot = np.full(len(unl_idx), 1e-2, dtype=np.float64)

        if cfg["use_fusion"]:
            p_bar = fuse_probs(p_cls, p_reg, cfg["alpha_fuse"])
        else:
            p_bar = p_cls
        pred = p_bar.argmax(1)
        u = normalised_epistemic(u_epi, K) if cfg["use_mc"] else u_epi

        # ---- v5: joint decoding of each tool life --------------------------- #
        # Labelled windows enter as one-hot evidence, so the decoded path is
        # forced through every inspection; unlabelled windows contribute the
        # network's MC-mean probabilities.
        pred_net = pred.copy()
        if cfg["path_decode"]:
            p_all = np.concatenate([np.eye(K)[y_cls[lab_idx]], p_bar], 0)
            if use_mb:
                pa, _ = monopath.decode_trajectories(
                    p_all, traj[trn_idx], order[trn_idx], lo_t, hi_t)
            else:
                # structure only: no inspection is used, labelled windows are
                # treated like any other window
                p_all[:len(lab_idx)] = 1.0 / K
                pa, _ = monopath.decode_trajectories(
                    p_all, traj[trn_idx], order[trn_idx])
            pred = pa[len(lab_idx):]
        elif use_mb:
            pred = np.clip(pred, lo_u, hi_u)

        # ---- gate 1: physics consistency between the two heads -------------- #
        # The discriminative head and the head implied by the wear thresholds
        # must nominate the same stage.  This is a hard consistency check that
        # no amount of soft-max over-confidence can pass on its own.
        cand = np.ones(len(unl_idx), dtype=bool)
        n_agree = -1
        if cfg["use_fusion"] and cfg["require_agreement"]:
            cand &= (p_cls.argmax(1) == p_reg.argmax(1))
            n_agree = int(cand.sum())

        # ---- gate 2: monotone tool-life consistency ------------------------- #
        proj = mu_bar.copy()
        if cfg["use_mono"]:
            km, proj = monotonicity_filter(mu_bar, v_tot, traj[unl_idx],
                                           order[unl_idx], kappa=cfg["kappa"])
            cand &= km
            if cfg["mono_relabel"]:
                # the isotonic projection pools every window of a tool life
                # under a precision-weighted monotone constraint, so the stage
                # it implies is far less noisy than a single window's vote
                pred_mono = np.digitize(proj, edges)
                cand &= (pred_mono == pred)
        if use_mb:
            # wear target: isotonic fit tied to the inspections, then clipped
            # into the bracket set by the neighbouring measurements
            mu_all = np.concatenate([y_reg[lab_idx].astype(np.float64), mu_bar])
            w_all = np.concatenate([np.full(len(lab_idx), 1e6),
                                    1.0 / np.maximum(v_tot, 1e-12)])
            pj = mu_all.copy()
            tt, oo = traj[trn_idx], order[trn_idx]
            from uncertainty import weighted_pava
            for g_ in np.unique(tt):
                ii = np.flatnonzero(tt == g_)
                ii = ii[np.argsort(oo[ii], kind="mergesort")]
                pj[ii] = weighted_pava(mu_all[ii], w_all[ii])
            proj = np.clip(pj[len(lab_idx):], ylo_u, yhi_u)
        n_cons = int(cand.sum())

        # ---- gate 3: adaptive per-class uncertainty quantile ---------------- #
        # Applied LAST and only inside the surviving candidate pool, so the
        # acceptance rate is equalised across stages after the consistency
        # gates have already removed whatever they remove.  Applying it first
        # would let the gates undo the class balance it is meant to create.
        rho = percentile_schedule(t, cfg["n_rounds"], cfg["rho0"], cfg["rho_max"])
        keep = np.zeros(len(unl_idx), dtype=bool)
        if cand.any():
            ci = np.flatnonzero(cand)
            if cfg["per_class_tau"]:
                tau = adaptive_thresholds(u[ci], pred[ci], K, rho)
                keep[ci] = uncertainty_mask(u[ci], pred[ci], tau)
            else:
                g = float(np.quantile(u[ci], rho))
                tau = np.full(K, g)
                keep[ci] = u[ci] <= g
        else:
            tau = np.full(K, -np.inf)
        if cfg["cert_only"]:
            keep[:] = False
        keep |= cert          # certified windows are always admitted
        n_unc = int(keep.sum())

        sel = unl_idx[keep]
        diag.append(dict(round=t, rho=float(rho),
                         tau=[float(x) for x in tau],
                         mean_u=float(u.mean()),
                         n_after_agreement=n_agree, n_after_consistency=n_cons,
                         n_after_uncertainty=n_unc,
                         n_unlabelled=int(len(unl_idx)),
                         n_certified=int(cert.sum()),
                         cert_precision=(float((pred[cert] == y_cls[unl_idx][cert]).mean())
                                         if cert.any() else float("nan")),
                         net_precision_all=float((pred_net == y_cls[unl_idx]).mean()),
                         path_precision_all=float((pred == y_cls[unl_idx]).mean()),
                         wear_target_mae=float(np.abs(proj - y_reg[unl_idx]).mean()),
                         wear_net_mae=float(np.abs(mu_bar - y_reg[unl_idx]).mean()),
                         **_pl_diag(sel, pred[keep], y_cls, K)))

        yc = y_cls.copy(); yrg = y_reg.copy()
        yc[sel] = pred[keep]
        yrg[sel] = proj[keep].astype(np.float32)
        w = np.ones(len(y_cls), dtype=np.float32)
        # certainty weighting relative to the class threshold: a sample sitting
        # exactly on tau_k contributes nothing, the most certain one contributes
        # in full.  This is scale-free, unlike a raw (1 - u) weight.
        tk = np.where(np.isfinite(tau[pred[keep]]), tau[pred[keep]], 1.0)
        w[sel] = np.clip(1.0 - u[keep] / np.maximum(tk, 1e-8), 0.0, 1.0)
        w[sel] *= cfg["lam_u"] * sigmoid_rampup(t + 1, cfg["n_rounds"])
        # a certified window carries a physically implied label, not a guess
        w[unl_idx[cert]] = 1.0
        idx = np.concatenate([lab_idx, sel])
        supervised_fit(m, X, yc, yrg, idx, cfg, seed + 101 * (t + 1),
                       cfg["e_round"], sample_w=w, K=K, val_idx=val_idx)

    return m, dict(pl=diag, con_loss=con_hist)
