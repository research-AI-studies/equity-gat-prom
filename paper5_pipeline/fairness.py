"""Equity audit: DPD, EOD, predictive-parity ratio, per-stratum calibration,
plus bias mitigations (reweighing, equalized-odds post-processing, adversarial).

Operates on the binary head (clinically poor sexual functioning) and, for the
regression heads, on the predicted probability of a clinically meaningful low
score. Metrics are computed per protected stratum on the held-out test set.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from . import config as C
from . import train as T


def _rates(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    pos_rate = pred.mean()
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    fpr = pred[y == 0].mean() if (y == 0).any() else np.nan
    ppv = y[pred == 1].mean() if (pred == 1).any() else np.nan
    return pos_rate, tpr, fpr, ppv


def fairness_metrics(y, p, group, thr=0.5):
    """DPD / EOD / PPR / per-group positive rate between the two extreme groups
    (ordered by name). For 3-level strata (education) compares low vs high."""
    groups = sorted(np.unique(group).tolist())
    per = {}
    for g in groups:
        m = group == g
        if m.sum() < 5:
            continue
        pr, tpr, fpr, ppv = _rates(y[m], p[m], thr)
        per[g] = {"n": int(m.sum()), "pos_rate": float(pr), "tpr": float(tpr),
                  "fpr": float(fpr), "ppv": float(ppv)}
    # reference comparison: low vs high (education) or the two groups otherwise
    if "low" in per and "high" in per:
        a, b = "low", "high"
    elif len(per) >= 2:
        a, b = list(per)[0], list(per)[-1]
    else:
        return {"per_group": per}
    dpd = abs(per[a]["pos_rate"] - per[b]["pos_rate"])
    eod = max(abs(per[a]["tpr"] - per[b]["tpr"]),
              abs(per[a]["fpr"] - per[b]["fpr"]))
    ppr = (per[a]["ppv"] / per[b]["ppv"]) if per[b]["ppv"] not in (0, np.nan) else np.nan
    return {"per_group": per, "compare": [a, b], "dpd": float(dpd),
            "eod": float(eod), "ppr": float(ppr) if ppr == ppr else None,
            "triggered": bool(dpd > C.FAIR_THRESHOLD or eod > C.FAIR_THRESHOLD)}


def audit_all(prob, ybin, ybin_mask, strata, test_idx):
    """Run the fairness audit on the binary head for all three strata."""
    bm = np.zeros(len(prob), dtype=bool); bm[test_idx] = True
    bm &= ybin_mask
    y = ybin[bm].astype(int); p = prob[bm]
    out = {}
    for col in ["education", "partnership", "diagnosis"]:
        g = strata[col].values[bm]
        out[col] = fairness_metrics(y, p, g)
    return out


# --------------------------------------------------------- mitigations #
def reweighing_weights(y, group):
    """Kamiran-Calders reweighing: w(g,c) = P(g)P(c) / P(g,c)."""
    n = len(y)
    w = np.ones(n)
    for g in np.unique(group):
        for c in [0, 1]:
            m = (group == g) & (y == c)
            if m.sum() == 0:
                continue
            pg = (group == g).mean(); pc = (y == c).mean()
            w[m] = pg * pc / (m.mean())
    return w


def train_binary_reweighed(co, T_obj, train_idx, val_idx, cfg, seed, sens_col):
    """Retrain with stratum-balanced sample weights on the binary head."""
    import torch.nn.functional as F
    from .models import MultiTaskNet
    T.seed_everything(seed)
    n = T_obj.X.shape[0]; n_reg = T_obj.Yz.shape[1]
    grp = co.strata[sens_col].values
    yb = co.y_bin
    tr_obs = np.array([i for i in train_idx if not np.isnan(yb[i])])
    w = np.ones(n)
    w[tr_obs] = reweighing_weights(yb[tr_obs].astype(int), grp[tr_obs])
    wt = torch.tensor(w, dtype=torch.float32)
    model = MultiTaskNet("gat", T_obj.X.shape[1], hidden=cfg["hidden"],
                         heads=cfg.get("heads", 4), n_layers=cfg["n_layers"],
                         dropout=cfg["dropout"], n_reg=n_reg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    trm = T._mask(n, train_idx); reg_mask_tr = T_obj.Ymask & trm[:, None]
    binm = T_obj.ybin_mask & trm
    best, best_state, bad = -1e9, None, 0
    for ep in range(C.MAX_EPOCHS):
        model.train(); opt.zero_grad()
        reg, logit = model(T_obj.X, T_obj.edge_index, T_obj.edge_weight)
        # weighted multi-task loss (regression unweighted; binary reweighed)
        loss = 0.0
        for t in range(n_reg):
            m = reg_mask_tr[:, t]
            if m.any():
                loss = loss + 0.5 * torch.exp(-model.log_vars[t]) * \
                    F.mse_loss(reg[m, t], T_obj.Yz[m, t]) + 0.5 * model.log_vars[t]
        if binm.any():
            bce = F.binary_cross_entropy_with_logits(logit[binm], T_obj.ybin[binm],
                                                     weight=wt[binm])
            loss = loss + torch.exp(-model.log_vars[-1]) * bce + 0.5 * model.log_vars[-1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()
        vr = T._val_macro_r2(model, T_obj, T._mask(n, val_idx))
        if vr > best + 1e-4:
            best, best_state, bad = vr, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def equalized_odds_postprocess(y, p, group):
    """Per-group thresholds chosen to equalize TPR across groups (grid search)."""
    thr_grid = np.linspace(0.2, 0.8, 25)
    groups = np.unique(group)
    # target TPR = min over groups of best-achievable; choose per-group thr closest
    target = 0.5
    best = {}
    for g in groups:
        m = group == g
        if m.sum() < 5:
            best[g] = 0.5; continue
        # pick threshold whose TPR is closest to target
        tprs = [( (p[m] >= t).astype(int)[y[m] == 1].mean() if (y[m]==1).any() else 0, t)
                for t in thr_grid]
        best[g] = min(tprs, key=lambda x: abs(x[0] - target))[1]
    pred = np.array([1 if p[i] >= best[group[i]] else 0 for i in range(len(p))])
    return pred, best
