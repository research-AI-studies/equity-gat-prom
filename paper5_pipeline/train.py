"""Training, cross-validation, random search, evaluation, and baselines.

Transductive node-level multi-task learning on the patient-similarity graph.
The 15% test set is constructed once and only touched by `evaluate`.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

# Limit thread over-subscription: many small GAT ops on a 1.4k-node graph run
# faster with a modest thread count than with all logical cores.
torch.set_num_threads(min(6, os.cpu_count() or 6))
from sklearn.metrics import (r2_score, roc_auc_score, f1_score,
                             recall_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C
from .models import MultiTaskNet


# ----------------------------------------------------------------- seeding #
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# ----------------------------------------------------------------- tensors #
@dataclass
class Tensors:
    X: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    Yz: torch.Tensor          # standardised regression targets (n, n_reg)
    Ymask: torch.Tensor       # observed mask (n, n_reg)
    ybin: torch.Tensor        # binary labels (n,)
    ybin_mask: torch.Tensor   # observed mask (n,)
    y_mean: np.ndarray        # per-target train mean (original scale)
    y_std: np.ndarray         # per-target train std


def make_tensors(co, edge_index, edge_weight, train_idx) -> Tensors:
    X = torch.tensor(co.X, dtype=torch.float32)
    Yreg = co.Y[C.REG_TARGETS].values.astype(float)        # (n, 7) raw 0-100
    Ymask = ~np.isnan(Yreg)
    # standardise targets using TRAIN nodes only (no leakage)
    y_mean = np.array([np.nanmean(Yreg[train_idx, t]) for t in range(Yreg.shape[1])])
    y_std = np.array([np.nanstd(Yreg[train_idx, t]) or 1.0 for t in range(Yreg.shape[1])])
    Yz = (Yreg - y_mean) / y_std
    Yz = np.nan_to_num(Yz, nan=0.0)
    ybin = co.y_bin.astype(float)
    ybin_mask = ~np.isnan(ybin)
    return Tensors(
        X=X,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
        Yz=torch.tensor(Yz, dtype=torch.float32),
        Ymask=torch.tensor(Ymask, dtype=torch.bool),
        ybin=torch.tensor(np.nan_to_num(ybin), dtype=torch.float32),
        ybin_mask=torch.tensor(ybin_mask, dtype=torch.bool),
        y_mean=y_mean, y_std=y_std,
    )


# -------------------------------------------------------------- splitting #
def make_splits(co, seed: int):
    """Stratified 70/15/15 train/val/test by binary label (observed only)."""
    n = co.X.shape[0]
    idx = np.arange(n)
    # stratify on the (near-complete) binary label; the handful of unobserved
    # labels are bucketed with class 0 for splitting only (training masks still
    # respect the observed mask, so these nodes contribute no binary loss).
    strat = np.nan_to_num(co.y_bin, nan=0).astype(int)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat)
    val, test = train_test_split(tmp, test_size=0.50, random_state=seed,
                                 stratify=strat[tmp])
    return np.sort(tr), np.sort(val), np.sort(test)


def _mask(n, idx):
    m = torch.zeros(n, dtype=torch.bool); m[idx] = True; return m


# -------------------------------------------------------------- training #
def train_model(kind, T: Tensors, train_idx, val_idx, cfg, seed,
                max_epochs=None, patience=C.PATIENCE):
    seed_everything(seed)
    n = T.X.shape[0]
    n_reg = T.Yz.shape[1]
    model = MultiTaskNet(kind, T.X.shape[1], hidden=cfg["hidden"],
                         heads=cfg.get("heads", 4), n_layers=cfg["n_layers"],
                         dropout=cfg["dropout"], n_reg=n_reg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    tr_m, va_m = _mask(n, train_idx), _mask(n, val_idx)
    reg_mask_tr = T.Ymask & tr_m[:, None]
    best_val, best_state, bad = -1e9, None, 0
    max_epochs = max_epochs or C.MAX_EPOCHS
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        reg, logit = model(T.X, T.edge_index, T.edge_weight)
        loss = model.loss(reg, logit, T.Yz, reg_mask_tr, T.ybin, T.ybin_mask & tr_m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()
        # validation macro-R² (standardised scale; scale-invariant)
        vr = _val_macro_r2(model, T, va_m)
        if vr > best_val + 1e-4:
            best_val, best_state, bad = vr, {k: v.detach().clone()
                                            for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def _val_macro_r2(model, T, val_m):
    model.eval()
    reg, _ = model(T.X, T.edge_index, T.edge_weight)
    reg = reg.numpy(); scores = []
    vm = val_m.numpy()
    for t in range(reg.shape[1]):
        m = vm & T.Ymask[:, t].numpy()
        if m.sum() > 5:
            scores.append(r2_score(T.Yz[m, t].numpy(), reg[m, t]))
    return float(np.mean(scores)) if scores else -1e9


# ------------------------------------------------------- random search (CV) #
def sample_configs(n, seed):
    rng = random.Random(seed)
    cfgs = []
    for _ in range(n):
        cfgs.append({key: rng.choice(vals) for key, vals in C.SEARCH_SPACE.items()})
    return cfgs


def cv_score(kind, T, train_pool, cfg, seed, folds=C.CV_FOLDS, max_epochs=120):
    strat = T.ybin.numpy()[train_pool].astype(int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(train_pool, strat):
        _, v = train_model(kind, T, train_pool[tr], train_pool[va], cfg,
                           seed, max_epochs=max_epochs, patience=20)
        scores.append(v)
    return float(np.mean(scores))


def random_search(kind, T, train_pool, seed, n=None, max_epochs=120):
    n = n or C.RANDOM_SEARCH_N
    cfgs = sample_configs(n, seed)
    history = []
    for i, cfg in enumerate(cfgs):
        s = cv_score(kind, T, train_pool, cfg, seed, max_epochs=max_epochs)
        history.append({"cfg": cfg, "cv_macro_r2": s})
    best = max(history, key=lambda h: h["cv_macro_r2"])
    return best["cfg"], history


def cv_score_graph(co, S, train_pool, cfg, seed, folds=C.CV_FOLDS, max_epochs=120):
    """Graph-aware CV: graph resolution k (in cfg) rebuilds the kNN edges."""
    from . import graph as G
    A = G.knn_adjacency(S, cfg["k"])
    ei, ew = G.build_edges(A, S)
    strat = np.nan_to_num(co.y_bin[train_pool], nan=0).astype(int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(train_pool, strat):
        Tn = make_tensors(co, ei, ew, train_pool[tr])
        _, v = train_model("gat", Tn, train_pool[tr], train_pool[va], cfg,
                           seed, max_epochs=max_epochs, patience=20)
        scores.append(v)
    return float(np.mean(scores))


def random_search_graph(co, S, train_pool, seed, n=None, max_epochs=120):
    n = n or C.RANDOM_SEARCH_N
    cfgs = sample_configs(n, seed)
    history = []
    for i, cfg in enumerate(cfgs):
        s = cv_score_graph(co, S, train_pool, cfg, seed, max_epochs=max_epochs)
        history.append({"cfg": cfg, "cv_macro_r2": s})
        print(f"    search {i+1}/{n}  k={cfg['k']} L={cfg['n_layers']} "
              f"h={cfg['hidden']} lr={cfg['lr']:.0e}  cv_R2={s:.4f}", flush=True)
    best = max(history, key=lambda h: h["cv_macro_r2"])
    return best["cfg"], history


# -------------------------------------------------------------- evaluation #
def _calib_slope_reg(y, yhat):
    if len(y) < 3 or np.std(yhat) < 1e-9:
        return float("nan")
    return float(np.polyfit(yhat, y, 1)[0])


def _binary_metrics(y, p):
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auc"] = float("nan")
    pred = (p >= 0.5).astype(int)
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["sensitivity"] = float(recall_score(y, pred, pos_label=1, zero_division=0))
    out["specificity"] = float(recall_score(y, pred, pos_label=0, zero_division=0))
    # calibration slope via logistic regression of outcome on logit(p)
    try:
        from sklearn.linear_model import LogisticRegression
        eps = 1e-6
        logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(logit.reshape(-1, 1), y)
        out["calibration_slope"] = float(lr.coef_[0][0])
    except Exception:
        out["calibration_slope"] = float("nan")
    return out


@torch.no_grad()
def predict(model, T):
    model.eval()
    reg, logit = model(T.X, T.edge_index, T.edge_weight)
    reg = reg.numpy() * T.y_std + T.y_mean         # back to original 0-100 scale
    prob = torch.sigmoid(logit).numpy()
    return reg, prob


def evaluate(model, T, test_idx, n_boot=C.N_BOOTSTRAP, seed=C.SEED):
    reg, prob = predict(model, T)
    Ymask = T.Ymask.numpy()
    Yreg_true = (T.Yz.numpy() * T.y_std + T.y_mean)
    res = {"per_head": {}, "macro_r2": None, "binary": {}}
    r2s = []
    rng = np.random.default_rng(seed)
    for t, name in enumerate(C.REG_TARGETS):
        m = np.zeros(reg.shape[0], dtype=bool); m[test_idx] = True
        m &= Ymask[:, t]
        yt, yp = Yreg_true[m, t], reg[m, t]
        if m.sum() < 5:
            continue
        r2 = r2_score(yt, yp)
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        entry = {"r2": float(r2), "mae": mae, "rmse": rmse,
                 "calibration_slope": _calib_slope_reg(yt, yp), "n": int(m.sum())}
        if n_boot:
            bo = []
            for _ in range(n_boot):
                s = rng.choice(len(yt), len(yt), replace=True)
                bo.append(r2_score(yt[s], yp[s]))
            entry["r2_lo"] = float(np.percentile(bo, 2.5))
            entry["r2_hi"] = float(np.percentile(bo, 97.5))
        res["per_head"][name] = entry
        r2s.append(r2)
    res["macro_r2"] = float(np.mean(r2s)) if r2s else float("nan")
    # binary head
    bm = np.zeros(prob.shape[0], dtype=bool); bm[test_idx] = True
    bm &= T.ybin_mask.numpy()
    yb, pb = T.ybin.numpy()[bm], prob[bm]
    res["binary"] = _binary_metrics(yb, pb)
    if n_boot:
        bo_auc = []
        for _ in range(n_boot):
            s = rng.choice(len(yb), len(yb), replace=True)
            try:
                bo_auc.append(roc_auc_score(yb[s], pb[s]))
            except Exception:
                pass
        if bo_auc:
            res["binary"]["auc_lo"] = float(np.percentile(bo_auc, 2.5))
            res["binary"]["auc_hi"] = float(np.percentile(bo_auc, 97.5))
    res["binary"]["n"] = int(bm.sum())
    return res


# -------------------------------------------------------------- XGBoost #
def xgboost_baseline(co, train_idx, test_idx, seed=C.SEED):
    import xgboost as xgb
    X = co.X
    res = {"per_head": {}, "binary": {}}
    r2s = []
    for name in C.REG_TARGETS:
        y = co.Y[name].values.astype(float)
        trm = np.array([i for i in train_idx if not np.isnan(y[i])])
        tem = np.array([i for i in test_idx if not np.isnan(y[i])])
        if len(tem) < 5:
            continue
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed)
        m.fit(X[trm], y[trm])
        yp = m.predict(X[tem])
        r2 = r2_score(y[tem], yp); r2s.append(r2)
        res["per_head"][name] = {"r2": float(r2),
                                 "mae": float(np.mean(np.abs(y[tem] - yp))),
                                 "rmse": float(np.sqrt(np.mean((y[tem] - yp) ** 2)))}
    res["macro_r2"] = float(np.mean(r2s)) if r2s else float("nan")
    yb = co.y_bin
    trm = np.array([i for i in train_idx if not np.isnan(yb[i])])
    tem = np.array([i for i in test_idx if not np.isnan(yb[i])])
    cm = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed,
                           eval_metric="logloss")
    cm.fit(X[trm], yb[trm].astype(int))
    pb = cm.predict_proba(X[tem])[:, 1]
    res["binary"] = _binary_metrics(yb[tem].astype(int), pb)
    return res
