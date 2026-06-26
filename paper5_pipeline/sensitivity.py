"""Sensitivity analyses: graph resolution (k), threshold-graph and cosine
variants, and feature ablation. Every configuration is trained across the full
seed set and reported as mean +/- SD, so robustness is assessed on the same
footing as the headline (seed-mean) result rather than a single lucky/unlucky
seed.
"""
from __future__ import annotations

import copy

import numpy as np

from . import config as C
from . import graph as G
from . import train as T


def _agg(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(np.nanmean(a)), "sd": float(np.nanstd(a)),
            "values": [float(x) for x in a]}


def _fit_eval_seeds(co, S, A, cfg, seeds, epochs=None):
    """Train GAT across `seeds` on a fixed adjacency; return aggregated R2/AUC."""
    ei, ew = G.build_edges(A, S)
    r2s, aucs = [], []
    for sd in seeds:
        tr, va, te = T.make_splits(co, sd)
        Tn = T.make_tensors(co, ei, ew, tr)
        m, _ = T.train_model("gat", Tn, tr, va, cfg, sd, max_epochs=epochs or C.SENS_EPOCHS)
        res = T.evaluate(m, Tn, te, n_boot=0)
        r2s.append(res["macro_r2"]); aucs.append(res["binary"]["auc"])
    return _agg(r2s), _agg(aucs)


def graph_resolution(co, S, cfg, seeds, ks=None):
    seeds = [seeds] if np.isscalar(seeds) else list(seeds)
    ks = ks or C.K_GRID
    out = []
    for k in ks:
        r2, auc = _fit_eval_seeds(co, S, G.knn_adjacency(S, k), cfg, seeds)
        out.append({"k": k, "macro_r2": r2, "auc": auc})
    return out


def threshold_variant(co, S, cfg, seeds, taus=None):
    """Threshold-graph robustness check. Pathologically dense low-threshold
    graphs (edges > MAX_SENS_EDGES) are recorded but skipped for tractability."""
    seeds = [seeds] if np.isscalar(seeds) else list(seeds)
    taus = taus or C.TAU_GRID
    out = []
    for t in taus:
        A = G.threshold_adjacency(S, t)
        n_edges = int(A.sum() // 2)
        if n_edges == 0 or n_edges > C.MAX_SENS_EDGES:
            out.append({"tau": t, "n_edges": n_edges, "skipped": bool(n_edges > C.MAX_SENS_EDGES)})
            continue
        r2, auc = _fit_eval_seeds(co, S, A, cfg, seeds)
        out.append({"tau": t, "n_edges": n_edges, "macro_r2": r2, "auc": auc})
    return out


def cosine_variant(co, cfg, seeds, k):
    seeds = [seeds] if np.isscalar(seeds) else list(seeds)
    Sc = G.cosine_matrix(co.X)
    r2, auc = _fit_eval_seeds(co, Sc, G.knn_adjacency(Sc, k), cfg, seeds)
    return {"macro_r2": r2, "auc": auc}


def feature_ablation(co, S, cfg, seeds, k):
    """Socio-only vs full feature set (graph fixed from full-feature similarity)."""
    seeds = [seeds] if np.isscalar(seeds) else list(seeds)
    A = G.knn_adjacency(S, k)
    ei, ew = G.build_edges(A, S)

    socio_cols = [i for i, nm in enumerate(co.feat_names)
                  if not (nm.startswith("comorb_") or nm.startswith("diagnosis_")
                          or nm.startswith("pre_op") or nm.startswith("breastcancer_first")
                          or nm.startswith("cancer_breast"))]
    mask = np.ones(co.X.shape[1], dtype=bool); mask[socio_cols] = False

    full_r2, full_auc, soc_r2, soc_auc = [], [], [], []
    for sd in seeds:
        tr, va, te = T.make_splits(co, sd)
        Tn = T.make_tensors(co, ei, ew, tr)
        mfull, _ = T.train_model("gat", Tn, tr, va, cfg, sd, max_epochs=C.SENS_EPOCHS)
        rf = T.evaluate(mfull, Tn, te, n_boot=0)
        full_r2.append(rf["macro_r2"]); full_auc.append(rf["binary"]["auc"])

        co2 = copy.copy(co)
        Xs = co.X.copy(); Xs[:, mask] = 0.0; co2.X = Xs
        Tn2 = T.make_tensors(co2, ei, ew, tr)
        msoc, _ = T.train_model("gat", Tn2, tr, va, cfg, sd, max_epochs=C.SENS_EPOCHS)
        rs = T.evaluate(msoc, Tn2, te, n_boot=0)
        soc_r2.append(rs["macro_r2"]); soc_auc.append(rs["binary"]["auc"])

    return {"full": {"macro_r2": _agg(full_r2), "auc": _agg(full_auc)},
            "socio_only": {"macro_r2": _agg(soc_r2), "auc": _agg(soc_auc)}}
