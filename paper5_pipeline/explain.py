"""Explainability: SHAP, LIME, and GNNExplainer feature attributions on the
binary head, attention neighbour-homophily analysis, and cross-method agreement.

Feature attributions perturb the focal patient's features while keeping the
graph and neighbours fixed (the correct local explanation given the graph).
Attention is analysed as neighbour importance (homophily) rather than as a
feature ranking, which is the conceptually appropriate object for GAT attention.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from . import config as C


def _focal_predict(model, T, node_idx):
    """Local prediction function for one patient. Restricts computation to the
    node's L-hop ego-subgraph (exact for an L-layer GNN) for large speed-ups."""
    from torch_geometric.utils import k_hop_subgraph
    n_hops = len(getattr(model.trunk, "convs", [None] * 2))
    subset, sub_ei, mapping, _ = k_hop_subgraph(
        int(node_idx), n_hops, T.edge_index, relabel_nodes=True)
    sub_X = T.X[subset].clone()
    pos = int((subset == int(node_idx)).nonzero(as_tuple=True)[0])

    @torch.no_grad()
    def f(Z):
        Z = np.atleast_2d(Z).astype(np.float32)
        out = np.empty(len(Z))
        for j, row in enumerate(Z):
            x = sub_X.clone()
            x[pos] = torch.from_numpy(row)
            _, logit = model(x, sub_ei)
            out[j] = torch.sigmoid(logit[pos]).item()
        return out
    return f


def shap_attributions(model, T, train_idx, subset_idx):
    import shap
    bg = shap.kmeans(T.X.numpy()[train_idx], C.SHAP_BG)
    mat = np.zeros((len(subset_idx), T.X.shape[1]))
    for r, i in enumerate(subset_idx):
        f = _focal_predict(model, T, i)
        expl = shap.KernelExplainer(f, bg)
        sv = expl.shap_values(T.X.numpy()[i], nsamples=C.SHAP_NSAMPLES, silent=True)
        mat[r] = np.asarray(sv).ravel()
    return mat   # (subset, d) signed SHAP values


def lime_attributions(model, T, train_idx, subset_idx, feat_names):
    from lime.lime_tabular import LimeTabularExplainer
    expl = LimeTabularExplainer(T.X.numpy()[train_idx], feature_names=feat_names,
                                mode="regression", discretize_continuous=False,
                                random_state=C.SEED)
    d = T.X.shape[1]
    mat = np.zeros((len(subset_idx), d))
    for r, i in enumerate(subset_idx):
        f = _focal_predict(model, T, i)
        e = expl.explain_instance(T.X.numpy()[i], f, num_features=d,
                                  num_samples=C.LIME_SAMPLES)
        for fi, w in e.local_exp[1]:
            mat[r, fi] = w
    return mat


def gnnexplainer_global(model, T, subset_idx):
    """PyG GNNExplainer feature mask on the binary head; global mean importance."""
    from torch_geometric.explain import Explainer, GNNExplainer

    class BinWrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, edge_index, **kw):
            _, logit = self.m(x, edge_index)
            return logit.unsqueeze(-1)

    wrap = BinWrap(model)
    explainer = Explainer(
        model=wrap, algorithm=GNNExplainer(epochs=C.GNNEXPLAINER_EPOCHS),
        explanation_type="model", node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="node", return_type="raw"),
    )
    masks = []
    for i in subset_idx:
        try:
            e = explainer(T.X, T.edge_index, index=int(i))
            masks.append(e.node_mask.abs().mean(0).detach().numpy())
        except Exception:
            continue
    if not masks:
        return np.zeros(T.X.shape[1])
    return np.mean(masks, axis=0)


def attention_homophily(model, T, strata, test_idx):
    """For each test node, find its top-attended neighbour in the first GAT layer
    and report how often it shares education / partnership with the focal node."""
    if model.kind != "gat":
        return {}
    model.eval()
    conv = model.trunk.convs[0]
    with torch.no_grad():
        _, (ei, alpha) = conv(T.X, T.edge_index, return_attention_weights=True)
    alpha = alpha.mean(1).numpy() if alpha.ndim > 1 else alpha.numpy()
    ei = ei.numpy()
    edu = strata["education"].values
    part = strata["partnership"].values
    share_edu, share_part, sims = [], [], []
    Sw = T.edge_weight.numpy()
    for node in test_idx:
        m = ei[1] == node                      # edges pointing to focal node
        if not m.any():
            continue
        js = ei[0][m]; aw = alpha[m]
        top = js[int(np.argmax(aw))]
        share_edu.append(edu[top] == edu[node])
        share_part.append(part[top] == part[node])
    return {
        "share_education_top_neighbour": float(np.mean(share_edu)) if share_edu else None,
        "share_partnership_top_neighbour": float(np.mean(share_part)) if share_part else None,
        "n": len(share_edu),
    }


def cross_method(shap_mat, lime_mat, gnn_global, feat_names):
    """Global rankings (mean |attr|) + pairwise Spearman; per-patient SHAP-LIME rho."""
    shap_glob = np.abs(shap_mat).mean(0)
    lime_glob = np.abs(lime_mat).mean(0)
    rankings = {"shap": shap_glob, "lime": lime_glob, "gnnexplainer": gnn_global}
    pairs = {}
    keys = list(rankings)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            rho, _ = spearmanr(rankings[keys[a]], rankings[keys[b]])
            pairs[f"{keys[a]}_vs_{keys[b]}"] = float(rho)
    # per-patient SHAP vs LIME rank correlation
    per = []
    for r in range(shap_mat.shape[0]):
        rho, _ = spearmanr(np.abs(shap_mat[r]), np.abs(lime_mat[r]))
        if rho == rho:
            per.append(rho)
    per = np.array(per)
    order = np.argsort(-shap_glob)
    top_features = [(feat_names[i], float(shap_glob[i])) for i in order[:10]]
    return {
        "global_ranking_shap": [(feat_names[i], float(shap_glob[i])) for i in order],
        "top_features": top_features,
        "pairwise_spearman": pairs,
        "shap_lime_per_patient_mean": float(per.mean()) if len(per) else None,
        "shap_lime_per_patient_median": float(np.median(per)) if len(per) else None,
        "shap_lime_frac_above_0.5": float((per >= 0.5).mean()) if len(per) else None,
        "n_patients": int(shap_mat.shape[0]),
    }
