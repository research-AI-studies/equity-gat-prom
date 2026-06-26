"""Patient-similarity graph: Gower similarity, k-NN construction, topology.

The primary construction is a symmetric k-nearest-neighbour graph on Gower
similarity (standard for patient-similarity GNNs; guarantees a sparse, locally
structured, fully connected graph). k is chosen by spectral-clustering
silhouette with a parsimony tie-break. Threshold and cosine graphs are retained
as sensitivity variants. Expensive path/spectral metrics are computed once for
the selected graph via scipy sparse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path, laplacian
from scipy.sparse.linalg import eigsh

from . import config as C


# ------------------------------------------------------------- similarities #
def gower_matrix(feat: pd.DataFrame) -> np.ndarray:
    """Gower similarity (1 - Gower distance) on mixed-type features in [0,1]."""
    n = len(feat)
    cont = list(C.CONTINUOUS)
    nom = list(C.NOMINAL) + list(C.BINARY)
    D = np.zeros((n, n), dtype=np.float64)
    p = len(cont) + len(nom)
    for c in cont:
        v = feat[c].astype(float).values
        rng = np.nanmax(v) - np.nanmin(v)
        rng = rng if rng > 0 else 1.0
        D += np.abs(v[:, None] - v[None, :]) / rng
    for c in nom:
        v = feat[c].astype(str).values
        D += (v[:, None] != v[None, :]).astype(float)
    D /= p
    S = 1.0 - D
    np.fill_diagonal(S, 0.0)
    return S


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = (Xn @ Xn.T + 1.0) / 2.0
    np.fill_diagonal(S, 0.0)
    return S


# --------------------------------------------------------------- adjacency #
def knn_adjacency(S: np.ndarray, k: int) -> np.ndarray:
    """Symmetric (union) k-NN adjacency from a similarity matrix."""
    n = S.shape[0]
    idx = np.argsort(-S, axis=1)[:, :k]
    A = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    A[rows, idx.ravel()] = True
    A = A | A.T
    np.fill_diagonal(A, False)
    return A


def threshold_adjacency(S: np.ndarray, tau: float) -> np.ndarray:
    A = (S >= tau).copy()
    np.fill_diagonal(A, False)
    return A


def build_edges(A: np.ndarray, S: np.ndarray):
    """Return (edge_index [2,E], edge_weight [E]) from a boolean adjacency."""
    iu, ju = np.where(np.triu(A, k=1))
    w = S[iu, ju]
    src = np.concatenate([iu, ju])
    dst = np.concatenate([ju, iu])
    return np.vstack([src, dst]).astype(np.int64), np.concatenate([w, w]).astype(np.float32)


# --------------------------------------------------------------- silhouette #
def spectral_silhouette(S: np.ndarray, A: np.ndarray, k: int = 2) -> float:
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    W = np.where(A, S, 0.0)
    if W.sum() == 0:
        return float("nan")
    try:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                assign_labels="discretize", random_state=C.SEED)
        labels = sc.fit_predict(W)
        if len(np.unique(labels)) < 2:
            return float("nan")
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        return float(silhouette_score(D, labels, metric="precomputed"))
    except Exception:
        return float("nan")


# --------------------------------------------------------------- topology #
def stats(S: np.ndarray, A: np.ndarray, key: str, val, with_silhouette=True) -> dict:
    n = S.shape[0]
    deg = A.sum(1)
    n_edges = int(A.sum() // 2)
    density = (2 * n_edges) / (n * (n - 1)) if n > 1 else 0.0
    nc, lbl = connected_components(csr_matrix(A.astype(np.int8)), directed=False)
    lcc = int(np.bincount(lbl).max())
    ew = S[np.triu(A, 1)] if n_edges else np.array([])
    out = {
        key: val, "n_nodes": int(n), "n_edges": n_edges, "density": float(density),
        "mean_degree": float(deg.mean()), "median_degree": float(np.median(deg)),
        "deg_q1": float(np.percentile(deg, 25)), "deg_q3": float(np.percentile(deg, 75)),
        "max_degree": int(deg.max()), "min_degree": int(deg.min()),
        "n_isolated": int((deg == 0).sum()), "n_components": int(nc),
        "lcc_size": lcc, "lcc_frac": float(lcc / n),
    }
    if ew.size:
        out.update({
            "edge_w_min": float(ew.min()), "edge_w_max": float(ew.max()),
            "edge_w_median": float(np.median(ew)), "edge_w_skew": float(skew(ew)),
            "edge_w_frac_gt_085": float((ew > 0.85).mean()),
        })
    if with_silhouette:
        out["silhouette"] = spectral_silhouette(S, A)
    return out


def full_topology(S: np.ndarray, A: np.ndarray, key="k", val=None) -> dict:
    out = stats(S, A, key, val)
    sp = csr_matrix(A.astype(np.int8))
    nc, lbl = connected_components(sp, directed=False)
    lcc_lbl = np.argmax(np.bincount(lbl))
    idx = np.where(lbl == lcc_lbl)[0]
    sub = sp[idx][:, idx]
    dist = shortest_path(sub, method="D", unweighted=True)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    if finite.size:
        out["lcc_diameter"] = int(finite.max())
        out["lcc_mean_path"] = float(finite.mean())
    out["clustering_coef"] = _avg_clustering(A[np.ix_(idx, idx)])
    try:
        W = csr_matrix(S[np.ix_(idx, idx)] * A[np.ix_(idx, idx)])
        L = laplacian(W, normed=False)
        ev = np.sort(eigsh(L, k=4, which="SM", return_eigenvectors=False))
        out["lambda2"], out["lambda3"] = float(ev[1]), float(ev[2])
    except Exception:
        pass
    return out


def _avg_clustering(A: np.ndarray) -> float:
    A = A.astype(np.float64)
    deg = A.sum(1)
    tri = np.einsum("ij,jk,ki->i", A, A, A)
    denom = deg * (deg - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(denom > 0, tri / denom, 0.0)
    return float(ci.mean())


# --------------------------------------------------------------- selection #
def select_k(S: np.ndarray, ks=None, tol=0.01):
    """Grid over k; pick the smallest k whose silhouette is within `tol` of the
    maximum among fully-connected graphs (parsimony tie-break)."""
    ks = ks or C.K_GRID
    grid = []
    for k in ks:
        A = knn_adjacency(S, k)
        grid.append(stats(S, A, "k", k))
    conn = [g for g in grid if g["lcc_frac"] >= 0.999 and not np.isnan(g.get("silhouette", np.nan))]
    pool = conn or grid
    smax = max(g.get("silhouette", -1) for g in pool)
    best = min((g for g in pool if g.get("silhouette", -1) >= smax - tol), key=lambda g: g["k"])
    return best["k"], grid


def threshold_grid(S: np.ndarray, taus=None):
    taus = taus or C.TAU_GRID
    return [stats(S, threshold_adjacency(S, t), "tau", t) for t in taus]
