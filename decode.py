"""Joint labelling of all windows of one participant.

Graph: video kNN + same-limb IMU kNN + successor candidates.
Each edge type is a directed weight matrix W_t that is symmetrised before the
types are added: A = sum_t lambda_t * S(W_t), with S(W) = W + W^T (the variant
used in the paper; a pair that is a neighbour in both directions gets the sum of
both weights) or S(W) = max(W, W^T). The mean (W + W^T) / 2 equals the sum up
to a constant factor, which the degree normalisation in decode() removes.

Decoder: mean-field style update with a soft count constraint per activity
class. A class bias is raised while fewer than lo windows have the class as
their arg max and lowered while more than hi windows have it. This is a
heuristic in the spirit of dual ascent on lo <= n_c <= hi; it is not inference
in a defined probabilistic model.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, spmatrix


def _standardise(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def knn_weights(x: np.ndarray, k: int = 10) -> csr_matrix:
    """Directed kNN graph on cosine similarity of standardised features, weight exp(5 (s - 1))."""
    n = len(x)
    k = min(k, n - 1)
    if k < 1:
        return csr_matrix((n, n))
    z = _standardise(x)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    s = z @ z.T
    np.fill_diagonal(s, -np.inf)
    nb = np.argpartition(-s, k - 1, axis=1)[:, :k]
    sv = np.take_along_axis(s, nb, 1)
    return csr_matrix((np.exp(5 * (sv.ravel() - 1)), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))


def link_weights(n: int, pairs: np.ndarray, scores: np.ndarray) -> csr_matrix:
    """Directed successor edges a -> b with weight r^2. Repeated pairs keep their largest score."""
    pairs = np.asarray(pairs)
    key = pairs[:, 0].astype(np.int64) * n + pairs[:, 1]
    order = np.lexsort((-np.asarray(scores), key))
    key, first = np.unique(key[order], return_index=True)
    r = np.asarray(scores)[order][first]
    return csr_matrix((r ** 2, (key // n, key % n)), shape=(n, n))


def symmetrise(W: spmatrix, how: str = "sum") -> spmatrix:
    if how == "sum":
        return W + W.T
    if how == "max":
        return W.maximum(W.T)
    raise ValueError("how must be 'sum' or 'max'")


def build_graph(video: np.ndarray, imu: np.ndarray, limb: np.ndarray, pairs: np.ndarray | None = None,
                scores: np.ndarray | None = None, k: int = 10, imu_w: float = 1.0, link_w: float = 10.0,
                how: str = "sum") -> csr_matrix:
    """video: (n, d) mean frame embedding; imu: (n, f) IMU features; limb: (n,) limb ids;
    pairs: (m, 2) successor candidates (a, b); scores: (m,) rescorer outputs in [0, 1]."""
    n = len(video)
    A = symmetrise(knn_weights(video, k), how)
    W = csr_matrix((n, n))
    for lb in np.unique(limb):
        m = np.flatnonzero(limb == lb)
        Wm = knn_weights(imu[m], k).tocoo()
        W = W + csr_matrix((Wm.data, (m[Wm.row], m[Wm.col])), shape=(n, n))
    A = A + imu_w * symmetrise(W, how)
    if pairs is not None and len(pairs):
        A = A + link_w * symmetrise(link_weights(n, pairs, scores), how)
    return A.tocsr()


def decode(logp: np.ndarray, A: spmatrix, beta: float = 1.5, lo: float | np.ndarray = 75,
           hi: float | np.ndarray = 95, iters: int = 30, step: float = 0.15, start: int = 5,
           null: int = 0, max_bias: float | None = None) -> np.ndarray:
    """logp: (n, C) unary log probabilities; A: symmetric (n, n) graph from build_graph.
    lo, hi: scalars or (C,) arrays of count bounds, lo <= hi. The class with index
    null has no bias. Bias steps of size step are taken after iterations start to
    iters - 2 (0-based); a step after the last iteration would not change the
    labels and is not taken. The bias that enters the labels is therefore at most
    step * (iters - start - 1) in absolute value (3.6 with the defaults); max_bias
    clips it further if given. Returns labels (n,)."""
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2:
        raise ValueError("logp must be a 2-D array (windows x classes)")
    n, C = logp.shape
    if A.shape != (n, n):
        raise ValueError("A must be an (n, n) matrix for n = len(logp)")
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (C,))
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (C,))
    if np.any(lo > hi):
        raise ValueError("every lower bound must be at most the upper bound")
    b = np.zeros(C)
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    q = np.exp(logp - logp.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    for it in range(iters):
        msg = (A @ q) / deg[:, None]
        z = logp + b + beta * np.log(msg + 1e-3)
        q = np.exp(z - z.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        if start <= it < iters - 1:
            cnt = np.bincount(q.argmax(1), minlength=C)
            up = cnt < lo
            down = cnt > hi
            up[null] = down[null] = False
            b[up] += step
            b[down] -= step
            if max_bias is not None:
                np.clip(b, -max_bias, max_bias, out=b)
    return q.argmax(1)
