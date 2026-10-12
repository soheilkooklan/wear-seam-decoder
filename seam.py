"""Seam distance between 1 s accelerometer windows of the same limb.

For windows a and b, the end of a and the start of b are extrapolated with
Newton's second-order formulas to the midpoint of the gap between them
(half a sample after the last sample of a). The seam distance is the
Euclidean distance between the two extrapolated 3-axis points.

The distance matrix is dense (n x n). It is meant to be computed for one
participant at a time; pairs with different limb ids get an infinite distance.
"""
from __future__ import annotations

import numpy as np


def seam_points(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w: (n, L, 3) windows, L >= 3. Returns the extrapolated end and start points, (n, 3) each."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 3 or w.shape[1] < 3:
        raise ValueError("w must have shape (n, L, 3) with L >= 3")
    va = w[:, -1] - w[:, -2]
    aa = w[:, -1] - 2 * w[:, -2] + w[:, -3]
    vb = w[:, 1] - w[:, 0]
    ab = w[:, 2] - 2 * w[:, 1] + w[:, 0]
    end = w[:, -1] + 0.5 * va + 0.375 * aa
    start = w[:, 0] - 0.5 * vb + 0.375 * ab
    return end, start


def seam_distance(w: np.ndarray, limb: np.ndarray | None = None) -> np.ndarray:
    """D[a, b] = seam distance for the hypothesis 'b follows a'.

    w: (n, L, 3); limb: optional (n,) limb ids. The diagonal and all
    cross-limb entries are set to inf.
    """
    end, start = seam_points(np.nan_to_num(w))
    d2 = (end ** 2).sum(1)[:, None] + (start ** 2).sum(1)[None] - 2 * end @ start.T
    D = np.sqrt(np.clip(d2, 0, None))
    if limb is not None:
        limb = np.asarray(limb)
        D[limb[:, None] != limb[None, :]] = np.inf
    np.fill_diagonal(D, np.inf)
    return D


def successor_ranks(w: np.ndarray, nxt: np.ndarray, limb: np.ndarray | None = None) -> np.ndarray:
    """Rank of the true successor among all candidates of each anchor (1 = best).

    nxt[i] is the index of the window that follows window i, or -1.
    Only anchors whose successor carries the same limb are returned.
    """
    D = seam_distance(w, limb)
    a = np.flatnonzero(nxt >= 0)
    a = a[np.isfinite(D[a, nxt[a]])]
    d_true = D[a, nxt[a]]
    return a, (D[a] < d_true[:, None]).sum(1) + 1


if __name__ == "__main__":
    # synthetic check: a smooth 3-axis signal cut into 1 s windows at 50 Hz and shuffled
    rng = np.random.default_rng(0)
    t = np.arange(50 * 200) / 50.0
    x = np.stack([np.sin(1.3 * t), np.cos(0.7 * t), np.sin(2.1 * t + 1)], 1)
    x += 0.01 * rng.standard_normal(x.shape)
    w = x.reshape(200, 50, 3)
    p = rng.permutation(200)
    inv = np.argsort(p)
    nxt = np.full(200, -1)
    for i in range(199):
        nxt[inv[i]] = inv[i + 1]
    a, r = successor_ranks(w[p], nxt)
    print("anchors %d  top-1 %.3f  top-5 %.3f" % (len(a), (r == 1).mean(), (r <= 5).mean()))
