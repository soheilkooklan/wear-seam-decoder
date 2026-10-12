"""Pairwise successor model: is window j the next second after window i (same subject)?"""
from __future__ import annotations
import numpy as np

KV = 30   # video candidates per row
KI = 12   # IMU-extrapolation candidates per row (same limb only)
KP = 15   # ridge-predicted-head candidates


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def ridge_fit(X, Y, lam=10.0):
    mu_x = X.mean(0); mu_y = Y.mean(0)
    Xc = X - mu_x; Yc = Y - mu_y
    A = Xc.T @ Xc + lam * len(X) / 1000.0 * np.eye(X.shape[1])
    B = np.linalg.solve(A, Xc.T @ Yc)
    return mu_x, mu_y, B.astype(np.float32)


def ridge_apply(model, X):
    mu_x, mu_y, B = model
    return (X - mu_x) @ B + mu_y


def next_inputs(v):
    """Inputs for predicting the next window's head (v: (n,15,K) centred)."""
    return np.concatenate([v[:, -3:].mean(1), v[:, -1], v[:, -1] - v[:, -4], v.mean(1)], 1)


def prev_inputs(v):
    return np.concatenate([v[:, :3].mean(1), v[:, 0], v[:, 0] - v[:, 3], v.mean(1)], 1)


def fit_predictors(units):
    """units: list of (v (n,15,K) centred, consecutive mask) from train recordings in time order."""
    Xn, Yn, Xp, Yp = [], [], [], []
    for v in units:
        Xn.append(next_inputs(v[:-1])); Yn.append(v[1:, :3].mean(1))
        Xp.append(prev_inputs(v[1:])); Yp.append(v[:-1, -3:].mean(1))
    return ridge_fit(np.concatenate(Xn), np.concatenate(Yn)), ridge_fit(np.concatenate(Xp), np.concatenate(Yp))


def imu_extrap(w):
    """Predicted next sample(s) from the tail of each window, and the first samples."""
    lin1 = 2 * w[:, -1] - w[:, -2]
    lin2 = 3 * w[:, -1] - 2 * w[:, -2]
    return lin1, lin2


def row_rank_greater(S, a, x, descending=True):
    """Number of entries in row a[i] of S that are strictly greater (or smaller if not descending) than x[i]."""
    S = np.asarray(S, dtype=np.float64)
    finite = np.isfinite(S)
    big = np.abs(S[finite]).max() + 1.0 if finite.any() else 1.0
    S = np.where(finite, S, big if not descending else -big)
    span = 2 * big + 1.0
    n, m = S.shape
    srt = np.sort(S, axis=1) + (np.arange(n)[:, None] * span)
    flat = srt.ravel()
    key = np.asarray(x, dtype=np.float64) + a * span
    if descending:
        pos = np.searchsorted(flat, key, side="right")
        return (a + 1) * m - pos
    pos = np.searchsorted(flat, key, side="left")
    return pos - a * m


def _topk(S, k):
    k = min(k, S.shape[1] - 1)
    return np.argpartition(-S, k - 1, axis=1)[:, :k]


def build_pairs(v, w, limb, probs, pred_models, imu_feat=None, extra_sims=None, k_extra=15):
    """Candidate successor pairs and their features for one unit (subject / recording).

    v: (n,15,K) video centred per unit; w: (n,50,3) IMU of the given limb; limb: (n,)
    probs: (n,C) window class probabilities (OOF for train); imu_feat: optional (n,F) standardized IMU features.
    """
    n = len(v)
    tail = unit(v[:, -3:].mean(1)); head = unit(v[:, :3].mean(1))
    last = unit(v[:, -1]); first = unit(v[:, 0]); mean = unit(v.mean(1))
    pn = unit(ridge_apply(pred_models[0], next_inputs(v)))
    pp = unit(ridge_apply(pred_models[1], prev_inputs(v)))
    TH = tail @ head.T; np.fill_diagonal(TH, -2)
    PH = pn @ head.T; np.fill_diagonal(PH, -2)          # predicted next head vs head
    PT = tail @ pp.T; np.fill_diagonal(PT, -2)          # tail vs predicted previous tail of j
    PB = 0.5 * (PH + PT)
    # IMU continuity distance (same limb only)
    lin1, lin2 = imu_extrap(w)
    D1 = np.sqrt(((lin1 ** 2).sum(1)[:, None] + (w[:, 0] ** 2).sum(1)[None, :] - 2 * lin1 @ w[:, 0].T).clip(0)).astype(np.float32)
    same = limb[:, None] == limb[None, :]
    D1s = np.where(same, D1, np.inf); np.fill_diagonal(D1s, np.inf)
    cand = [_topk(TH, KV), _topk(PB, KP)]
    ci = np.argpartition(D1s, min(KI, n - 2), axis=1)[:, :KI]
    cand.append(ci)
    if extra_sims is not None:
        for Sx in extra_sims:
            Sx = Sx.copy(); np.fill_diagonal(Sx, -np.inf); cand.append(_topk(Sx, k_extra))
    c = np.concatenate(cand, 1)
    rows = np.repeat(np.arange(n), c.shape[1])
    pairs = np.unique(np.stack([rows, c.ravel()], 1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    a, b = pairs[:, 0], pairs[:, 1]

    def rank_feats(S, a, b):
        x = S[a, b]
        r = row_rank_greater(S, a, x)
        c_ = row_rank_greater(S.T, b, x)
        return [x, x - S.max(1)[a], x - S.max(0)[b], np.log1p(r), np.log1p(c_)]

    f = []
    f += rank_feats(TH, a, b)
    f += rank_feats(PB, a, b)
    f += [PH[a, b], PT[a, b], TH[b, a], (last[a] * first[b]).sum(1), (mean[a] * mean[b]).sum(1),
          (head[a] * head[b]).sum(1), (tail[a] * tail[b]).sum(1)]
    srt = np.sort(TH, 1)
    f += [srt[a, -1] - srt[a, -2]]
    sm = same[a, b].astype(np.float32)
    d1 = D1[a, b]
    d2 = np.linalg.norm(lin2[a] - w[b, 1], axis=1)
    scale = w[a].std(axis=1).mean(1) + w[b].std(axis=1).mean(1) + 1e-3
    # rank of b among a's same-limb candidates by D1, and of a among b's
    r1 = np.where(sm > 0, row_rank_greater(D1s, a, d1, descending=False), 0)
    r2 = np.where(sm > 0, row_rank_greater(D1s.T, b, d1, descending=False), 0)
    f += [sm, np.where(sm > 0, np.log1p(d1), -1), np.where(sm > 0, np.log1p(d2), -1),
          np.where(sm > 0, d1 / scale, -1), np.where(sm > 0, np.log1p(r1), -1), np.where(sm > 0, np.log1p(r2), -1)]
    # cross-limb magnitude continuity
    ma = np.linalg.norm(w[a, -5:], axis=2).mean(1); mb = np.linalg.norm(w[b, :5], axis=2).mean(1)
    f += [np.abs(ma - mb), np.abs(w[a].std(1).mean(1) - w[b].std(1).mean(1))]
    f += [limb[a].astype(np.float32), limb[b].astype(np.float32)]
    if probs is not None:
        f += [np.sqrt(probs[a] * probs[b]).sum(1), (probs[a].argmax(1) == probs[b].argmax(1)).astype(np.float32)]
    if imu_feat is not None:
        f += [np.where(sm > 0, np.linalg.norm(imu_feat[a] - imu_feat[b], axis=1), -1)]
    if extra_sims is not None:
        for Sx in extra_sims:
            Sx = Sx.copy(); np.fill_diagonal(Sx, -2)
            f += rank_feats(Sx, a, b)
            f += [Sx[b, a]]
    X = np.stack(f, 1).astype(np.float32)
    return pairs, X
