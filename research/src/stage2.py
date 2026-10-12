"""Second-stage window features: graph-propagated first-stage log-probs, video and per-limb IMU features."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, diags
from decode_lib import build_graph
import os
STACK_LW = float(os.environ.get('STACK_LW', '10'))


def unit_std(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def stage2_features(logP, video_feats, imu_feats, limb, emb, pairs, scores, link_w=STACK_LW, knn_k=10):
    """All inputs for one unit (subject / recording). Returns (n, F) float32 matrix."""
    n = len(logP)
    A = build_graph(emb, pairs, scores, knn_k=knn_k, link_w=link_w)
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    H = diags(1.0 / deg) @ A                     # row-normalised
    H2 = H @ H
    vf = unit_std(video_feats)
    fi = imu_feats.copy()
    for k in range(4):
        m = limb == k
        if m.sum() > 1:
            fi[m] = unit_std(fi[m])
    oh = np.eye(4, dtype=np.float32)[limb]
    out = [fi, oh, vf, logP, H @ logP, H2 @ logP, H @ vf]
    for k in range(4):
        m = (limb == k).astype(np.float32)
        cov1 = H @ m; cov2 = H2 @ m
        agg = (H2 @ (fi * m[:, None])) / (cov2[:, None] + 1e-6)
        agg1 = (H @ (fi * m[:, None])) / (cov1[:, None] + 1e-6)
        w = np.clip(cov1, 0, 1)[:, None]
        out += [agg, w * agg1, cov1[:, None], cov2[:, None]]
    return np.concatenate(out, 1).astype(np.float32)


def stage2_extra(comp_logps, imu_logp, limb, emb, pairs, scores, link_w=STACK_LW, knn_k=10):
    """Extra features: component log-probs, plus per-limb neighbourhood averages of the IMU log-probs."""
    A = build_graph(emb, pairs, scores, knn_k=knn_k, link_w=link_w)
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    H = diags(1.0 / deg) @ A; H2 = H @ H
    out = list(comp_logps)
    for k in range(4):
        m = (limb == k).astype(np.float32)
        cov2 = H2 @ m
        out.append((H2 @ (imu_logp * m[:, None])) / (cov2[:, None] + 1e-6))
    return np.concatenate(out, 1).astype(np.float32)
