"""Full per-subject decoder: kNN + successor-link graph, propagation, clustering, size-bounded assignment."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from common import N_CLASSES


def knn(e, k):
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    s = e @ e.T; np.fill_diagonal(s, -np.inf)
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    return idx, np.take_along_axis(s, idx, 1)


def build_graph(emb, pairs, scores, knn_k=10, knn_w=1.0, link_w=3.0, link_pow=2.0):
    n = len(emb)
    e = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
    nb, s = knn(e, knn_k)
    r = [np.repeat(np.arange(n), knn_k)]; c = [nb.ravel()]; w = [knn_w * np.exp(5 * (s.ravel() - 1))]
    if pairs is not None and link_w > 0:
        r.append(pairs[:, 0]); c.append(pairs[:, 1]); w.append(link_w * scores ** link_pow)
    r = np.concatenate(r); c = np.concatenate(c); w = np.concatenate(w)
    return csr_matrix((np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))


def propagate(P0, A, alpha=0.8, iters=30):
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    Q = P0.copy()
    for _ in range(iters):
        Q = (1 - alpha) * P0 + alpha * (A @ Q) / deg[:, None]
    return Q / Q.sum(1, keepdims=True)


def cluster_assign(Sbar, sizes, lo=70, hi=125, iters=300, step=0.05):
    M, C = Sbar.shape
    lam = np.zeros(C); best = None
    for it in range(iters):
        lab = (Sbar + lam).argmax(1)
        cnt = np.bincount(lab, weights=sizes, minlength=C)
        viol = 0.0; d = step * (1 - it / iters) + 0.005
        for c in range(1, C):
            if cnt[c] < lo: lam[c] += d; viol += lo - cnt[c]
            elif cnt[c] > hi: lam[c] -= d; viol += cnt[c] - hi
        if best is None or viol < best[0]:
            best = (viol, lab.copy())
        if viol == 0: break
    return best[1]


def decode_unit(P, emb, pairs, scores, alpha=0.8, per=25, wq=3.0, wv=1.0, lo=70, hi=125, return_all=False,
                assign='lagr', pen=20.0, tl=10, link_w=3.0, knn_k=10, base_src='Q', temp=1.0):
    n = len(P)
    A = build_graph(emb, pairs, scores, knn_k=knn_k, link_w=link_w)
    Q = propagate(P, A, alpha)
    e = (emb - emb.mean(0)) / (emb.std(0) + 1e-6); e = e / np.sqrt(e.shape[1])
    feat = np.concatenate([wq * np.sqrt(Q), wv * e], 1)
    cl = AgglomerativeClustering(n_clusters=max(20, n // per), linkage='ward', connectivity=A).fit_predict(feat)
    base = np.log(np.clip(Q if base_src == 'Q' else P, 1e-6, 1)) / temp; Mc = cl.max() + 1
    sizes = np.bincount(cl, minlength=Mc).astype(float)
    Ssum = np.zeros((Mc, N_CLASSES)); np.add.at(Ssum, cl, base)
    lab = None
    if assign == 'milp':
        lab = milp_assign(Ssum, sizes, lo, hi, pen=pen, time_limit=tl)
    if lab is None:
        lab = cluster_assign(Ssum / sizes[:, None], sizes, lo, hi)
    lab = lab[cl]
    if return_all:
        return lab, Q, cl, A
    return lab


def milp_assign(Ssum, sizes, lo=70, hi=125, pen=0.5, time_limit=20):
    """Exact-ish assignment: max sum_k sum_c Ssum[k,c] x_kc, one class per cluster,
    per-activity-class window counts within [lo, hi] (soft, slack penalised by pen per window)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    M, C = Ssum.shape
    nx = M * C; ns = 2 * (C - 1)
    c = np.concatenate([-Ssum.ravel(), np.full(ns, pen)])
    A = lil_matrix((M + 2 * (C - 1), nx + ns))
    lb = []; ub = []
    for k in range(M):
        A[k, k * C:(k + 1) * C] = 1; lb.append(1); ub.append(1)
    r = M
    for j, cl in enumerate(range(1, C)):
        cols = np.arange(M) * C + cl
        # sum size x + s_lo >= lo
        A[r, cols] = sizes; A[r, nx + 2 * j] = 1; lb.append(lo); ub.append(np.inf); r += 1
        # sum size x - s_hi <= hi
        A[r, cols] = sizes; A[r, nx + 2 * j + 1] = -1; lb.append(-np.inf); ub.append(hi); r += 1
    integrality = np.concatenate([np.ones(nx), np.zeros(ns)])
    bounds = Bounds(np.zeros(nx + ns), np.concatenate([np.ones(nx), np.full(ns, np.inf)]))
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integrality, bounds=bounds,
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None
    x = res.x[:nx].reshape(M, C)
    return x.argmax(1)
