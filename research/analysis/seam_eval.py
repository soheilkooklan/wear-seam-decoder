# same-limb successor ranking with different seam distances, all training recordings
import sys; sys.path.insert(0, '.')
from common import *
from sim import pick_limbs

T = pd.read_pickle(WORK/'tiles_orig.pkl'); LIMB = pick_limbs(T, 2)

def ar_fit(X, p):
    L = X.shape[1]; A = np.stack([X[:, i:L - p + i] for i in range(p)], 2)
    AtA = np.einsum('mtp,mtq->mpq', A, A) + 1e-3 * np.eye(p); Aty = np.einsum('mtp,mt->mp', A, X[:, p:])
    return np.linalg.solve(AtA, Aty[..., None])[..., 0]

def ar_pred(X, c, k):
    p = c.shape[1]; buf = X[:, -p:].copy(); out = []
    for _ in range(k):
        nx = (buf * c).sum(1); out.append(nx); buf = np.concatenate([buf[:, 1:], nx[:, None]], 1)
    return np.stack(out, 1)

def dist(P, Q):
    return np.sqrt(((P ** 2).sum(1)[:, None] + (Q ** 2).sum(1)[None] - 2 * P @ Q.T).clip(0))

def measures(W):
    k = len(W)
    va = W[:, -1] - W[:, -2]; aa = W[:, -1] - 2 * W[:, -2] + W[:, -3]
    vb = W[:, 1] - W[:, 0]; ab = W[:, 2] - 2 * W[:, 1] + W[:, 0]
    ja = W[:, -1] - 3 * W[:, -2] + 3 * W[:, -3] - W[:, -4]; jb = W[:, 3] - 3 * W[:, 2] + 3 * W[:, 1] - W[:, 0]
    qa = W[:, -1] + 0.5 * va + 0.375 * aa; qb = W[:, 0] - 0.5 * vb + 0.375 * ab
    D = {}
    D['endpoint'] = dist(W[:, -1], W[:, 0])
    D['linear, one side'] = dist(2 * W[:, -1] - W[:, -2], W[:, 0])
    D['linear, both sides'] = dist(W[:, -1] + 0.5 * va, W[:, 0] - 0.5 * vb)
    D['quadratic, both sides'] = dist(qa, qb)
    D['cubic, both sides'] = dist(qa + 0.3125 * ja, qb - 0.3125 * jb)
    Xa = W.transpose(0, 2, 1).reshape(k * 3, 50); pa = ar_pred(Xa, ar_fit(Xa, 6), 4).reshape(k, 3, 4).transpose(0, 2, 1)
    Xb = W[:, ::-1].transpose(0, 2, 1).reshape(k * 3, 50); pb = ar_pred(Xb, ar_fit(Xb, 6), 4).reshape(k, 3, 4).transpose(0, 2, 1)
    Df = np.sqrt(((pa[:, None] - W[None, :, :4]) ** 2).sum((2, 3)))
    Db = np.sqrt(((pb[None, :] - W[:, None, ::-1][:, :, :4]) ** 2).sum((2, 3)))
    D['AR(6), both sides'] = Df + Db
    return D

top1, rank, ncand = {}, {}, []
for r, g in T.groupby('rec', sort=False):
    idx = g.index.to_numpy(); n = len(idx); cube = np.nan_to_num(np.asarray(load_cube(r), np.float32))
    l = LIMB[idx]
    W = cube[np.arange(n)[:, None] * 50 + np.arange(50)[None]][np.arange(n)[:, None], np.arange(50)[None], l[:, None]].astype(np.float64)
    for lb in range(4):
        m = np.flatnonzero(l == lb)
        if len(m) < 20: continue
        pos = {v: i for i, v in enumerate(m)}
        pr = [(i, pos[v + 1]) for i, v in enumerate(m) if v + 1 in pos]
        if not pr: continue
        ai = np.array([a for a, _ in pr]); bi = np.array([b for _, b in pr]); ncand += [len(m) - 1] * len(ai)
        for name, D in measures(W[m]).items():
            D = D.copy(); np.fill_diagonal(D, np.inf)
            rk = (D[ai] < D[ai, bi][:, None]).sum(1)
            top1.setdefault(name, []).append(rk == 0); rank.setdefault(name, []).append(rk + 1)
print('anchors', len(ncand), 'median candidates per anchor', int(np.median(ncand)))
for name in top1:
    t = np.concatenate(top1[name]); rk = np.concatenate(rank[name])
    print(f'{name:24s} top1 {t.mean():.3f}  top5 {(rk <= 5).mean():.3f}  median rank {np.median(rk):.0f}')
