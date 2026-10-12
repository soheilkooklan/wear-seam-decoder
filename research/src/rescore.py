"""Re-score successor candidate pairs: base successor score in context (row/column competition),
content-invariant frame self-similarity features, and optional deep bi-encoder similarities.
usage: python3 rescore.py OUT_TAG [BASE_TR BASE_TE]   env LINK=1 adds work/link bi-encoder features."""
import sys, time, pickle, os; sys.path.insert(0, '.')
from common import *
from test_data import load_test
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
OUT = sys.argv[1]; BTR = sys.argv[2] if len(sys.argv) > 2 else 'succ_scores_cv_f.pkl'; BTE = sys.argv[3] if len(sys.argv) > 3 else 'test_succ_f.pkl'
LINK = os.environ.get('LINK', '0') == '1'; LK = WORK / os.environ.get('LINK_DIR', 'link')
OFF = np.arange(15)[None, :] - np.arange(15)[:, None]

def norm_frames(v):
    v = v - v.reshape(-1, v.shape[-1]).mean(0); v = v / (v.reshape(-1, v.shape[-1]).std(0) + 1e-6)
    return (v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-8)).astype(np.float32)

def within(Fn):
    G = np.matmul(Fn, Fn.transpose(0, 2, 1))
    return np.stack([np.mean([G[:, i, i + k] for i in range(15 - k)], 0) for k in range(1, 15)], 1).astype(np.float32)

def ssm(Fn, a, b):
    out = []
    for s in range(0, len(a), 4000):
        C = np.matmul(Fn[a[s:s+4000]], Fn[b[s:s+4000]].transpose(0, 2, 1))
        diag = np.stack([C[:, OFF == d].mean(1) for d in range(-14, 15)], 1)
        out.append(np.concatenate([diag, C[:, 14, :], C[:, :, 0], C[:, -3:, :3].mean((1, 2))[:, None], C.max((1, 2))[:, None]], 1))
    return np.concatenate(out).astype(np.float32)

from scipy.optimize import linear_sum_assignment
HUNG = os.environ.get('HUNG', '0') == '1'
def hungarian(a, b, s, n):
    C = np.full((n, n), 20.0, np.float32); C[a, b] = -np.log(np.clip(s, 1e-4, 1))
    r, c = linear_sum_assignment(C)
    mb = np.full(n, -1); mb[r] = c; ma = np.full(n, -1); ma[c] = r
    inm = (mb[a] == b).astype(np.float32)
    sm = pd.Series(s, index=a.astype(np.int64) * n + b)
    row_m = sm.reindex(a.astype(np.int64) * n + mb[a]).fillna(0).to_numpy(); col_m = sm.reindex(ma[b].astype(np.int64) * n + b).fillna(0).to_numpy()
    return np.stack([inm, s - row_m, s - col_m], 1).astype(np.float32)

def context(a, b, s, n):
    d = pd.DataFrame({'a': a, 'b': b, 's': s})
    rmax = d.groupby('a').s.transform('max').to_numpy(); cmax = d.groupby('b').s.transform('max').to_numpy()
    rr = d.groupby('a').s.rank(ascending=False).to_numpy(); cr = d.groupby('b').s.rank(ascending=False).to_numpy()
    r2 = d.groupby('a').s.transform(lambda x: np.sort(x.to_numpy())[-2] if len(x) > 1 else -1).to_numpy()
    key = pd.Series(s, index=a.astype(np.int64) * n + b)
    rev = key.reindex(b.astype(np.int64) * n + a).fillna(-1).to_numpy()
    f = np.stack([s, s - rmax, s - cmax, np.log1p(rr), np.log1p(cr), rmax - r2, rev], 1).astype(np.float32)
    return np.concatenate([f, hungarian(a, b, s, n)], 1) if HUNG else f

def sim_feats(S, a, b):
    x = S[a, b]; S2 = S.copy(); np.fill_diagonal(S2, -9); S2T = np.ascontiguousarray(S2.T)
    rr = np.zeros(len(a), np.int32); cr = np.zeros(len(a), np.int32)
    for c in range(0, len(a), 20000):
        sl = slice(c, c + 20000)
        rr[sl] = (S2[a[sl]] > x[sl, None]).sum(1); cr[sl] = (S2T[b[sl]] > x[sl, None]).sum(1)
    return np.stack([x, x - S2.max(1)[a], x - S2.max(0)[b], np.log1p(rr), np.log1p(cr), S[b, a]], 1).astype(np.float32)

PROBS = os.environ.get('PROBS', '')
def prob_feats(p, a, b):
    q = np.sqrt(p[a] * p[b]).sum(1); am = (p[a].argmax(1) == p[b].argmax(1)).astype(np.float32)
    ent = -(p * np.log(p + 1e-9)).sum(1)
    return np.stack([q, am, ent[a], ent[b], p[a].max(1), p[b].max(1), (p[a] * p[b]).sum(1)], 1).astype(np.float32)

IMUQ = os.environ.get('IMUQ', '0') == '1'
def imu_cont(w, limb, a, b):
    """Same-limb IMU continuity: quadratic half-step extrapolation from both window ends (mid_quad) and variants,
    with ranks among all same-limb windows. NaN for cross-limb pairs."""
    out = np.full((len(a), 10), np.nan, np.float32)
    for lb in range(4):
        m = np.flatnonzero(limb == lb)
        if len(m) < 3: continue
        Wm = w[m].astype(np.float64); pos = np.full(len(w), -1); pos[m] = np.arange(len(m))
        va = Wm[:, -1] - Wm[:, -2]; aa = Wm[:, -1] - 2 * Wm[:, -2] + Wm[:, -3]
        vb = Wm[:, 1] - Wm[:, 0]; ab = Wm[:, 2] - 2 * Wm[:, 1] + Wm[:, 0]
        qa = Wm[:, -1] + 0.5 * va + 0.375 * aa; qb = Wm[:, 0] - 0.5 * vb + 0.375 * ab
        va2 = 0.5 * (Wm[:, -1] - Wm[:, -3]); vb2 = 0.5 * (Wm[:, 2] - Wm[:, 0])
        sc = np.median(np.abs(np.diff(Wm, 2, axis=1)), axis=(1, 2)) + 1e-3
        dist = lambda P, Q: np.sqrt(((P ** 2).sum(1)[:, None] + (Q ** 2).sum(1)[None] - 2 * P @ Q.T).clip(0))
        Dq = dist(qa, qb); np.fill_diagonal(Dq, np.inf)
        D2 = dist(Wm[:, -1] + 0.5 * va2, Wm[:, 0] - 0.5 * vb2); np.fill_diagonal(D2, np.inf)
        Dl = dist(2 * Wm[:, -1] - Wm[:, -2], Wm[:, 0]); np.fill_diagonal(Dl, np.inf)
        Dv = dist(va, vb)
        sel = np.flatnonzero((limb[a] == lb) & (limb[b] == lb)); ia = pos[a[sel]]; ib = pos[b[sel]]
        x = Dq[ia, ib]; x2 = D2[ia, ib]
        rr = (Dq[ia] < x[:, None]).sum(1); cr = (Dq[:, ib].T < x[:, None]).sum(1)
        rr2 = (D2[ia] < x2[:, None]).sum(1)
        ns = 0.5 * (sc[ia] + sc[ib])
        out[sel] = np.stack([x, x / ns, np.log1p(rr), np.log1p(cr), x - Dq.min(1)[ia], x - Dq.min(0)[ib],
                             x2 / ns, np.log1p(rr2), Dl[ia, ib] / ns, Dv[ia, ib] / ns], 1)
    return out

ILINK = os.environ.get('ILINK', '0') == '1'; ILK = WORK / os.environ.get('ILINK_DIR', 'imu_link')
def unit_features(Fn, pairs, s, emb=None, probs=None, iemb=None, limb=None, w=None):
    a, b = pairs[:, 0], pairs[:, 1]; Wn = within(Fn)
    f = [ssm(Fn, a, b), Wn[a], Wn[b], context(a, b, s, len(Fn))]
    if w is not None: f.append(imu_cont(w, limb, a, b))
    if iemb is not None:
        Si = sum(iemb[:, k, 0] @ iemb[:, k, 1].T for k in range(iemb.shape[1])) / iemb.shape[1]
        f.append(sim_feats(Si.astype(np.float32), a, b))
        f.append(np.stack([limb[a], limb[b], (limb[a] == limb[b])], 1).astype(np.float32))
    if probs is not None: f.append(prob_feats(probs, a, b))
    if emb is not None:
        S = sum(emb[:, k, 0] @ emb[:, k, 1].T for k in range(emb.shape[1])) / emb.shape[1]
        f.append(sim_feats(S.astype(np.float32), a, b))
    return np.concatenate(f, 1)

T = pd.read_pickle(WORK/'tiles.pkl'); V = np.load(WORK/'train_video_f16.npy', mmap_mode='r')
from sim import pick_limbs
LIMB = pick_limbs(T, 2)
PTR = np.load(WORK/f'{PROBS}_oof.npy') if PROBS else None; PTE = np.load(WORK/f'{PROBS}_pte.npy') if PROBS else None
SC = pickle.load(open(WORK/BTR, 'rb'))
X, Y, SUB, ANC, UNIT = [], [], [], [], []
for ui, (r, g) in enumerate(T.groupby('rec', sort=False)):
    idx = g.index.to_numpy(); pairs, s = SC[r]
    emb = np.load(LK/'train'/f'{r}_emb.npy').astype(np.float32) if LINK else None
    iemb = None
    if ILINK:
        ie = np.load(ILK/'train'/f'{r}_emb.npy'); iemb = ie[np.arange(len(idx)), LIMB[idx]].astype(np.float32)
    wq = None
    if IMUQ:
        n = len(idx); cube = np.nan_to_num(np.asarray(load_cube(r), np.float32))
        wq = cube[np.arange(n)[:, None] * 50 + np.arange(50)[None]][np.arange(n)[:, None], np.arange(50)[None], LIMB[idx][:, None]]
    X.append(unit_features(norm_frames(V[idx].astype(np.float32)), pairs, s.astype(np.float32), emb, PTR[idx] if PROBS else None, iemb, LIMB[idx], wq))
    Y.append(pairs[:, 1] == pairs[:, 0] + 1); SUB.append(np.full(len(pairs), subject_of(r))); ANC.append(pairs[:, 0] + 100000 * ui)
X = np.concatenate(X); Y = np.concatenate(Y); SUB = np.concatenate(SUB); ANC = np.concatenate(ANC)
log('train features', X.shape)
def top1(score):
    d = pd.DataFrame({'a': ANC, 's': score, 'y': Y}).sort_values('s').groupby('a').tail(1)
    return d.y.sum() / pd.Series(Y).groupby(ANC).any().sum()
def clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100, early_stopping=False, random_state=0)
log('base top1 (anchors with true pair in candidates)', round(top1(X[:, 89]), 4))
usb = np.unique(SUB); oof = np.zeros(len(Y), np.float32); rng = np.random.default_rng(0)
SKIP = os.environ.get('SKIP_CV', '0') == '1'
keep_all = Y | (rng.random(len(Y)) < 0.2)
for fi, (tr_s, va_s) in enumerate(GroupKFold(5).split(usb, groups=usb) if not SKIP else []):
    tr = np.isin(SUB, usb[tr_s]); va = ~tr; k = tr & keep_all
    oof[va] = clf().fit(X[k], Y[k]).predict_proba(X[va])[:, 1]; log('fold', fi)
if not SKIP: log('rescored top1', round(top1(oof), 4))
out = {}; pos = 0
for r, g in T.groupby('rec', sort=False):
    pairs, _ = SC[r]; out[r] = (pairs, oof[pos:pos + len(pairs)].astype(np.float64)); pos += len(pairs)
if not SKIP: pickle.dump(out, open(WORK/f'succ_scores_cv_{OUT}.pkl', 'wb'))
m = clf().fit(X[keep_all], Y[keep_all]); del X; import gc; gc.collect()
meta, tv, tw, tl = load_test(); ST = pickle.load(open(WORK/BTE, 'rb'))
LTE = np.load(LK/'test_emb.npy').astype(np.float32) if LINK else None
ITE = np.load(ILK/'test_emb.npy').astype(np.float32) if ILINK else None
res = {}
for s_, (idx, pairs, sc) in ST.items():
    Xt = unit_features(norm_frames(tv[idx]), pairs, sc.astype(np.float32), LTE[idx] if LINK else None, PTE[idx] if PROBS else None,
                       ITE[idx] if ILINK else None, tl[idx], tw[idx] if IMUQ else None)
    res[s_] = (idx, pairs, m.predict_proba(Xt)[:, 1]); log('test subject', s_, len(pairs))
pickle.dump(res, open(WORK/f'test_succ_{OUT}.pkl', 'wb')); log('done')
