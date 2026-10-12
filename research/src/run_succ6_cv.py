"""Successor model with contrastive video-embedding features. CV scores (train sim) + test scores."""
import sys, time, pickle
sys.path.insert(0, '.')
from common import *
from sim import pick_limbs
from successor import *
import successor as _s
from contrast_lib import end_feats, start_feats, train_contrastive, embed
CE_DIR = WORK / 'contrast'
from test_data import load_test
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
_s.KV, _s.KP = 30, 15
TAG = sys.argv[1] if len(sys.argv) > 1 else 'c'
VAL = int(sys.argv[2]); TRD = [d for d in (0, 1, 2) if d != VAL]
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
T = pd.read_pickle(WORK/'tiles.pkl'); y = T.y.to_numpy(); sb = T.sbj.to_numpy(); recs = T.rec.to_numpy(); ar = np.arange(len(T))
V = np.load(WORK/'train_video_f16.npy').astype(np.float32)
W = dict(v_mlp=0.2, v_deep=0.2, i_mlp=0.2, i_hgb=0.3, i_deep=0.1)
OO = {'v_mlp': np.load(WORK/'oof_video_snorm.npy'), 'v_deep': np.load(WORK/'oof_video_deep.npy'), 'i_mlp': np.load(WORK/'oof_imu_mlp_exact.npy'),
      'i_hgb': np.load(WORK/'oof_imu_hgb_exact.npy'), 'i_deep': np.load(WORK/'oof_imu_deep.npy')}
def fusedP(limb):
    z = sum(W[k] * np.log((OO[k] if OO[k].ndim == 2 else OO[k][ar, limb]) + 1e-6) for k in W)
    P = np.exp(z - z.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)
limbs = [pick_limbs(T, s) for s in range(3)]; PL = [fusedP(l) for l in limbs]
CUBES = {r: np.nan_to_num(np.asarray(load_cube(r), dtype=np.float32)) for r in T.rec.unique()}
VC = {}; CD = {}
for r in T.rec.unique():
    idx = np.flatnonzero(recs == r); v = V[idx]; v = v - v.reshape(-1, v.shape[-1]).mean(0); VC[r] = v
    vs = v / (v.reshape(-1, v.shape[-1]).std(0) + 1e-6); CD[r] = (end_feats(vs).astype(np.float32), start_feats(vs).astype(np.float32))
def unit_w(r, limb):
    idx = np.flatnonzero(recs == r); n = len(idx); l = limb[idx]
    w = CUBES[r][np.arange(n)[:, None] * 50 + np.arange(50)[None]][np.arange(n)[:, None], np.arange(50)[None], l[:, None]]
    return idx, w, l
def clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100, early_stopping=False, random_state=0)
FULL = {}
def full_sim(r):
    if r not in FULL:
        a = np.load(CE_DIR / 'train' / f'{r}.npy').astype(np.float32)
        FULL[r] = sum(a[:, k, 0] @ a[:, k, 1].T for k in range(a.shape[1])) / a.shape[1]
    return FULL[r]
def csim(te, ts_, xe, xs, r=None):
    ue, us = embed(te, ts_, xe, xs); return ue @ us.T
usb = np.unique(sb); folds = list(GroupKFold(5).split(usb, groups=usb))
out = {}; top1 = []
for fi, (tr_s, va_s) in enumerate(folds):
    tr_recs = [r for r in T.rec.unique() if subject_of(r) in usb[tr_s]]; va_recs = [r for r in T.rec.unique() if subject_of(r) in usb[va_s]]
    te_, ts_ = train_contrastive(CD, tr_recs, iters=3000, hidden=False, tau=0.1, lr=2e-3, out=256, seed=fi)
    pm = fit_predictors([VC[r] for r in tr_recs])
    Xs, Ys = [], []; rng = np.random.default_rng(fi)
    for seed in TRD:
        for r in tr_recs:
            idx, w, l = unit_w(r, limbs[seed])
            pairs, X = build_pairs(VC[r], w, l, PL[seed][idx], pm, extra_sims=[csim(te_, ts_, *CD[r]), full_sim(r)])
            lab = pairs[:, 1] == pairs[:, 0] + 1; keep = lab | (rng.random(len(lab)) < 0.2)
            Xs.append(X[keep].astype(np.float32)); Ys.append(lab[keep]); del X
    m = clf().fit(np.concatenate(Xs), np.concatenate(Ys))
    for r in va_recs:
        idx, w, l = unit_w(r, limbs[VAL])
        pairs, X = build_pairs(VC[r], w, l, PL[VAL][idx], pm, extra_sims=[csim(te_, ts_, *CD[r]), full_sim(r)])
        s = m.predict_proba(X)[:, 1]; out[r] = (pairs, s)
        best = pd.DataFrame({'a': pairs[:, 0], 'b': pairs[:, 1], 's': s}).sort_values('s').groupby('a').tail(1)
        top1.append((best.b.to_numpy() == best.a.to_numpy() + 1).mean())
    log('fold', fi, 'mean top1 so far', round(np.mean(top1), 4))
pickle.dump(out, open(WORK/f'succ_scores_cv_{TAG}.pkl', 'wb'))
