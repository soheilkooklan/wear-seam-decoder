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
    idx = np.flatnonzero(recs == r); v = V[idx]; v = v - v.reshape(-1, v.shape[-1]).mean(0); VC[r] = v.astype(np.float16)
    vs = v / (v.reshape(-1, v.shape[-1]).std(0) + 1e-6); CD[r] = (end_feats(vs).astype(np.float32), start_feats(vs).astype(np.float32))
def unit_w(r, limb):
    idx = np.flatnonzero(recs == r); n = len(idx); l = limb[idx]
    w = CUBES[r][np.arange(n)[:, None] * 50 + np.arange(50)[None]][np.arange(n)[:, None], np.arange(50)[None], l[:, None]]
    return idx, w, l
def clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100, early_stopping=False, random_state=0)
FULL = {}
def full_sim(r):
    a = np.load(CE_DIR / 'train' / f'{r}.npy').astype(np.float32)
    return (sum(a[:, k, 0] @ a[:, k, 1].T for k in range(a.shape[1])) / a.shape[1]).astype(np.float32)
def csim(te, ts_, xe, xs, r=None):
    ue, us = embed(te, ts_, xe, xs); return ue @ us.T
del V; import gc; gc.collect()
# final only
import gc

te_, ts_ = train_contrastive(CD, list(T.rec.unique()), iters=3000, hidden=False, tau=0.1, lr=2e-3, out=256, seed=99)
pm = fit_predictors([VC[r].astype(np.float32) for r in T.rec.unique()])
Xs, Ys = [], []; rng = np.random.default_rng(7)
for seed in (0, 1):
    for r in T.rec.unique():
        idx, w, l = unit_w(r, limbs[seed])
        pairs, X = build_pairs(VC[r].astype(np.float32), w, l, PL[seed][idx], pm, extra_sims=[csim(te_, ts_, *CD[r]), full_sim(r)])
        lab = pairs[:, 1] == pairs[:, 0] + 1; keep = lab | (rng.random(len(lab)) < 0.12)
        Xs.append(X[keep].astype(np.float32)); Ys.append(lab[keep]); del X; gc.collect()
XX = np.concatenate(Xs); YY = np.concatenate(Ys); del Xs, Ys; gc.collect()
m = clf().fit(XX, YY); del XX, YY; gc.collect()
pickle.dump((pm, m), open(WORK/f'succ_model_{TAG}.pkl', 'wb')); log('model saved')
del VC, CD, CUBES; gc.collect()
meta, tv, tw, tl = load_test(); P1t = np.load(WORK/'test_P1_v4.npy'); TEST_EMB = np.load(CE_DIR / 'test_emb.npy')
res = {}
for s_ in sorted(meta.sbj_id.unique()):
    idx = np.flatnonzero(meta.sbj_id.to_numpy() == s_)
    v = tv[idx]; v = v - v.reshape(-1, v.shape[-1]).mean(0); vs = v / (v.reshape(-1, v.shape[-1]).std(0) + 1e-6)
    Sc = csim(te_, ts_, end_feats(vs).astype(np.float32), start_feats(vs).astype(np.float32))
    ta = TEST_EMB[idx].astype(np.float32); Sf = sum(ta[:, k, 0] @ ta[:, k, 1].T for k in range(ta.shape[1])) / ta.shape[1]
    pairs, X = build_pairs(v, tw[idx], tl[idx], P1t[idx], pm, extra_sims=[Sc, Sf])
    res[s_] = (idx, pairs, m.predict_proba(X)[:, 1])
    log('test subject', s_, 'mean max score', round(pd.Series(res[s_][2]).groupby(pairs[:, 0]).max().mean(), 4))
pickle.dump(res, open(WORK/f'test_succ_{TAG}.pkl', 'wb'))
log('done')
