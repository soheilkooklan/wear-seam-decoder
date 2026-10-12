# How optimistic is non-nested stacking on this data?
# Two-stage pipeline (window HGB -> graph-feature HGB), evaluated with plain out-of-fold stacking
# and with nested cross-validation over the same five participant folds.
import sys, time, json
sys.path.insert(0, '.')
from common import *
from sim import pick_limbs
from decode_lib import knn
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

T = pd.read_pickle(WORK/'tiles_orig.pkl'); y = T.y.to_numpy(); sbj = T.sbj.to_numpy(); n = len(T)
LIMB = pick_limbs(T, 2)
FOLD = {1: 0, 6: 0, 11: 0, 16: 0, 21: 0, 0: 1, 5: 1, 10: 1, 15: 1, 20: 1, 4: 2, 9: 2, 14: 2, 19: 2,
        3: 3, 8: 3, 13: 3, 18: 3, 2: 4, 7: 4, 12: 4, 17: 4}
fold = np.array([FOLD[s] for s in sbj])
bad = np.zeros(n, bool)
for r, k in [('sbj_2', 112), ('sbj_10', 1419)]:
    bad[T[T.rec == r].sort_values('t').tail(k).index] = True
ok = ~bad
single = ~np.isin(sbj, (0, 2, 3, 14))

def std(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
FI = np.load(WORK/'imu_feats_exact.npy', mmap_mode='r')
F = np.stack([FI[i, l] for i, l in enumerate(LIMB)]).astype(np.float32)
Fs = np.zeros_like(F)
for s in np.unique(sbj):
    for l in range(4):
        m = (sbj == s) & (LIMB == l)
        if m.sum() > 1: Fs[m] = std(F[m])
E = np.load(WORK/'E_mean.npy').astype(np.float32)
Es = np.zeros_like(E)
for s in np.unique(sbj): m = sbj == s; Es[m] = std(E[m])
Zv = PCA(32, random_state=0).fit_transform(Es).astype(np.float32)
X1 = np.nan_to_num(np.concatenate([Fs, np.eye(4, dtype=np.float32)[LIMB], Zv], 1))
log('stage-1 features', X1.shape)

def m1():
    return HistGradientBoostingClassifier(max_iter=50, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight='balanced', early_stopping=False, random_state=0)
def m2():
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50,
                                          l2_regularization=1.0, class_weight='balanced', early_stopping=False, random_state=0)

SUB = np.random.default_rng(0).random(n) < 0.5
def fit_predict(train_mask, pred_mask, X, model, sub=False):
    tr = train_mask & ok & (SUB if sub else True)
    m = model().fit(X[tr], y[tr])
    return m.predict_proba(X[pred_mask])

# stage 1: plain out-of-fold
P1 = np.zeros((n, 19))
for k in range(5):
    P1[fold == k] = fit_predict(fold != k, fold == k, X1, m1, True)
log('stage-1 OOF window F1', round(macro_f1(y[ok], P1[ok].argmax(1)), 4))

# stage 1: inner out-of-fold for every outer fold
P1in = {}
for k in range(5):
    Q = np.zeros((n, 19))
    for j in range(5):
        if j == k: continue
        tr = (fold != k) & (fold != j)
        Q[fold == j] = fit_predict(tr, fold == j, X1, m1, True)
    P1in[k] = Q
    log('inner OOF done for outer fold', k)

# participant graphs (video kNN only, no labels)
H = {}
for r, g in T.groupby('rec', sort=False):
    idx = g.index.to_numpy(); m = len(idx)
    nb, s = knn(std(E[idx]), 10)
    W = csr_matrix((np.exp(5 * (s.ravel() - 1)), (np.repeat(np.arange(m), 10), nb.ravel())), shape=(m, m))
    A = W + W.T; d = np.asarray(A.sum(1)).ravel() + 1e-9
    H[r] = (idx, diags(1 / d) @ A)
def graph_feats(P):
    lp = np.log(P + 1e-6); out = np.zeros((n, 57), np.float32)
    for r, (idx, Hr) in H.items():
        a = lp[idx]; b = Hr @ a; c = Hr @ b
        out[idx] = np.concatenate([a, b, c], 1)
    return out

X2 = graph_feats(P1)
P2_plain = np.zeros((n, 19)); P2_nested = np.zeros((n, 19))
for k in range(5):
    P2_plain[fold == k] = fit_predict(fold != k, fold == k, X2, m2)
    X2k = graph_feats(np.where((fold != k)[:, None], P1in[k], P1))
    P2_nested[fold == k] = fit_predict(fold != k, fold == k, X2k, m2)
    log('stage-2 fold', k)

def sc(P): return [round(macro_f1(y[ok], P[ok].argmax(1)), 4), round(macro_f1(y[ok & single], P[ok & single].argmax(1)), 4)]
res = {'stage1_oof': sc(P1), 'stage2_plain': sc(P2_plain), 'stage2_nested': sc(P2_nested)}
log(res)
json.dump(res, open('../analysis/nested_check.json', 'w'), indent=1)
np.save(WORK/'nested_P2_plain.npy', P2_plain.astype(np.float32)); np.save(WORK/'nested_P2_nested.npy', P2_nested.astype(np.float32))
log('done')
