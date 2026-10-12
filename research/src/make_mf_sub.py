import sys, pickle; sys.path.insert(0,'.')
from common import *
from decode_lib import build_graph
from video_feats import pooled
from test_data import load_test
import os
from decode_lib import knn
from scipy.sparse import csr_matrix
LW = float(os.environ.get('LW', '10')); BETA = float(os.environ.get('BETA', '2.0')); IMU_W = float(os.environ.get('IMU_W', '0')); LO = float(os.environ.get('LO', '70')); HI = float(os.environ.get('HI', '125')); IMU_K = int(os.environ.get('IMU_K', '10'))
tag, out, succ = sys.argv[1], sys.argv[2], sys.argv[3]
a = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
keys = (sys.argv[5] if len(sys.argv) > 5 else f'{tag}_pte_stage3,{tag}_pte_stage4').split(',')
def mean_field(lp, A, beta=2.0, iters=30, lo=70, hi=125):
    n = len(lp); b = np.zeros(19); deg = np.asarray(A.sum(1)).ravel() + 1e-9
    q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    for it in range(iters):
        msg = (A @ q) / deg[:, None]; zz = lp + b + beta * np.log(msg + 1e-3)
        q = np.exp(zz - zz.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        if it >= 5:
            cnt = np.bincount(q.argmax(1), minlength=19)
            for c in range(1, 19):
                if cnt[c] < lo: b[c] += 0.15
                elif cnt[c] > hi: b[c] -= 0.15
    return q.argmax(1)
meta, tv, tw, tl = load_test(); Et = pooled(tv)['mean']
P1t = np.load(WORK/f'test_P1_{tag}.npy')
S = [np.log(np.load(WORK/f'{k}.npy') + 1e-6) for k in keys]
z = a * sum(S) / len(S) + (1 - a) * np.log(P1t + 1e-6); P = np.exp(z - z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
ST = pickle.load(open(WORK/succ, 'rb'))
if IMU_W > 0:
    from imu_feats import imu_features
    FT = imu_features(tw).astype(np.float32); print('test imu feats', FT.shape)
def imu_graph(idx, k, w):
    n = len(idx); l = tl[idx]; r_, c_, w_ = [], [], []
    for lb in range(4):
        m = np.flatnonzero(l == lb)
        if len(m) <= k + 1: continue
        f = FT[idx[m]]; f = (f - f.mean(0)) / (f.std(0) + 1e-6); f = np.nan_to_num(f)
        nb, s_ = knn(f, k)
        r_.append(np.repeat(m, k)); c_.append(m[nb.ravel()]); w_.append(w * np.exp(5 * (s_.ravel() - 1)))
    r_ = np.concatenate(r_); c_ = np.concatenate(c_); w_ = np.concatenate(w_)
    return csr_matrix((np.concatenate([w_, w_]), (np.concatenate([r_, c_]), np.concatenate([c_, r_]))), shape=(n, n))
pred = P.argmax(1).copy()
for s, (idx, pairs, sc) in ST.items():
    A = build_graph(Et[idx], pairs, sc, link_w=LW)
    if IMU_W > 0: A = A + imu_graph(idx, IMU_K, IMU_W)
    pred[idx] = mean_field(np.log(P[idx] + 1e-6), A, beta=BETA, lo=LO, hi=HI)
    c = np.bincount(pred[idx], minlength=19); print(s, 'null', c[0], 'acts', c[1:].tolist())
sub = pd.DataFrame({'id': meta['id'].to_numpy(), 'target_feature': pred.astype(int)})
sub.to_csv(ROOT/'submissions'/f'submission_{out}.csv', index=False)
for v in ['v8', 'v9']:
    o = pd.read_csv(ROOT/'submissions'/f'submission_{v}.csv'); print('agree', v, round((o.target_feature.values == pred).mean(), 4))
