"""Fit window models on all train tiles and predict the test windows (v1: MLPs)."""
import sys, time
sys.path.insert(0, '.')
from common import *
from video_feats import pooled, subject_standardize
from test_data import load_test, test_imu_features
from mlp import MLP

t0 = time.time()
T = pd.read_pickle(WORK / 'tiles.pkl'); y = T.y.to_numpy(); sb = T.sbj.to_numpy()
V = np.load(WORK / 'train_video_f16.npy').astype(np.float32)
meta, tv, tw, tl = load_test(); ts = meta.sbj_id.to_numpy()
keys = ['mean', 'std', 'head', 'tail', 'absdiff']
Xv = subject_standardize(np.concatenate([pooled(V)[k] for k in keys], 1), sb)
Xv_t = subject_standardize(np.concatenate([pooled(tv)[k] for k in keys], 1), ts)
cw = len(y) / (N_CLASSES * np.bincount(y, minlength=N_CLASSES))
pv = np.zeros((len(meta), N_CLASSES))
for seed in range(2):
    m = MLP([Xv.shape[1], 512, 256, N_CLASSES], epochs=15, dropout=0.3, in_dropout=0.1, lr=1e-3, batch=256, seed=seed, wd=1e-4)
    m.fit(Xv, y, sample_weight=cw[y]); pv += m.predict_proba(Xv_t) / 2
    print('video seed', seed, time.time() - t0, flush=True)
np.save(WORK / 'test_pv_v1.npy', pv)
# IMU
F = np.load(WORK / 'imu_feats_exact.npy')
ok = np.stack([T[f'ok{l}'].to_numpy() for l in range(4)], 1)
Fn = np.zeros_like(F)
for s in np.unique(sb):
    for l in range(4):
        mm = (sb == s) & ok[:, l]
        Fn[sb == s, l] = (F[sb == s, l] - F[mm, l].mean(0)) / (F[mm, l].std(0) + 1e-6)
X = np.concatenate([F, Fn], 2)
Xtr, ytr = [], []
for l in range(4):
    oh = np.zeros((ok[:, l].sum(), 4), np.float32); oh[:, l] = 1
    Xtr.append(np.concatenate([X[ok[:, l], l], oh], 1)); ytr.append(y[ok[:, l]])
Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
cwi = len(ytr) / (N_CLASSES * np.bincount(ytr, minlength=N_CLASSES))
Xt = test_imu_features(tw, tl, ts)
oh = np.zeros((len(tl), 4), np.float32); oh[np.arange(len(tl)), tl] = 1
Xt = np.concatenate([Xt, oh], 1)
pi = np.zeros((len(meta), N_CLASSES))
for seed in range(2):
    m = MLP([Xtr.shape[1], 512, 256, N_CLASSES], epochs=12, dropout=0.3, lr=1e-3, batch=512, seed=seed).fit(Xtr, ytr, sample_weight=cwi[ytr])
    pi += m.predict_proba(Xt) / 2
    print('imu seed', seed, time.time() - t0, flush=True)
np.save(WORK / 'test_pi_v1.npy', pi)
print('test pred dist video', np.bincount(pv.argmax(1), minlength=19))
print('test pred dist imu', np.bincount(pi.argmax(1), minlength=19))
