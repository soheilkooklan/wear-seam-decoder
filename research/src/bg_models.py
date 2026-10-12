"""Background: IMU HGB/MLP with exact labels (CV + final), video HGB (CV + final)."""
import sys, time
sys.path.insert(0, '.')
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from video_feats import pooled, subject_standardize
from test_data import load_test, test_imu_features
from mlp import MLP

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
T = pd.read_pickle(WORK / 'tiles.pkl'); y = T.y.to_numpy(); sb = T.sbj.to_numpy()
ok = np.stack([T[f'ok{l}'].to_numpy() for l in range(4)], 1)
F = np.load(WORK / 'imu_feats_exact.npy')
Fn = np.zeros_like(F)
for s in np.unique(sb):
    for l in range(4):
        mm = (sb == s) & ok[:, l]
        Fn[sb == s, l] = (F[sb == s, l] - F[mm, l].mean(0)) / (F[mm, l].std(0) + 1e-6)
X = np.concatenate([F, Fn], 2)
meta, tv, tw, tl = load_test(); ts = meta.sbj_id.to_numpy()
Xt = test_imu_features(tw, tl, ts)
usb = np.unique(sb); folds = list(GroupKFold(5).split(usb, groups=usb))

def imu_rows(mask, frac, rng, onehot):
    Xs, ys = [], []
    for l in range(4):
        m = np.flatnonzero(mask & ok[:, l])
        if frac < 1: m = rng.choice(m, int(len(m) * frac), replace=False)
        extra = np.full((len(m), 1), l, np.float32) if not onehot else np.eye(4, dtype=np.float32)[np.full(len(m), l)]
        Xs.append(np.concatenate([X[m, l], extra], 1)); ys.append(y[m])
    return np.concatenate(Xs), np.concatenate(ys)

def hgb():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight='balanced', categorical_features=[X.shape[2]],
                                          early_stopping=False, random_state=0)

# ---- IMU MLP exact (CV + final)
oof = np.zeros((len(T), 4, N_CLASSES), np.float32)
for fi, (tr_s, va_s) in enumerate(folds):
    tr = np.isin(sb, usb[tr_s]); va = np.isin(sb, usb[va_s])
    Xtr, ytr = imu_rows(tr, 1.0, None, True)
    cw = len(ytr) / (N_CLASSES * np.bincount(ytr, minlength=N_CLASSES))
    m = MLP([Xtr.shape[1], 512, 256, N_CLASSES], epochs=12, dropout=0.3, lr=1e-3, batch=512, seed=fi).fit(Xtr, ytr, sample_weight=cw[ytr])
    for l in range(4):
        oof[va, l] = m.predict_proba(np.concatenate([X[va, l], np.eye(4, dtype=np.float32)[np.full(va.sum(), l)]], 1))
    log('imu mlp fold', fi)
np.save(WORK / 'oof_imu_mlp_exact.npy', oof)
log('IMU MLP exact OOF', [round(macro_f1(y[ok[:, l]], oof[ok[:, l], l].argmax(1)), 4) for l in range(4)])

# ---- IMU HGB (CV + final)
oof = np.zeros((len(T), 4, N_CLASSES), np.float32)
for fi, (tr_s, va_s) in enumerate(folds):
    tr = np.isin(sb, usb[tr_s]); va = np.isin(sb, usb[va_s])
    Xtr, ytr = imu_rows(tr, 0.5, np.random.default_rng(fi), False)
    m = hgb().fit(Xtr, ytr)
    for l in range(4):
        oof[va, l] = m.predict_proba(np.concatenate([X[va, l], np.full((va.sum(), 1), l, np.float32)], 1))
    log('imu hgb fold', fi, [round(macro_f1(y[va & ok[:, l]], oof[va & ok[:, l], l].argmax(1)), 4) for l in range(4)])
np.save(WORK / 'oof_imu_hgb_exact.npy', oof)
log('IMU HGB exact OOF', [round(macro_f1(y[ok[:, l]], oof[ok[:, l], l].argmax(1)), 4) for l in range(4)])
Xtr, ytr = imu_rows(np.ones(len(T), bool), 0.5, np.random.default_rng(99), False)
m = hgb().fit(Xtr, ytr)
np.save(WORK / 'test_pi_hgb.npy', m.predict_proba(np.concatenate([Xt, tl[:, None].astype(np.float32)], 1)))
log('IMU HGB final done')

# ---- video HGB on subject-standardized mean+std (CV + final)
V = np.load(WORK / 'train_video_f16.npy').astype(np.float32)
Pd = pooled(V); Pt = pooled(tv)
Xv = subject_standardize(np.concatenate([Pd['mean'], Pd['std'], Pd['absdiff'][:, :64]], 1), sb)
Xvt = subject_standardize(np.concatenate([Pt['mean'], Pt['std'], Pt['absdiff'][:, :64]], 1), ts)
def vhgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight='balanced', early_stopping=False, random_state=0)
oofv = np.zeros((len(T), N_CLASSES), np.float32)
for fi, (tr_s, va_s) in enumerate(folds):
    tr = np.isin(sb, usb[tr_s]); va = np.isin(sb, usb[va_s])
    m = vhgb().fit(Xv[tr], y[tr]); oofv[va] = m.predict_proba(Xv[va])
    log('video hgb fold', fi, round(macro_f1(y[va], oofv[va].argmax(1)), 4))
np.save(WORK / 'oof_video_hgb.npy', oofv)
log('VIDEO HGB OOF', round(macro_f1(y, oofv.argmax(1)), 4))
m = vhgb().fit(Xv, y)
np.save(WORK / 'test_pv_hgb.npy', m.predict_proba(Xvt))
log('video HGB final done')
