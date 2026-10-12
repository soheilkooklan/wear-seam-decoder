import sys, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import *
from video_feats import pooled, subject_standardize
from mlp import MLP

variant = sys.argv[1]
T = pd.read_pickle(WORK / 'tiles.pkl'); V = np.load(WORK / 'train_video_f16.npy').astype(np.float32)
y = T.y.to_numpy(); sb = T.sbj.to_numpy()
P = pooled(V)
keys = ['mean', 'std', 'head', 'tail', 'absdiff'] if variant != 'meanonly' else ['mean']
X = np.concatenate([P[k] for k in keys], 1)
if variant in ('snorm', 'meanonly'):
    X = subject_standardize(X, sb)
elif variant == 'both':
    X = np.concatenate([X, subject_standardize(X, sb)], 1)
usb = np.unique(sb)
folds = list(GroupKFold(5).split(np.zeros(len(usb)), groups=usb))
oof = np.zeros((len(T), N_CLASSES), np.float32)
t0 = time.time()
for fi, (tr_s, va_s) in enumerate(folds):
    tr = np.isin(sb, usb[tr_s]); va = np.isin(sb, usb[va_s])
    cw = tr.sum() / (N_CLASSES * np.bincount(y[tr], minlength=N_CLASSES))
    m = MLP([X.shape[1], 512, 256, N_CLASSES], epochs=15, dropout=0.3, in_dropout=0.1, lr=1e-3, batch=256, seed=fi, wd=1e-4)
    m.fit(X[tr], y[tr], sample_weight=cw[y[tr]])
    oof[va] = m.predict_proba(X[va])
    print(f'fold {fi} F1 {macro_f1(y[va], oof[va].argmax(1)):.4f} t {time.time()-t0:.0f}', flush=True)
np.save(WORK / f'oof_video_{variant}.npy', oof)
print(variant, 'VIDEO OOF', round(macro_f1(y, oof.argmax(1)), 4))
