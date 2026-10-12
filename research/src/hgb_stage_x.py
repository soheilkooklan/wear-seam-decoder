"""Generic HGB stacking stage on train (CV OOF) and test (full fit).
args: out_tag prev_tr prev_te dec_tr dec_te"""
import sys, time, pickle, os; sys.path.insert(0,'.')
from common import *
from sim import pick_limbs
from stage2 import stage2_extra
from decode_lib import build_graph
from video_feats import pooled
from test_data import load_test
from scipy.sparse import diags
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
out_tag, prev_tr, prev_te, dec_tr, dec_te = sys.argv[1:6]
t0 = time.time()
T = pd.read_pickle(WORK/os.environ.get('TILES', 'tiles.pkl')); y = T.y.to_numpy(); LIMB = pick_limbs(T, 2); ar = np.arange(len(T))
Vd = pooled(np.load(WORK/'train_video_f16.npy').astype(np.float32)); E = Vd['mean']; VF = np.concatenate([Vd['mean'][:, :64], Vd['std'][:, :32]], 1)
meta, tv, tw, tl = load_test(); Vt = pooled(tv); Et = Vt['mean']; VFt = np.concatenate([Vt['mean'][:, :64], Vt['std'][:, :32]], 1)
SC = pickle.load(open(WORK/os.environ.get('SUCC_TR', 'succ_scores_cv.pkl'), 'rb')); ST = pickle.load(open(WORK/os.environ.get('SUCC_TE', 'test_succ_v2.pkl'), 'rb'))
CT = [np.load(WORK/os.environ.get('VMLP_OOF', 'oof_video_snorm.npy')), np.load(WORK/'oof_video_deep.npy'), np.load(WORK/os.environ.get('IMLP_OOF', 'oof_imu_mlp_exact.npy'))[ar, LIMB],
      np.load(WORK/'oof_imu_hgb_exact.npy')[ar, LIMB], np.load(WORK/'oof_imu_deep.npy')[ar, LIMB]]
CE = [np.load(WORK/os.environ.get('VMLP_TE', 'test_pv_v1.npy')), np.load(WORK/'test_pv_deep.npy'), np.load(WORK/os.environ.get('IMLP_TE', 'test_pi_v1.npy')), np.load(WORK/'test_pi_hgb.npy'), np.load(WORK/'test_pi_deep.npy')]
for _n in [x for x in os.environ.get('EXTRA_COMPS', '').split(',') if x]:
    _a = np.load(WORK/f'oof_{_n}.npy'); CT.append(_a[ar, LIMB] if _a.ndim == 3 else _a); CE.append(np.load(WORK/f'test_{_n}.npy'))
CT = [np.log(c + 1e-6).astype(np.float32) for c in CT]; CE = [np.log(c + 1e-6).astype(np.float32) for c in CE]
P1, P1t = np.load(WORK/os.environ.get('P1_SIM', 'sim_P1_v7.npy')), np.load(WORK/os.environ.get('P1_TE', 'test_P1_v7.npy'))
Ptr, Pte = np.load(WORK/prev_tr), np.load(WORK/prev_te)
Dtr, Dte = np.load(WORK/dec_tr), np.load(WORK/dec_te)
from scipy.sparse import csr_matrix
CHAIN = os.environ.get('CHAIN', '0') == '1'
def chain_feats(n, pairs, sc, lpp, oh, imu_lp, limb):
    w = sc.astype(np.float64) ** 2
    Fm = csr_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    F = diags(1 / (np.asarray(Fm.sum(1)).ravel() + 1e-9)) @ Fm
    Bt = Fm.T.tocsr(); B = diags(1 / (np.asarray(Bt.sum(1)).ravel() + 1e-9)) @ Bt
    C = 0.5 * (F + B); C2 = C @ C; C4 = C2 @ C2
    out = []
    for x in (lpp, oh):
        out += [F @ x, B @ x, C2 @ x, C4 @ x]
    for k in range(4):
        m = (limb == k).astype(np.float64)
        for M in (C2, C4):
            cov = M @ m; out += [(M @ (imu_lp * m[:, None])) / (cov[:, None] + 1e-6), cov[:, None]]
    return np.concatenate(out, 1).astype(np.float32)

def feats(units, P1_, Pp, D, comps, limb, emb, vfa, n):
    X = None
    for idx, pairs, sc in units:
        A = build_graph(emb[idx], pairs, sc, link_w=float(os.environ.get('STACK_LW', '10'))); deg = np.asarray(A.sum(1)).ravel() + 1e-9; H = diags(1/deg) @ A; H2 = H @ H
        lp1 = np.log(P1_[idx] + 1e-6); lpp = np.log(Pp[idx] + 1e-6); oh = np.eye(19, dtype=np.float32)[D[idx]]
        imu_lp = (comps[2][idx] + comps[3][idx] + comps[4][idx]) / 3
        ex = stage2_extra([c[idx] for c in comps], imu_lp, limb[idx], emb[idx], pairs, sc)
        vf = vfa[idx]; vf = (vf - vf.mean(0)) / (vf.std(0) + 1e-6)
        f = np.concatenate([lp1, H @ lp1, H2 @ lp1, lpp, H @ lpp, H2 @ lpp, oh, H @ oh, H2 @ oh, ex, vf, np.eye(4)[limb[idx]]], 1).astype(np.float32)
        if CHAIN: f = np.concatenate([f, chain_feats(len(idx), pairs, sc, lpp, oh, imu_lp, limb[idx])], 1)
        if X is None: X = np.zeros((n, f.shape[1]), np.float32)
        X[idx] = f
    return X
units_tr = [(g.index.to_numpy(), SC[r][0], SC[r][1]) for r, g in T.groupby('rec', sort=False)]
units_te = [(idx, pairs, sc) for s, (idx, pairs, sc) in ST.items()]
X = feats(units_tr, P1, Ptr, Dtr, CT, LIMB, E, VF, len(T)); Xt = feats(units_te, P1t, Pte, Dte, CE, tl, Et, VFt, len(meta))
print('features', X.shape, Xt.shape, round(time.time()-t0), flush=True)
def hgb(seed=0):
    E_ = os.environ
    return HistGradientBoostingClassifier(max_iter=int(E_.get('HGB_ITERS', 150)), learning_rate=float(E_.get('HGB_LR', 0.1)),
                                          max_leaf_nodes=int(E_.get('HGB_LEAVES', 31)), min_samples_leaf=50, l2_regularization=1.0,
                                          max_features=float(E_.get('HGB_MAXFEAT', 1.0)), class_weight='balanced', early_stopping=False,
                                          random_state=int(E_.get('HGB_SEED', seed)))
sb = T.sbj.to_numpy(); usb = np.unique(sb); folds = list(GroupKFold(5).split(usb, groups=usb))
oof = np.zeros((len(T), N_CLASSES), np.float32)
for fi, (tr_s, va_s) in enumerate(folds):
    tr = np.isin(sb, usb[tr_s]) & (y >= 0); va = np.isin(sb, usb[va_s])
    oof[va] = hgb().fit(X[tr], y[tr]).predict_proba(X[va])
    print('fold', fi, round(macro_f1(y[va], oof[va].argmax(1)), 4), round(time.time()-t0), flush=True)
np.save(WORK/f'{out_tag}_oof.npy', oof)
print('window OOF', round(macro_f1(y, oof.argmax(1)), 4), flush=True)
pt = hgb().fit(X[y >= 0], y[y >= 0]).predict_proba(Xt); np.save(WORK/f'{out_tag}_pte.npy', pt)
print('test done', round(time.time()-t0), flush=True)
