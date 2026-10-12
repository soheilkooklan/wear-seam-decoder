"""Iterative stacking on train (subject-CV simulation) and test, sharing one code path.

Stage 1: window probabilities P (video + IMU fusion).
Stage k>=2: features = own IMU/video features, graph-propagated log-probs, per-limb neighbourhood IMU
features, plus (k>=3) decoded labels of the previous stage. MLP trained with subject CV; test predicted
by the average of the fold models. Each stage output is mixed with stage 1 and decoded.
"""
import sys, time, pickle, json
sys.path.insert(0, '.')
from common import *
from sim import pick_limbs
from stage2 import stage2_features, stage2_extra
import os
from decode_lib import build_graph, propagate, decode_unit
from video_feats import pooled
from test_data import load_test
from imu_feats import imu_features
from mlp import MLP
from scipy.sparse import diags
from sklearn.model_selection import GroupKFold

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
N_STAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else 'v3'
MIX = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
DEC = sys.argv[4] if len(sys.argv) > 4 else 'cl'
SUCC_TR = sys.argv[5] if len(sys.argv) > 5 else 'succ_scores_cv.pkl'
SUCC_TE = sys.argv[6] if len(sys.argv) > 6 else 'test_succ_v2.pkl'

# ---------------- train side (simulation)
T = pd.read_pickle(WORK/os.environ.get('TILES', 'tiles.pkl')); y = T.y.to_numpy(); recs = T.rec.to_numpy()
LIMB = pick_limbs(T, int(os.environ.get('DRAW', '2')))
ar = np.arange(len(T))
Lv = np.log(np.load(WORK/'oof_video_snorm.npy')+1e-6); Lm = np.log(np.load(WORK/'oof_imu_mlp_exact.npy')[ar, LIMB]+1e-6); Lh = np.log(np.load(WORK/'oof_imu_hgb_exact.npy')[ar, LIMB]+1e-6)
z = 0.4*Lv + 0.3*Lm + 0.3*Lh; P1 = np.exp(z - z.max(1, keepdims=True)); P1 /= P1.sum(1, keepdims=True)
import os
if os.path.exists(WORK/f'sim_P1_{TAG}.npy'):
    P1 = np.load(WORK/f'sim_P1_{TAG}.npy').astype(np.float64); log('using stage-1', f'sim_P1_{TAG}.npy')
elif os.environ.get('DRAW', '2') != '2':
    _W = dict(v_mlp=0.2, v_deep=0.2, i_mlp=0.2, i_hgb=0.3, i_deep=0.1)
    _O = {'v_mlp': np.load(WORK/'oof_video_snorm.npy'), 'v_deep': np.load(WORK/'oof_video_deep.npy'), 'i_mlp': np.load(WORK/'oof_imu_mlp_exact.npy')[ar, LIMB],
          'i_hgb': np.load(WORK/'oof_imu_hgb_exact.npy')[ar, LIMB], 'i_deep': np.load(WORK/'oof_imu_deep.npy')[ar, LIMB]}
    _z = sum(_W[k] * np.log(_O[k] + 1e-6) for k in _W); P1 = np.exp(_z - _z.max(1, keepdims=True)); P1 /= P1.sum(1, keepdims=True)
    log('stage-1 recomputed for draw', os.environ['DRAW'])
elif os.path.exists(WORK/f'sim_P1_{TAG}.npy'):
    P1 = np.load(WORK/f'sim_P1_{TAG}.npy'); log('using stage-1', f'sim_P1_{TAG}.npy')
Vd = pooled(np.load(WORK/'train_video_f16.npy').astype(np.float32))
VF = np.concatenate([Vd['mean'][:, :64], Vd['std'][:, :32]], 1); E = Vd['mean']
F = np.load(WORK/'imu_feats_exact.npy')[ar, LIMB]
SC = pickle.load(open(WORK/SUCC_TR, 'rb'))
units_tr = [(rec, g.index.to_numpy(), SC[rec][0], SC[rec][1]) for rec, g in T.groupby('rec', sort=False)]

EXTRA = os.environ.get('EXTRA', '0') == '1'
if EXTRA:
    CT = [np.load(WORK/os.environ.get('VMLP_OOF', 'oof_video_snorm.npy')), np.load(WORK/'oof_video_deep.npy'), np.load(WORK/os.environ.get('IMLP_OOF', 'oof_imu_mlp_exact.npy'))[ar, LIMB],
          np.load(WORK/'oof_imu_hgb_exact.npy')[ar, LIMB], np.load(WORK/'oof_imu_deep.npy')[ar, LIMB]]
    for _n in [x for x in os.environ.get('EXTRA_COMPS', '').split(',') if x]:
        _a = np.load(WORK/f'oof_{_n}.npy'); CT.append(_a[ar, LIMB] if _a.ndim == 3 else _a)
    CT = [np.log(c + 1e-6).astype(np.float32) for c in CT]
# ---------------- test side
meta, tv, tw, tl = load_test(); ts = meta.sbj_id.to_numpy()
P1t = np.load(WORK/f'test_P1_{TAG}.npy') if os.path.exists(WORK/f'test_P1_{TAG}.npy') else np.load(WORK/'test_P_v2.npy')
Vt = pooled(tv); VFt = np.concatenate([Vt['mean'][:, :64], Vt['std'][:, :32]], 1); Et = Vt['mean']
Ft = imu_features(tw)
ST = pickle.load(open(WORK/SUCC_TE, 'rb'))
units_te = [(s, idx, pairs, sc) for s, (idx, pairs, sc) in ST.items()]
if EXTRA:
    CE = [np.load(WORK/os.environ.get('VMLP_TE', 'test_pv_v1.npy')), np.load(WORK/'test_pv_deep.npy'), np.load(WORK/os.environ.get('IMLP_TE', 'test_pi_v1.npy')), np.load(WORK/'test_pi_hgb.npy'), np.load(WORK/'test_pi_deep.npy')]
    for _n in [x for x in os.environ.get('EXTRA_COMPS', '').split(',') if x]:
        CE.append(np.load(WORK/f'test_{_n}.npy'))
    CE = [np.log(c + 1e-6).astype(np.float32) for c in CE]


def mix(Pk, P1_):
    z = MIX*np.log(Pk+1e-6) + (1-MIX)*np.log(P1_+1e-6); P = np.exp(z - z.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)


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


def decode_all(P, units, emb):
    pred = np.zeros(len(P), int)
    for _, idx, pairs, sc in units:
        if DEC == 'mf':
            pred[idx] = mean_field(np.log(P[idx] + 1e-6), build_graph(emb[idx], pairs, sc, link_w=10))
        else:
            pred[idx] = decode_unit(P[idx], emb[idx], pairs, sc, link_w=10)
    return pred


def features(units, P1_, Pprev, dec, emb, vf, fi, limb, comps=None):
    X = None
    for _, idx, pairs, sc in units:
        base = stage2_features(np.log(P1_[idx]+1e-6), vf[idx], fi[idx], limb[idx], emb[idx], pairs, sc)
        parts = [base]
        if comps is not None:
            imu_lp = (comps[2][idx] + comps[3][idx] + comps[4][idx]) / 3
            parts.append(stage2_extra([c[idx] for c in comps], imu_lp, limb[idx], emb[idx], pairs, sc))
        if dec is not None:
            A = build_graph(emb[idx], pairs, sc, link_w=float(os.environ.get('STACK_LW', '10'))); deg = np.asarray(A.sum(1)).ravel() + 1e-9; H = diags(1/deg) @ A
            oh = np.eye(19, dtype=np.float32)[dec[idx]]
            lp = np.log(Pprev[idx]+1e-6)
            parts += [oh, H @ oh, H @ (H @ oh), lp, H @ lp]
        f = np.concatenate(parts, 1).astype(np.float32)
        if X is None: X = np.zeros((len(P1_), f.shape[1]), np.float32)
        X[idx] = f
    return X


sb = T.sbj.to_numpy(); usb = np.unique(sb); folds = list(GroupKFold(5).split(usb, groups=usb))
Pk_tr, Pk_te = P1, P1t
dec_tr = decode_all(P1, units_tr, E); dec_te = decode_all(P1t, units_te, Et)
log('stage 1 decode F1 (sim)', round(macro_f1(y, dec_tr), 4))
history = {'stage1': {'sim_f1': macro_f1(y, dec_tr)}}
np.save(WORK/f'{TAG}_dec_te_stage1.npy', dec_te)
for stage in range(2, N_STAGES + 1):
    use_dec = stage >= 3
    Xtr = features(units_tr, P1, Pk_tr, dec_tr if use_dec else None, E, VF, F, LIMB, CT if EXTRA else None)
    Xte = features(units_te, P1t, Pk_te, dec_te if use_dec else None, Et, VFt, Ft, tl, CE if EXTRA else None)
    log(f'stage {stage} features', Xtr.shape, Xte.shape)
    oof = np.zeros((len(T), N_CLASSES), np.float32); pte = np.zeros((len(meta), N_CLASSES), np.float32)
    for fi_, (tr_s, va_s) in enumerate(folds):
        tr = np.isin(sb, usb[tr_s]) & (y >= 0); va = np.isin(sb, usb[va_s])
        cw = tr.sum() / (N_CLASSES * np.bincount(y[tr], minlength=N_CLASSES))
        m = MLP([Xtr.shape[1], 512, 256, N_CLASSES], epochs=12, dropout=0.3, in_dropout=0.1, lr=1e-3, batch=256, seed=fi_ + 100 * stage, wd=1e-4)
        m.fit(Xtr[tr], y[tr], sample_weight=cw[y[tr]])
        oof[va] = m.predict_proba(Xtr[va]); pte += m.predict_proba(Xte) / 5
    Pk_tr, Pk_te = oof, pte
    dec_tr = decode_all(mix(oof, P1), units_tr, E); dec_te = decode_all(mix(pte, P1t), units_te, Et)
    f_win = macro_f1(y, oof.argmax(1)); f_dec = macro_f1(y, dec_tr)
    log(f'stage {stage}: window OOF {f_win:.4f}  decode (sim) {f_dec:.4f}  test pred dist', np.bincount(dec_te, minlength=19).tolist())
    history[f'stage{stage}'] = {'window_f1': f_win, 'sim_f1': f_dec}
    np.save(WORK/f'{TAG}_oof_stage{stage}.npy', oof); np.save(WORK/f'{TAG}_pte_stage{stage}.npy', pte)
    np.save(WORK/f'{TAG}_dec_te_stage{stage}.npy', dec_te); np.save(WORK/f'{TAG}_dec_tr_stage{stage}.npy', dec_tr)
json.dump(history, open(WORK/f'{TAG}_history.json', 'w'), indent=1)
