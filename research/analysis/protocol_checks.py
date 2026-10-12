# Protocol checks for the report: decoder settings chosen inside the folds,
# sensitivity of the video graph to the data used for the projection,
# confidence of the window models on OOF and test windows, per-participant spread.
import json
src = open('../analysis/paper_numbers.py').read()
exec(src.split('# ---------------------------------------------------------------- A.')[0])
from sklearn.metrics import f1_score
base = np.load(WORK/'sim_pred_final_check.npy'); PRED['final'] = base
R = {}
L3 = np.mean([L('hgb3h'), L('hgb3hm'), L('hgb3i3f')], 0)

# ---------------------------------------------------------------- 1. nested choice of decoder settings and mixing weight
grid = [(a, lo, hi, be) for a in (0.5, 0.6, 0.7) for lo in (70, 75, 80, 85) for hi in (90, 95, 100, 105, 110) for be in (1.25, 1.5, 1.75, 2.0)]
P = {}
for g in grid:
    a, lo, hi, be = g
    P[g] = decode(a * L3 + (1 - a) * L1, lo=lo, hi=hi, beta=be)
log('grid decoded', len(grid))
def pooled_f1(pred, mask): return macro_f1(y[mask], pred[mask])
sel = np.zeros(len(T), int); chosen = {}
for k in range(5):
    tr = okA & (fold != k)
    best = max(grid, key=lambda g: pooled_f1(P[g], tr))
    chosen[k] = best; sel[fold == k] = P[best][fold == k]
PRED['nested decoder'] = sel
R['nested_decoder'] = dict(f1=f1s(sel), chosen={int(k): list(v) for k, v in chosen.items()}, boot=B2('nested decoder', 'final'))
best_all = max(grid, key=lambda g: pooled_f1(P[g], okA))
R['grid_best_pooled'] = dict(config=list(best_all), f1=f1s(P[best_all]))
log('nested decoder', R['nested_decoder'], 'grid best', R['grid_best_pooled'])

# ---------------------------------------------------------------- 2. projection fitted on different data (decoder video graph only)
from sklearn.decomposition import PCA
meta_te = pd.read_csv(ROOT/'pack'/'test_meta_data.csv')
q = np.load(ROOT/'pack'/'test_video_q.npy').astype(np.float32) * np.load(ROOT/'pack'/'pca_scale.npy').astype(np.float32)
Et = q.mean(1)
E0 = E.copy()
def with_video(Ealt, name):
    global E
    E = Ealt; VID.clear(); GC.clear()
    p = decode(LP22); E = E0; VID.clear(); GC.clear()
    PRED[name] = p; return p
K = 128
Ea = np.zeros((len(T), K), np.float32)
for k in range(5):
    pca = PCA(K, random_state=0).fit(E0[fold != k]); Ea[fold == k] = pca.transform(E0[fold == k])
with_video(Ea, 'proj train folds')
pca = PCA(K, random_state=0).fit(E0); with_video(pca.transform(E0).astype(np.float32), 'proj all 22')
pca = PCA(K, random_state=0).fit(np.concatenate([E0, Et])); with_video(pca.transform(E0).astype(np.float32), 'proj 22 + test')
R['projection'] = {n: f1s(PRED[n]) for n in ('proj train folds', 'proj all 22', 'proj 22 + test')}
R['projection']['train_folds - 22+test'] = B2('proj train folds', 'proj 22 + test')
R['projection']['22+test - final'] = B2('proj 22 + test', 'final')
log('projection', R['projection'])

# ---------------------------------------------------------------- 3. confidence of window models, OOF vs test
def conf(p): p = p / p.sum(1, keepdims=True); return round(float(p.max(1).mean()), 3)
def ld(k):
    a = np.load(WORK/f'{k}.npy'); return a[ar, LIMB] if a.ndim == 3 else a
pairs = [('video MLP (refit)', 'oof_video_snorm', 'test_pv_v1'), ('video transformer (fold mean)', 'oof_video_deep', 'test_pv_deep'),
         ('IMU MLP (refit)', 'oof_imu_mlp_exact', 'test_pi_v1'), ('IMU boosting (refit)', 'oof_imu_hgb_exact', 'test_pi_hgb'),
         ('IMU CNN (fold mean)', 'oof_imu_deep', 'test_pi_deep')]
R['confidence'] = {n: dict(oof=conf(ld(a)[okA]), test=conf(np.load(WORK/f'{b}.npy'))) for n, a, b in pairs}
R['confidence']['stage-1 fusion'] = dict(oof=conf(np.exp(L1)[okA]), test=conf(np.load(WORK/'test_P1_v15.npy')))
for n, a in [('stage 3 chain A', 'hgb3h'), ('stage 3 chain B', 'hgb3hm'), ('stage 3 chain C', 'hgb3i3f')]:
    R['confidence'][n] = dict(oof=conf(np.load(WORK/f'{a}_oof.npy')[okA]), test=conf(np.load(WORK/f'{a}_pte.npy')))
log('confidence', R['confidence'])

# ---------------------------------------------------------------- 4. per-participant spread
pp = {}
for s_i in np.unique(sbj):
    m = okA & (sbj == s_i); lab = sorted(set(y[m]) | set(base[m]))
    pp[int(s_i)] = f1_score(y[m], base[m], labels=lab, average='macro')
v_all = np.array(list(pp.values())); v_single = np.array([v for s_, v in pp.items() if s_ not in DOUBLE])
R['per_participant_mean_sd'] = dict(all=[round(float(v_all.mean()), 4), round(float(v_all.std(ddof=1)), 4)],
                                    single=[round(float(v_single.mean()), 4), round(float(v_single.std(ddof=1)), 4)])
for nm, pr in [('labels excluded', decode(unary(('hgb3h', 'hgb3hm', 'hgb3i3x'))))]:
    PRED[nm] = pr
    q2 = {}
    for s_i in np.unique(sbj):
        m = okA & (sbj == s_i); lab = sorted(set(y[m]) | set(pr[m])); q2[int(s_i)] = f1_score(y[m], pr[m], labels=lab, average='macro')
    va = np.array(list(q2.values())); vs = np.array([v for s_, v in q2.items() if s_ not in DOUBLE])
    R['per_participant_mean_sd_excluded'] = dict(all=[round(float(va.mean()), 4), round(float(va.std(ddof=1)), 4)], single=[round(float(vs.mean()), 4), round(float(vs.std(ddof=1)), 4)])
    R['excluded_f1'] = f1s(pr)
log('per participant', R['per_participant_mean_sd'], R.get('per_participant_mean_sd_excluded'))
json.dump(R, open('../analysis/protocol_checks.json', 'w'), indent=1, default=str)
log('done')
