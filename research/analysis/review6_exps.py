# Checks asked for in the sixth review round: bounds per class from training counts,
# bounds scaled per participant, and chain C averaged over its three stage-2 seeds.
import json
src = open('../analysis/paper_numbers.py').read()
exec(src.split('# ---------------------------------------------------------------- A.')[0])
base = np.load(WORK/'sim_pred_final_check.npy'); PRED['final'] = base
R = {'final': f1s(base)}

# true counts per single-performance participant and class
cnt = {}
for s in np.unique(sbj[okS]):
    m = (sbj == s) & okS
    cnt[s] = np.bincount(y[m], minlength=19)
for qlo, qhi in [(0.1, 0.9), (0.25, 0.75)]:
    lo_c, hi_c = {}, {}
    for k in range(5):
        C = np.array([cnt[s] for s in cnt if FOLD[s] != k])
        lo_c[k] = np.quantile(C, qlo, axis=0); hi_c[k] = np.quantile(C, qhi, axis=0)
    name = f'class bounds q{qlo}-q{qhi}'
    PRED[name] = decode(LP22, lo_c=lo_c, hi_c=hi_c)
    R[name] = dict(f1=f1s(PRED[name]), boot=B2(name, 'final')); log(name, R[name])

# bounds scaled by the number of windows the unary term does not call null
def decode_scaled(LP, a, b, beta=1.5):
    pred = np.zeros(len(T), int)
    for rec, idx in RECS:
        lp = LP[idx]; A = graph(rec, idx); s = subject_of(rec)
        m = (lp.argmax(1) != 0).sum() / 18.0
        lo, hi = a * m, b * m
        if s in DOUBLE: lo, hi = min(lo, 80), 125
        pred[idx] = mean_field(lp, A, beta, lo, hi)
    return pred
for a, b in [(0.8, 1.0), (0.85, 1.05), (0.9, 1.1)]:
    name = f'scaled bounds {a}-{b}'
    PRED[name] = decode_scaled(LP22, a, b)
    R[name] = dict(f1=f1s(PRED[name]), boot=B2(name, 'final')); log(name, R[name])

# chain C averaged over its three seeds
LC = np.mean([L('hgb3i3f'), L('hgb3i3fs1'), L('hgb3i3fs2')], 0)
LPs = 0.6 * np.mean([L('hgb3h'), L('hgb3hm'), LC], 0) + 0.4 * L1
PRED['seed average'] = decode(LPs)
R['seed average'] = dict(f1=f1s(PRED['seed average']), boot=B2('seed average', 'final')); log('seed average', R['seed average'])
json.dump(R, open('../analysis/review6_exps.json', 'w'), indent=1, default=str)
