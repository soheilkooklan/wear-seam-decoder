# Most conservative variant: spans left out of training and decoder settings chosen inside the folds.
import json
src = open('../analysis/paper_numbers.py').read()
exec(src.split('# ---------------------------------------------------------------- A.')[0])
base = np.load(WORK/'sim_pred_final_check.npy'); PRED['final'] = base
L3x = np.mean([L('hgb3h'), L('hgb3hm'), L('hgb3i3x')], 0)
grid = [(a, lo, hi, be) for a in (0.5, 0.6, 0.7) for lo in (70, 75, 80, 85) for hi in (90, 95, 100, 105, 110) for be in (1.25, 1.5, 1.75, 2.0)]
P = {g: decode(g[0] * L3x + (1 - g[0]) * L1, lo=g[1], hi=g[2], beta=g[3]) for g in grid}
sel = np.zeros(len(T), int); chosen = {}
for k in range(5):
    tr = okA & (fold != k)
    best = max(grid, key=lambda g: macro_f1(y[tr], P[g][tr])); chosen[k] = best; sel[fold == k] = P[best][fold == k]
PRED['conservative'] = sel
PRED['excluded'] = P[(0.6, 75, 95, 1.5)]
R = dict(f1=f1s(sel), chosen={int(k): list(v) for k, v in chosen.items()}, boot_vs_final=B2('conservative', 'final'), excluded_default=f1s(PRED['excluded']))
log(R)
json.dump(R, open('../analysis/protocol_checks2.json', 'w'), indent=1, default=str)
