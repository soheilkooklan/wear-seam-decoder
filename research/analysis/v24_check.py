# Simulation score of submission v24 (chain C with spans left out, bounds 85/100) and grid best for that chain.
import json
src = open('../analysis/paper_numbers.py').read()
exec(src.split('# ---------------------------------------------------------------- A.')[0])
base = np.load(WORK/'sim_pred_final_check.npy'); PRED['final'] = base
L3x = np.mean([L('hgb3h'), L('hgb3hm'), L('hgb3i3x')], 0)
grid = [(a, lo, hi, be) for a in (0.5, 0.6, 0.7) for lo in (70, 75, 80, 85) for hi in (90, 95, 100, 105, 110) for be in (1.25, 1.5, 1.75, 2.0)]
P = {g: decode(g[0] * L3x + (1 - g[0]) * L1, lo=g[1], hi=g[2], beta=g[3]) for g in grid}
PRED['v24'] = P[(0.6, 85, 100, 1.5)]
best = max(grid, key=lambda g: macro_f1(y[okA], P[g][okA]))
R = dict(v24=f1s(PRED['v24']), v24_vs_final=B2('v24', 'final'), grid_best=list(best), grid_best_f1=f1s(P[best]))
log(R)
json.dump(R, open('../analysis/v24_check.json', 'w'), indent=1, default=str)
