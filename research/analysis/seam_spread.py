# Seam distances: spread over participants and activities, plus simpler baselines.
import sys, json
sys.path.insert(0, '.')
from common import *
from sim import pick_limbs

T = pd.read_pickle(WORK/'tiles_orig.pkl'); LIMB = pick_limbs(T, 2)

def ar_fit(X, p):
    L = X.shape[1]; A = np.stack([X[:, i:L - p + i] for i in range(p)], 2)
    AtA = np.einsum('mtp,mtq->mpq', A, A) + 1e-3 * np.eye(p); Aty = np.einsum('mtp,mt->mp', A, X[:, p:])
    return np.linalg.solve(AtA, Aty[..., None])[..., 0]
def ar_pred(X, c, k):
    p = c.shape[1]; buf = X[:, -p:].copy(); out = []
    for _ in range(k):
        nx = (buf * c).sum(1); out.append(nx); buf = np.concatenate([buf[:, 1:], nx[:, None]], 1)
    return np.stack(out, 1)
def dist(P, Q): return np.sqrt(((P ** 2).sum(1)[:, None] + (Q ** 2).sum(1)[None] - 2 * P @ Q.T).clip(0))

def measures(W):
    k = len(W)
    va = W[:, -1] - W[:, -2]; aa = W[:, -1] - 2 * W[:, -2] + W[:, -3]
    vb = W[:, 1] - W[:, 0]; ab = W[:, 2] - 2 * W[:, 1] + W[:, 0]
    ja = W[:, -1] - 3 * W[:, -2] + 3 * W[:, -3] - W[:, -4]; jb = W[:, 3] - 3 * W[:, 2] + 3 * W[:, 1] - W[:, 0]
    qa = W[:, -1] + 0.5 * va + 0.375 * aa; qb = W[:, 0] - 0.5 * vb + 0.375 * ab
    sc = np.median(np.abs(np.diff(W, 2, axis=1)), axis=(1, 2)) + 1e-3
    D = {}
    D['endpoint'] = dist(W[:, -1], W[:, 0])
    D['end-velocity difference'] = dist(va, vb)
    D['end-acceleration difference'] = dist(aa, ab)
    D['linear, from a only'] = dist(2 * W[:, -1] - W[:, -2], W[:, 0])
    D['linear, both sides'] = dist(W[:, -1] + 0.5 * va, W[:, 0] - 0.5 * vb)
    D['quadratic, both sides'] = dist(qa, qb)
    D['quadratic, both sides, normalised'] = dist(qa, qb) / (0.5 * (sc[:, None] + sc[None]))
    D['cubic, both sides'] = dist(qa + 0.3125 * ja, qb - 0.3125 * jb)
    Xa = W.transpose(0, 2, 1).reshape(k * 3, 50); pa = ar_pred(Xa, ar_fit(Xa, 6), 4).reshape(k, 3, 4).transpose(0, 2, 1)
    Xb = W[:, ::-1].transpose(0, 2, 1).reshape(k * 3, 50); pb = ar_pred(Xb, ar_fit(Xb, 6), 4).reshape(k, 3, 4).transpose(0, 2, 1)
    D['AR(6), both sides'] = np.sqrt(((pa[:, None] - W[None, :, :4]) ** 2).sum((2, 3))) + np.sqrt(((pb[None, :] - W[:, None, ::-1][:, :, :4]) ** 2).sum((2, 3)))
    return D

rows = []
for r, g in T.groupby('rec', sort=False):
    idx = g.index.to_numpy(); n = len(idx); cube = np.nan_to_num(np.asarray(load_cube(r), np.float32))
    l = LIMB[idx]; yy = g.y.to_numpy()
    W = cube[np.arange(n)[:, None] * 50 + np.arange(50)[None]][np.arange(n)[:, None], np.arange(50)[None], l[:, None]].astype(np.float64)
    for lb in range(4):
        m = np.flatnonzero(l == lb)
        if len(m) < 20: continue
        pos = {v: i for i, v in enumerate(m)}
        pr = [(i, pos[v + 1]) for i, v in enumerate(m) if v + 1 in pos]
        if not pr: continue
        ai = np.array([a for a, _ in pr]); bi = np.array([b for _, b in pr])
        res = {}
        for name, D in measures(W[m]).items():
            D = D.copy(); np.fill_diagonal(D, np.inf)
            res[name] = (D[ai] < D[ai, bi][:, None]).sum(1) + 1
        for j in range(len(ai)):
            rows.append(dict(sbj=subject_of(r), y_a=int(yy[m[ai[j]]]), y_b=int(yy[m[bi[j]]]), ncand=len(m) - 1,
                             **{k: int(v[j]) for k, v in res.items()}))
R = pd.DataFrame(rows); names = [c for c in R.columns if c not in ('sbj', 'y_a', 'y_b', 'ncand')]
out = {'anchors': len(R), 'median_candidates': int(R.ncand.median())}
rng = np.random.default_rng(0); S = R.sbj.unique()
for nm in names:
    t1 = (R[nm] == 1); per = t1.groupby(R.sbj).mean()
    boots = []
    for _ in range(2000):
        ss = rng.choice(S, len(S)); boots.append(np.concatenate([t1[R.sbj == s].to_numpy() for s in ss]).mean())
    out[nm] = dict(top1=round(float(t1.mean()), 3), top5=round(float((R[nm] <= 5).mean()), 3),
                   ci=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                   per_participant_min=round(float(per.min()), 3), per_participant_max=round(float(per.max()), 3))
q = (R['quadratic, both sides'] == 1)
out['quadratic_same_label'] = round(float(q[R.y_a == R.y_b].mean()), 3)
out['quadratic_label_change'] = round(float(q[R.y_a != R.y_b].mean()), 3); out['n_label_change'] = int((R.y_a != R.y_b).sum())
out['quadratic_null_anchor'] = round(float(q[R.y_a == 0].mean()), 3)
out['quadratic_activity_anchor'] = round(float(q[R.y_a > 0].mean()), 3)
out['quadratic_per_activity'] = {int(c): round(float(q[R.y_a == c].mean()), 3) for c in range(19)}
lin = (R['linear, from a only'] == 1)
per_q = q.groupby(R.sbj).mean(); per_l = lin.groupby(R.sbj).mean()
out['quadratic_beats_linear_in_participants'] = f'{int((per_q > per_l).sum())}/{len(per_q)}'
json.dump(out, open('../analysis/seam_spread.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
