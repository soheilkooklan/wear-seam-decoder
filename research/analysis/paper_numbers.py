# All simulation numbers of the report, one protocol.
# Participants who performed activities twice: 0, 2, 3, 14 and 10 (its last, null-labelled session repeats activities).
import sys, pickle, time, json
sys.path.insert(0, '.')
from common import *
from sim import pick_limbs
from decode_lib import knn
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
OUT = {}

T = pd.read_pickle(WORK/'tiles_orig.pkl'); y = T.y.to_numpy(); sbj = T.sbj.to_numpy(); ar = np.arange(len(T))
LIMB = pick_limbs(T, 2)
E = np.load(WORK/'E_mean.npy').astype(np.float32)
FIall = np.load(WORK/'imu_feats_exact.npy', mmap_mode='r')
SC = {k: pickle.load(open(WORK/f'succ_scores_cv_{k}.pkl', 'rb')) for k in ('f', 'h', 'h3', 'hl9', 'hli', 'hliq')}
DOUBLE = (0, 2, 3, 10, 14); single = ~np.isin(sbj, DOUBLE)
FOLD = {1: 0, 6: 0, 11: 0, 16: 0, 21: 0, 0: 1, 5: 1, 10: 1, 15: 1, 20: 1, 4: 2, 9: 2, 14: 2, 19: 2,
        3: 3, 8: 3, 13: 3, 18: 3, 2: 4, 7: 4, 12: 4, 17: 4}
fold = np.array([FOLD[s] for s in sbj])
bad = np.zeros(len(T), bool)
for r, n in [('sbj_2', 112), ('sbj_10', 1419)]:
    bad[T[T.rec == r].sort_values('t').tail(n).index] = True
okA = ~bad; okS = okA & single
RECS = [(r, g.index.to_numpy()) for r, g in T.groupby('rec', sort=False)]
OUT['n_scored'] = [int(okA.sum()), int(okS.sum())]

def L(k): return np.log(np.load(WORK/f'{k}_oof.npy').astype(np.float64) + 1e-6)
def softmax_log(z):
    z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)
L1 = np.log(np.load(WORK/'sim_P1_v15.npy').astype(np.float64) + 1e-6)
def unary(k3s, p1='v15'):
    return 0.6 * np.mean([L(k) for k in k3s], 0) + 0.4 * np.log(np.load(WORK/f'sim_P1_{p1}.npy').astype(np.float64) + 1e-6)
LP22 = unary(('hgb3h', 'hgb3hm', 'hgb3i3f'))
def F(pred, m): return macro_f1(y[m], pred[m])
def f1s(pred): return [round(F(pred, okA), 4), round(F(pred, okS), 4)]

# ---------------------------------------------------------------- graph and decoder
def std(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
def knn_W(x, k, n):
    nb, s = knn(np.nan_to_num(x), min(k, n - 1)); kk = nb.shape[1]
    return csr_matrix((np.exp(5 * (s.ravel() - 1)), (np.repeat(np.arange(n), kk), nb.ravel())), shape=(n, n))
VID, IMU = {}, {}
def vid_W(rec, idx, pc=0):
    if (rec, pc) not in VID:
        e = std(E[idx])
        if pc:
            _, _, vt = np.linalg.svd(e, full_matrices=False); e = e - (e @ vt[:pc].T) @ vt[:pc]
        VID[(rec, pc)] = knn_W(e, 10, len(idx))
    return VID[(rec, pc)]
def imu_W(rec, idx, limb, key):
    if (rec, key) not in IMU:
        n = len(idx); W = csr_matrix((n, n)); l = limb[idx]
        Fm = np.stack([FIall[i, l_] for i, l_ in zip(idx, l)]).astype(np.float32)
        for lb in range(4):
            m = np.flatnonzero(l == lb)
            if len(m) <= 11: continue
            Wm = knn_W(std(Fm[m]), 10, len(m)).tocoo()
            W = W + csr_matrix((Wm.data, (m[Wm.row], m[Wm.col])), shape=(n, n))
        IMU[(rec, key)] = W.tocsr()
    return IMU[(rec, key)]
def link_W(rec, n, links, mode, perfect_idx=None):
    if links == 'true':
        g = T.loc[perfect_idx].sort_values('t'); pos = pd.Series(np.arange(n), index=perfect_idx)
        o = pos[g.index].to_numpy(); return csr_matrix((np.ones(n - 1), (o[:-1], o[1:])), shape=(n, n))
    pairs, sc = SC[links][rec]; a, b = pairs[:, 0], pairs[:, 1]; r = sc.astype(np.float64)
    keep = np.ones(len(r), bool)
    if mode.startswith('top'):
        k = int(mode[3:]); d = pd.DataFrame({'a': a, 'r': r}); keep = (d.groupby('a').r.rank(ascending=False, method='first') <= k).to_numpy()
    elif mode == 'mutual':
        d = pd.DataFrame({'a': a, 'b': b, 'r': r})
        keep = (r >= d.groupby('a').r.transform('max').to_numpy()) & (r >= d.groupby('b').r.transform('max').to_numpy())
    elif mode.startswith('thr'):
        keep = r > float(mode[3:])
    return csr_matrix((r[keep] ** 2, (a[keep], b[keep])), shape=(n, n))
def sym(W, how): return (W + W.T) if how == 'sum' else W.maximum(W.T)
GC = {}
def graph(rec, idx, links='hli', lw=10, kw=1.0, iw=1.0, how='sum', lmode='all', limb=None, lkey=2, pc=0):
    limb = LIMB if limb is None else limb
    key = (rec, links, lw, kw, iw, how, lmode, lkey, pc)
    if key not in GC:
        if len(GC) > 150: GC.clear()
        n = len(idx); A = csr_matrix((n, n))
        if kw > 0: A = A + kw * sym(vid_W(rec, idx, pc), how)
        if iw > 0: A = A + iw * sym(imu_W(rec, idx, limb, lkey), how)
        if lw > 0: A = A + lw * sym(link_W(rec, n, links, lmode, idx), how)
        GC[key] = A.tocsr()
    return GC[key]
def mean_field(lp, A, beta, lo, hi, iters=30, step=0.15, hist=False):
    C = lp.shape[1]; lo = np.broadcast_to(lo, C); hi = np.broadcast_to(hi, C)
    b = np.zeros(C); deg = np.asarray(A.sum(1)).ravel() + 1e-9
    q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True); H = []
    for it in range(iters):
        msg = (A @ q) / deg[:, None]; z = lp + b + beta * np.log(msg + 1e-3)
        q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        if it >= 5:
            cnt = np.bincount(q.argmax(1), minlength=C)
            for c in range(1, C):
                if cnt[c] < lo[c]: b[c] += step
                elif cnt[c] > hi[c]: b[c] -= step
        if hist: H.append(b.copy())
    return (q.argmax(1), np.array(H)) if hist else q.argmax(1)
def decode(LP, lo=75, hi=95, beta=1.5, iters=30, step=0.15, null_off=0.0, extra=None, w_extra=0.0, hist=False, lo_c=None, hi_c=None, **gk):
    pred = np.zeros(len(T), int); HB = {}
    for rec, idx in RECS:
        lp = LP[idx].copy()
        if null_off: lp[:, 0] -= null_off
        if extra is not None and w_extra: lp = lp + w_extra * extra[idx]
        A = graph(rec, idx, **gk); s = subject_of(rec)
        if lo_c is not None:
            l_, h_ = lo_c[FOLD[s]], hi_c[FOLD[s]]
            if s in DOUBLE: l_, h_ = np.minimum(l_, 80), np.full(19, 125.0)
        else:
            dbl = s in DOUBLE and np.max(hi) < 1e8
            l_ = np.minimum(lo, 80) if dbl else lo; h_ = 125 if dbl else hi
        r_ = mean_field(lp, A, beta, l_, h_, iters, step, hist)
        if hist: pred[idx], HB[rec] = r_
        else: pred[idx] = r_
    return (pred, HB) if hist else pred

# ---------------------------------------------------------------- bootstrap over participants
def cm_by_sbj(pred, mask):
    D = {}
    for s in np.unique(sbj):
        m = (sbj == s) & mask
        if m.sum(): D[s] = np.bincount(y[m] * 19 + pred[m], minlength=361).reshape(19, 19)
    return D
def f1_cm(C, present=False):
    tp = np.diag(C); fp = C.sum(0) - tp; fn = C.sum(1) - tp; den = 2 * tp + fp + fn
    f = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0)
    return f[den > 0].mean() if present else f.mean()
PRED = {}
def boot(a, b, mask, B=2000):
    A = cm_by_sbj(PRED[a], mask); Bc = cm_by_sbj(PRED[b], mask); S = sorted(A); rng = np.random.default_rng(0)
    d0 = f1_cm(sum(A[s] for s in S)) - f1_cm(sum(Bc[s] for s in S)); ds = []
    for _ in range(B):
        ss = rng.choice(S, len(S)); ds.append(f1_cm(sum(A[s] for s in ss)) - f1_cm(sum(Bc[s] for s in ss)))
    wins = sum(f1_cm(A[s], True) > f1_cm(Bc[s], True) for s in S)
    return dict(diff=round(float(d0), 4), ci=[round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)], wins=f'{wins}/{len(S)}')
def B2(a, b): return {'all': boot(a, b, okA), 'single': boot(a, b, okS)}
def run(name, pred):
    PRED[name] = pred; OUT.setdefault('f1', {})[name] = f1s(pred); log(name, OUT['f1'][name]); return pred

# ---------------------------------------------------------------- A. window-level components
comp = {}
def ld(k):
    a = np.load(WORK/f'{k}.npy'); return a[ar, LIMB] if a.ndim == 3 else a
for nm, k in [('video MLP', 'oof_video_snorm'), ('video transformer', 'oof_video_deep'), ('IMU MLP', 'oof_imu_mlp_exact'),
              ('IMU boosting', 'oof_imu_hgb_exact'), ('IMU CNN', 'oof_imu_deep'), ('joint 1', 'oof_joint'), ('joint 2', 'oof_joint2'),
              ('joint mean', 'oof_jointavg'), ('sequence model', 'oof_chainm')]:
    comp[nm] = round(F(ld(k).argmax(1), okA), 4)
for nm in ('v11', 'v14', 'v15f', 'v15x', 'v15q'):
    comp[f'stage2 {nm}'] = round(F(np.load(WORK/f'{nm}_oof_stage2.npy').argmax(1), okA), 4)
for nm in ('hgb3h', 'hgb3hm', 'hgb3i3f', 'hgb3i3', 'hgb3i3x', 'hgb3i3q'):
    comp[nm] = round(F(np.load(WORK/f'{nm}_oof.npy').argmax(1), okA), 4)
OUT['components'] = comp; log('components', comp)

# ---------------------------------------------------------------- B. build-up, oracles
base = run('final', decode(LP22))
run('stage-1 fusion argmax', L1.argmax(1)); run('stacked argmax', LP22.argmax(1))
run('p1 + decoder', decode(L1)); run('no count constraint', decode(LP22, lo=0, hi=10 ** 9))
run('no links', decode(LP22, lw=0)); run('no video kNN', decode(LP22, kw=0)); run('no IMU kNN', decode(LP22, iw=0))
run('true links', decode(LP22, links='true'))
GA = np.array([4, 5, 8, 9, 10, 15, 16, 17, 18]); grp = np.zeros(len(T), int)
for rec, g in T.groupby('rec', sort=False):
    s_ = g.sort_values('t'); idx = s_.index.to_numpy(); yy = y[idx]; act = np.flatnonzero(yy > 0)
    near = act[np.abs(np.arange(len(idx))[:, None] - act[None]).argmin(1)]; grp[idx] = np.where(np.isin(yy[near], GA), 1, 2)
Msk = np.ones((len(T), 19))
for c in range(1, 19): Msk[:, c] = np.where(np.isin(c, GA), grp == 1, grp == 2)
run('session oracle', decode(LP22 + np.log(np.maximum(Msk, 1e-3))))
def viterbi(lp, pen):
    n, C = lp.shape; d = lp[0].copy(); bp = np.zeros((n, C), int)
    for t in range(1, n):
        bi = d.argmax(); sw = d[bi] - pen; bp[t] = np.where(d >= sw, np.arange(C), bi); d = np.maximum(d, sw) + lp[t]
    o = np.zeros(n, int); o[-1] = d.argmax()
    for t in range(n - 1, 0, -1): o[t - 1] = bp[t, o[t]]
    return o
for pen in (6, 10):
    p = np.zeros(len(T), int)
    for rec, g in T.groupby('rec', sort=False):
        o = g.sort_values('t').index.to_numpy(); p[o] = viterbi(LP22[o], pen)
    run(f'viterbi true order {pen}', p)
D2 = T.sort_values(['rec', 't']); seg2 = ((D2.y != D2.y.shift()) | (D2.rec != D2.rec.shift())).cumsum().to_numpy(); idx2 = D2.index.to_numpy()
for nm, LPx in [('segment oracle final', LP22), ('segment oracle stage1', L1)]:
    p = np.zeros(len(T), int)
    for sid in np.unique(seg2):
        ii = idx2[seg2 == sid]; p[ii] = LPx[ii].mean(0).argmax()
    run(nm, p)

# ---------------------------------------------------------------- C. links table
def top1(S):
    hit = {True: [0, 0], False: [0, 0]}
    for rec, g in T.groupby('rec', sort=False):
        idx = g.index.to_numpy(); p, s = S[rec]; n = len(idx)
        o = g.sort_values('t').index.to_numpy(); pos = pd.Series(np.arange(n), index=idx)
        nxt = np.full(n, -1); nxt[pos[o[:-1]].to_numpy()] = pos[o[1:]].to_numpy()
        d = pd.DataFrame({'a': p[:, 0], 'b': p[:, 1], 's': s}).sort_values('s').groupby('a').tail(1)
        best = np.full(n, -1); best[d.a.to_numpy()] = d.b.to_numpy()
        for a in np.flatnonzero(nxt >= 0):
            same = bool(LIMB[idx[a]] == LIMB[idx[nxt[a]]]); hit[same][0] += 1; hit[same][1] += best[a] == nxt[a]
    return [round((hit[True][1] + hit[False][1]) / (hit[True][0] + hit[False][0]), 3), round(hit[True][1] / hit[True][0], 3), round(hit[False][1] / hit[False][0], 3)]
OUT['links'] = {}
for k in SC:
    OUT['links'][k] = top1(SC[k]); run(f'links {k}', decode(LP22, links=k))
run('seam rebuilt stacking', decode(unary(('hgb3h', 'hgb3hm', 'hgb3i3q')), links='hliq'))

# ---------------------------------------------------------------- D. submissions
P14 = unary(('hgb3h', 'hgb3hm', 'hgb3hl'), 'v14'); P15 = unary(('hgb3h', 'hgb3hm', 'hgb3i3'))
run('sub base 70/125/2', decode(P14, lo=70, hi=125, beta=2.0)); run('sub 80/125/2', decode(P15, lo=80, hi=125, beta=2.0))
run('sub 80/100/1.5', decode(P15, lo=80, hi=100)); run('sub seam 80/100', decode(P15, lo=80, hi=100, links='hliq'))
run('sub 80/105', decode(P15, lo=80, hi=105)); run('sub 80/95', decode(P15, lo=80, hi=95)); run('sub 80/92', decode(P15, lo=80, hi=92))
run('sub 75/95', decode(P15)); OUT['f1']['sub pseudo 75/95'] = OUT['f1']['final']

# ---------------------------------------------------------------- E. graph variants, null offset, PCs, IMU weight, kNN prior
run('max symmetrisation', decode(LP22, how='max'))
for m in ('top1', 'top3', 'mutual', 'thr0.1'): run(f'links {m}', decode(LP22, lmode=m))
for d in (-0.25, 0.25, 0.5, 1.0): run(f'null offset {d}', decode(LP22, null_off=d))
for pc in (10, 30, 50): run(f'pc removal {pc}', decode(LP22, pc=pc))
IM = np.mean([np.log(ld(k).astype(np.float64) + 1e-6) for k in ('oof_imu_hgb_exact', 'oof_imu_mlp_exact', 'oof_imu_deep')], 0)
for w in (0.1, 0.2): run(f'imu weight {w}', decode(LP22, extra=IM, w_extra=w))
Es = np.zeros_like(E)
for s_ in np.unique(sbj): m = sbj == s_; Es[m] = std(E[m])
KV = np.zeros((len(T), 19))
for f_ in range(5):
    tr = fold != f_; pca = PCA(128, random_state=0).fit(Es[tr]); Z = pca.transform(Es)
    tri = np.flatnonzero(tr & okA); te = np.flatnonzero(fold == f_)
    Zt = Z[tri] / (np.linalg.norm(Z[tri], axis=1, keepdims=True) + 1e-8)
    for c0 in range(0, len(te), 500):
        q = Z[te[c0:c0 + 500]]; q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        s = q @ Zt.T; nb = np.argpartition(-s, 20, axis=1)[:, :20]; w = np.exp(5 * (np.take_along_axis(s, nb, 1) - 1))
        Hh = np.zeros((len(q), 19)); np.add.at(Hh, (np.repeat(np.arange(len(q)), 20), y[tri][nb].ravel()), w.ravel()); KV[te[c0:c0 + 500]] = Hh
KV = np.log((KV + 0.05) / (KV + 0.05).sum(1, keepdims=True))
run('kNN prior alone', KV.argmax(1))
for w in (0.1, 0.25, 0.5): run(f'kNN prior {w}', decode(LP22, extra=KV, w_extra=w))

# ---------------------------------------------------------------- F. count constraint
cnt = T[single & okA].groupby(['sbj', 'y']).size().unstack(fill_value=0)
OUT['class_counts_q10_q50_q90'] = {int(c): [int(cnt[c].quantile(q)) for q in (0.1, 0.5, 0.9)] for c in range(1, 19)}
for qlo, qhi in [(0.1, 0.9), (0.25, 0.75)]:
    lo_c, hi_c = {}, {}
    for f_ in range(5):
        trs = [s for s in cnt.index if FOLD[s] != f_]
        a_ = np.zeros(19); b_ = np.full(19, 1e9)
        a_[1:] = [cnt.loc[trs, c].quantile(qlo) for c in range(1, 19)]; b_[1:] = [cnt.loc[trs, c].quantile(qhi) for c in range(1, 19)]
        lo_c[f_], hi_c[f_] = a_, b_
    run(f'class bounds q{qlo}-q{qhi}', decode(LP22, lo_c=lo_c, hi_c=hi_c))
nob = PRED['no count constraint']; ch = (nob != base) & okA
OUT['count_changes'] = dict(changed=int(ch.sum()), now_correct=int((ch & (base == y)).sum()), now_wrong=int((ch & (nob == y)).sum()),
                            both_wrong=int((ch & (base != y) & (nob != y)).sum()))
_, HB = decode(LP22, hist=True)
inb, osc = [], []
cerr_b, cerr_n = np.zeros(18), np.zeros(18); nrec = 0
for rec, idx in RECS:
    if subject_of(rec) in DOUBLE: continue
    m = okA[idx]
    c_ = np.bincount(base[idx], minlength=19)[1:]; inb.append(((c_ >= 75) & (c_ <= 95)).mean())
    d = np.diff(HB[rec], axis=0)[:, 1:]; sg = np.sign(d)
    osc.append(np.mean([(np.diff(sg[:, c][sg[:, c] != 0]) != 0).sum() >= 2 for c in range(18)]))
    tc = np.bincount(y[idx][m], minlength=19)[1:]
    cerr_b += np.abs(np.bincount(base[idx][m], minlength=19)[1:] - tc); cerr_n += np.abs(np.bincount(nob[idx][m], minlength=19)[1:] - tc); nrec += 1
OUT['within_bounds'] = round(float(np.mean(inb)), 3); OUT['oscillating'] = round(float(np.mean(osc)), 3)
OUT['count_err_with'] = round(float(cerr_b.sum() / (18 * nrec)), 1); OUT['count_err_without'] = round(float(cerr_n.sum() / (18 * nrec)), 1)
OUT['count_err_per_class_with'] = [round(float(v / nrec), 1) for v in cerr_b]; OUT['count_err_per_class_without'] = [round(float(v / nrec), 1) for v in cerr_n]
log('count', OUT['count_changes'], OUT['within_bounds'], OUT['oscillating'], OUT['count_err_with'], OUT['count_err_without'])
SENS = {}
for lo in (70, 75, 80, 85):
    for hi in (90, 95, 100, 105, 110, 125): SENS[f'l{lo}_h{hi}'] = f1s(decode(LP22, lo=lo, hi=hi))
for beta in (1.0, 1.25, 1.75, 2.0, 2.5): SENS[f'beta{beta}'] = f1s(decode(LP22, beta=beta))
for step, iters in [(0.1, 20), (0.1, 30), (0.1, 40), (0.15, 20), (0.15, 40), (0.2, 20), (0.2, 30), (0.2, 40)]:
    SENS[f'step{step}_it{iters}'] = f1s(decode(LP22, step=step, iters=iters))
OUT['sensitivity'] = SENS; log('sensitivity done')
run('h=100', decode(LP22, hi=100)); run('l80 h95', decode(LP22, lo=80))

# ---------------------------------------------------------------- G. label regimes
run('labels original', decode(unary(('hgb3h', 'hgb3hm', 'hgb3i3')))); run('labels excluded', decode(unary(('hgb3h', 'hgb3hm', 'hgb3i3x'))))
PRED['labels pseudo'] = base

# ---------------------------------------------------------------- H. errors, classes, participants
D = T.copy(); D['p'] = base; D = D[okA].sort_values(['rec', 't'])
D['seg'] = ((D.y != D.y.shift()) | (D.rec != D.rec.shift())).cumsum()
D['pos'] = D.groupby('seg').cumcount(); D['n'] = D.groupby('seg').y.transform('size'); D['dist'] = np.minimum(D.pos, D.n - 1 - D.pos)
sg_ = D.groupby('seg').agg(rec=('rec', 'first'), y=('y', 'first')).reset_index()
sg_['prev'] = sg_.groupby('rec').y.shift(1); sg_['next'] = sg_.groupby('rec').y.shift(-1)
sg_['pause'] = (sg_.y == 0) & (sg_.prev == sg_.next) & (sg_.prev > 0)
D = D.merge(sg_[['seg', 'pause']], on='seg', how='left'); e = D[D.p != D.y]
OUT['errors'] = dict(total=int(len(e)), of=int(len(D)), act_to_null=int(((e.y > 0) & (e.p == 0)).sum()), null_to_act=int((e.y == 0).sum()),
                     act_to_act=int(((e.y > 0) & (e.p > 0)).sum()), boundary2=int((e.dist <= 2).sum()), dist0=int((e.dist == 0).sum()),
                     dist10=int((e.dist >= 10).sum()), inside_act=int(((e.dist > 2) & (e.y > 0)).sum()),
                     inside_pause=int(((e.dist > 2) & (e.y == 0) & e.pause).sum()), inside_null=int(((e.dist > 2) & (e.y == 0) & ~e.pause).sum()),
                     pause_windows=int(D.pause.sum()), pause_null=round(float((D[D.pause].p == 0).mean()), 3))
log('errors', OUT['errors'])
p_, r_, f_, s_ = precision_recall_fscore_support(y[okA], base[okA], labels=list(range(19)), zero_division=0)
rows = []
for c in range(19):
    miss = okA & (y == c) & (base != c)
    rows.append(dict(c=c, P=round(float(p_[c]), 3), R=round(float(r_[c]), 3), F=round(float(f_[c]), 3), n=int(s_[c]),
                     q=OUT['class_counts_q10_q50_q90'].get(c), gain=int((ch & (y == c) & (base == y)).sum()), loss=int((ch & (y == c) & (nob == y)).sum()),
                     miss_to_null=round(float((base[miss] == 0).mean()), 2) if c else None,
                     top_confusion=(pd.Series(base[miss]).value_counts().head(2).to_dict() if c else None),
                     count_err=OUT['count_err_per_class_with'][c - 1] if c else None))
OUT['classes'] = rows
ps = {}
for s_i in np.unique(sbj):
    m = okA & (sbj == s_i); lab = sorted(set(y[m]) | set(base[m]))
    ps[int(s_i)] = round(float(f1_score_ := __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(y[m], base[m], labels=lab, average='macro')), 3)
OUT['per_participant'] = ps
OUT['pooled_0_10'] = round(F(base, okA & (sbj <= 10)), 4); OUT['pooled_11_21'] = round(F(base, okA & (sbj > 10)), 4)
log('participants', min(ps.values()), max(ps.values()), OUT['pooled_0_10'], OUT['pooled_11_21'])

# ---------------------------------------------------------------- I. limb draws
ld_ = {}
for d in (0, 1, 2):
    Ld = pick_limbs(T, d)
    def comp_d(k):
        a = np.load(WORK/f'{k}.npy'); a = a[ar, Ld] if a.ndim == 3 else a
        return np.log(a.astype(np.float64) + 1e-6)
    Lp1 = 0.2 * comp_d('oof_video_snorm') + 0.2 * comp_d('oof_video_deep') + 0.2 * comp_d('oof_imu_mlp_exact') + 0.3 * comp_d('oof_imu_hgb_exact') + 0.1 * comp_d('oof_imu_deep')
    ld_[d] = dict(window=f1s(Lp1.argmax(1)), no_count=f1s(decode(Lp1, lw=0, limb=Ld, lkey=d, lo=0, hi=10 ** 9)), count=f1s(decode(Lp1, lw=0, limb=Ld, lkey=d)))
    log('limb draw', d, ld_[d])
OUT['limb_draws'] = ld_

# ---------------------------------------------------------------- J. nested check, decoded
Pp = np.load(WORK/'nested_P2_plain.npy').astype(np.float64); Pn = np.load(WORK/'nested_P2_nested.npy').astype(np.float64)
run('nested plain argmax', Pp.argmax(1)); run('nested nested argmax', Pn.argmax(1))
run('nested plain decoded', decode(np.log(Pp + 1e-6), lw=0)); run('nested nested decoded', decode(np.log(Pn + 1e-6), lw=0))

# ---------------------------------------------------------------- K. bootstrap
BOOT = {}
for a, b in [('final', 'no count constraint'), ('final', 'p1 + decoder'), ('final', 'stacked argmax'), ('final', 'no links'),
             ('final', 'no video kNN'), ('final', 'no IMU kNN'), ('links hliq', 'final'), ('seam rebuilt stacking', 'final'),
             ('links f', 'final'), ('max symmetrisation', 'final'), ('links top1', 'final'), ('links top3', 'final'), ('links mutual', 'final'),
             ('links thr0.1', 'final'), ('null offset 0.25', 'final'), ('null offset 0.5', 'final'), ('class bounds q0.1-q0.9', 'final'),
             ('class bounds q0.25-q0.75', 'final'), ('kNN prior 0.1', 'final'), ('kNN prior 0.25', 'final'), ('pc removal 30', 'final'),
             ('imu weight 0.1', 'final'), ('final', 'h=100'), ('l80 h95', 'final'), ('labels excluded', 'labels original'),
             ('labels pseudo', 'labels original'), ('labels pseudo', 'labels excluded'), ('nested plain argmax', 'nested nested argmax'),
             ('nested plain decoded', 'nested nested decoded')]:
    BOOT[f'{a} - {b}'] = B2(a, b); log('boot', a, '-', b, BOOT[f'{a} - {b}'])
OUT['bootstrap'] = BOOT
json.dump(OUT, open('../analysis/paper_numbers.json', 'w'), indent=1, default=str)
np.save(WORK/'sim_pred_final_check.npy', base)
log('done')
