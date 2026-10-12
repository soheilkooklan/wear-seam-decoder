# Run-to-run variation: chain C retrained with other stage-2 seeds (stage 3 is deterministic).
import json
src = open('../analysis/paper_numbers.py').read()
exec(src.split('# ---------------------------------------------------------------- A.')[0])
res = {}
for tag, st2 in [('hgb3i3f', 'v15f'), ('hgb3i3fs1', 'v15fs1'), ('hgb3i3fs2', 'v15fs2')]:
    p = decode(unary(('hgb3h', 'hgb3hm', tag)))
    PRED[tag] = p
    res[tag] = dict(stage2=f1s(np.load(WORK/f'{st2}_oof_stage2.npy').argmax(1)), stage3=f1s(np.load(WORK/f'{tag}_oof.npy').argmax(1)), final=f1s(p))
    log(tag, res[tag])
fin = np.array([res[t]['final'] for t in ('hgb3i3f', 'hgb3i3fs1', 'hgb3i3fs2')])
res['final_mean'] = fin.mean(0).round(4).tolist(); res['final_sd'] = fin.std(0, ddof=1).round(4).tolist()
res['final_range'] = (fin.max(0) - fin.min(0)).round(4).tolist()
res['s1-s0'] = B2('hgb3i3fs1', 'hgb3i3f'); res['s2-s0'] = B2('hgb3i3fs2', 'hgb3i3f')
res['windows_changed_s1'] = int((PRED['hgb3i3fs1'] != PRED['hgb3i3f']).sum()); res['windows_changed_s2'] = int((PRED['hgb3i3fs2'] != PRED['hgb3i3f']).sum())
log(res)
json.dump(res, open('../analysis/seed_eval.json', 'w'), indent=1)
