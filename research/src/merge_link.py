"""Merge bi-encoder embeddings from several Kaggle runs (seed axis) into work/link9/."""
import sys, zipfile, io; sys.path.insert(0, '.')
from common import *
zips = sys.argv[1:]
out = WORK / 'link9'; (out / 'train').mkdir(parents=True, exist_ok=True)
T = pd.read_pickle(WORK / 'tiles.pkl')
parts_tr = {r: [np.load(WORK / 'link' / 'train' / f'{r}_emb.npy')] for r in T.rec.unique()}
parts_te = [np.load(WORK / 'link' / 'test_emb.npy')]
for zp in zips:
    zf = zipfile.ZipFile(zp)
    inner = [n for n in zf.namelist() if n.endswith('.zip')]
    if inner: zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
    for r in parts_tr: parts_tr[r].append(np.load(io.BytesIO(zf.read(f'train/{r}_emb.npy'))))
    parts_te.append(np.load(io.BytesIO(zf.read('test_emb.npy'))))
t1 = []
for r, ps in parts_tr.items():
    e = np.concatenate(ps, 1); np.save(out / 'train' / f'{r}_emb.npy', e)
    ef = e.astype(np.float32); S = sum(ef[:, k, 0] @ ef[:, k, 1].T for k in range(ef.shape[1])); np.fill_diagonal(S, -1e9)
    t1.append((S[:-1].argmax(1) == np.arange(1, len(S))).mean())
te = np.concatenate(parts_te, 1); np.save(out / 'test_emb.npy', te)
print('seeds', te.shape[1], 'held-out top1 (all seeds)', round(np.mean(t1), 4))
