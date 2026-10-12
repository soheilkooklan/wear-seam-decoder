"""Unpack wear_imu_link.zip into work/imu_link/ and report held-out top-1 on the simulation limb draw."""
import sys, zipfile, io; sys.path.insert(0, '.')
from common import *
from sim import pick_limbs
out = WORK / 'imu_link'; (out / 'train').mkdir(parents=True, exist_ok=True)
zf = zipfile.ZipFile(sys.argv[1])
inner = [n for n in zf.namelist() if n.endswith('.zip')]
if inner: zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
for n in zf.namelist():
    if n.endswith('.npy'): (out / n).parent.mkdir(parents=True, exist_ok=True); (out / n).write_bytes(zf.read(n))
T = pd.read_pickle(WORK / 'tiles.pkl'); LIMB = pick_limbs(T, 2); t1, same, t1s = [], [], []
for r, g in T.groupby('rec', sort=False):
    idx = g.index.to_numpy(); n = len(idx)
    e = np.load(out / 'train' / f'{r}_emb.npy'); assert len(e) == n
    e = e[np.arange(n), LIMB[idx]].astype(np.float32)
    S = sum(e[:, k, 0] @ e[:, k, 1].T for k in range(e.shape[1])); np.fill_diagonal(S, -1e9)
    ok = S[:-1].argmax(1) == np.arange(1, n); sm = LIMB[idx][:-1] == LIMB[idx][1:]
    t1.append(ok.mean()); same.append(ok[sm].mean()); t1s.append(ok[~sm].mean())
print('IMU bi-encoder held-out top1 (draw 2):', round(np.mean(t1), 4), ' same limb', round(np.mean(same), 4), ' different limb', round(np.mean(t1s), 4))
print('test emb', np.load(out / 'test_emb.npy').shape)
