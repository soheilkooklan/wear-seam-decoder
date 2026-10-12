"""Unpack wear_chainmodel.zip -> work/oof_chainm.npy (tile order) and work/test_chainm.npy."""
import sys, zipfile, io; sys.path.insert(0, '.')
from common import *
zf = zipfile.ZipFile(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else 'chainm'
inner = [n for n in zf.namelist() if n.endswith('.zip')]
if inner: zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
T = pd.read_pickle(WORK/'tiles.pkl'); y = T.y.to_numpy(); oof = np.zeros((len(T), N_CLASSES), np.float32)
for r, g in T.groupby('rec', sort=False):
    a = np.load(io.BytesIO(zf.read(f'train/{r}.npy'))); assert len(a) == len(g); oof[g.index.to_numpy()] = a
np.save(WORK/f'oof_{tag}.npy', oof); np.save(WORK/f'test_{tag}.npy', np.load(io.BytesIO(zf.read('test.npy'))))
print('chain model OOF F1', round(macro_f1(y, oof.argmax(1)), 4))
