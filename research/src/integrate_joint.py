"""Align Kaggle joint-model outputs with the tile table."""
import sys, zipfile, io; sys.path.insert(0, '.')
from common import *
from pathlib import Path
zp = sys.argv[1]
with zipfile.ZipFile(zp) as zf:
    inner = [n for n in zf.namelist() if n.endswith('wear_joint.zip')]
    if inner:
        zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
    arr = {n: np.load(io.BytesIO(zf.read(n))) for n in zf.namelist() if n.endswith('.npy')}
    tiles_k = pd.read_csv(io.BytesIO(zf.read('tiles.csv')))
T = pd.read_pickle(WORK / 'tiles.pkl')
pos = pd.Series(np.arange(len(tiles_k)), index=(tiles_k.rec + '|' + tiles_k.t.astype(str)).values).reindex((T.rec + '|' + T.t.astype(str)).values).to_numpy()
assert not np.isnan(pos.astype(float)).any(); pos = pos.astype(int)
print('label agreement', (tiles_k.y.to_numpy()[pos] == T.y.to_numpy()).mean())
np.save(WORK / 'oof_joint.npy', arr['oof_joint.npy'][pos]); np.save(WORK / 'oof_jvideo.npy', arr['oof_jvideo.npy'][pos])
np.save(WORK / 'test_joint.npy', arr['test_joint.npy']); np.save(WORK / 'test_jvideo.npy', arr['test_jvideo.npy'])
y = T.y.to_numpy(); oj = arr['oof_joint.npy'][pos]
print('joint OOF F1 per limb', [round(macro_f1(y, oj[:, l].argmax(1)), 4) for l in range(4)])
print('joint video-head OOF F1', round(macro_f1(y, arr['oof_jvideo.npy'][pos].argmax(1)), 4))
