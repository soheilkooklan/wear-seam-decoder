"""Align the Kaggle deep-model outputs with the offline tile table and report OOF scores."""
import sys, zipfile, io; sys.path.insert(0, '.')
from common import *
from pathlib import Path
up = sorted((WORK / 'deep').glob('wear_deep*.zip'))
if not up:
    # the zip may be nested inside a Kaggle 'results' archive
    for z in sorted(Path('/mnt/user-data/uploads').rglob('*.zip')):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith('wear_deep.zip')]
            if names:
                (WORK / 'deep').mkdir(exist_ok=True)
                open(WORK / 'deep' / 'wear_deep.zip', 'wb').write(zf.read(names[0])); up = [WORK / 'deep' / 'wear_deep.zip']
assert up, 'wear_deep.zip not found'
with zipfile.ZipFile(up[-1]) as zf:
    arr = {n: np.load(io.BytesIO(zf.read(n))) for n in zf.namelist() if n.endswith('.npy')}
    tiles_k = pd.read_csv(io.BytesIO(zf.read('tiles.csv')))
T = pd.read_pickle(WORK / 'tiles.pkl')
key_k = tiles_k.rec + '|' + tiles_k.t.astype(str); key = T.rec + '|' + T.t.astype(str)
pos = pd.Series(np.arange(len(tiles_k)), index=key_k.values).reindex(key.values).to_numpy()
assert not np.isnan(pos.astype(float)).any(), 'tile mismatch'
pos = pos.astype(int)
print('label agreement with offline tiles', (tiles_k.y.to_numpy()[pos] == T.y.to_numpy()).mean())
ov = arr['oof_video_deep.npy'][pos]; oi = arr['oof_imu_deep.npy'][pos]
np.save(WORK / 'oof_video_deep.npy', ov); np.save(WORK / 'oof_imu_deep.npy', oi)
np.save(WORK / 'test_pv_deep.npy', arr['test_video_deep.npy']); np.save(WORK / 'test_pi_deep.npy', arr['test_imu_deep.npy'])
y = T.y.to_numpy(); ok = np.stack([T[f'ok{l}'].to_numpy() for l in range(4)], 1)
print('deep video OOF F1', round(macro_f1(y, ov.argmax(1)), 4))
print('deep imu OOF F1 per limb', [round(macro_f1(y[ok[:, l]], oi[ok[:, l], l].argmax(1)), 4) for l in range(4)])
