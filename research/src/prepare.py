"""Unpack the Kaggle packs and build the exact tile table + features."""
import sys, zipfile, json, time
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd
from common import *
from imu_feats import imu_features

t0 = time.time()
up = Path('/mnt/user-data/uploads')
PACK.mkdir(parents=True, exist_ok=True)
for name in []:
    hits = sorted(up.rglob(name.replace('.zip', '*.zip')))
    if not hits:
        print('missing', name); continue
    with zipfile.ZipFile(hits[-1]) as zf:
        zf.extractall(PACK)
    print('extracted', hits[-1], round(time.time() - t0, 1))
T = tile_table()
T.to_pickle(WORK / 'tiles.pkl')
print('tiles', T.shape, 'unlabelled', int((T.y < 0).sum()), 'impure', float((T.purity < 1).mean()))
W = tile_imu(T)
F = np.stack([imu_features(W[:, l]) for l in range(4)], 1)
np.save(WORK / 'imu_feats_exact.npy', F)
print('done', round(time.time() - t0, 1))
