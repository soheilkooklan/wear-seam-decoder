"""Test arrays in the same layout as the train tiles."""
from __future__ import annotations
import numpy as np, pandas as pd
from common import PACK, LIMBS, WORK
from video_feats import load_test_video
from imu_feats import imu_features


def load_test():
    meta = pd.read_csv(PACK / 'test_meta_data.csv')
    v = load_test_video()
    w = np.load(PACK / 'test_inertial.npy').astype(np.float32)
    limb = meta.sensor_location.map({l: i for i, l in enumerate(LIMBS)}).to_numpy()
    assert not np.isnan(limb.astype(float)).any()
    return meta, v, np.nan_to_num(w), limb.astype(int)


def test_imu_features(w, limb, sbj):
    F = imu_features(w)
    Fn = np.zeros_like(F)
    for s in np.unique(sbj):
        for l in range(4):
            m = (sbj == s) & (limb == l)
            Fn[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-6)
    return np.concatenate([F, Fn], 1)
