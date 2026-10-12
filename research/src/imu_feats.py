"""Hand-crafted features for one 50x3 accelerometer window (50 Hz)."""
from __future__ import annotations
import numpy as np

FS = 50.0


def imu_features(w: np.ndarray) -> np.ndarray:
    """w: (n, 50, 3) float32 -> (n, F) float32."""
    w = np.asarray(w, dtype=np.float32)
    n = len(w)
    mag = np.linalg.norm(w, axis=2, keepdims=True)
    s = np.concatenate([w, mag], axis=2)  # n,50,4
    mean = s.mean(1); std = s.std(1) + 1e-6
    mn = s.min(1); mx = s.max(1)
    q = np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1)  # 5,n,4
    c = s - mean[:, None, :]
    skew = (c ** 3).mean(1) / std ** 3
    kurt = (c ** 4).mean(1) / std ** 4
    d1 = np.diff(s, axis=1); d2 = np.diff(d1, axis=1)
    d1s = d1.std(1); d2s = d2.std(1)
    mob = d1s / std; comp = (d2s / (d1s + 1e-6)) / (mob + 1e-6)
    zc = (np.sign(c[:, 1:]) != np.sign(c[:, :-1])).mean(1)
    spec = np.abs(np.fft.rfft(c, axis=1))[:, 1:, :]  # n,25,4
    pw = spec ** 2; pws = pw.sum(1, keepdims=True) + 1e-9
    pn = pw / pws
    ent = -(pn * np.log(pn + 1e-12)).sum(1)
    freqs = np.fft.rfftfreq(50, 1 / FS)[1:]
    centroid = (pn * freqs[None, :, None]).sum(1)
    domf = freqs[spec.argmax(1)]
    logspec = np.log1p(spec[:, :15, :]).reshape(n, -1)
    # autocorrelation at a few lags (periodicity)
    ac = []
    denom = (c ** 2).sum(1) + 1e-6
    for lag in (2, 5, 10, 15, 20, 25):
        ac.append((c[:, lag:] * c[:, :-lag]).sum(1) / denom)
    ac = np.stack(ac, 1).reshape(n, -1)
    # axis correlations
    cc = w - w.mean(1, keepdims=True)
    sd = cc.std(1) + 1e-6
    corr = np.stack([(cc[:, :, i] * cc[:, :, j]).mean(1) / (sd[:, i] * sd[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))], 1)
    # gravity direction (unit mean vector) and its tilt angles
    g = w.mean(1); gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
    # half-window means (trend)
    h1 = s[:, :25].mean(1); h2 = s[:, 25:].mean(1)
    feats = [mean, std, mn, mx, mx - mn, *q, skew, kurt, d1s, d2s, mob, comp, zc, ent, centroid,
             domf, logspec, ac, corr, gn, h2 - h1, np.abs(d1).mean(1), (s[:, :, 3:] ** 2).mean(1)]
    out = np.concatenate([f.reshape(n, -1) for f in feats], axis=1)
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
