"""Test-like simulation on train recordings: one random limb per tile, per-recording decoding."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import N_CLASSES, macro_f1


def pick_limbs(T: pd.DataFrame, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ok = np.stack([T[f"ok{l}"].to_numpy() for l in range(4)], 1)
    limb = rng.integers(0, 4, len(T))
    bad = ~ok[np.arange(len(T)), limb]
    for i in np.flatnonzero(bad):
        limb[i] = int(np.flatnonzero(ok[i])[0])
    return limb


def evaluate(T: pd.DataFrame, probs: np.ndarray, decoder=None, seed: int = 0, units: str = "rec",
             verbose: bool = False) -> dict:
    """probs: (n, 4, C) per-limb probabilities or (n, C). decoder(idx, p) -> labels for one unit."""
    limb = pick_limbs(T, seed)
    p = probs[np.arange(len(T)), limb] if probs.ndim == 3 else probs
    y = T.y.to_numpy()
    pred = p.argmax(1).copy()
    if decoder is not None:
        for u, g in T.groupby(units, sort=False):
            idx = g.index.to_numpy()
            pred[idx] = decoder(idx, p[idx], limb[idx])
    m = y >= 0
    out = {"f1": macro_f1(y[m], pred[m])}
    if verbose:
        per = {}
        for u, g in T.groupby(units, sort=False):
            idx = g.index.to_numpy(); mm = y[idx] >= 0
            per[u] = round(macro_f1(y[idx][mm], pred[idx][mm]), 3)
        out["per_unit"] = per
    out["pred"] = pred
    return out
