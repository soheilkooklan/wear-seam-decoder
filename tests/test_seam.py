"""Tests for seam.py. Run with pytest, or with python tests/test_seam.py."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from seam import seam_distance, seam_points, successor_ranks


def smooth_windows(n=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(50 * n) / 50.0
    x = np.stack([np.sin(1.3 * t), np.cos(0.7 * t), np.sin(2.1 * t + 1)], 1)
    x = x + 0.01 * rng.standard_normal(x.shape)
    return x.reshape(n, 50, 3)


def test_quadratic_signal_meets_at_the_seam():
    # both extrapolations are exact for a quadratic signal
    t = np.arange(100, dtype=float)
    x = np.stack([0.01 * t ** 2, -0.02 * t ** 2 + t, 3 + 0 * t], 1)
    end, start = seam_points(x.reshape(2, 50, 3))
    assert np.allclose(end[0], start[1])


def test_true_successor_ranks_first():
    w = smooth_windows()
    p = np.random.default_rng(1).permutation(len(w))
    inv = np.argsort(p)
    nxt = np.full(len(w), -1)
    for i in range(len(w) - 1):
        nxt[inv[i]] = inv[i + 1]
    a, r = successor_ranks(w[p], nxt)
    assert len(a) == len(w) - 1
    assert (r == 1).mean() > 0.9


def test_diagonal_and_cross_limb_entries_are_infinite():
    w = smooth_windows(6)
    limb = np.array([0, 0, 1, 1, 0, 1])
    D = seam_distance(w, limb)
    assert np.isinf(np.diag(D)).all()
    assert np.isinf(D[0, 2]) and np.isfinite(D[0, 1])


def test_cross_limb_successors_are_skipped():
    w = smooth_windows(4)
    a, r = successor_ranks(w, np.array([1, 2, 3, -1]), np.array([0, 1, 1, 1]))
    assert list(a) == [1, 2]


def test_single_window():
    D = seam_distance(smooth_windows(1))
    assert D.shape == (1, 1) and np.isinf(D[0, 0])


def test_short_windows_are_rejected():
    try:
        seam_points(np.zeros((3, 2, 3)))
    except ValueError:
        return
    raise AssertionError("windows with fewer than three samples were accepted")


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok ", name)
