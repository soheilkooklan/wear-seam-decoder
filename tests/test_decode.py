"""Tests for decode.py. Run with pytest, or with python tests/test_decode.py."""
import glob
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, identity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from decode import decode, knn_weights, link_weights, symmetrise


def graded_unary(n=300):
    # class 1 preferred over class 2 by a margin that grows from 0.01 to 1; null far below
    lp = np.full((n, 3), -10.0)
    lp[:, 1] = 0.0
    lp[:, 2] = -np.linspace(0.01, 1.0, n)
    return lp


def test_knn_small_inputs():
    assert knn_weights(np.random.rand(1, 4)).nnz == 0
    assert knn_weights(np.random.rand(4, 4), k=10).nnz == 4 * 3


def test_repeated_link_pairs_keep_the_largest_score():
    W = link_weights(3, np.array([[0, 1], [0, 1], [1, 2]]), np.array([0.5, 0.9, 0.2]))
    assert np.isclose(W[0, 1], 0.81) and np.isclose(W[1, 2], 0.04) and W.nnz == 2


def test_symmetrisation():
    W = csr_matrix(np.array([[0.0, 1.0], [3.0, 0.0]]))
    assert np.allclose(symmetrise(W).toarray(), [[0, 4], [4, 0]])
    assert np.allclose(symmetrise(W, "max").toarray(), [[0, 3], [3, 0]])
    try:
        symmetrise(W, "mean")
    except ValueError:
        return
    raise AssertionError("an unknown symmetrisation was accepted")


def test_bounds_are_checked():
    try:
        decode(graded_unary(), identity(300, format="csr"), lo=100, hi=50)
    except ValueError:
        return
    raise AssertionError("lo > hi was accepted")


def test_lower_bound_raises_a_count():
    A = identity(300, format="csr")
    free = decode(graded_unary(), A, beta=0.0, lo=0, hi=1000)
    bound = decode(graded_unary(), A, beta=0.0, lo=100, hi=1000)
    assert (free == 2).sum() == 0
    assert (bound == 2).sum() >= 100


def test_bias_is_bounded_and_null_has_none():
    lp = np.full((50, 3), -10.0)
    lp[:, 0] = 0.0
    pred = decode(lp, identity(50, format="csr"), beta=0.0, lo=20, hi=1000)
    assert (pred == 0).all()


def test_bias_steps_stop_before_the_last_iteration():
    # 0.15 per step: 24 steps (3.6) do not overcome a margin of 3.65, 25 steps would
    lp = np.full((10, 3), -10.0)
    lp[:, 1] = 0.0
    lp[:, 2] = -3.65
    pred = decode(lp, identity(10, format="csr"), beta=0.0, lo=10, hi=1000)
    assert (pred == 1).all()
    lp[:, 2] = -3.55
    pred = decode(lp, identity(10, format="csr"), beta=0.0, lo=10, hi=1000)
    assert (pred == 2).all()


def test_max_bias_clips_the_bias():
    pred = decode(graded_unary(), identity(300, format="csr"), beta=0.0, lo=100, hi=1000, max_bias=0.2)
    assert 0 < (pred == 2).sum() < 100


def test_examples_reproduce_the_final_system():
    files = sorted(glob.glob(os.path.join(HERE, "..", "examples", "*.npz")))
    assert files
    for f in files:
        d = np.load(f)
        A = csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["A_shape"]))
        pred = decode(d["logp"], A, beta=float(d["beta"]), lo=float(d["lo"]), hi=float(d["hi"]))
        assert (pred == d["expected"]).all()


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok ", name)
