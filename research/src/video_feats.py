"""Video tile features from the PCA-int8 pack."""
from __future__ import annotations
import numpy as np
from common import PACK

_SCALE = None


def scale() -> np.ndarray:
    global _SCALE
    if _SCALE is None:
        _SCALE = np.load(PACK / "pca_scale.npy").astype(np.float32)
    return _SCALE


def load_train_video(rec: str) -> np.ndarray:
    q = np.load(PACK / "video_q" / f"{rec}.npy")
    return q.astype(np.float32) * scale()


def load_test_video() -> np.ndarray:
    q = np.load(PACK / "test_video_q.npy")
    return q.astype(np.float32) * scale()


def pooled(v: np.ndarray, k: int | None = None) -> dict[str, np.ndarray]:
    """v: (n, 15, K). Returns named pooled descriptors (optionally first k PCA dims)."""
    if k is not None:
        v = v[..., :k]
    return {
        "mean": v.mean(1),
        "std": v.std(1),
        "head": v[:, :3].mean(1),
        "tail": v[:, -3:].mean(1),
        "first": v[:, 0],
        "last": v[:, -1],
        "absdiff": np.abs(np.diff(v, axis=1)).mean(1),
    }


def subject_standardize(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for g in np.unique(groups):
        m = groups == g
        mu = x[m].mean(0); sd = x[m].std(0) + 1e-6
        out[m] = (x[m] - mu) / sd
    return out
