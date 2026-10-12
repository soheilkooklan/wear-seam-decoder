"""Shared constants and data access for the WEAR 2026 work."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(os.environ.get("WEAR_ROOT", Path(__file__).resolve().parents[1]))
REF = ROOT / "ref"
PACK = ROOT / "pack"
WORK = ROOT / "work"
N_CLASSES = 19
LIMBS = ("right_arm", "right_leg", "left_leg", "left_arm")
GROUP_A = (4, 5, 8, 9, 10, 15, 16, 17, 18)
GROUP_B = (1, 2, 3, 6, 7, 11, 12, 13, 14)
CLASS_NAMES = ("null", "jogging", "jogging (rotating arms)", "jogging (skipping)",
               "jogging (sidesteps)", "jogging (butt-kicks)", "stretching (triceps)",
               "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
               "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
               "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips")


def _missing_file_hint(kind, err, tb):
    sys.__excepthook__(kind, err, tb)
    if issubclass(kind, FileNotFoundError):
        sys.stderr.write(f"\nA file is missing. WEAR_ROOT is {ROOT}. Files in pack/ and ref/ come from "
                         "step 1 of research/STEPS.md, files in work/ from the later steps.\n")


sys.excepthook = _missing_file_hint


def require(path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (WEAR_ROOT = {ROOT}); see research/STEPS.md")
    return path


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(N_CLASSES)), zero_division=0))


def recordings() -> list[str]:
    meta = json.loads(require(REF / "cache/meta.json").read_text())
    return [Path(f[0]).stem for f in meta["files"]]


def subject_of(rec: str) -> int:
    return int(rec.split("_")[1])


def load_cube(rec: str) -> np.ndarray:
    return np.load(require(REF / "cache/cubes" / f"{rec}.npy"), mmap_mode="r")


def sample_labels(rec: str, n: int) -> np.ndarray:
    """Per-sample labels. Exact labels from the pack if present, else rebuilt from pure windows."""
    exact = PACK / "labels" / f"{rec}.npy"
    if exact.is_file():
        return np.load(exact).astype(np.int64)
    u = _unique()
    g = u[u.recording == rec]
    lab = np.full(n, -1, dtype=np.int64)
    for s, t in zip(g.imu_start.to_numpy(), g.target.to_numpy()):
        lab[s:s + 50] = t
    return lab


_U = None
def _unique():
    global _U
    if _U is None:
        _U = pd.read_csv(require(REF / "cache/unique.csv"))
    return _U


def tile_table() -> pd.DataFrame:
    """One row per 1-second tile of every train recording (IMU [50t,50t+50), video [30t+8,30t+23))."""
    cache = WORK / "tiles.parquet"
    rows = []
    for rec in recordings():
        cube = load_cube(rec)
        n = len(cube)
        info = _pack_info().get(rec)
        n_tiles = info["n_tiles"] if info else n // 50
        lab = sample_labels(rec, n)
        L = lab[: n_tiles * 50].reshape(n_tiles, 50)
        # majority label among known samples, purity = share of that label
        maj = np.full(n_tiles, -1); pur = np.zeros(n_tiles)
        for t in range(n_tiles):
            v = L[t][L[t] >= 0]
            if len(v):
                c = np.bincount(v, minlength=N_CLASSES)
                maj[t] = c.argmax(); pur[t] = c.max() / 50
        mid = L[:, 25]
        fin = np.isfinite(np.asarray(cube[: n_tiles * 50])).reshape(n_tiles, 50, 4, 3).all(axis=(1, 3))
        rows.append(pd.DataFrame({"rec": rec, "sbj": subject_of(rec), "t": np.arange(n_tiles),
                                  "y": maj, "y_mid": mid, "purity": pur,
                                  "ok0": fin[:, 0], "ok1": fin[:, 1], "ok2": fin[:, 2], "ok3": fin[:, 3]}))
    return pd.concat(rows, ignore_index=True)


_INFO = None
def _pack_info() -> dict:
    global _INFO
    if _INFO is None:
        p = PACK / "train_info.json"
        _INFO = json.loads(p.read_text()) if p.is_file() else {}
    return _INFO


def tile_imu(tiles: pd.DataFrame) -> np.ndarray:
    """(n_tiles, 4, 50, 3) float32 for the rows of ``tiles`` (in order)."""
    out = np.zeros((len(tiles), 4, 50, 3), dtype=np.float32)
    for rec, g in tiles.groupby("rec", sort=False):
        cube = np.asarray(load_cube(rec), dtype=np.float32)
        idx = g.t.to_numpy()[:, None] * 50 + np.arange(50)[None, :]
        w = cube[idx]  # n,50,4,3
        out[g.index.to_numpy() if False else np.flatnonzero(tiles.rec.to_numpy() == rec)] = np.transpose(w, (0, 2, 1, 3))
    return np.nan_to_num(out)
