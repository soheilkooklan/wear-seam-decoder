# WEAR 2026: compact data pack for offline development.
# Run on Kaggle (CPU is enough) with the competition data attached.
# Produces two files in /kaggle/working:
#   wear_pack_test.zip   (test arrays, train labels, PCA basis, diagnostics)
#   wear_pack_train.zip  (train VideoMAE tiles, PCA-compressed to int8)
import json, time, zipfile, io
from pathlib import Path
import numpy as np
import pandas as pd

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)",
           "jogging (sidesteps)", "jogging (butt-kicks)", "stretching (triceps)",
           "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
           "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
           "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
L2I = {n: i for i, n in enumerate(CLASSES)}
LIMBS = ["right_arm", "right_leg", "left_leg", "left_arm"]
ACC = [f"{s}_acc_{a}" for s in LIMBS for a in "xyz"]
OUT = Path("/kaggle/working")

# ---------------------------------------------------------------- locate files
hits = sorted(Path("/kaggle/input").rglob("test_meta_data.csv"))
assert hits, "Competition data is not attached to this notebook."
TEST_DIR = hits[0].parent
ROOT = TEST_DIR.parent
TRAIN_DIR = ROOT / "train"
npy_by_stem = {p.stem: p for p in TRAIN_DIR.rglob("*.npy") if not p.name.startswith(".")}
csvs = sorted(p for p in TRAIN_DIR.rglob("*.csv") if not p.name.startswith("."))
recs = [(p, npy_by_stem[p.stem]) for p in csvs if p.stem in npy_by_stem]
skipped = [str(p.relative_to(ROOT)) for p in csvs if p.stem not in npy_by_stem]
log("root", ROOT)
log("train recordings with video:", len(recs), "| csv without npy:", skipped[:10])
assert len(recs) >= 20, "Train VideoMAE .npy files are missing in this data version."

def encode(raw: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(raw):
        return raw.fillna(0).astype(np.int64).to_numpy()
    t = raw.astype("string").str.strip().fillna("null")
    t = t.replace({"": "null", "nan": "null", "None": "null", "<NA>": "null"})
    m = t.map(L2I)
    bad = m.isna()
    if bad.any():
        m[bad] = pd.to_numeric(t[bad], errors="coerce")
    assert not m.isna().any(), t[m.isna()].unique()[:10]
    return m.astype(np.int64).to_numpy()

def as_frames(v: np.ndarray) -> np.ndarray:
    if v.ndim == 2 and v.shape[1] != 768 and v.shape[0] == 768:
        v = v.T
    assert v.ndim == 2 and v.shape[1] == 768, v.shape
    return v

# ---------------------------------------------------------------- train pass
info, tiles, labels = {}, {}, {}
imu_stats = []
S = np.zeros(768); SS = np.zeros((768, 768)); NF = 0
for csv_path, npy_path in recs:
    rec = csv_path.stem
    df = pd.read_csv(csv_path, low_memory=False)
    y = encode(df["label"]).astype(np.int8)
    acc = df[ACC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32).reshape(-1, 4, 3)
    v = as_frames(np.load(npy_path, mmap_mode="r"))
    n_imu, n_vid = len(df), v.shape[0]
    n_tiles = 0
    while (n_tiles + 1) * 50 <= n_imu and 30 * n_tiles + 23 <= n_vid:
        n_tiles += 1
    idx = (30 * np.arange(n_tiles) + 8)[:, None] + np.arange(15)[None, :]
    fr = np.asarray(v[idx.ravel()], dtype=np.float32)
    fr = np.nan_to_num(fr)
    X = fr.astype(np.float64)
    S += X.sum(0); SS += X.T @ X; NF += len(X)
    tiles[rec] = fr.reshape(n_tiles, 15, 768)
    labels[rec] = y
    info[rec] = {"sbj_id": int(df["sbj_id"].iloc[0]), "n_imu": int(n_imu),
                 "n_video": int(n_vid), "n_tiles": int(n_tiles),
                 "video_dtype": str(v.dtype), "nan_imu": int(np.isnan(acc).sum())}
    # IMU statistics on tiles, per limb (to compare with test later)
    w = acc[: n_tiles * 50].reshape(n_tiles, 50, 4, 3)
    for li, limb in enumerate(LIMBS):
        x = w[:, :, li, :]
        ok = np.isfinite(x).all(axis=(1, 2))
        x = x[ok]
        imu_stats.append({"split": "train", "rec": rec, "limb": limb,
                          "mean_norm": float(np.linalg.norm(x.mean(1), axis=1).mean()),
                          "std": float(x.std(1).mean()),
                          "diff_std": float(np.diff(x, axis=1).std(1).mean()),
                          "mean_x": float(x[..., 0].mean()), "mean_y": float(x[..., 1].mean()),
                          "mean_z": float(x[..., 2].mean())})
    log(f"{rec}: imu {n_imu}  video {n_vid}  ratio {n_vid / max(n_imu, 1):.4f}  tiles {n_tiles}  dtype {v.dtype}")
    del df, acc, v, X

# ---------------------------------------------------------------- test
meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
log("test meta columns:", list(meta.columns))
tv = np.load(TEST_DIR / "test_videomae_data.npy", mmap_mode="r")
ti = np.load(TEST_DIR / "test_inertial_data.npy", mmap_mode="r")
log("test video", tv.shape, tv.dtype, "| test inertial", ti.shape, ti.dtype)
tv = np.asarray(tv, dtype=np.float32)
if tv.shape[1] == 768:
    tv = np.transpose(tv, (0, 2, 1))
tv = np.ascontiguousarray(np.nan_to_num(tv))
ti = np.asarray(ti, dtype=np.float32)
if ti.shape[1] == 3:
    ti = np.transpose(ti, (0, 2, 1))
ti = np.ascontiguousarray(ti)
X = tv.reshape(-1, 768).astype(np.float64)
S += X.sum(0); SS += X.T @ X; NF += len(X)
del X

# ---------------------------------------------------------------- PCA
mu = S / NF
C = SS / NF - np.outer(mu, mu)
evals, evecs = np.linalg.eigh(C)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
cum = np.cumsum(evals) / evals.sum()
for k in (32, 64, 96, 128, 160, 192, 224, 256, 320):
    log(f"PCA {k:3d} comps: explained variance {cum[k-1]:.4f}")
K = int(np.searchsorted(cum, 0.95) + 1)
K = int(min(224, max(128, int(np.ceil(K / 32) * 32))))
log("PCA dims kept:", K, f"(EV {cum[K-1]:.4f})")
W = evecs[:, :K].astype(np.float32)
mu32 = mu.astype(np.float32)

rng = np.random.default_rng(0)
sample = np.concatenate([tiles[r].reshape(-1, 768)[rng.choice(tiles[r].shape[0] * 15, min(2000, tiles[r].shape[0] * 15), replace=False)]
                         for r in tiles] + [tv.reshape(-1, 768)[rng.choice(len(tv) * 15, min(20000, len(tv) * 15), replace=False)]])
zs = (sample - mu32) @ W
scale = (np.percentile(np.abs(zs), 99.99, axis=0) / 127.0).astype(np.float32)
scale = np.maximum(scale, 1e-6)

def quant(fr: np.ndarray) -> np.ndarray:
    shp = fr.shape
    z = (fr.reshape(-1, 768) - mu32) @ W
    q = np.clip(np.rint(z / scale), -127, 127).astype(np.int8)
    return q.reshape(*shp[:-1], K)

# reconstruction check: cosine between original and dequantized-reconstructed frames
chk = sample[:5000]
rec_ = ((quant(chk).astype(np.float32) * scale) @ W.T) + mu32
cos = (chk * rec_).sum(1) / (np.linalg.norm(chk, axis=1) * np.linalg.norm(rec_, axis=1) + 1e-9)
log(f"reconstruction cosine: mean {cos.mean():.4f}  p5 {np.percentile(cos, 5):.4f}")

# ---------------------------------------------------------------- diagnostics
diag = {"n_test": int(len(meta)), "meta_columns": list(meta.columns)}
subj_col = [c for c in meta.columns if "sub" in c.lower() or "sbj" in c.lower()]
sens_col = [c for c in meta.columns if "sensor" in c.lower() or "location" in c.lower()]
if subj_col:
    vc = meta[subj_col[0]].value_counts().sort_index()
    diag["per_subject"] = {str(k): int(v) for k, v in vc.items()}
    log("test windows per subject:", diag["per_subject"])
if sens_col:
    vc = meta[sens_col[0]].value_counts()
    diag["per_sensor"] = {str(k): int(v) for k, v in vc.items()}
    log("test windows per sensor:", diag["per_sensor"])
if subj_col and sens_col:
    print(pd.crosstab(meta[subj_col[0]], meta[sens_col[0]]))

# exact frame sharing between different test windows (would reveal overlapping windows)
r768 = rng.standard_normal(768)
h = np.round(tv.reshape(-1, 768).astype(np.float64) @ r768, 6)
win = np.repeat(np.arange(len(tv)), 15)
dfh = pd.DataFrame({"h": h, "w": win})
g = dfh.groupby("h")["w"].nunique()
diag["test_frames_shared_by_2plus_windows"] = int((g > 1).sum())
diag["test_windows_with_shared_frame"] = int(dfh[dfh["h"].isin(g[g > 1].index)]["w"].nunique())
log("test frames shared across windows:", diag["test_frames_shared_by_2plus_windows"],
    "| windows involved:", diag["test_windows_with_shared_frame"])

# exact IMU sample-pair sharing between different test windows
r6 = rng.standard_normal(6)
pairs = np.concatenate([ti[:, :-1, :], ti[:, 1:, :]], axis=2).astype(np.float64)  # N,49,6
hp = np.round(pairs.reshape(-1, 6) @ r6, 9)
wp = np.repeat(np.arange(len(ti)), 49)
dfp = pd.DataFrame({"h": hp, "w": wp})
gp = dfp.groupby("h")["w"].nunique()
diag["test_imu_pairs_shared"] = int((gp > 1).sum())
diag["test_imu_windows_with_shared_pair"] = int(dfp[dfp["h"].isin(gp[gp > 1].index)]["w"].nunique())
log("test IMU sample pairs shared across windows:", diag["test_imu_pairs_shared"],
    "| windows involved:", diag["test_imu_windows_with_shared_pair"])

# test IMU stats per sensor
if sens_col:
    for limb, grp in meta.groupby(sens_col[0]):
        x = ti[grp.index.to_numpy()]
        imu_stats.append({"split": "test", "rec": "test", "limb": str(limb),
                          "mean_norm": float(np.linalg.norm(x.mean(1), axis=1).mean()),
                          "std": float(x.std(1).mean()),
                          "diff_std": float(np.diff(x, axis=1).std(1).mean()),
                          "mean_x": float(x[..., 0].mean()), "mean_y": float(x[..., 1].mean()),
                          "mean_z": float(x[..., 2].mean())})
st = pd.DataFrame(imu_stats)
print(st.groupby(["split", "limb"])[["mean_norm", "std", "diff_std", "mean_x", "mean_y", "mean_z"]].mean().round(4))
diag["pca_dims"] = K
diag["pca_cum_ev"] = [float(c) for c in cum[:512]]
diag["recon_cos_mean"] = float(cos.mean())

# ---------------------------------------------------------------- write zips
def put(zf, name, arr):
    buf = io.BytesIO(); np.save(buf, arr); zf.writestr(name, buf.getvalue())

with zipfile.ZipFile(OUT / "wear_pack_test.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    put(zf, "pca_mean.npy", mu32); put(zf, "pca_W.npy", W); put(zf, "pca_scale.npy", scale)
    put(zf, "test_video_q.npy", quant(tv))
    put(zf, "test_inertial.npy", ti)
    zf.writestr("test_meta_data.csv", meta.to_csv(index=False))
    for extra in ["sample_submission.csv", "participant_meta_data.txt", "participant_meta.txt"]:
        p = ROOT / extra
        if p.is_file():
            zf.writestr(extra, p.read_bytes())
    for rec, y in labels.items():
        put(zf, f"labels/{rec}.npy", y)
    zf.writestr("train_info.json", json.dumps(info, indent=1))
    zf.writestr("diagnostics.json", json.dumps(diag, indent=1))
    zf.writestr("imu_stats.csv", st.to_csv(index=False))
log("wrote wear_pack_test.zip", round((OUT / "wear_pack_test.zip").stat().st_size / 1e6, 1), "MB")

with zipfile.ZipFile(OUT / "wear_pack_train.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for rec, fr in tiles.items():
        put(zf, f"video_q/{rec}.npy", quant(fr))
log("wrote wear_pack_train.zip", round((OUT / "wear_pack_train.zip").stat().st_size / 1e6, 1), "MB")
log("DONE")
