# WEAR 2026: full-dimensional contrastive "next second" embeddings (CPU is enough).
# For every train 1-second tile it writes out-of-fold end/start embeddings, and for every
# test window embeddings from models trained on all train recordings.
import io, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

FOLD = {1: 0, 6: 0, 11: 0, 16: 0, 21: 0, 0: 1, 5: 1, 10: 1, 15: 1, 20: 1, 4: 2, 9: 2, 14: 2, 19: 2,
        3: 3, 8: 3, 13: 3, 18: 3, 2: 4, 7: 4, 12: 4, 17: 4}
SEEDS = [0, 1]; OUT_DIM = 128; ITERS = 3000; TAU = 0.1; LR = 2e-3
OUT = Path("/kaggle/working")
hits = sorted(Path("/kaggle/input").rglob("test_meta_data.csv"))
assert hits, "Competition data is not attached."
TEST_DIR = hits[0].parent; ROOT = TEST_DIR.parent; TRAIN_DIR = ROOT / "train"
npy_by_stem = {p.stem: p for p in TRAIN_DIR.rglob("*.npy") if not p.name.startswith(".")}
recs = [(p, npy_by_stem[p.stem]) for p in sorted(TRAIN_DIR.rglob("*.csv")) if p.stem in npy_by_stem]


def end_feats(v):
    return np.concatenate([v[:, -3:].mean(1), v[:, -1], v[:, -1] - v[:, -5], v.mean(1)], 1).astype(np.float32)


def start_feats(v):
    return np.concatenate([v[:, :3].mean(1), v[:, 0], v[:, 4] - v[:, 0], v.mean(1)], 1).astype(np.float32)


def standardize(v):
    fr = v.reshape(-1, v.shape[-1])
    return (v - fr.mean(0)) / (fr.std(0) + 1e-6)

# ---------------------------------------------------------------- train tiles
DATA, META = {}, []
for csv_path, npy_path in recs:
    sbj = int(pd.read_csv(csv_path, usecols=["sbj_id"], nrows=1)["sbj_id"].iloc[0])
    n_imu = sum(1 for _ in open(csv_path)) - 1
    v = np.load(npy_path, mmap_mode="r")
    if v.shape[1] != 768:
        v = v.T
    n_tiles = 0
    while (n_tiles + 1) * 50 <= n_imu and 30 * n_tiles + 23 <= v.shape[0]:
        n_tiles += 1
    idx = (30 * np.arange(n_tiles) + 8)[:, None] + np.arange(15)[None, :]
    tv_ = standardize(np.nan_to_num(np.asarray(v[idx.ravel()], dtype=np.float32)).reshape(n_tiles, 15, 768))
    DATA[csv_path.stem] = (end_feats(tv_), start_feats(tv_))
    META.append((csv_path.stem, sbj, n_tiles))
    log(csv_path.stem, "tiles", n_tiles)
    del v, tv_


class Tower:
    def __init__(self, din, out, rng):
        self.W = (rng.standard_normal((din, out)) / np.sqrt(din)).astype(np.float32)
        self.m = np.zeros_like(self.W); self.v = np.zeros_like(self.W)

    def forward(self, x):
        self.x = x; z = x @ self.W
        self.n = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8; self.u = z / self.n
        return self.u

    def step(self, gu, t, wd=1e-4):
        gz = (gu - self.u * (gu * self.u).sum(1, keepdims=True)) / self.n
        g = self.x.T @ gz + wd * self.W
        self.m = 0.9 * self.m + 0.1 * g; self.v = 0.999 * self.v + 0.001 * g * g
        self.W -= LR * (self.m / (1 - 0.9 ** t)) / (np.sqrt(self.v / (1 - 0.999 ** t)) + 1e-8)


def train(rec_names, seed):
    rng = np.random.default_rng(seed)
    din = DATA[rec_names[0]][0].shape[1]
    te, ts = Tower(din, OUT_DIM, rng), Tower(din, OUT_DIM, rng)
    eye = np.eye(512, dtype=np.float32)
    for it in range(ITERS):
        xe, xs = DATA[rec_names[rng.integers(len(rec_names))]]; n = len(xe)
        st = rng.integers(0, n - 9, 64); a = (st[:, None] + np.arange(8)[None]).ravel()
        ue = te.forward(xe[a]); us = ts.forward(xs[a + 1])
        L = ue @ us.T / TAU
        Pr = np.exp(L - L.max(1, keepdims=True)); Pr /= Pr.sum(1, keepdims=True)
        Pc = np.exp(L - L.max(0, keepdims=True)); Pc /= Pc.sum(0, keepdims=True)
        G = ((Pr - eye) + (Pc - eye)) / (len(a) * TAU)
        gue, gus = G @ us, G.T @ ue
        te.step(gue, it + 1); ts.step(gus, it + 1)
    return te, ts


def embed(model, x):
    z = x @ model.W
    return (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)).astype(np.float16)


def top1(te, ts, name):
    xe, xs = DATA[name]; ue = embed(te, xe).astype(np.float32); us = embed(ts, xs).astype(np.float32)
    S = ue @ us.T; np.fill_diagonal(S, -9); a = np.arange(len(S) - 1)
    return float(((S[a] > S[a, a + 1][:, None]).sum(1) == 0).mean())

# ---------------------------------------------------------------- CV embeddings
names = [m[0] for m in META]
fold_of = {m[0]: FOLD[m[1]] for m in META}
emb = {nm: np.zeros((len(DATA[nm][0]), len(SEEDS), 2, OUT_DIM), np.float16) for nm in names}
for fold in range(5):
    tr = [nm for nm in names if fold_of[nm] != fold]; va = [nm for nm in names if fold_of[nm] == fold]
    for k, seed in enumerate(SEEDS):
        te, ts = train(tr, 100 * fold + seed)
        for nm in va:
            emb[nm][:, k, 0] = embed(te, DATA[nm][0]); emb[nm][:, k, 1] = embed(ts, DATA[nm][1])
        log(f"fold {fold} seed {seed}: held-out top1", round(np.mean([top1(te, ts, nm) for nm in va]), 4))

# ---------------------------------------------------------------- test
meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
tv = np.asarray(np.load(TEST_DIR / "test_videomae_data.npy", mmap_mode="r"), dtype=np.float32)
if tv.shape[1] == 768:
    tv = np.transpose(tv, (0, 2, 1))
tv = np.nan_to_num(tv)
te_emb = np.zeros((len(meta), len(SEEDS), 2, OUT_DIM), np.float16)
for k, seed in enumerate(SEEDS):
    te, ts = train(names, 1000 + seed)
    for s in meta["sbj_id"].unique():
        m = (meta["sbj_id"] == s).to_numpy()
        v = standardize(tv[m])
        te_emb[m, k, 0] = embed(te, end_feats(v)); te_emb[m, k, 1] = embed(ts, start_feats(v))
    log("test seed", seed, "done")

buf = io.BytesIO()
with zipfile.ZipFile(OUT / "wear_contrast.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for nm in names:
        b = io.BytesIO(); np.save(b, emb[nm]); zf.writestr(f"train/{nm}.npy", b.getvalue())
    b = io.BytesIO(); np.save(b, te_emb); zf.writestr("test_emb.npy", b.getvalue())
    zf.writestr("meta.json", json.dumps({"recs": META, "seeds": SEEDS, "out_dim": OUT_DIM}))
log("wrote wear_contrast.zip", round((OUT / "wear_contrast.zip").stat().st_size / 1e6, 1), "MB")
