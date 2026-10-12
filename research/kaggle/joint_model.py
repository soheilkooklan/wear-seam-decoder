# WEAR 2026: joint video + IMU window model (GPU).
# One network sees the 15x768 VideoMAE clip and the 1-second IMU window of one limb.
# Writes out-of-fold (all 4 limbs) and test probabilities to /kaggle/working/wear_joint.zip.
import io, json, math, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("device", DEV, torch.__version__)
torch.backends.cudnn.benchmark = True
CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)",
           "jogging (sidesteps)", "jogging (butt-kicks)", "stretching (triceps)",
           "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
           "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
           "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
L2I = {n: i for i, n in enumerate(CLASSES)}
NC = 19
LIMBS = ["right_arm", "right_leg", "left_leg", "left_arm"]
ACC = [f"{s}_acc_{a}" for s in LIMBS for a in "xyz"]
# subject -> fold (identical to the offline pipeline)
FOLD = {1: 0, 6: 0, 11: 0, 16: 0, 21: 0, 0: 1, 5: 1, 10: 1, 15: 1, 20: 1, 4: 2, 9: 2, 14: 2, 19: 2,
        3: 3, 8: 3, 13: 3, 18: 3, 2: 4, 7: 4, 12: 4, 17: 4}
SEEDS = [0, 1]
EPOCHS = 15
OUT = Path("/kaggle/working")

hits = sorted(Path("/kaggle/input").rglob("test_meta_data.csv"))
assert hits, "Competition data is not attached."
TEST_DIR = hits[0].parent; ROOT = TEST_DIR.parent; TRAIN_DIR = ROOT / "train"
npy_by_stem = {p.stem: p for p in TRAIN_DIR.rglob("*.npy") if not p.name.startswith(".")}
recs = [(p, npy_by_stem[p.stem]) for p in sorted(TRAIN_DIR.rglob("*.csv")) if p.stem in npy_by_stem]
assert len(recs) >= 20, "Train VideoMAE .npy files are missing."


def encode(raw):
    if pd.api.types.is_numeric_dtype(raw):
        return raw.fillna(0).astype(np.int64).to_numpy()
    t = raw.astype("string").str.strip().fillna("null")
    t = t.replace({"": "null", "nan": "null", "None": "null", "<NA>": "null"})
    m = t.map(L2I); bad = m.isna()
    if bad.any():
        m[bad] = pd.to_numeric(t[bad], errors="coerce")
    return m.astype(np.int64).to_numpy()


def majority(lab):  # lab: (m, 50) int labels -> most frequent label per row
    out = np.zeros(len(lab), np.int64)
    for i in range(0, len(lab), 20000):
        blk = lab[i:i + 20000]
        cnt = (blk[:, :, None] == np.arange(NC)[None, None, :]).sum(1)
        out[i:i + 20000] = cnt.argmax(1)
    return out

# ------------------------------------------------------------------ load train
REC = []   # per recording dict
for csv_path, npy_path in recs:
    df = pd.read_csv(csv_path, low_memory=False)
    y = encode(df["label"])
    acc = df[ACC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32).reshape(-1, 4, 3)
    v = np.load(npy_path, mmap_mode="r")
    if v.shape[1] != 768:
        v = v.T
    n_imu, n_vid = len(df), v.shape[0]
    n_tiles = 0
    while (n_tiles + 1) * 50 <= n_imu and 30 * n_tiles + 23 <= n_vid:
        n_tiles += 1
    finite = np.isfinite(acc).all(axis=2)          # (n, 4)
    REC.append(dict(name=csv_path.stem, sbj=int(df["sbj_id"].iloc[0]), y=y, acc=np.nan_to_num(acc),
                    finite=finite, v=np.asarray(v, dtype=np.float16), n_tiles=n_tiles))
    log(csv_path.stem, "tiles", n_tiles)
    del df

# subject-level normalisation statistics
subj = sorted({r["sbj"] for r in REC})
vstat, istat = {}, {}
for s in subj:
    rs = [r for r in REC if r["sbj"] == s]
    frames = np.concatenate([r["v"][::5].astype(np.float32) for r in rs])
    vstat[s] = (frames.mean(0), frames.std(0) + 1e-6)
    a = np.concatenate([r["acc"] for r in rs]); f = np.concatenate([r["finite"] for r in rs])
    istat[s] = [(a[f[:, l], l].mean(0), a[f[:, l], l].std(0) + 1e-6) for l in range(4)]

# tile index table (alignment key for the offline pipeline)
tiles = []
for r in REC:
    t = np.arange(r["n_tiles"])
    lab = r["y"][: r["n_tiles"] * 50].reshape(-1, 50)
    tiles.append(pd.DataFrame({"rec": r["name"], "sbj": r["sbj"], "t": t, "y": majority(lab)}))
tiles = pd.concat(tiles, ignore_index=True)
log("tiles", len(tiles))


def video_clips(r, starts):
    idx = starts[:, None] + np.arange(15)[None, :]
    mu, sd = vstat[r["sbj"]]
    return ((r["v"][idx].astype(np.float32) - mu) / sd).astype(np.float16)

# video training windows: tiles (offset 0) and half-second shifted windows (offset 25 samples)
VX, VY, VS, VT, IX, IF = [], [], [], [], [], []
for r in REC:
    nt = r["n_tiles"]
    for off in (0, 25):
        t = np.arange(nt if off == 0 else nt - 1)
        s_imu = 50 * t + off
        s_vid = np.rint(s_imu * 0.6).astype(int) + 8
        lab = r["y"][s_imu[:, None] + np.arange(50)[None, :]]
        VX.append(video_clips(r, s_vid)); VY.append(majority(lab))
        blk = r["acc"][s_imu[:, None] + np.arange(50)[None, :]]            # (m,50,4,3)
        fin = r["finite"][s_imu[:, None] + np.arange(50)[None, :]].all(1)   # (m,4)
        IX.append(blk.astype(np.float16)); IF.append(fin)
        VS.append(np.full(len(t), r["sbj"])); VT.append(np.full(len(t), off == 0))
VX = np.concatenate(VX); VY = np.concatenate(VY); VS = np.concatenate(VS); VT = np.concatenate(VT)
IX = np.concatenate(IX); IF = np.concatenate(IF)
log("video windows", VX.shape, "imu blocks", IX.shape)

# ------------------------------------------------------------------ test
meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
tv = np.load(TEST_DIR / "test_videomae_data.npy", mmap_mode="r")
tv = np.asarray(tv, dtype=np.float32)
if tv.shape[1] == 768:
    tv = np.transpose(tv, (0, 2, 1))
tv = np.nan_to_num(tv)
ti = np.asarray(np.load(TEST_DIR / "test_inertial_data.npy"), dtype=np.float32)
if ti.shape[1] == 3:
    ti = np.transpose(ti, (0, 2, 1))
ti = np.nan_to_num(ti)
tsb = meta["sbj_id"].to_numpy()
tlimb = meta["sensor_location"].map({l: i for i, l in enumerate(LIMBS)}).to_numpy().astype(int)
TVX = np.zeros_like(tv, dtype=np.float16)
for s in np.unique(tsb):
    m = tsb == s
    fr = tv[m].reshape(-1, 768)
    TVX[m] = ((tv[m] - fr.mean(0)) / (fr.std(0) + 1e-6)).astype(np.float16)
TI_STAT = {}
for s in np.unique(tsb):
    for l in range(4):
        m = (tsb == s) & (tlimb == l)
        x = ti[m].reshape(-1, 3)
        TI_STAT[(s, l)] = (x.mean(0), x.std(0) + 1e-6)
log("test", tv.shape, ti.shape)


# ------------------------------------------------------------------ joint model
class JNet(nn.Module):
    def __init__(self, d=768, h=256, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h))
        self.pos = nn.Parameter(torch.zeros(1, 15, h))
        layer = nn.TransformerEncoderLayer(h, 4, 512, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2)
        self.stat = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Dropout(drop))
        conv = lambda i, o, k: nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.imu = nn.Sequential(conv(7, 64, 7), conv(64, 64, 5), nn.MaxPool1d(2), conv(64, 128, 5), conv(128, 128, 3),
                                 nn.MaxPool1d(2), conv(128, 256, 3), conv(256, 256, 3))
        self.limb = nn.Embedding(4, 32)
        self.vhead = nn.Linear(2 * h, NC)
        self.ihead = nn.Linear(512 + 32, NC)
        self.jhead = nn.Sequential(nn.LayerNorm(2 * h + 512 + 32), nn.Dropout(drop), nn.Linear(2 * h + 512 + 32, 512), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(512, NC))

    def forward(self, x, w, limb, vdrop=None, idrop=None):
        z = self.enc(self.inp(x) + self.pos)
        v = torch.cat([z.mean(1), self.stat(torch.cat([x.mean(1), x.std(1)], 1))], 1)
        h = self.imu(w)
        i = torch.cat([h.mean(2), h.amax(2), self.limb(limb)], 1)
        vj, ij = v, i
        if vdrop is not None:
            vj = v * vdrop[:, None]; ij = i * idrop[:, None]
        return self.jhead(torch.cat([vj, ij], 1)), self.vhead(v), self.ihead(i)


def imu_input(w, stats):
    mag = np.linalg.norm(w, axis=2, keepdims=True)
    z = (w - stats[0][:, None, :]) / stats[1][:, None, :]
    return np.concatenate([w, mag, z], 2).transpose(0, 2, 1).astype(np.float32)


def rand_rot(n, rng, max_deg=10.0):
    ax = rng.normal(size=(n, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    th = np.deg2rad(rng.uniform(-max_deg, max_deg, n))
    K = np.zeros((n, 3, 3)); K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -ax[:, 2], ax[:, 1], -ax[:, 0]
    K = K - K.transpose(0, 2, 1)
    return (np.eye(3)[None] + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * (K @ K)).astype(np.float32)


SUBJ_OF_ROW = VS
def batch_imu(rows, limbs, rng=None):
    w = IX[rows, :, limbs, :].astype(np.float32)          # (B,50,3)
    if rng is not None:
        w = np.einsum("bij,btj->bti", rand_rot(len(w), rng), w) * rng.uniform(0.9, 1.1, (len(w), 1, 1)).astype(np.float32)
        w = w + rng.normal(0, 0.01, w.shape).astype(np.float32)
    mu = np.stack([istat[s][l][0] for s, l in zip(SUBJ_OF_ROW[rows], limbs)])
    sd = np.stack([istat[s][l][1] for s, l in zip(SUBJ_OF_ROW[rows], limbs)])
    return imu_input(w, (mu, sd))


def pick_limbs(rows, rng):
    ok = IF[rows]
    l = rng.integers(0, 4, len(rows))
    bad = ~ok[np.arange(len(rows)), l]
    for k in np.flatnonzero(bad):
        f = np.flatnonzero(ok[k]); l[k] = f[0] if len(f) else 0
    return l


def class_weights(y):
    c = np.bincount(y, minlength=NC).astype(np.float64)
    return torch.tensor(len(y) / (NC * np.maximum(c, 1)), dtype=torch.float32, device=DEV)


def train_joint(tr_idx, seed, bs=512):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = JNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    steps = EPOCHS * math.ceil(len(tr_idx) / bs)
    sc = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.15)
    cw = class_weights(VY[tr_idx])
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(tr_idx); tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            l = pick_limbs(b, rng)
            x = torch.from_numpy(VX[b].astype(np.float32)).to(DEV); x = x + 0.15 * torch.randn_like(x)
            w = torch.from_numpy(batch_imu(b, l, rng)).to(DEV); lb = torch.from_numpy(l).to(DEV)
            yb = torch.from_numpy(VY[b]).to(DEV)
            vdrop = (torch.rand(len(b), device=DEV) > 0.1).float()
            idrop = (torch.rand(len(b), device=DEV) > 0.1).float()
            idrop = torch.where(vdrop == 0, torch.ones_like(idrop), idrop)   # never drop both modalities
            oj, ov, oi = model(x, w, lb, vdrop, idrop)
            loss = (F.cross_entropy(oj, yb, weight=cw, label_smoothing=0.05)
                    + 0.3 * F.cross_entropy(ov, yb, weight=cw, label_smoothing=0.05)
                    + 0.3 * F.cross_entropy(oi, yb, weight=cw, label_smoothing=0.05))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sc.step()
            tot += loss.item() * len(b)
        log(f"  joint seed {seed} epoch {ep+1}/{EPOCHS} loss {tot/len(perm):.4f}")
    return model


@torch.no_grad()
def predict(model, X, W, L, bs=1024):
    model.eval(); oj, ov = [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(DEV)
        w = torch.from_numpy(W[i:i + bs]).to(DEV); lb = torch.from_numpy(L[i:i + bs]).to(DEV)
        a, b_, _ = model(x, w, lb)
        oj.append(F.softmax(a.float(), 1).cpu().numpy()); ov.append(F.softmax(b_.float(), 1).cpu().numpy())
    return np.concatenate(oj), np.concatenate(ov)

# ------------------------------------------------------------------ cross-validation
vfold = np.array([FOLD[s] for s in VS])
vx_tile_rows = []
pos = 0
for r in REC:
    vx_tile_rows.append(np.arange(pos, pos + r["n_tiles"])); pos += r["n_tiles"] + r["n_tiles"] - 1
vx_tile_rows = np.concatenate(vx_tile_rows)
assert len(vx_tile_rows) == len(tiles) and VT[vx_tile_rows].all()
Xti = imu_input(ti, (np.stack([TI_STAT[(s, l)][0] for s, l in zip(tsb, tlimb)]),
                     np.stack([TI_STAT[(s, l)][1] for s, l in zip(tsb, tlimb)])))
oof_j = np.zeros((len(tiles), 4, NC), np.float32); oof_v = np.zeros((len(tiles), NC), np.float32)
test_j = np.zeros((len(meta), NC), np.float32); test_v = np.zeros((len(meta), NC), np.float32)
tile_fold = vfold[vx_tile_rows]
for fold in range(5):
    tr = np.flatnonzero(vfold != fold)
    va_t = np.flatnonzero(tile_fold == fold); rows = vx_tile_rows[va_t]
    for seed in SEEDS:
        m = train_joint(tr, seed * 10 + fold)
        for l in range(4):
            L = np.full(len(rows), l)
            pj, pv = predict(m, VX[rows], batch_imu(rows, L), L)
            oof_j[va_t, l] += pj / len(SEEDS)
            if l == 0: oof_v[va_t] += pv / len(SEEDS)
        pj, pv = predict(m, TVX, Xti, tlimb)
        test_j += pj / (5 * len(SEEDS)); test_v += pv / (5 * len(SEEDS))
        del m; torch.cuda.empty_cache()
    yv = tiles["y"].to_numpy()[va_t]
    log(f"fold {fold}: joint acc {np.mean([(oof_j[va_t, l].argmax(1) == yv).mean() for l in range(4)]):.4f}  video-head acc {(oof_v[va_t].argmax(1) == yv).mean():.4f}")

from sklearn.metrics import f1_score
yall = tiles["y"].to_numpy()
log("OOF macro F1 joint per limb", [round(f1_score(yall, oof_j[:, l].argmax(1), average="macro"), 4) for l in range(4)],
    "video head", round(f1_score(yall, oof_v.argmax(1), average="macro"), 4))


def put(zf, name, arr):
    b = io.BytesIO(); np.save(b, arr); zf.writestr(name, b.getvalue())

with zipfile.ZipFile(OUT / "wear_joint.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    put(zf, "oof_joint.npy", oof_j); put(zf, "oof_jvideo.npy", oof_v)
    put(zf, "test_joint.npy", test_j); put(zf, "test_jvideo.npy", test_v)
    zf.writestr("tiles.csv", tiles.to_csv(index=False))
log("wrote wear_joint.zip", round((OUT / "wear_joint.zip").stat().st_size / 1e6, 1), "MB")
