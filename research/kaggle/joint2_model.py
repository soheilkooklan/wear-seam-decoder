# WEAR 2026: joint video + IMU window model, version 2 (GPU, about 1.5 hours).
# Same network idea as version 1, but every training window is cut at a random sub-second offset
# (IMU offset 0..49 samples, video offset aligned to it), so the network sees ~50x more distinct windows.
# Writes out-of-fold (all 4 limbs) and test probabilities to /kaggle/working/wear_joint2.zip.
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
assert DEV == "cuda", "Turn on the GPU accelerator (Settings -> Accelerator -> GPU)."
log("device", DEV, torch.__version__)
CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)",
           "jogging (sidesteps)", "jogging (butt-kicks)", "stretching (triceps)",
           "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
           "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
           "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
L2I = {n: i for i, n in enumerate(CLASSES)}
NC = 19
LIMBS = ["right_arm", "right_leg", "left_leg", "left_arm"]
ACC = [f"{s}_acc_{a}" for s in LIMBS for a in "xyz"]
FOLD = {1: 0, 6: 0, 11: 0, 16: 0, 21: 0, 0: 1, 5: 1, 10: 1, 15: 1, 20: 1, 4: 2, 9: 2, 14: 2, 19: 2,
        3: 3, 8: 3, 13: 3, 18: 3, 2: 4, 7: 4, 12: 4, 17: 4}
SEEDS = [0, 1, 2]
EPOCHS = 30
BS = 512
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

# ------------------------------------------------------------------ load train into flat GPU tensors
REC = []
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
    REC.append(dict(name=csv_path.stem, sbj=int(df["sbj_id"].iloc[0]), y=y, acc=acc,
                    v=np.nan_to_num(np.asarray(v, dtype=np.float32)).astype(np.float16), n_tiles=n_tiles))
    log(csv_path.stem, "tiles", n_tiles)
    del df

SUBJ = sorted({r["sbj"] for r in REC}); SIDX = {s: i for i, s in enumerate(SUBJ)}
IMU_MU = np.zeros((len(SUBJ), 4, 3), np.float32); IMU_SD = np.ones((len(SUBJ), 4, 3), np.float32)
vparts, iparts, fparts, yparts = [], [], [], []
tiles, voff, ioff = [], 0, 0
for s in SUBJ:
    rs = [r for r in REC if r["sbj"] == s]
    fr = np.concatenate([r["v"][::5].astype(np.float32) for r in rs]); mu, sd = fr.mean(0), fr.std(0) + 1e-6
    a = np.concatenate([r["acc"] for r in rs]); fin = np.isfinite(a).all(2)
    for l in range(4):
        IMU_MU[SIDX[s], l] = a[fin[:, l], l].mean(0); IMU_SD[SIDX[s], l] = a[fin[:, l], l].std(0) + 1e-6
    for r in rs:
        vparts.append(((r["v"].astype(np.float32) - mu) / sd).astype(np.float16))
        fin_r = np.isfinite(r["acc"]).all(2)
        iparts.append(np.nan_to_num(r["acc"])); fparts.append(fin_r); yparts.append(r["y"])
        lab = r["y"][: r["n_tiles"] * 50].reshape(-1, 50)
        maj = np.array([np.bincount(row, minlength=NC).argmax() for row in lab])
        tiles.append(pd.DataFrame({"rec": r["name"], "sbj": s, "t": np.arange(r["n_tiles"]), "y": maj,
                                   "voff": voff, "ioff": ioff, "n_tiles": r["n_tiles"]}))
        voff += len(r["v"]); ioff += len(r["acc"])
        del r["v"], r["acc"]
tiles = pd.concat(tiles, ignore_index=True)
GV = torch.from_numpy(np.concatenate(vparts)).to(DEV); del vparts
GI = torch.from_numpy(np.concatenate(iparts)).to(DEV); GF = torch.from_numpy(np.concatenate(fparts)).to(DEV)
GY = torch.from_numpy(np.concatenate(yparts)).to(DEV); del iparts, fparts, yparts
MU_T = torch.from_numpy(IMU_MU).to(DEV); SD_T = torch.from_numpy(IMU_SD).to(DEV)
T_VOFF = torch.from_numpy(tiles["voff"].to_numpy()).to(DEV); T_IOFF = torch.from_numpy(tiles["ioff"].to_numpy()).to(DEV)
T_T = torch.from_numpy(tiles["t"].to_numpy()).to(DEV); T_N = torch.from_numpy(tiles["n_tiles"].to_numpy()).to(DEV)
T_S = torch.from_numpy(tiles["sbj"].map(SIDX).to_numpy()).to(DEV)
log("tiles", len(tiles), "GPU memory", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")
AR15 = torch.arange(15, device=DEV); AR50 = torch.arange(50, device=DEV)


def rand_rot(n, max_deg=10.0):
    ax = F.normalize(torch.randn(n, 3, device=DEV), dim=1)
    th = torch.deg2rad((torch.rand(n, device=DEV) * 2 - 1) * max_deg)
    K = torch.zeros(n, 3, 3, device=DEV)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -ax[:, 2], ax[:, 1], -ax[:, 0]
    K = K - K.transpose(1, 2)
    return torch.eye(3, device=DEV)[None] + torch.sin(th)[:, None, None] * K + (1 - torch.cos(th))[:, None, None] * (K @ K)


def imu_input(w, mu, sd):
    """w (B,50,3) raw acceleration; mu/sd (B,3) subject+limb statistics -> (B,7,50)."""
    mag = w.norm(dim=2, keepdim=True)
    return torch.cat([w, mag, (w - mu[:, None]) / sd[:, None]], 2).transpose(1, 2).contiguous()


def make_batch(rows, train, limb=None):
    """rows: LongTensor of tile rows. Random sub-second offset when training."""
    t = T_T[rows]
    if train:
        off = torch.randint(0, 50, (len(rows),), device=DEV)
        off = torch.where(t + 1 < T_N[rows], off, torch.zeros_like(off))      # last tile keeps offset 0
    else:
        off = torch.zeros_like(t)
    s_imu = 50 * t + off
    s_vid = torch.round(s_imu.float() * 0.6).long() + 8
    x = GV[T_VOFF[rows, None] + s_vid[:, None] + AR15[None]].float()           # (B,15,768)
    ii = T_IOFF[rows, None] + s_imu[:, None] + AR50[None]                      # (B,50)
    yb = F.one_hot(GY[ii].clamp(0, NC - 1), NC).sum(1).argmax(1)
    fin = GF[ii].all(1)                                                        # (B,4)
    if limb is None:
        score = torch.rand(len(rows), 4, device=DEV) + fin.float() * 2         # random finite limb
        limb = score.argmax(1)
    w = GI[ii, :, :][torch.arange(len(rows), device=DEV), :, limb]             # (B,50,3)
    if train:
        w = torch.einsum("bij,btj->bti", rand_rot(len(w)), w) * (0.9 + 0.2 * torch.rand(len(w), 1, 1, device=DEV))
        w = w + 0.01 * torch.randn_like(w)
        x = x + 0.15 * torch.randn_like(x)
    sb = T_S[rows]
    return x, imu_input(w, MU_T[sb, limb], SD_T[sb, limb]), limb, yb

# ------------------------------------------------------------------ test
meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
tv = np.asarray(np.load(TEST_DIR / "test_videomae_data.npy", mmap_mode="r"), dtype=np.float32)
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
    m = tsb == s; fr = tv[m].reshape(-1, 768)
    TVX[m] = ((tv[m] - fr.mean(0)) / (fr.std(0) + 1e-6)).astype(np.float16)
tmu = np.zeros((len(meta), 3), np.float32); tsd = np.ones((len(meta), 3), np.float32)
for s in np.unique(tsb):
    for l in range(4):
        m = (tsb == s) & (tlimb == l)
        if m.any():
            x = ti[m].reshape(-1, 3); tmu[m] = x.mean(0); tsd[m] = x.std(0) + 1e-6
TX = torch.from_numpy(TVX).to(DEV)
TW = imu_input(torch.from_numpy(ti).to(DEV), torch.from_numpy(tmu).to(DEV), torch.from_numpy(tsd).to(DEV))
TL = torch.from_numpy(tlimb).to(DEV)
del tv, TVX
log("test", len(meta))


# ------------------------------------------------------------------ model
class JNet(nn.Module):
    def __init__(self, d=768, h=256, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 15, h))
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


def class_weights(y):
    c = np.bincount(y, minlength=NC).astype(np.float64)
    return torch.tensor(len(y) / (NC * np.maximum(c, 1)), dtype=torch.float32, device=DEV)


def train_joint(tr_rows, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = JNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    steps = EPOCHS * math.ceil(len(tr_rows) / BS)
    sc = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.15)
    cw = class_weights(tiles["y"].to_numpy()[tr_rows])
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(tr_rows); tot = 0.0
        for i in range(0, len(perm), BS):
            rows = torch.from_numpy(perm[i:i + BS]).to(DEV)
            x, w, lb, yb = make_batch(rows, True)
            vdrop = (torch.rand(len(rows), device=DEV) > 0.1).float()
            idrop = (torch.rand(len(rows), device=DEV) > 0.1).float()
            idrop = torch.where(vdrop == 0, torch.ones_like(idrop), idrop)   # never drop both modalities
            with torch.autocast("cuda", dtype=torch.float16):
                oj, ov, oi = model(x, w, lb, vdrop, idrop)
            oj, ov, oi = oj.float(), ov.float(), oi.float()
            loss = (F.cross_entropy(oj, yb, weight=cw, label_smoothing=0.05)
                    + 0.3 * F.cross_entropy(ov, yb, weight=cw, label_smoothing=0.05)
                    + 0.3 * F.cross_entropy(oi, yb, weight=cw, label_smoothing=0.05))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sc.step()
            tot += loss.item() * len(rows)
        if ep in (0, EPOCHS // 2, EPOCHS - 1):
            log(f"  seed {seed} epoch {ep + 1}/{EPOCHS} loss {tot / len(perm):.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict_rows(model, rows, limb):
    oj, ov = [], []
    for i in range(0, len(rows), 1024):
        r = torch.from_numpy(rows[i:i + 1024]).to(DEV)
        x, w, lb, _ = make_batch(r, False, torch.full((len(r),), limb, device=DEV, dtype=torch.long))
        with torch.autocast("cuda", dtype=torch.float16):
            a, b, _ = model(x, w, lb)
        oj.append(F.softmax(a.float(), 1).cpu().numpy()); ov.append(F.softmax(b.float(), 1).cpu().numpy())
    return np.concatenate(oj), np.concatenate(ov)


@torch.no_grad()
def predict_test(model):
    oj, ov = [], []
    for i in range(0, len(TX), 1024):
        with torch.autocast("cuda", dtype=torch.float16):
            a, b, _ = model(TX[i:i + 1024].float(), TW[i:i + 1024], TL[i:i + 1024])
        oj.append(F.softmax(a.float(), 1).cpu().numpy()); ov.append(F.softmax(b.float(), 1).cpu().numpy())
    return np.concatenate(oj), np.concatenate(ov)

# ------------------------------------------------------------------ cross-validation
tfold = tiles["sbj"].map(FOLD).to_numpy()
oof_j = np.zeros((len(tiles), 4, NC), np.float32); oof_v = np.zeros((len(tiles), NC), np.float32)
test_j = np.zeros((len(meta), NC), np.float32); test_v = np.zeros((len(meta), NC), np.float32)
yall = tiles["y"].to_numpy()
for fold in range(5):
    tr = np.flatnonzero(tfold != fold); va = np.flatnonzero(tfold == fold)
    for seed in SEEDS:
        m = train_joint(tr, seed * 10 + fold)
        for l in range(4):
            pj, pv = predict_rows(m, va, l)
            oof_j[va, l] += pj / len(SEEDS)
            if l == 0:
                oof_v[va] += pv / len(SEEDS)
        pj, pv = predict_test(m)
        test_j += pj / (5 * len(SEEDS)); test_v += pv / (5 * len(SEEDS))
        del m; torch.cuda.empty_cache()
    log(f"fold {fold}: joint acc {np.mean([(oof_j[va, l].argmax(1) == yall[va]).mean() for l in range(4)]):.4f}  "
        f"video-head acc {(oof_v[va].argmax(1) == yall[va]).mean():.4f}")

from sklearn.metrics import f1_score
log("OOF macro F1 joint per limb", [round(f1_score(yall, oof_j[:, l].argmax(1), average="macro"), 4) for l in range(4)],
    "video head", round(f1_score(yall, oof_v.argmax(1), average="macro"), 4))


def put(zf, name, arr):
    b = io.BytesIO(); np.save(b, arr); zf.writestr(name, b.getvalue())

with zipfile.ZipFile(OUT / "wear_joint2.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    put(zf, "oof_joint2.npy", oof_j); put(zf, "oof_jvideo2.npy", oof_v)
    put(zf, "test_joint2.npy", test_j); put(zf, "test_jvideo2.npy", test_v)
    zf.writestr("tiles.csv", tiles[["rec", "sbj", "t", "y"]].to_csv(index=False))
log("wrote wear_joint2.zip", round((OUT / "wear_joint2.zip").stat().st_size / 1e6, 1), "MB")
