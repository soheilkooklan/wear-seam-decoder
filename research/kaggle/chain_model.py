# WEAR 2026: sequence model over reconstructed "chains" of 1-second windows (GPU, about 1 hour).
# The offline pipeline links every window to its most likely next / previous second. Walking those links gives
# for each window a pseudo-sequence of up to 8 seconds before and after it (wear_chains.zip, attached as a
# Kaggle dataset). A transformer reads the 17 windows (video summary + IMU of whichever limb each window has)
# and predicts the label of the centre window. Out-of-fold predictions use the chains built from out-of-fold
# link scores, so the train noise matches the test noise. Output: /kaggle/working/wear_chainmodel.zip
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
SEEDS = [0, 1]
EPOCHS = 12
BS = 128
OUT = Path("/kaggle/working")

hits = sorted(Path("/kaggle/input").rglob("test_meta_data.csv"))
assert hits, "Competition data is not attached."
TEST_DIR = hits[0].parent; ROOT = TEST_DIR.parent; TRAIN_DIR = ROOT / "train"
npy_by_stem = {p.stem: p for p in TRAIN_DIR.rglob("*.npy") if not p.name.startswith(".")}
recs = [(p, npy_by_stem[p.stem]) for p in sorted(TRAIN_DIR.rglob("*.csv")) if p.stem in npy_by_stem]
assert len(recs) >= 20, "Train files are missing."

# ------------------------------------------------------------------ chain file (attached dataset)
CH = {}
zips = sorted(Path("/kaggle/input").rglob("wear_chains.zip"))
if zips:
    with zipfile.ZipFile(zips[0]) as zf:
        for n in zf.namelist():
            if n.endswith(".npy"):
                CH[n] = np.load(io.BytesIO(zf.read(n)))
else:
    tn = sorted(Path("/kaggle/input").rglob("test_nb.npy"))
    assert tn, "Attach the wear_chains dataset (the uploaded wear_chains.zip)."
    base = tn[0].parent
    for p in base.rglob("*.npy"):
        CH[str(p.relative_to(base))] = np.load(p)
K = (CH["test_nb.npy"].shape[1] - 1) // 2
log("chain arrays", len(CH), "K", K)


def encode(raw):
    if pd.api.types.is_numeric_dtype(raw):
        return raw.fillna(0).astype(np.int64).to_numpy()
    t = raw.astype("string").str.strip().fillna("null")
    t = t.replace({"": "null", "nan": "null", "None": "null", "<NA>": "null"})
    m = t.map(L2I); bad = m.isna()
    if bad.any():
        m[bad] = pd.to_numeric(t[bad], errors="coerce")
    return m.astype(np.int64).to_numpy()


def vid_summary(clips):
    """clips (n,15,768) standardised -> (n, 3*768): mean, std, last-first."""
    return np.concatenate([clips.mean(1), clips.std(1), clips[:, -3:].mean(1) - clips[:, :3].mean(1)], 1)

# ------------------------------------------------------------------ train tiles
VS, IW, IF, YY, SB, NB, NS, LB, NAMES, OFFS = [], [], [], [], [], [], [], [], [], []
off = 0
REC = []
for csv_path, npy_path in recs:
    df = pd.read_csv(csv_path, low_memory=False)
    y = encode(df["label"]); acc = df[ACC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32).reshape(-1, 4, 3)
    v = np.load(npy_path, mmap_mode="r")
    if v.shape[1] != 768:
        v = v.T
    n_tiles = 0
    while (n_tiles + 1) * 50 <= len(df) and 30 * n_tiles + 23 <= v.shape[0]:
        n_tiles += 1
    REC.append(dict(name=csv_path.stem, sbj=int(df["sbj_id"].iloc[0]), y=y, acc=acc, v=v, n=n_tiles))
    del df
for s in sorted({r["sbj"] for r in REC}):
    rs = [r for r in REC if r["sbj"] == s]
    fr = np.concatenate([np.nan_to_num(np.asarray(r["v"][::5], dtype=np.float32)) for r in rs]); mu, sd = fr.mean(0), fr.std(0) + 1e-6
    a = np.concatenate([r["acc"] for r in rs]); fin = np.isfinite(a).all(2)
    imu_mu = np.stack([a[fin[:, l], l].mean(0) for l in range(4)]); imu_sd = np.stack([a[fin[:, l], l].std(0) + 1e-6 for l in range(4)])
    for r in rs:
        n = r["n"]; idx = (30 * np.arange(n) + 8)[:, None] + np.arange(15)[None]
        clips = (np.nan_to_num(np.asarray(r["v"][idx.ravel()], dtype=np.float32)).reshape(n, 15, 768) - mu) / sd
        r["vs"] = vid_summary(clips).astype(np.float16)
        w = r["acc"][: n * 50].reshape(n, 50, 4, 3).transpose(0, 2, 1, 3)          # (n,4,50,3)
        r["fin"] = np.isfinite(w).all((2, 3)); w = np.nan_to_num(w)
        r["iw"] = ((w - imu_mu[None, :, None]) / imu_sd[None, :, None]).astype(np.float32); r["raw"] = w.astype(np.float32)
        lab = r["y"][: n * 50].reshape(n, 50)
        r["yt"] = np.array([np.bincount(np.clip(row, 0, NC - 1), minlength=NC).argmax() for row in lab])
        del r["v"], r["acc"], r["y"]
    log("subject", s, "prepared")
for r in REC:
    nm = r["name"]; NAMES.append(nm); OFFS.append(off)
    nb = CH[f"train/{nm}_nb.npy"]; assert len(nb) == r["n"], (nm, len(nb), r["n"])
    NB.append(np.where(nb >= 0, nb + off, -1)); NS.append(CH[f"train/{nm}_ns.npy"]); LB.append(CH[f"train/{nm}_limb.npy"].astype(np.int64))
    VS.append(r["vs"]); IW.append(np.concatenate([r["raw"], r["iw"]], 3)); IF.append(r["fin"]); YY.append(r["yt"]); SB.append(np.full(r["n"], r["sbj"]))
    off += r["n"]
G_V = torch.from_numpy(np.concatenate(VS)).to(DEV)                 # (N, 2304) fp16
G_I = torch.from_numpy(np.concatenate(IW)).to(DEV)                 # (N, 4, 50, 6)
G_F = torch.from_numpy(np.concatenate(IF)).to(DEV)                 # (N, 4)
G_Y = torch.from_numpy(np.concatenate(YY)).to(DEV)
G_NB = torch.from_numpy(np.concatenate(NB)).long().to(DEV); G_NS = torch.from_numpy(np.concatenate(NS)).float().to(DEV)
G_L = torch.from_numpy(np.concatenate(LB)).to(DEV)
SBJ = np.concatenate(SB); N = len(SBJ)
log("train tiles", N, "GPU", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")

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
tsb = meta["sbj_id"].to_numpy(); tlimb = meta["sensor_location"].map({l: i for i, l in enumerate(LIMBS)}).to_numpy().astype(int)
TVS = np.zeros((len(meta), 3 * 768), np.float16); TIW = np.zeros((len(meta), 50, 6), np.float32)
for s in np.unique(tsb):
    m = tsb == s; fr = tv[m].reshape(-1, 768)
    TVS[m] = vid_summary((tv[m] - fr.mean(0)) / (fr.std(0) + 1e-6)).astype(np.float16)
    for l in range(4):
        ml = m & (tlimb == l)
        if ml.any():
            x = ti[ml]; f = x.reshape(-1, 3)
            TIW[ml] = np.concatenate([x, (x - f.mean(0)) / (f.std(0) + 1e-6)], 2)
T_V = torch.from_numpy(TVS).to(DEV); T_I = torch.from_numpy(TIW).to(DEV); T_L = torch.from_numpy(tlimb).long().to(DEV)
T_NB = torch.from_numpy(CH["test_nb.npy"]).long().to(DEV); T_NS = torch.from_numpy(CH["test_ns.npy"]).float().to(DEV)
del tv, ti
log("test", len(meta))


class ChainNet(nn.Module):
    def __init__(self, d=256, drop=0.2):
        super().__init__()
        self.vid = nn.Sequential(nn.Dropout(0.1), nn.Linear(3 * 768, d), nn.GELU(), nn.Linear(d, d))
        conv = lambda i, o, k: nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.imu = nn.Sequential(conv(7, 64, 7), nn.MaxPool1d(2), conv(64, 128, 5), nn.MaxPool1d(2), conv(128, 128, 3))
        self.imu_out = nn.Linear(256 + 16, d); self.limb = nn.Embedding(4, 16)
        self.pos = nn.Parameter(0.02 * torch.randn(1, 2 * K + 1, d)); self.score = nn.Linear(1, d); self.pad = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NC))

    def window(self, v, w, limb):
        """v (M,2304), w (M,50,6) raw+z, limb (M,) -> (M,d)."""
        x = torch.cat([w[..., :3], w[..., :3].norm(dim=2, keepdim=True), w[..., 3:]], 2).transpose(1, 2)
        h = self.imu(x); hi = torch.cat([h.mean(2), h.amax(2), self.limb(limb)], 1)
        return self.vid(v.float()) + self.imu_out(hi)

    def forward(self, v, w, limb, valid, score):
        """v (B,L,2304), w (B,L,50,6), limb (B,L), valid (B,L) bool, score (B,L)."""
        B, L = valid.shape
        e = self.window(v.reshape(B * L, -1), w.reshape(B * L, 50, 6), limb.reshape(-1)).reshape(B, L, -1)
        e = torch.where(valid[..., None], e, self.pad.expand(B, L, -1))
        e = e + self.pos + self.score(torch.log(score.clamp_min(1e-4))[..., None])
        z = self.enc(e, src_key_padding_mask=~valid)
        return self.head(z)                                       # (B,L,NC)


def gather_train(centers, rng_limbs):
    nb = G_NB[centers]; valid = nb >= 0; g = nb.clamp_min(0)
    if rng_limbs:
        score = torch.rand(g.shape + (4,), device=DEV) + G_F[g].float() * 2
        limb = torch.where(torch.rand(g.shape, device=DEV) < 0.5, score.argmax(2), G_L[g])
    else:
        limb = G_L[g]
    w = G_I[g, limb]                                              # (B,L,50,6)
    return G_V[g], w, limb, valid, G_NS[centers], G_Y[g]


def class_weights(y):
    c = np.bincount(y, minlength=NC).astype(np.float64)
    return torch.tensor(len(y) / (NC * np.maximum(c, 1)), dtype=torch.float32, device=DEV)


def train(tr_idx, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = ChainNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    steps = EPOCHS * math.ceil(len(tr_idx) / BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    cw = class_weights(G_Y[torch.from_numpy(tr_idx).to(DEV)].cpu().numpy()); scaler = torch.amp.GradScaler("cuda")
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(tr_idx); tot = 0.0
        for i in range(0, len(perm), BS):
            c = torch.from_numpy(perm[i:i + BS]).to(DEV)
            v, w, limb, valid, score, yy = gather_train(c, True)
            v = v + 0.1 * torch.randn_like(v.float()).half()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(v, w, limb, valid, score).float()
            ce = F.cross_entropy(out.reshape(-1, NC), yy.reshape(-1), weight=cw, label_smoothing=0.05, reduction="none").reshape(yy.shape)
            ce = ce * valid.float()
            loss = ce[:, K].mean() + 0.3 * (ce.sum(1) - ce[:, K]).mean() / (2 * K)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(c)
        if ep in (0, EPOCHS // 2, EPOCHS - 1):
            log(f"  seed {seed} epoch {ep + 1}/{EPOCHS} loss {tot / len(perm):.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict_train(model, idx):
    out = []
    for i in range(0, len(idx), 512):
        c = torch.from_numpy(idx[i:i + 512]).to(DEV)
        v, w, limb, valid, score, _ = gather_train(c, False)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.softmax(model(v, w, limb, valid, score)[:, K].float(), 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_test(model):
    out = []
    for i in range(0, len(meta), 512):
        c = torch.arange(i, min(i + 512, len(meta)), device=DEV)
        nb = T_NB[c]; valid = nb >= 0; g = nb.clamp_min(0)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(T_V[g], T_I[g], T_L[g], valid, T_NS[c])[:, K].float()
        out.append(F.softmax(o, 1).cpu().numpy())
    return np.concatenate(out)

# ------------------------------------------------------------------ cross-validation
tfold = np.array([FOLD[s] for s in SBJ]); yall = G_Y.cpu().numpy()
oof = np.zeros((N, NC), np.float32); pte = np.zeros((len(meta), NC), np.float32)
from sklearn.metrics import f1_score
for fold in range(5):
    tr = np.flatnonzero(tfold != fold); va = np.flatnonzero(tfold == fold)
    for seed in SEEDS:
        m = train(tr, 10 * seed + fold)
        oof[va] += predict_train(m, va) / len(SEEDS); pte += predict_test(m) / (5 * len(SEEDS))
        del m; torch.cuda.empty_cache()
    log(f"fold {fold}: macro F1 {f1_score(yall[va], oof[va].argmax(1), average='macro'):.4f}")
log("OOF macro F1", round(f1_score(yall, oof.argmax(1), average="macro"), 4))


def put(zf, name, arr):
    b = io.BytesIO(); np.save(b, arr); zf.writestr(name, b.getvalue())

with zipfile.ZipFile(OUT / "wear_chainmodel.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for nm, o in zip(NAMES, OFFS):
        n = len(CH[f"train/{nm}_nb.npy"]); put(zf, f"train/{nm}.npy", oof[o:o + n])
    put(zf, "test.npy", pte)
    zf.writestr("meta.json", json.dumps({"recs": NAMES}))
log("wrote wear_chainmodel.zip", round((OUT / "wear_chainmodel.zip").stat().st_size / 1e6, 1), "MB")
