# WEAR 2026: IMU "next second" bi-encoder (GPU, about 30 minutes).
# End-of-window and start-of-window towers over one limb's 1-second accelerometer window.
# The limb of each window is random (as in the test data), so the towers learn continuity both within a limb
# (exact signal continuation) and across limbs (shared rhythm and phase). Random sub-second offsets are used
# as augmentation. Embeddings are written for all four limbs of every train tile (out-of-fold) and for the
# given limb of every test window. Output: /kaggle/working/wear_imu_link.zip
import io, json, time, zipfile
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
LIMBS = ["right_arm", "right_leg", "left_leg", "left_arm"]
ACC = [f"{s}_acc_{a}" for s in LIMBS for a in "xyz"]
FOLD = {1: 0, 6: 0, 11: 0, 16: 0, 21: 0, 0: 1, 5: 1, 10: 1, 15: 1, 20: 1, 4: 2, 9: 2, 14: 2, 19: 2,
        3: 3, 8: 3, 13: 3, 18: 3, 2: 4, 7: 4, 12: 4, 17: 4}
SEEDS = [0, 1, 2]
D_OUT = 32
TAU = 0.1
STEPS = 2000
EDGE = 8             # samples at the window edge given explicitly to the towers
OUT = Path("/kaggle/working")

hits = sorted(Path("/kaggle/input").rglob("test_meta_data.csv"))
assert hits, "Competition data is not attached."
TEST_DIR = hits[0].parent; ROOT = TEST_DIR.parent; TRAIN_DIR = ROOT / "train"
npy_by_stem = {p.stem: p for p in TRAIN_DIR.rglob("*.npy") if not p.name.startswith(".")}
recs = [(p, npy_by_stem[p.stem]) for p in sorted(TRAIN_DIR.rglob("*.csv")) if p.stem in npy_by_stem]
assert len(recs) >= 20, "Train files are missing."

# ------------------------------------------------------------------ train IMU
REC = []
for csv_path, npy_path in recs:
    df = pd.read_csv(csv_path, usecols=["sbj_id"] + ACC, low_memory=False)
    acc = df[ACC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32).reshape(-1, 4, 3)
    n_vid = np.load(npy_path, mmap_mode="r").shape
    n_vid = n_vid[0] if n_vid[1] == 768 else n_vid[1]
    n_tiles = 0
    while (n_tiles + 1) * 50 <= len(acc) and 30 * n_tiles + 23 <= n_vid:
        n_tiles += 1
    REC.append(dict(name=csv_path.stem, sbj=int(df["sbj_id"].iloc[0]), acc=acc, n_tiles=n_tiles))
    log(csv_path.stem, "tiles", n_tiles)
    del df
for s in sorted({r["sbj"] for r in REC}):
    rs = [r for r in REC if r["sbj"] == s]
    a = np.concatenate([r["acc"] for r in rs]); fin = np.isfinite(a).all(2)
    mu = np.stack([a[fin[:, l], l].mean(0) for l in range(4)]); sd = np.stack([a[fin[:, l], l].std(0) + 1e-6 for l in range(4)])
    for r in rs:
        r["fin"] = torch.from_numpy(np.isfinite(r["acc"]).all(2)).to(DEV)
        r["g"] = torch.from_numpy(np.nan_to_num(r["acc"])).to(DEV)
        r["mu"] = torch.from_numpy(mu.astype(np.float32)).to(DEV); r["sd"] = torch.from_numpy(sd.astype(np.float32)).to(DEV)
        del r["acc"]
AR50 = torch.arange(50, device=DEV)

# ------------------------------------------------------------------ test IMU
meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
ti = np.asarray(np.load(TEST_DIR / "test_inertial_data.npy"), dtype=np.float32)
if ti.shape[1] == 3:
    ti = np.transpose(ti, (0, 2, 1))
ti = np.nan_to_num(ti)
tsb = meta["sbj_id"].to_numpy()
tlimb = meta["sensor_location"].map({l: i for i, l in enumerate(LIMBS)}).to_numpy().astype(int)
tmu = np.zeros((len(meta), 3), np.float32); tsd = np.ones((len(meta), 3), np.float32)
for s in np.unique(tsb):
    for l in range(4):
        m = (tsb == s) & (tlimb == l)
        if m.any():
            x = ti[m].reshape(-1, 3); tmu[m] = x.mean(0); tsd[m] = x.std(0) + 1e-6


def prep(w, mu, sd):
    """w (B,50,3) raw; mu/sd (B,3) -> (B,7,50)."""
    z = (w - mu[:, None]) / sd[:, None]
    return torch.cat([w, w.norm(dim=2, keepdim=True), z], 2).transpose(1, 2).contiguous()


TX = prep(torch.from_numpy(ti).to(DEV), torch.from_numpy(tmu).to(DEV), torch.from_numpy(tsd).to(DEV))
TL = torch.from_numpy(tlimb).to(DEV)
log("test windows", len(meta))


def windows(r, starts, limb):
    """IMU windows of one recording: starts (B,) sample index, limb (B,) -> (B,7,50)."""
    idx = starts[:, None] + AR50[None]
    w = r["g"][idx][torch.arange(len(starts), device=DEV), :, limb]          # (B,50,3)
    return prep(w, r["mu"][limb], r["sd"][limb])


def rand_limbs(r, starts, gen):
    fin = r["fin"][starts[:, None] + AR50[None]].all(1)                        # (B,4)
    score = torch.rand(len(starts), 4, device=DEV, generator=gen) + fin.float() * 2
    return score.argmax(1)


class Tower(nn.Module):
    def __init__(self, end, d=D_OUT):
        super().__init__()
        self.end = end
        conv = lambda i, o, k: nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.cnn = nn.Sequential(conv(7, 64, 7), conv(64, 64, 5), nn.MaxPool1d(2), conv(64, 128, 5), conv(128, 128, 3))
        self.limb = nn.Embedding(4, 16)
        self.head = nn.Sequential(nn.Linear(128 * 3 + 16 + 4 * EDGE * 4, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d))

    def forward(self, x, limb):
        h = self.cnn(x)                                                          # (B,128,25)
        edge_h = h[:, :, -3:].mean(2) if self.end else h[:, :, :3].mean(2)
        e = x[:, 3:7, -EDGE:] if self.end else x[:, 3:7, :EDGE]                  # z-scored axes + magnitude
        oh = F.one_hot(limb, 4).float()
        gated = (oh[:, :, None] * e.reshape(len(x), 1, -1)).reshape(len(x), -1)  # limb-gated edge samples
        z = torch.cat([h.mean(2), h.amax(2), edge_h, self.limb(limb), gated], 1)
        return F.normalize(self.head(z), dim=1)


@torch.no_grad()
def embed(bi, X, L, bs=4096):
    E, S = bi; ue, us = [], []
    for i in range(0, len(X), bs):
        ue.append(E(X[i:i + bs], L[i:i + bs]).float()); us.append(S(X[i:i + bs], L[i:i + bs]).float())
    return torch.cat(ue), torch.cat(us)


def train_bi(tr, seed, val=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    E, S = Tower(True).to(DEV), Tower(False).to(DEV)
    params = list(E.parameters()) + list(S.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)
    for it in range(STEPS):
        E.train(); S.train(); loss = 0.0
        for ri in rng.choice(len(tr), 2, replace=False):
            r = tr[ri]; off = int(rng.integers(50)); n = r["n_tiles"] - 1
            b = rng.integers(0, n - 9, 128)
            a = torch.from_numpy(np.unique((b[:, None] + np.arange(8)[None]).ravel())).to(DEV)
            sa = 50 * a + off; sb = sa + 50
            la = rand_limbs(r, sa, gen); lb = rand_limbs(r, sb, gen)
            ue = E(windows(r, sa, la), la); us = S(windows(r, sb, lb), lb)
            L = ue @ us.T / TAU; t = torch.arange(len(a), device=DEV)
            loss = loss + 0.5 * (F.cross_entropy(L, t) + F.cross_entropy(L.T, t))
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        if val is not None and (it + 1) % 500 == 0:
            E.eval(); S.eval()
            log(f"    seed {seed} step {it + 1}: loss {loss.item() / 2:.3f}  held-out top1 {np.mean([top1_rec((E, S), r) for r in val]):.4f}")
    E.eval(); S.eval()
    return E, S


@torch.no_grad()
def top1_rec(bi, r, bis=None):
    n = r["n_tiles"]; st = 50 * torch.arange(n, device=DEV)
    gen = torch.Generator(device=DEV); gen.manual_seed(123)
    lb = rand_limbs(r, st, gen); X = windows(r, st, lb)
    S = None
    for m in (bis or [bi]):
        ue, us = embed(m, X, lb); s = ue @ us.T; S = s if S is None else S + s
    S.fill_diagonal_(-1e4)
    return (S[:-1].argmax(1) == torch.arange(1, n, device=DEV)).float().mean().item()


@torch.no_grad()
def emb_all_limbs(bis, r):
    n = r["n_tiles"]; st = 50 * torch.arange(n, device=DEV)
    out = np.zeros((n, 4, len(bis), 2, D_OUT), np.float16)
    for l in range(4):
        L = torch.full((n,), l, device=DEV, dtype=torch.long); X = windows(r, st, L)
        for k, bi in enumerate(bis):
            ue, us = embed(bi, X, L); out[:, l, k, 0] = ue.cpu().numpy(); out[:, l, k, 1] = us.cpu().numpy()
    return out

# ------------------------------------------------------------------ cross-validation
EMB = {}
for fold in range(5):
    tr = [r for r in REC if FOLD[r["sbj"]] != fold]; va = [r for r in REC if FOLD[r["sbj"]] == fold]
    bis = [train_bi(tr, 100 * fold + s, va if fold == 0 else None) for s in SEEDS]
    for r in va:
        EMB[r["name"]] = emb_all_limbs(bis, r)
    log(f"fold {fold}: held-out top1 (random limbs, {len(SEEDS)} seeds) {np.mean([top1_rec(None, r, bis) for r in va]):.4f}")
    del bis; torch.cuda.empty_cache()

bis = [train_bi(REC, 1000 + s) for s in SEEDS]
TEST_EMB = np.zeros((len(meta), len(SEEDS), 2, D_OUT), np.float16)
for k, bi in enumerate(bis):
    ue, us = embed(bi, TX, TL); TEST_EMB[:, k, 0] = ue.cpu().numpy(); TEST_EMB[:, k, 1] = us.cpu().numpy()


def put(zf, name, arr):
    b = io.BytesIO(); np.save(b, arr); zf.writestr(name, b.getvalue())

with zipfile.ZipFile(OUT / "wear_imu_link.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for nm in EMB:
        put(zf, f"train/{nm}_emb.npy", EMB[nm])
    put(zf, "test_emb.npy", TEST_EMB)
    zf.writestr("meta.json", json.dumps({"recs": [(r["name"], r["sbj"], r["n_tiles"]) for r in REC], "seeds": SEEDS, "d": D_OUT}))
log("wrote wear_imu_link.zip", round((OUT / "wear_imu_link.zip").stat().st_size / 1e6, 1), "MB")
