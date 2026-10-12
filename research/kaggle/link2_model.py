# WEAR 2026: deep "next second" bi-encoder (GPU, about 40 minutes on a T4 / P100).
# An end-of-window tower and a start-of-window tower (video only), trained with InfoNCE.
# Every one of the 30 possible frame phases is used as augmentation (test windows use phase 0).
# Training is kept short on purpose: longer runs memorise the training routes and generalise worse.
# Writes out-of-fold outputs (train tiles) and test outputs to /kaggle/working/wear_link.zip
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
torch.backends.cuda.matmul.allow_tf32 = True
FOLD = {1: 0, 6: 0, 11: 0, 16: 0, 21: 0, 0: 1, 5: 1, 10: 1, 15: 1, 20: 1, 4: 2, 9: 2, 14: 2, 19: 2,
        3: 3, 8: 3, 13: 3, 18: 3, 2: 4, 7: 4, 12: 4, 17: 4}
SEEDS = [0, 1, 2]
D_OUT = 128          # embedding size
TAU = 0.1            # InfoNCE temperature
BI_STEPS = 1200      # bi-encoder optimisation steps (longer runs overfit the training routes)
CR_STEPS = 4000      # cross-encoder optimisation steps
N_CAND = 16          # candidates per anchor while training the cross-encoder
K_SCORE = 20         # candidates per window (both directions) scored at inference
OUT = Path("/kaggle/working")

hits = sorted(Path("/kaggle/input").rglob("test_meta_data.csv"))
assert hits, "Competition data is not attached."
TEST_DIR = hits[0].parent; ROOT = TEST_DIR.parent; TRAIN_DIR = ROOT / "train"
npy_by_stem = {p.stem: p for p in TRAIN_DIR.rglob("*.npy") if not p.name.startswith(".")}
recs = [(p, npy_by_stem[p.stem]) for p in sorted(TRAIN_DIR.rglob("*.csv")) if p.stem in npy_by_stem]
assert len(recs) >= 20, "Train VideoMAE .npy files are missing."

# ------------------------------------------------------------------ load train video
REC = []
for csv_path, npy_path in recs:
    sbj = int(pd.read_csv(csv_path, usecols=["sbj_id"], nrows=1)["sbj_id"].iloc[0])
    n_imu = sum(1 for _ in open(csv_path)) - 1
    v = np.load(npy_path, mmap_mode="r")
    if v.shape[1] != 768:
        v = v.T
    n_tiles = 0
    while (n_tiles + 1) * 50 <= n_imu and 30 * n_tiles + 23 <= v.shape[0]:
        n_tiles += 1
    REC.append(dict(name=csv_path.stem, sbj=sbj, v=np.nan_to_num(np.asarray(v, dtype=np.float32)).astype(np.float16),
                    n_tiles=n_tiles))
    log(csv_path.stem, "subject", sbj, "tiles", n_tiles, "frames", v.shape[0])

# subject-level standardisation; the frames live on the GPU as float16
for s in sorted({r["sbj"] for r in REC}):
    rs = [r for r in REC if r["sbj"] == s]
    fr = np.concatenate([r["v"][::5].astype(np.float32) for r in rs])
    mu, sd = fr.mean(0), fr.std(0) + 1e-6
    for r in rs:
        r["g"] = torch.from_numpy(((r["v"].astype(np.float32) - mu) / sd).astype(np.float16)).to(DEV)
        del r["v"]
log("train video on GPU", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")

# ------------------------------------------------------------------ test video
meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
tv = np.asarray(np.load(TEST_DIR / "test_videomae_data.npy", mmap_mode="r"), dtype=np.float32)
if tv.shape[1] == 768:
    tv = np.transpose(tv, (0, 2, 1))
tv = np.nan_to_num(tv)
tsb = meta["sbj_id"].to_numpy()
TG = {}
for s in np.unique(tsb):
    m = np.flatnonzero(tsb == s); x = tv[m]; fr = x.reshape(-1, 768)
    TG[s] = (m, torch.from_numpy(((x - fr.mean(0)) / (fr.std(0) + 1e-6)).astype(np.float16)).to(DEV))
del tv
log("test windows", len(meta), "subjects", len(TG))

AR15 = torch.arange(15, device=DEV)


def grid(r, phase):
    """Start frames of the 1-second windows of a recording for a given frame phase (phase 0 = the tiles)."""
    n = r["n_tiles"] if phase == 0 else r["n_tiles"] - 1
    return 30 * torch.arange(n, device=DEV) + 8 + phase


def clips(g, starts):
    return g[starts[:, None] + AR15[None, :]]            # (B, 15, 768) float16


# ------------------------------------------------------------------ models
def encoder(h, layers):
    layer = nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, layers)


class Tower(nn.Module):
    """end=True embeds how a window ends, end=False how it starts. Linear branch on hand-made summaries
    (the model that already works offline) plus a small transformer branch."""
    def __init__(self, end, h=256, d=D_OUT):
        super().__init__()
        self.end = end
        self.lin = nn.Linear(4 * 768, d)
        self.inp = nn.Linear(768, h)
        self.pos = nn.Parameter(0.02 * torch.randn(1, 15, h))
        self.enc = encoder(h, 2)
        self.head = nn.Sequential(nn.LayerNorm(3 * h), nn.Linear(3 * h, d))

    def forward(self, x):
        if self.end:
            hand = torch.cat([x[:, -3:].mean(1), x[:, -1], x[:, -1] - x[:, -5], x.mean(1)], 1)
        else:
            hand = torch.cat([x[:, :3].mean(1), x[:, 0], x[:, 4] - x[:, 0], x.mean(1)], 1)
        z = self.enc(self.inp(F.dropout(x, 0.1, self.training)) + self.pos)
        if self.end:
            pool = torch.cat([z[:, -1], z[:, -3:].mean(1), z.mean(1)], 1)
        else:
            pool = torch.cat([z[:, 0], z[:, :3].mean(1), z.mean(1)], 1)
        return F.normalize(self.lin(F.dropout(hand, 0.1, self.training)) + self.head(pool), dim=1)


class Cross(nn.Module):
    """Scores 'clip B is the second right after clip A' from both clips jointly."""
    def __init__(self, h=256):
        super().__init__()
        self.inp = nn.Linear(768, h)
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, h))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 31, h))
        self.enc = encoder(h, 3)
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))

    def forward(self, xa, xb):
        z = self.inp(F.dropout(torch.cat([xa, xb], 1), 0.1, self.training))
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos
        return self.out(self.enc(z)[:, 0]).squeeze(1)


@torch.no_grad()
def embed(bi, X, bs=2048):
    """bi = (end tower, start tower); X (n,15,768) float16 -> (n,D), (n,D) float32."""
    E, S = bi; ue, us = [], []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].float()
        with torch.autocast("cuda", dtype=torch.float16):
            ue.append(E(x).float()); us.append(S(x).float())
    return torch.cat(ue), torch.cat(us)


@torch.no_grad()
def sim_matrix(bis, X):
    S = None
    for bi in bis:
        ue, us = embed(bi, X); s = ue @ us.T
        S = s if S is None else S + s
    S = S / len(bis); S.fill_diagonal_(-1e4)
    return S


def top1(S):
    n = len(S)
    return (S[:-1].argmax(1) == torch.arange(1, n, device=DEV)).float().mean().item()


def train_bi(tr, seed, val=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    E, S = Tower(True).to(DEV), Tower(False).to(DEV)
    params = list(E.parameters()) + list(S.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=BI_STEPS, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    for it in range(BI_STEPS):
        E.train(); S.train(); loss = 0.0
        for ri in rng.choice(len(tr), 2, replace=False):
            r = tr[ri]; st = grid(r, int(rng.integers(30))); n = len(st)
            b = rng.integers(0, n - 9, 128)
            a = torch.from_numpy(np.unique((b[:, None] + np.arange(8)[None]).ravel())).to(DEV)
            xa = clips(r["g"], st[a]).float(); xb = clips(r["g"], st[a + 1]).float()
            xa = xa + 0.1 * torch.randn_like(xa); xb = xb + 0.1 * torch.randn_like(xb)
            with torch.autocast("cuda", dtype=torch.float16):
                ue, us = E(xa), S(xb)
            L = ue.float() @ us.float().T / TAU; t = torch.arange(len(a), device=DEV)
            loss = loss + 0.5 * (F.cross_entropy(L, t) + F.cross_entropy(L.T, t))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if it == 199:
            log(f"    bi seed {seed}: 200 steps, loss {loss.item() / 2:.3f}")
        if val is not None and (it + 1) % 300 == 0:
            E.eval(); S.eval()
            t1 = np.mean([top1(sim_matrix([(E, S)], clips(r["g"], grid(r, 0)))) for r in val])
            log(f"    bi seed {seed} step {it + 1}: loss {loss.item() / 2:.3f}  held-out top1 {t1:.4f}")
    E.eval(); S.eval()
    return E, S


def train_cross(tr, bis, seed, val=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    C = Cross().to(DEV)
    opt = torch.optim.AdamW(C.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=CR_STEPS, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    for it in range(CR_STEPS):
        C.train()
        r = tr[rng.integers(len(tr))]; X = clips(r["g"], grid(r, int(rng.integers(30)))); n = len(X)
        S = sim_matrix(bis, X)
        a = torch.from_numpy(rng.choice(n - 1, min(256, n - 1), replace=False)).to(DEV)
        cand = S[a].topk(N_CAND, 1).indices; pos = a + 1
        miss = ~(cand == pos[:, None]).any(1)
        cand[miss, -1] = pos[miss]
        tgt = (cand == pos[:, None]).float().argmax(1)
        xa = X[a].float()[:, None].expand(-1, N_CAND, -1, -1).reshape(-1, 15, 768)
        xb = X[cand.reshape(-1)].float()
        xa = xa + 0.1 * torch.randn_like(xa); xb = xb + 0.1 * torch.randn_like(xb)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = C(xa, xb).float().view(-1, N_CAND)
        loss = F.cross_entropy(logit, tgt)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(C.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if it == 199:
            log(f"    cross: 200 steps, loss {loss.item():.3f}")
        if val is not None and (it + 1) % 1000 == 0:
            C.eval()
            log(f"    cross step {it + 1}: loss {loss.item():.3f}  held-out top1 "
                f"{np.mean([score_unit(bis, C, clips(r['g'], grid(r, 0)))[3] for r in val]):.4f}")
    C.eval()
    return C


@torch.no_grad()
def score_unit(bis, C, X):
    """Candidate pairs (top-K successors of every window and top-K predecessors of every window) and cross scores."""
    S = sim_matrix(bis, X); n = len(X); k = min(K_SCORE, n - 1)
    s_idx = S.topk(k, 1).indices; p_idx = S.topk(k, 0).indices
    ar = torch.arange(n, device=DEV)
    a = torch.cat([ar[:, None].expand(-1, k).reshape(-1), p_idx.reshape(-1)])
    b = torch.cat([s_idx.reshape(-1), ar[None, :].expand(k, -1).reshape(-1)])
    pr = torch.unique(torch.stack([a, b], 1), dim=0)
    sc = []
    for i in range(0, len(pr), 8192):
        p = pr[i:i + 8192]
        with torch.autocast("cuda", dtype=torch.float16):
            sc.append(C(X[p[:, 0]].float(), X[p[:, 1]].float()).float())
    sc = torch.cat(sc)
    # cross top-1: best scored successor of each window among its candidates
    best = torch.full((n,), -1e9, device=DEV).scatter_reduce(0, pr[:, 0], sc, reduce="amax")
    win = sc >= best[pr[:, 0]]                                   # pairs holding the best score of their anchor
    hit = torch.zeros(n, device=DEV).index_put_((pr[win, 0],), (pr[win, 1] == pr[win, 0] + 1).float(), accumulate=True)
    t1 = (hit[:-1] > 0).float().mean().item()
    return S, pr.cpu().numpy().astype(np.int32), sc.cpu().numpy().astype(np.float32), t1


def emb_array(bis, X):
    out = np.zeros((len(X), len(bis), 2, D_OUT), np.float16)
    for k, bi in enumerate(bis):
        ue, us = embed(bi, X); out[:, k, 0] = ue.cpu().numpy(); out[:, k, 1] = us.cpu().numpy()
    return out

# ------------------------------------------------------------------ cross-validation (out-of-fold)
EMB = {}
for fold in range(5):
    tr = [r for r in REC if FOLD[r["sbj"]] != fold]; va = [r for r in REC if FOLD[r["sbj"]] == fold]
    log(f"fold {fold}: train recordings {len(tr)}, held-out {len(va)}")
    bis = [train_bi(tr, 100 * fold + s, va if fold == 0 else None) for s in SEEDS]
    t_bi = []
    for r in va:
        X = clips(r["g"], grid(r, 0))
        EMB[r["name"]] = emb_array(bis, X); t_bi.append(top1(sim_matrix(bis, X)))
    log(f"fold {fold}: held-out top1  bi-encoder ({len(SEEDS)} seeds) {np.mean(t_bi):.4f}")
    del bis; torch.cuda.empty_cache()

# ------------------------------------------------------------------ test (models trained on all recordings)
bis = [train_bi(REC, 1000 + s) for s in SEEDS]
TEST_EMB = np.zeros((len(meta), len(SEEDS), 2, D_OUT), np.float16)
for s, (m, X) in TG.items():
    TEST_EMB[m] = emb_array(bis, X)
    log("test subject", s, "windows", len(m))


def put(zf, name, arr):
    b = io.BytesIO(); np.save(b, arr); zf.writestr(name, b.getvalue())

with zipfile.ZipFile(OUT / "wear_link.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for nm in EMB:
        put(zf, f"train/{nm}_emb.npy", EMB[nm])
    put(zf, "test_emb.npy", TEST_EMB)
    zf.writestr("meta.json", json.dumps({"recs": [(r["name"], r["sbj"], r["n_tiles"]) for r in REC],
                                         "seeds": SEEDS, "d": D_OUT}))
log("wrote wear_link.zip", round((OUT / "wear_link.zip").stat().st_size / 1e6, 1), "MB")
