"""Numpy InfoNCE successor embedding: end-encoder(window i) . start-encoder(window j)."""
from __future__ import annotations
import numpy as np


def end_feats(v, variant='A'):
    if variant == 'A':
        return np.concatenate([v[:, -3:].mean(1), v[:, -1], v[:, -1] - v[:, -5], v.mean(1)], 1)
    # B: last 5 frames individually + velocity + mean + std
    return np.concatenate([v[:, -5:].reshape(len(v), -1), v[:, -1] - v[:, -8], v.mean(1), v.std(1)], 1)


def start_feats(v, variant='A'):
    if variant == 'A':
        return np.concatenate([v[:, :3].mean(1), v[:, 0], v[:, 4] - v[:, 0], v.mean(1)], 1)
    return np.concatenate([v[:, :5].reshape(len(v), -1), v[:, 7] - v[:, 0], v.mean(1), v.std(1)], 1)


class Tower:
    """x -> (optional ReLU hidden) -> linear -> unit-normalised embedding."""
    def __init__(self, din, h, out, rng, hidden=True):
        self.hidden = hidden
        if hidden:
            self.W1 = (rng.standard_normal((din, h)) * np.sqrt(2 / din)).astype(np.float32); self.b1 = np.zeros(h, np.float32)
            self.W2 = (rng.standard_normal((h, out)) / np.sqrt(h)).astype(np.float32)
        else:
            self.W2 = (rng.standard_normal((din, out)) / np.sqrt(din)).astype(np.float32)
        self.params = [self.W1, self.b1, self.W2] if hidden else [self.W2]
        self.m = [np.zeros_like(p) for p in self.params]; self.v = [np.zeros_like(p) for p in self.params]

    def forward(self, x, train=False, rng=None, drop=0.1):
        self.x = x
        if self.hidden:
            self.a = x @ self.W1 + self.b1; self.h = np.maximum(self.a, 0)
            if train and drop > 0:
                self.mask = (rng.random(self.h.shape, dtype=np.float32) > drop) / (1 - drop); self.h = self.h * self.mask
            else:
                self.mask = None
            z = self.h @ self.W2
        else:
            z = x @ self.W2
        self.n = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        self.u = z / self.n
        return self.u

    def backward(self, gu, wd=1e-4):
        gz = (gu - self.u * (gu * self.u).sum(1, keepdims=True)) / self.n
        if self.hidden:
            gW2 = self.h.T @ gz; gh = gz @ self.W2.T
            if self.mask is not None: gh = gh * self.mask
            ga = gh * (self.a > 0)
            gW1 = self.x.T @ ga; gb1 = ga.sum(0)
            grads = [gW1 + wd * self.W1, gb1, gW2 + wd * self.W2]
        else:
            grads = [self.x.T @ gz + wd * self.W2]
        return grads

    def adam(self, grads, lr, step):
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            p -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)


def train_contrastive(data, rec_list, iters=4000, hidden=True, h=512, out=128, tau=0.07, lr=1e-3, seed=0,
                      blocks=64, block_len=8, log=None, eval_fn=None, gap=1):
    """data[rec] = (end_feats (n,d), start_feats (n,d)) in time order."""
    rng = np.random.default_rng(seed)
    din = data[rec_list[0]][0].shape[1]
    te = Tower(din, h, out, rng, hidden); ts = Tower(din, h, out, rng, hidden)
    for it in range(iters):
        r = rec_list[rng.integers(len(rec_list))]; xe, xs = data[r]; n = len(xe)
        st = rng.integers(0, n - block_len - gap, blocks); a = (st[:, None] + np.arange(block_len)[None]).ravel()
        ue = te.forward(xe[a], True, rng); us = ts.forward(xs[a + gap], True, rng)
        L = ue @ us.T / tau; B = len(a)
        Pr = np.exp(L - L.max(1, keepdims=True)); Pr /= Pr.sum(1, keepdims=True)
        Pc = np.exp(L - L.max(0, keepdims=True)); Pc /= Pc.sum(0, keepdims=True)
        G = ((Pr - np.eye(B, dtype=np.float32)) + (Pc - np.eye(B, dtype=np.float32))) / (B * tau)
        ge = te.backward(G @ us); gs = ts.backward(G.T @ ue)
        te.adam(ge, lr, it + 1); ts.adam(gs, lr, it + 1)
        if eval_fn is not None and (it + 1) % 1000 == 0:
            log(f'iter {it+1} val top1 {eval_fn(te, ts):.4f}')
    return te, ts


def embed(te, ts, xe, xs, bs=4096):
    ue = np.concatenate([te.forward(xe[i:i + bs]) for i in range(0, len(xe), bs)])
    us = np.concatenate([ts.forward(xs[i:i + bs]) for i in range(0, len(xs), bs)])
    return ue, us
