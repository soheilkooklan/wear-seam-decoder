"""Small numpy MLP classifier (Adam, dropout, label smoothing, class weights)."""
from __future__ import annotations
import numpy as np


def _gelu(x):
    return np.maximum(x, 0)


def _gelu_grad(x):
    return (x > 0).astype(np.float32)


class MLP:
    def __init__(self, sizes, seed=0, dropout=0.2, in_dropout=0.0, wd=1e-4, lr=1e-3,
                 smooth=0.05, epochs=20, batch=512, verbose=False):
        self.sizes = list(sizes); self.seed = seed; self.dropout = dropout; self.in_dropout = in_dropout
        self.wd = wd; self.lr = lr; self.smooth = smooth; self.epochs = epochs; self.batch = batch
        self.verbose = verbose

    def _init(self, rng):
        self.W, self.b = [], []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))
            self.b.append(np.zeros(b, dtype=np.float32))
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.step = 0

    def fit(self, X, y, sample_weight=None, n_classes=None, X_val=None, y_val=None, callback=None):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        k = n_classes or int(y.max()) + 1
        assert self.sizes[-1] == k
        self._init(rng)
        sw = np.ones(len(y), np.float32) if sample_weight is None else np.asarray(sample_weight, np.float32)
        n = len(Xs); steps_per = int(np.ceil(n / self.batch)); total = steps_per * self.epochs
        Y = np.full((n, k), self.smooth / k, dtype=np.float32)
        Y[np.arange(n), y] += 1.0 - self.smooth
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            loss_sum = 0.0
            for i in range(steps_per):
                idx = perm[i * self.batch:(i + 1) * self.batch]
                lr = self.lr * 0.5 * (1 + np.cos(np.pi * self.step / total))
                loss_sum += self._train_step(Xs[idx], Y[idx], sw[idx], lr, rng)
            if self.verbose:
                msg = f"epoch {ep+1}/{self.epochs} loss {loss_sum/steps_per:.4f}"
                if callback is not None:
                    msg += " " + str(callback(self))
                print(msg, flush=True)
        return self

    def _train_step(self, x, Y, sw, lr, rng):
        if self.in_dropout > 0:
            x = x * (rng.random(x.shape, dtype=np.float32) > self.in_dropout) / (1 - self.in_dropout)
        acts, pres, masks = [x], [], []
        h = x
        L = len(self.W)
        for li in range(L):
            z = h @ self.W[li] + self.b[li]
            if li < L - 1:
                pres.append(z)
                h = _gelu(z)
                if self.dropout > 0:
                    m = (rng.random(h.shape, dtype=np.float32) > self.dropout) / (1 - self.dropout)
                    h = h * m; masks.append(m)
                else:
                    masks.append(None)
                acts.append(h)
            else:
                logits = z
        logits = logits - logits.max(1, keepdims=True)
        e = np.exp(logits); p = e / e.sum(1, keepdims=True)
        w = sw / sw.sum()
        loss = float(-(w[:, None] * Y * np.log(p + 1e-9)).sum())
        g = (p - Y) * w[:, None]
        grads_W = [None] * L; grads_b = [None] * L
        for li in range(L - 1, -1, -1):
            grads_W[li] = acts[li].T @ g + self.wd * self.W[li]
            grads_b[li] = g.sum(0)
            if li > 0:
                g = g @ self.W[li].T
                if masks[li - 1] is not None:
                    g = g * masks[li - 1]
                g = g * _gelu_grad(pres[li - 1])
        self.step += 1
        b1, b2 = 0.9, 0.999
        params = self.W + self.b; grads = grads_W + grads_b
        for j, (p_, g_) in enumerate(zip(params, grads)):
            self.m[j] = b1 * self.m[j] + (1 - b1) * g_
            self.v[j] = b2 * self.v[j] + (1 - b2) * g_ * g_
            mh = self.m[j] / (1 - b1 ** self.step); vh = self.v[j] / (1 - b2 ** self.step)
            p_ -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return loss

    def logits(self, X, batch=8192):
        X = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        out = []
        for i in range(0, len(X), batch):
            h = X[i:i + batch]
            for li in range(len(self.W)):
                h = h @ self.W[li] + self.b[li]
                if li < len(self.W) - 1:
                    h = _gelu(h)
            out.append(h)
        return np.concatenate(out)

    def predict_proba(self, X):
        z = self.logits(X); z = z - z.max(1, keepdims=True)
        e = np.exp(z); return e / e.sum(1, keepdims=True)
