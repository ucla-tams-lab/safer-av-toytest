"""Sequence-conditioned regressor shared by every learned method.

Input encoding for (x, suffix g_{k:K}):
  x / [3, 3, 5, 5]                                              (4)
  goal coordinates g_k..g_{k+L-1} / 3, zero-padded to LMAX goals (2 * LMAX)
  mask of valid goals                                           (LMAX)
plus optional extra scalars (e.g. the scaled simulation value Z_S for a calibration model).

Training budget (identical for every method and dataset size): gradient steps, not epochs. Training stops after
PATIENCE_STEPS steps without validation improvement, at most MAX_STEPS steps; the best-validation weights are kept.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from . import maze as M

LMAX = 8
X_SCALE = np.array([3.0, 3.0, 5.0, 5.0])
G_SCALE = 3.0
PATIENCE_STEPS, MAX_STEPS = 3000, 30000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(X, suffixes, extra=None):
    """X: (n,4); suffixes: list of goal-name lists (len 1..LMAX); extra: (n,e) or None."""
    n = len(X)
    G = np.zeros((n, LMAX, 2)); mask = np.zeros((n, LMAX))
    for i, s in enumerate(suffixes):
        L = len(s)
        G[i, :L] = [M.GOALS[g] for g in s]
        mask[i, :L] = 1.0
    feats = [np.asarray(X) / X_SCALE, G.reshape(n, -1) / G_SCALE, mask]
    if extra is not None:
        feats.append(np.asarray(extra).reshape(n, -1))
    return np.concatenate(feats, axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return self.net(f).squeeze(-1)


class Regressor:
    """MLP (3 x 256, SiLU) with target scaling, MSE loss, AdamW, early stopping on a validation set."""

    def __init__(self, d_in, hidden=256, depth=3, seed=0):
        torch.manual_seed(seed)
        self.model = MLP(d_in, hidden, depth).to(DEVICE)
        self.y_scale = 1.0

    def fit(self, F, y, F_val, y_val, lr=1e-3, wd=1e-5, batch=512, seed=0):
        g = torch.Generator().manual_seed(seed)
        steps_per_epoch = max(1, math.ceil(len(F) / batch))
        patience = max(1, math.ceil(PATIENCE_STEPS / steps_per_epoch))
        epochs = max(1, math.ceil(MAX_STEPS / steps_per_epoch))
        self.y_scale = float(max(np.std(y), 1e-3))
        Ft, yt = torch.tensor(F, device=DEVICE), torch.tensor(y / self.y_scale, dtype=torch.float32, device=DEVICE)
        Fv, yv = torch.tensor(F_val, device=DEVICE), torch.tensor(y_val / self.y_scale, dtype=torch.float32, device=DEVICE)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        best, best_state, bad = np.inf, None, 0
        n = len(Ft)
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n, generator=g).to(DEVICE)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                loss = ((self.model(Ft[idx]) - yt[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                v = ((self.model(Fv) - yv) ** 2).mean().item()
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return dict(val_mse=best * self.y_scale ** 2, epochs=ep + 1)

    def predict(self, F, batch=65536):
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(F), batch):
                out.append(self.model(torch.tensor(F[i:i + batch], device=DEVICE)).cpu().numpy())
        return np.concatenate(out) * self.y_scale if out else np.zeros(0)


def frozen(reg):
    """Deep copy of a fitted Regressor with gradients disabled (the previous stage's value in backward training)."""
    c = copy.deepcopy(reg)
    for p in c.model.parameters():
        p.requires_grad_(False)
    return c
