"""Helpers of backward learning over the remaining-goal length, and two baselines built on them.

Goal-sequence reuse: a transition (x, g, y, x+) whose start x is near goal `prev` serves every valid
remaining-goal sequence (g, g_2, ..., g_L) ("suffix"): a non-backtracking walk on the goal graph starting
with g (g_2 != prev), matching how test sequences are generated. Its target for length L is
    y + W_{L-1}(x+; (g_2, ..., g_L)),     W_0 = 0,
with W_{L-1} the frozen value from the previous length.

Baselines here:
  target-only                     : the same backward Bellman learning on the real side of the paired data only
                                    (no simulation data).
  segment-wise outcome correction : SCAPE-style h_out(x, g, y_S) -> y_R, summed along simulation rollouts of the
                                    target sequence (J_seg; ignores the terminal-state mismatch).
"""
import numpy as np

from . import maze as M
from .data import nearest_goal
from .models import LMAX, X_DIM, PerLength, Regressor, encode, frozen


def sample_suffixes(prev, g, L, rng, n):
    """Up to n distinct non-backtracking walks of length L starting with g (g_2 != prev)."""
    out, seen = [], set()
    for _ in range(4 * n):
        s, p, c = [g], prev, g
        for _ in range(L - 1):
            nb = [m for m in M.NEIGHBORS[c] if m != p]
            p, c = c, nb[rng.integers(len(nb))]
            s.append(c)
        t = tuple(s)
        if t not in seen:
            seen.add(t); out.append(s)
            if len(out) == n:
                break
    return out


def build_rows(recs, env, cost, L, rng, n_suf):
    """Rows (x, suffix) of suffix length L with the env's local cost and terminal state.
    Returns X, suffixes, y_local, x_plus, tails, rec_index."""
    X, S, Y, XP, T, I = [], [], [], [], [], []
    for i, r in enumerate(recs):
        prev = nearest_goal(r["x"][:2])
        if prev == r["g"]:  # start already at the goal's own position; treat the goal it left as unknown
            prev = None
        for s in sample_suffixes(prev, r["g"], L, rng, n_suf):
            X.append(r["x"]); S.append(s); Y.append(r[env][f"y_{cost}"]); XP.append(r[env]["x_plus"]); T.append(s[1:]); I.append(i)
    return np.array(X), S, np.array(Y), np.array(XP), T, np.array(I)


def W_eval(W, X, tails):
    """W(x; tail) with W(.; []) = 0."""
    out = np.zeros(len(X))
    idx = [i for i, t in enumerate(tails) if len(t) > 0]
    if W is not None and idx:
        out[idx] = W.predict(encode(X[idx], [tails[i] for i in idx]))
    return out


def predict_W(W, x1s, routes):
    return W.predict(encode(np.array(x1s), [list(r) for r in routes]))


def train_target_only(paired, paired_val, cost, seed=0, L_max=LMAX, n_suf=6, per_length=False):
    """Target-only: W trained on real targets y_R + W_{L-1}(x_R+; tail) from paired data only.
    per_length=True: a separate W_L for each remaining-goal length (variant)."""
    rng = np.random.default_rng(seed)
    W, W_prev = Regressor(X_DIM + 3 * LMAX, seed=seed), None
    W_by_L = {}
    hist = []
    for L in range(1, L_max + 1):
        if per_length:
            W = Regressor(X_DIM + 3 * LMAX, seed=seed + 1000 * L)
        F, Yt, Fv, Yv = [], [], [], []
        for ell in ([L] if per_length else range(1, L + 1)):  # shared: refit all lengths <= L with the frozen previous model
            for recs, Fo, Yo in ((paired, F, Yt), (paired_val, Fv, Yv)):
                X, S, y, XP, T, _ = build_rows(recs, "R", cost, ell, rng, n_suf)
                Fo.append(encode(X, S)); Yo.append(y + W_eval(W_prev, XP, T))
        info = W.fit(np.vstack(F), np.concatenate(Yt), np.vstack(Fv), np.concatenate(Yv), seed=seed + L)
        hist.append(dict(L=L, **info))
        if per_length:
            W_by_L[L] = frozen(W)
            W_prev = PerLength(W_by_L)
        else:
            W_prev = frozen(W)
    return W_prev, hist


def train_outcome_calibrator(paired, paired_val, cost, seed=0):
    """Segment-wise outcome correction (SCAPE-style): h_out(x, g, y_S) ~ E[y_R | x, g, y_S]."""
    def feats(recs):
        return encode(np.array([r["x"] for r in recs]), [[r["g"]] for r in recs],
                      extra=np.array([[r["S"][f"y_{cost}"]] for r in recs]))
    y = np.array([r["R"][f"y_{cost}"] for r in paired]); yv = np.array([r["R"][f"y_{cost}"] for r in paired_val])
    h = Regressor(X_DIM + 3 * LMAX + 1, seed=seed)
    info = h.fit(feats(paired), y, feats(paired_val), yv, seed=seed)
    return h, info


def outcome_only_estimate(h, rollouts, goals, cost):
    """J_seg: mean over target-sequence simulation rollouts of sum_k h_out(X_k, g_k, y_k^S)."""
    tot = []
    for r in rollouts:
        F = encode(r["X"][:-1], [[g] for g in goals], extra=r["y"][cost][:, None])
        tot.append(h.predict(F).sum())
    return float(np.mean(tot))
