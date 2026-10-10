"""Residual variant of backward Bellman calibration (METHODOLOGY section 7): W_R(x; s) = W_S(x; s) + D(x; s).

W_S is computed by running the simulator from x along the suffix (MC_REPS rollouts); only the residual D is learned:
    D(x; s) <- (y_R - y_S) + [W_S(x_R+; tail) - W_S(x_S+; tail)] + D_prev(x_R+; tail)
               local gap     terminal-state effect (sim value)      propagated residual
Calibration model for simulation rows (same role as h_phi in safer_av/ours.py):
    h_D(x, s, Z_S) -> D target,   Z_S = y_S + W_S(x_S+; tail)
Residual direct estimate:  J_R(x1; g) ~= J_S(x1; g) [target-sequence simulation rollouts] + D(x1; g).

Suffixes are sampled once per transition and length (fixed rows), so all Monte Carlo values are precomputed
once per dataset (scripts/make_rows.py) and shared by the five costs.
"""
import numpy as np

from .backward import sample_suffixes
from .data import _W, nearest_goal
from .models import LMAX, Regressor, encode, frozen
from .ours import Z_SCALE, paired_repeats
from .route import run_route
from .segment import COSTS

MC_REPS = 2


# ---------------- Monte Carlo sim values (pool workers use data.init_worker) ----------------
def ws_jobs(args):
    """args = (seed, [(x, tail), ...]) -> array (n, len(COSTS)) of mean route costs over MC_REPS sim rollouts."""
    seed, jobs = args
    _W["S"].reseed(seed)
    out = np.zeros((len(jobs), len(COSTS)))
    for j, (x, tail) in enumerate(jobs):
        if not tail:
            continue
        acc = np.zeros(len(COSTS))
        for _ in range(MC_REPS):
            _, infos = run_route(_W["S"], _W["pol"], x, tail, _W["cfg"])
            acc += [sum(i[f"y_{c}"] for i in infos) for c in COSTS]
        out[j] = acc / MC_REPS
    return out


def mc_values(pool, jobs, seed, chunk=200):
    rng = np.random.default_rng(seed)
    parts = pool.map(ws_jobs, [(int(rng.integers(1 << 31)), jobs[i:i + chunk]) for i in range(0, len(jobs), chunk)])
    return np.vstack(parts) if parts else np.zeros((0, len(COSTS)))


def make_rows(recs, paired, n_suf, seed, L_max=LMAX):
    """One row per (transition, sampled suffix), lengths 1..L_max."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, r in enumerate(recs):
        prev = nearest_goal(r["x"][:2])
        prev = None if prev == r["g"] else prev
        for L in range(1, L_max + 1):
            for s in sample_suffixes(prev, r["g"], L, rng, n_suf):
                row = dict(i=i, x=r["x"], s=s, tail=s[1:], L=L, yS={c: r["S"][f"y_{c}"] for c in COSTS}, xS=r["S"]["x_plus"])
                if paired:
                    row.update(yR={c: r["R"][f"y_{c}"] for c in COSTS}, xR=r["R"]["x_plus"])
                rows.append(row)
    return rows


def attach_mc(pool, rows, paired, seed):
    """Adds W_S at the sim terminal state (and at the real terminal state for paired rows)."""
    jobs = [(r["xS"], r["tail"]) for r in rows]
    if paired:
        jobs += [(r["xR"], r["tail"]) for r in rows]
    V = mc_values(pool, jobs, seed)
    n = len(rows)
    for k, r in enumerate(rows):
        r["WS_S"] = dict(zip(COSTS, V[k]))
        if paired:
            r["WS_R"] = dict(zip(COSTS, V[n + k]))
    return rows


# ---------------- training ----------------
def _D_eval(D, X, tails):
    out = np.zeros(len(X))
    idx = [i for i, t in enumerate(tails) if t]
    if D is not None and idx:
        out[idx] = D.predict(encode(np.asarray(X)[idx], [tails[i] for i in idx]))
    return out


def train_residual(p_rows, v_rows, s_rows, cost, seed=0):
    d = 4 + 3 * LMAX
    D, h, D_prev = Regressor(d, seed=seed), Regressor(d + 1, seed=seed + 100), None
    arr = lambda rows, f: np.array([f(r) for r in rows])
    hist = []
    for L in range(1, LMAX + 1):
        P = [r for r in p_rows if r["L"] <= L]; V = [r for r in v_rows if r["L"] <= L]; S = [r for r in s_rows if r["L"] <= L]

        def targets(rows):
            loc = arr(rows, lambda r: r["yR"][cost] - r["yS"][cost])
            term = arr(rows, lambda r: r["WS_R"][cost] - r["WS_S"][cost])
            prop = _D_eval(D_prev, arr(rows, lambda r: r["xR"]), [r["tail"] for r in rows])
            return loc + term + prop
        zs = lambda rows: arr(rows, lambda r: r["yS"][cost] + r["WS_S"][cost])
        F = lambda rows: encode(arr(rows, lambda r: r["x"]), [r["s"] for r in rows])
        Fh = lambda rows: encode(arr(rows, lambda r: r["x"]), [r["s"] for r in rows], extra=(zs(rows) * Z_SCALE[cost])[:, None])
        tP, tV = targets(P), targets(V)
        Fp, Fv = F(P), F(V)
        hinfo = h.fit(Fh(P), tP, Fh(V), tV, seed=seed + L)
        Fs, tS = F(S), h.predict(Fh(S))
        rep = paired_repeats(len(Fp), len(Fs))
        F_all, t_all = np.vstack([np.repeat(Fp, rep, axis=0), Fs]), np.concatenate([np.repeat(tP, rep), tS])
        info = D.fit(F_all, t_all, Fv, tV, batch=4096, seed=seed + 10 * L)
        hist.append(dict(L=L, D_val_mse=info["val_mse"], h_val_mse=hinfo["val_mse"], n_paired=len(P),
                         target_mean=float(tP.mean()), target_std=float(tP.std())))
        D_prev = frozen(D)
    return D_prev, hist


def predict_residual(D, x1s, routes, raw_sim):
    return np.asarray(raw_sim) + D.predict(encode(np.array(x1s), [list(r) for r in routes]))
