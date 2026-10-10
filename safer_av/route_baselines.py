"""Bellman-free long-horizon baselines learned from the SAME real chains (equal real budget).

Windows: every (chain, start step i, length L) inside the first n paired transitions gives an observed real
return J_R(x_i; g_i..g_{i+L-1}) = sum of y_R along the chain. For each window the sim return J_S is computed by
running the simulator from x_i along the same goals (MC_REPS rollouts; scripts/make_rows.py).
  B1  MC real regression : J_R(x; suffix) regressed directly (no bootstrapping, no suffix reuse)
  B2  route-level SCAPE  : h_route(x, suffix, J_S) -> J_R ; at test time J_S comes from test-route sim rollouts
"""
import numpy as np

from .data import _W
from .models import LMAX, Regressor, encode
from .ours import Z_SCALE
from .route import run_route
from .segment import COSTS

MC_REPS = 3


def windows(recs):
    """All (start, length) windows of the chains in `recs` (chains are consecutive records, step 0 starts one)."""
    chains, cur = [], None
    for r in recs:
        if r["step"] == 0 or cur is None:
            cur = []; chains.append(cur)
        cur.append(r)
    out = []
    for c in chains:
        for i in range(len(c)):
            for L in range(1, len(c) - i + 1):
                seg = c[i:i + L]
                out.append(dict(x=c[i]["x"], s=[r["g"] for r in seg], JR={k: float(sum(r["R"][f"y_{k}"] for r in seg)) for k in COSTS}))
    return out


def js_job(args):
    """args = (seed, [(x, suffix), ...]) -> mean sim return per cost along each suffix (pool worker)."""
    seed, jobs = args
    _W["S"].reseed(seed); out = []
    for x, s in jobs:
        acc = np.zeros(len(COSTS))
        for _ in range(MC_REPS):
            _, inf = run_route(_W["S"], _W["pol"], x, s, _W["cfg"])
            acc += [sum(i[f"y_{k}"] for i in inf) for k in COSTS]
        out.append(acc / MC_REPS)
    return out


def attach_js(pool, ws, rng, chunk=50):
    J = [(w["x"], w["s"]) for w in ws]
    V = [v for c in pool.map(js_job, [(int(rng.integers(1 << 30)), J[i:i + chunk]) for i in range(0, len(J), chunk)]) for v in c]
    for w, v in zip(ws, V):
        w["JS"] = dict(zip(COSTS, map(float, v)))
    return ws


def _F(ws):
    return encode(np.array([w["x"] for w in ws]), [w["s"] for w in ws])


def _Fh(ws, cost):
    return encode(np.array([w["x"] for w in ws]), [w["s"] for w in ws], extra=(np.array([w["JS"][cost] for w in ws]) * Z_SCALE[cost])[:, None])


def b1_mc_real(tr, va, cost, x1s, routes, seed=0):
    y, yv = np.array([w["JR"][cost] for w in tr]), np.array([w["JR"][cost] for w in va])
    m = Regressor(4 + 3 * LMAX, seed=seed); m.fit(_F(tr), y, _F(va), yv, seed=seed)
    return m.predict(encode(np.array(x1s), routes))


def b2_route_scape(tr, va, cost, x1s, routes, js_test, seed=0):
    y, yv = np.array([w["JR"][cost] for w in tr]), np.array([w["JR"][cost] for w in va])
    m = Regressor(4 + 3 * LMAX + 1, seed=seed); m.fit(_Fh(tr, cost), y, _Fh(va, cost), yv, seed=seed)
    return m.predict(encode(np.array(x1s), routes, extra=(np.asarray(js_test) * Z_SCALE[cost])[:, None]))
