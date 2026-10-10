"""Long rollout: consecutive segments with no state reset, and random routes on the goal graph."""
import numpy as np

from . import maze as M
from .segment import SegmentConfig, run_segment


def run_route(env, policy, x1, goals, cfg=SegmentConfig()):
    """goals: sequence of goal names. Returns (states, infos);
    states[k] is the start state of segment k (states[0] = x1), states[K] the final state."""
    x = np.asarray(x1, dtype=np.float64).copy()
    states, infos = [x.copy()], []
    for g in goals:
        x, info = run_segment(env, policy, x, M.GOALS[g], cfg)
        states.append(x.copy()); infos.append(info)
    return states, infos


def random_route(rng, K, start=None):
    """Non-backtracking random walk of K goals on the goal graph from goal `start` (random if None).
    Returns (start_name, [g_1, ..., g_K])."""
    start = start if start is not None else list(M.GOALS)[rng.integers(len(M.GOALS))]
    cur, prev, seq = start, None, []
    for _ in range(K):
        nbrs = [n for n in M.NEIGHBORS[cur] if n != prev]
        nxt = nbrs[rng.integers(len(nbrs))]
        seq.append(nxt)
        prev, cur = cur, nxt
    return start, seq
