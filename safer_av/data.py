"""Start-state samplers, short-transition datasets, test queries with ground truth, and paired real chains.

x = (px, py, vx, vy), g = goal name. Every place W / h are evaluated lies within eps of a goal (terminal
states, query starts at rest), so the sim start-state mixture concentrates there:
  near_goal : position within R_NEAR of a goal; speed U[0, V_HI], direction half corridor-aligned (+-10 deg),
              half any direction; g uniform over the goal's neighbours.
  at_rest   : exactly at a goal with zero velocity (the query start state).
  harvest   : terminal state of one S segment from a near_goal start (S-reachable arrival states).
  seed_R    : a real state of the paired chains (start or terminal), slightly jittered; puts sim data where
              W is evaluated. g uniform over the neighbours of the reached goal.
Paired data: the real system runs along random goal routes without reset (chains); at every segment the sim is
set to the current real state and runs the same goal once.
"""
import numpy as np

from . import maze as M
from .envs import make_env
from .policy import PDPolicy
from .route import random_route, run_route
from .segment import COSTS, SegmentConfig, clearance, run_segment

GOAL_NAMES = list(M.GOALS)
GOAL_XY = np.array([M.GOALS[k] for k in GOAL_NAMES])
R_NEAR, V_HI = 0.25, 5.0
MIX = dict(near_goal=0.6, at_rest=0.1, harvest=0.3)


def nearest_goal(p):
    return GOAL_NAMES[int(np.argmin(np.linalg.norm(GOAL_XY - p, axis=1)))]


def random_velocity(rng):
    if rng.uniform() < 0.5:  # corridor-aligned, like real arrival velocities
        ang = rng.integers(4) * np.pi / 2 + np.radians(rng.uniform(-10, 10))
    else:
        ang = rng.uniform(0, 2 * np.pi)
    spd = rng.uniform(0, V_HI)
    return np.array([spd * np.cos(ang), spd * np.sin(ang)])


def free_position(rng, center, radius):
    while True:
        ang, rad = rng.uniform(0, 2 * np.pi), radius * np.sqrt(rng.uniform())
        p = center + rad * np.array([np.cos(ang), np.sin(ang)])
        if clearance(p) > 0.0:
            return p


def pick_goal(rng, near):
    nb = M.NEIGHBORS[near]
    return nb[rng.integers(len(nb))]


def sample_start(rng, kind, harvest_env=None, policy=None, cfg=None, seeds=None):
    """Returns (x, g, source). source = index into `seeds` for seed_R (provenance), else None."""
    if kind == "near_goal":
        a = GOAL_NAMES[rng.integers(len(GOAL_NAMES))]
        return np.r_[free_position(rng, M.GOALS[a], R_NEAR), random_velocity(rng)], pick_goal(rng, a), None
    if kind == "at_rest":
        a = GOAL_NAMES[rng.integers(len(GOAL_NAMES))]
        return np.r_[M.GOALS[a], 0.0, 0.0], pick_goal(rng, a), None
    if kind == "harvest":
        x0, g0, _ = sample_start(rng, "near_goal")
        xp, info = run_segment(harvest_env, policy, x0, M.GOALS[g0], cfg)
        if info["fail"]:
            return sample_start(rng, "near_goal")
        return xp, pick_goal(rng, g0), None
    if kind == "seed_R":
        j = int(rng.integers(len(seeds)))
        src, xr, reached = seeds[j]
        x = xr + np.r_[rng.normal(0, 0.03, 2), rng.normal(0, 0.15, 2)]
        if clearance(x[:2]) <= 0.0:
            x = xr.copy()
        return x, pick_goal(rng, reached), src
    raise ValueError(kind)


def segment_record(env, policy, x, g, cfg):
    xp, info = run_segment(env, policy, x, M.GOALS[g], cfg)
    return dict(**{f"y_{c}": info[f"y_{c}"] for c in COSTS}, min_ttc=info["min_ttc"], x_plus=xp,
                fail=info["fail"], contact=info["contact"], tau=info["tau"], min_clear=info["min_clear"])


# ---------------- worker-side generation (multiprocessing pool initialised with init_worker) ----------------
_W = {}


def init_worker(env_cfg, policy_cfg, seg_cfg):
    noise = env_cfg["noise_std"]
    _W["S"] = make_env("S", noise_std=noise, **env_cfg["S"])
    _W["R"] = make_env("R", noise_std=noise, **env_cfg["R"])
    _W["S_harvest"] = make_env("S", noise_std=noise, **env_cfg["S"])
    _W["pol"] = PDPolicy(policy_cfg["kp"], policy_cfg["kd"])
    _W["cfg"] = SegmentConfig(**seg_cfg)


def gen_chunk(args):
    """args = (seed, n, kinds, paired, seeds). Returns a list of transition dicts."""
    seed, n, kinds, paired, seeds = args
    rng = np.random.default_rng(seed)
    for k in ("S", "R", "S_harvest"):
        _W[k].reseed(int(rng.integers(1 << 31)))
    out = []
    for i in range(n):
        kind = kinds[i]
        x, g, src = sample_start(rng, kind, _W["S_harvest"], _W["pol"], _W["cfg"], seeds)
        rec = dict(x=x, g=g, kind=kind, source=src, S=segment_record(_W["S"], _W["pol"], x, g, _W["cfg"]))
        if paired:
            rec["R"] = segment_record(_W["R"], _W["pol"], x, g, _W["cfg"])
        out.append(rec)
    return out


def gen_queries_gt(args):
    """args = (seed, start, goals, n_mc). Ground truth J_R (and J_S for reference) by Monte Carlo."""
    seed, start, goals, n_mc = args
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("R", "S"):
        _W[kind].reseed(int(rng.integers(1 << 31)))
        vals = {c: [] for c in COSTS}
        for _ in range(n_mc):
            _, infos = run_route(_W[kind], _W["pol"], np.r_[M.GOALS[start], 0, 0], goals, _W["cfg"])
            for c in vals:
                vals[c].append(sum(i[f"y_{c}"] for i in infos))
        out[kind] = {c: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v)))) for c, v in vals.items()}
    return dict(start=start, goals=goals, K=len(goals), **out)


def make_query_set(rng, Ks, n_per_K, suffix_group_K=None):
    """Random non-backtracking routes, plus 'same first edge, all suffixes' groups for one K."""
    qs = [(*random_route(rng, K), "random") for K in Ks for _ in range(n_per_K)]
    if suffix_group_K:
        for a, b in M.GOAL_EDGES:
            for s, t in ((a, b), (b, a)):
                for suf in _walks(s, t, suffix_group_K - 1):
                    qs.append((s, [t] + suf, f"group:{s}{t}"))
    return qs


def _walks(prev, cur, L):
    if L == 0:
        return [[]]
    return [[n] + w for n in M.NEIGHBORS[cur] if n != prev for w in _walks(cur, n, L - 1)]


def gen_sim_rollouts(args):
    """args = (seed, start, goals, n_roll). n_roll sim rollouts of a query: states and per-segment costs."""
    seed, start, goals, n_roll = args
    rng = np.random.default_rng(seed)
    _W["S"].reseed(int(rng.integers(1 << 31)))
    out = []
    for _ in range(n_roll):
        X, infos = run_route(_W["S"], _W["pol"], np.r_[M.GOALS[start], 0, 0], goals, _W["cfg"])
        out.append(dict(X=np.array(X), y={c: np.array([i[f"y_{c}"] for i in infos]) for c in COSTS}))
    return out


def gen_paired_chain(args):
    """args = (seed, x0, start_goal_name, goals, kind). Run R along `goals` from x0 without reset;
    at every segment the sim is set to the current REAL state and runs the same goal once."""
    seed, x0, start, goals, kind = args
    rng = np.random.default_rng(seed)
    _W["S"].reseed(int(rng.integers(1 << 31))); _W["R"].reseed(int(rng.integers(1 << 31)))
    x, out = np.asarray(x0, dtype=np.float64), []
    for k, g in enumerate(goals):
        rec = dict(x=x.copy(), g=g, kind=kind, step=k, chain_start=start,
                   S=segment_record(_W["S"], _W["pol"], x, g, _W["cfg"]),
                   R=segment_record(_W["R"], _W["pol"], x, g, _W["cfg"]))
        out.append(rec)
        x = rec["R"]["x_plus"]
    return out


def make_chain_specs(rng, n_target, forbidden_prefixes, L_range=(2, 8), prefix_len=4):
    """Chain specs until their total length >= n_target. Half start at rest at a goal, half at a
    moving near-goal state. A rest-start chain may not share its first `prefix_len` goals with a
    test query (so no long test route appears in the training data); its route is resampled."""
    specs, total = [], 0
    while total < n_target:
        a = GOAL_NAMES[rng.integers(len(GOAL_NAMES))]
        L = int(rng.integers(L_range[0], L_range[1] + 1))
        _, goals = random_route(rng, L, start=a)
        if rng.uniform() < 0.5:
            tries = 0
            while L >= prefix_len and (a, tuple(goals[:prefix_len])) in forbidden_prefixes and tries < 100:
                _, goals = random_route(rng, L, start=a); tries += 1
            if L >= prefix_len and (a, tuple(goals[:prefix_len])) in forbidden_prefixes:
                goals = goals[:prefix_len - 1]; L = len(goals)
            x0, kind = np.r_[M.GOALS[a], 0.0, 0.0], "chain_rest"
        else:
            x0, kind = np.r_[free_position(rng, M.GOALS[a], R_NEAR), random_velocity(rng)], "chain_moving"
        specs.append((int(rng.integers(1 << 31)), x0, a, goals, kind))
        total += L
    return specs


def chain_seeds(chain):
    """Real states of the paired chains used to seed sim data (seed_R): every start state (with the goal it is
    near) and every real terminal state (with the goal reached). Rest states are dropped (no jitter there)."""
    seeds = [(i, r["x"], nearest_goal(r["x"][:2]) if r["step"] > 0 else r["chain_start"]) for i, r in enumerate(chain)]
    seeds += [(i, r["R"]["x_plus"], r["g"]) for i, r in enumerate(chain) if not r["R"]["fail"]]
    return [(i, x, g) for i, x, g in seeds if np.linalg.norm(x[2:]) > 1e-6]
