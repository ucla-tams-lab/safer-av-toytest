"""Simulation-only transitions, paired real chains, and evaluation queries with ground truth (the toy study's data.py
for the car; x = the 7-number car state, g = goal name).

Evaluation queries start either at rest on a goal, facing the first waypoint (as the toy), or "moving": at the state
the real system reaches after driving 1-3 goals of a held-out route (so the value is also tested where terminal
states matter). Simulation-only start states come from
simulated driving (states the car actually reaches):
  sim_route : the segments of simulated routes of 8 goals from rest, so every start state is one the car reaches
  seed_R    : a real state of the paired chains (start or terminal) with small jitter (position 0.05 m, yaw, speed),
              driven to a neighbour of the nearest goal that is not behind the car: puts simulation data where the
              real system goes
Paired data: the real system drives random goal routes without reset (chains; half start at rest, half moving near
a goal, as the toy); at every segment the simulator is set to the current real state and drives the same goal once.
"""
import numpy as np

from . import maze as M
from .segment import COSTS, SegmentConfig, clearance, run_route, run_segment

GOAL_NAMES = list(M.GOALS)
GOAL_XY = np.array([M.GOALS[k] for k in GOAL_NAMES])


def nearest_goal(p):
    return GOAL_NAMES[int(np.argmin(np.linalg.norm(GOAL_XY - p, axis=1)))]


def pick_goal(rng, near, yaw=None):
    """A random neighbour of goal `near`; given the car's heading, never one behind the car (goal sequences do not turn
    back, and a car arriving at a junction cannot turn round in the corridor)."""
    nb = list(M.NEIGHBORS[near])
    if yaw is not None:
        h = np.array([np.cos(yaw), np.sin(yaw)])
        d = {k: (M.GOALS[k] - M.GOALS[near]) / np.linalg.norm(M.GOALS[k] - M.GOALS[near]) for k in nb}
        nb = [k for k in nb if d[k] @ h > -0.5] or nb
    return nb[rng.integers(len(nb))]


def rest_start(world, start, first_goal):
    """Standing still on goal `start`, facing the first waypoint towards `first_goal`."""
    p = M.GOALS[start]
    w = M.waypoint(p, M.GOALS[first_goal])
    return world.rest_state(p, float(np.arctan2(w[1] - p[1], w[0] - p[0])))


def segment_record(env, policy, x, g, cfg):
    xp, info = run_segment(env, policy, x, M.GOALS[g], cfg)
    return dict(**{f"y_{c}": info[f"y_{c}"] for c in COSTS}, min_ttc=info["min_ttc"], x_plus=xp, fail=info["fail"],
                fell=info["fell"], contact=info["contact"], tau=info["tau"], min_clear=info["min_clear"])


# ---------------- worker-side generation (multiprocessing pool initialised with init_worker) ----------------
_W = {}


def init_worker(env_cfg, policy_cfg, seg_cfg):
    from .car import CarWorld
    from .policy import CarPolicy
    noise = env_cfg["noise_std"]
    _W["S"] = CarWorld("S", noise_std=noise, **env_cfg.get("S", {}))
    _W["R"] = CarWorld("R", noise_std=noise, **env_cfg.get("R", {}))
    _W["pol"] = CarPolicy(**policy_cfg)
    _W["cfg"] = SegmentConfig(**seg_cfg)


def gen_sim_route(args):
    """args = (seed, start, goals). Simulation-only transitions: the segments of one simulated route from rest."""
    seed, start, goals = args
    _W["S"].reseed(seed)
    x, out = rest_start(_W["S"], start, goals[0]), []
    for k, g in enumerate(goals):
        rec = dict(x=x.copy(), g=g, kind="sim_route", step=k, source=None, S=segment_record(_W["S"], _W["pol"], x, g, _W["cfg"]))
        out.append(rec)
        x = rec["S"]["x_plus"]
    return out


SIM_MIX = dict(near_goal=0.6, at_rest=0.1, harvest=0.3)        # the toy's simulation-only start mix


def gen_sim_toy(args):
    """args = (seed, kinds). The toy's simulation-only transitions, for the car (make_data.py with CARMAZE_SIM=toy):
    moving near a goal (moving_start; 60%), at rest on a goal (10%), or where such a simulated segment ended (30%);
    each drives to a neighbouring goal that is not behind the car."""
    seed, kinds = args
    rng = np.random.default_rng(seed)
    _W["S"].reseed(int(rng.integers(1 << 31)))
    out = []
    for kind in kinds:
        a = GOAL_NAMES[rng.integers(len(GOAL_NAMES))]
        g = pick_goal(rng, a)
        x = rest_start(_W["S"], a, g) if kind == "at_rest" else moving_start(rng, a, g)
        if kind == "harvest":
            first = segment_record(_W["S"], _W["pol"], x, g, _W["cfg"])
            if first["fail"]:
                kind = "near_goal"
            else:
                x = first["x_plus"]; g = pick_goal(rng, g, yaw=x[2])
        out.append(dict(x=x, g=g, kind=kind, step=0, source=None, S=segment_record(_W["S"], _W["pol"], x, g, _W["cfg"])))
    return out


def gen_seed_R(args):
    """args = (seed, n, seeds). n simulation-only transitions started near real states of the paired chains."""
    seed, n, seeds = args
    rng = np.random.default_rng(seed)
    _W["S"].reseed(int(rng.integers(1 << 31)))
    out = []
    for _ in range(n):
        j = int(rng.integers(len(seeds)))
        src, xr, near = seeds[j]
        x = np.asarray(xr, dtype=np.float64).copy()
        x[:2] += rng.normal(0, 0.05, 2)
        x[2] += rng.normal(0, 0.05)
        x[3] = max(0.0, x[3] + rng.normal(0, 0.2))
        if clearance(x[:2]) <= 0.0:
            x = np.asarray(xr, dtype=np.float64).copy()
        g = pick_goal(rng, near, yaw=x[2])
        out.append(dict(x=x, g=g, kind="seed_R", source=src, S=segment_record(_W["S"], _W["pol"], x, g, _W["cfg"])))
    return out


def gen_queries_gt(args):
    """args = (seed, start, warmup, goals, n_mc). Start: at rest on `start` (no warm-up), or the real state after
    driving the warm-up goals from there. Ground truth J_R (and J_S for reference) by Monte Carlo from that state.
    Returns None if the car fell during the warm-up (never for the car)."""
    seed, start, warmup, goals, n_mc = args
    rng = np.random.default_rng(seed)
    x1 = rest_start(_W["R"], start, (warmup or goals)[0])
    if warmup:
        _W["R"].reseed(int(rng.integers(1 << 31)))
        X, infos = run_route(_W["R"], _W["pol"], x1, warmup, _W["cfg"])
        if any(i["fell"] for i in infos):
            return None
        x1 = X[-1]
    out = dict(start=start, warmup=list(warmup), goals=goals, K=len(goals), x1=np.asarray(x1))
    for kind in ("R", "S"):
        _W[kind].reseed(int(rng.integers(1 << 31)))
        vals = {c: [] for c in COSTS}
        falls, fails = 0, 0
        for _ in range(n_mc):
            _, infos = run_route(_W[kind], _W["pol"], out["x1"], goals, _W["cfg"])
            for c in vals:
                vals[c].append(sum(i[f"y_{c}"] for i in infos))
            falls += any(i["fell"] for i in infos)
            fails += sum(i["fail"] for i in infos)
        out[kind] = {c: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v)))) for c, v in vals.items()}
        out[kind]["fall_rate"] = falls / n_mc
        out[kind]["fail_rate"] = fails / (n_mc * len(goals))           # share of segments that miss their goal
    return out


def make_query_set(rng, Ks, n_per_K, suffix_group_K=None):
    """Rest-start queries (start, warmup=[], goals, tag): random non-backtracking routes, plus 'same first edge, all
    suffixes' groups for one K (as the toy)."""
    qs = [(s, [], g, "random") for K in Ks for _ in range(n_per_K) for s, g in [M.random_route(rng, K)]]
    if suffix_group_K:
        for a, b in M.GOAL_EDGES:
            for s, t in ((a, b), (b, a)):
                for suf in _walks(s, t, suffix_group_K - 1):
                    qs.append((s, [], [t] + suf, f"group:{s}{t}"))
    return qs


def make_moving_query_set(rng, Ks, n_per_K):
    """Moving-start queries: a random route of w + K goals (w = 1..3); the first w goals are the warm-up."""
    out = []
    for K in Ks:
        for _ in range(n_per_K):
            w = int(rng.integers(1, 4))
            s, route = M.random_route(rng, w + K)
            out.append((s, route[:w], route[w:], "moving"))
    return out


def _walks(prev, cur, L):
    if L == 0:
        return [[]]
    return [[n] + w for n in M.NEIGHBORS[cur] if n != prev for w in _walks(cur, n, L - 1)]


def gen_sim_rollouts(args):
    """args = (seed, x1, goals, n_roll). n_roll sim rollouts of a query: states and per-segment costs."""
    seed, x1, goals, n_roll = args
    _W["S"].reseed(seed)
    out = []
    for _ in range(n_roll):
        X, infos = run_route(_W["S"], _W["pol"], x1, goals, _W["cfg"])
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


R_NEAR, V_HI = 0.25 * M.CELL, 5.0          # moving chain starts: within 0.25 cell of a goal (the toy's), up to top speed


def moving_start(rng, a, g1):
    """A car moving near goal `a` as if it had just arrived from a neighbour n != g1 (so it goes straight on or turns,
    never back): up to R_NEAR before the junction centre on n's side, heading from n to a (+-10 deg), speed uniform
    in [0, V_HI] m/s (the toy's moving chain start, for a car)."""
    nb = [k for k in M.NEIGHBORS[a] if k != g1]
    n = nb[rng.integers(len(nb))]
    d = (M.GOALS[a] - M.GOALS[n]) / np.linalg.norm(M.GOALS[a] - M.GOALS[n])
    p = M.GOALS[a] - d * rng.uniform(0, R_NEAR) + np.array([-d[1], d[0]]) * rng.uniform(-0.3, 0.3)
    yaw = np.arctan2(d[1], d[0]) + np.radians(rng.uniform(-10, 10))
    return np.array([p[0], p[1], yaw, rng.uniform(0, V_HI), 0.0, 0.0, 0.0])


def make_chain_specs(rng, n_target, forbidden_prefixes, world, L_range=(2, 8), prefix_len=4):
    """Chain specs until their total length >= n_target (as the toy). Half start at rest on a goal, half moving near a
    goal (moving_start). A rest-start chain may not share its start and first `prefix_len` goals with a test query (no
    long test route appears in the training data); its route is resampled, and cut to prefix_len - 1 goals if every
    route is taken."""
    specs, total = [], 0
    while total < n_target:
        a = GOAL_NAMES[rng.integers(len(GOAL_NAMES))]
        L = int(rng.integers(L_range[0], L_range[1] + 1))
        _, goals = M.random_route(rng, L, start=a)
        if rng.uniform() < 0.5:
            tries = 0
            while L >= prefix_len and (a, tuple(goals[:prefix_len])) in forbidden_prefixes and tries < 100:
                _, goals = M.random_route(rng, L, start=a); tries += 1
            if L >= prefix_len and (a, tuple(goals[:prefix_len])) in forbidden_prefixes:
                goals = goals[:prefix_len - 1]; L = len(goals)
            x0, kind = rest_start(world, a, goals[0]), "chain_rest"
        else:
            x0, kind = moving_start(rng, a, goals[0]), "chain_moving"
        specs.append((int(rng.integers(1 << 31)), x0, a, goals, kind))
        total += L
    return specs


def chain_seeds(chain):
    """Real states of the paired chains used to seed sim data (seed_R): every moving start state (with the goal it is
    near) and every real terminal state that did not fall (with the goal reached)."""
    seeds = [(i, r["x"], nearest_goal(r["x"][:2])) for i, r in enumerate(chain) if r["step"] > 0]
    seeds += [(i, r["R"]["x_plus"], r["g"]) for i, r in enumerate(chain) if not r["R"]["fail"]]
    return seeds
