"""Gate check: can the cost of one segment be learned from (x, g)?
Training data: sim segments along random sim routes (8 goals from rest) and from the states of real routes (other
goals, other noise seeds). Test: 400 segment starts of held-out real routes, each run 30 times in sim and in real
(different action-noise seeds); truth = the 30-run mean. Model: MLP on (normalized x, goal position).
Reported per cost: R^2 of the model against the sim truth, the share of single-run variance that the start state
explains (the most any estimator can reach), and the correlation of sim and real truths across states.
Usage: python scripts/check_learn.py [--chains 4000] [--workers 150]"""
import argparse, os, sys, time  # noqa: E401
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
NOISE, M_RUNS = 0.05, 30
W = {}


def init_worker():
    from carmaze.car import CarWorld
    from carmaze.policy import CarPolicy
    W["pol"] = CarPolicy()
    W["S"], W["R"] = CarWorld("S", noise_std=NOISE), CarWorld("R", noise_std=NOISE)


def start_state(world, start, first_goal):
    from carmaze import maze as M
    p = M.GOALS[start]
    w = M.waypoint(p, M.GOALS[first_goal])
    return world.rest_state(p, float(np.arctan2(w[1] - p[1], w[0] - p[0])))


def ys(info):
    from carmaze.segment import COSTS
    return [info[f"y_{c}"] for c in COSTS]


def chain(args):
    """A route of 8 goals in world kind; returns [(x0, goal, y)] for every segment."""
    from carmaze import maze as M
    from carmaze.segment import run_segment
    kind, seed = args
    rng = np.random.default_rng(seed)
    start, goals = M.random_route(rng, 8)
    w = W[kind]
    x, out = start_state(w, start, goals[0]), []
    for k, g in enumerate(goals):
        w.reseed(seed * 100 + k)
        xe, info = run_segment(w, W["pol"], x, M.GOALS[g])
        out.append((x, g, ys(info)))
        x = xe
    return out


def from_real_state(args):
    """Sim segment from a real state to a random neighbour of the nearest goal."""
    from carmaze import maze as M
    from carmaze.segment import run_segment
    x, seed = args
    rng = np.random.default_rng(seed)
    near = min(M.GOALS, key=lambda k: np.linalg.norm(M.GOALS[k] - x[:2]))
    g = M.NEIGHBORS[near][rng.integers(len(M.NEIGHBORS[near]))]
    W["S"].reseed(seed)
    _, info = run_segment(W["S"], W["pol"], x, M.GOALS[g])
    return (x, g, ys(info))


def mc(args):
    from carmaze import maze as M
    from carmaze.segment import run_segment
    x, g, seed = args
    out = {}
    for kind in ("S", "R"):
        r = []
        for m in range(M_RUNS):
            W[kind].reseed(seed * 1000 + m)
            r.append(ys(run_segment(W[kind], W["pol"], x, M.GOALS[g])[1]))
        out[kind] = np.array(r)
    return out


def features(X, G):
    from carmaze import maze as M
    from carmaze.models import state_features
    return np.concatenate([state_features(X), np.array([M.GOALS[g] for g in G]) / M.CELL], 1)


def fit_predict(Xtr, ytr, Xte, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ysd = ytr.mean(), ytr.std() + 1e-9
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    xt, yt, xe = T((Xtr - mu) / sd), T((ytr - ym) / ysd), T((Xte - mu) / sd)
    n = len(xt); perm = torch.randperm(n); va, tr = perm[: n // 10], perm[n // 10:]
    net = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, state, bad = np.inf, None, 0
    for ep in range(300):
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), 512):
            b = p[i:i + 512]
            loss = ((net(xt[b]).squeeze(-1) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = ((net(xt[va]).squeeze(-1) - yt[va]) ** 2).mean().item()
        if v < best - 1e-5:
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return net(xe).squeeze(-1).numpy() * ysd + ym


def main():
    from carmaze.segment import COSTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=150)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers, initializer=init_worker) as p:
        sim = [s for c in p.map(chain, [("S", 10_000 + i) for i in range(a.chains)], chunksize=4) for s in c]
        real = [s for c in p.map(chain, [("R", 20_000 + i) for i in range(a.chains // 4)], chunksize=4) for s in c]
        seedR = p.map(from_real_state, [(x, 30_000 + i) for i, (x, _, _) in enumerate(real)], chunksize=8)
        test = [s for c in p.map(chain, [("R", 90_000 + i) for i in range(50)]) for s in c]
        truth = p.map(mc, [(x, g, 40_000 + i) for i, (x, g, _) in enumerate(test)], chunksize=1)
    print(f"data: {len(sim)} sim-route segments + {len(seedR)} from real states; {len(test)} test states x {M_RUNS} runs; "
          f"{time.time() - t0:.0f}s", flush=True)
    train = sim + seedR
    Xtr = features(np.array([s[0] for s in train]), [s[1] for s in train])
    Xte = features(np.array([s[0] for s in test]), [s[1] for s in test])
    YS = np.stack([t["S"] for t in truth]); YR = np.stack([t["R"] for t in truth])     # (states, runs, costs)
    print(f"{'cost':8s} {'R2 (sim)':>9s} {'state share sim':>16s} {'state share real':>17s} {'corr(sim,real)':>15s} {'real-sim':>9s}")
    for c, name in enumerate(COSTS):
        ytr = np.array([s[2][c] for s in train])
        pred = fit_predict(Xtr, ytr, Xte)
        tS, tR = YS[:, :, c].mean(1), YR[:, :, c].mean(1)
        r2 = 1 - ((pred - tS) ** 2).mean() / tS.var()
        share = lambda Y: tS.var() and (Y.mean(1).var() - Y.var(1, ddof=1).mean() / M_RUNS) / (Y.mean(1).var() - Y.var(1, ddof=1).mean() / M_RUNS + Y.var(1, ddof=1).mean())
        print(f"{name:8s} {r2:9.2f} {share(YS[:, :, c]):16.0%} {share(YR[:, :, c]):17.0%} {np.corrcoef(tS, tR)[0, 1]:15.2f} "
              f"{(tR.mean() - tS.mean()) / max(abs(tS.mean()), 1e-9):+9.0%}", flush=True)


if __name__ == "__main__":
    main()
