"""Gate: do the costs differ between scenes (goal sequences and start states)? Per-scene estimation is only meaningful
if the real cost varies across queries much more than the noise of its Monte Carlo truth.
Queries: rest-start and moving-start (after 1-3 real warm-up goals), K = 4 and 8; each run --mc times in real and sim.
Reported per start type, K and cost: spread of the real truth across queries (std / mean), the same spread over the
truth's standard error, the mean real - sim gap, and the correlation of sim and real truths across queries.
Usage: python scripts/check_scenes.py [--queries 60] [--mc 30] [--noise 0.05] [--accel 4 --decel 3.5 --mu 0.8 --mu_wet 0.5]"""
import argparse, json, os, sys, time  # noqa: E401
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from carmaze import data as D  # noqa: E402
from carmaze.segment import COSTS  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--mc", type=int, default=30)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--accel", type=float, default=None)
    ap.add_argument("--decel", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--mu_wet", type=float, default=None)
    ap.add_argument("--workers", type=int, default=150)
    a = ap.parse_args()
    real = {k: v for k, v in (("accel", a.accel), ("decel", a.decel), ("mu", a.mu), ("mu_wet", a.mu_wet)) if v is not None}
    env_cfg = dict(S={}, R=real, noise_std=a.noise)
    rng = np.random.default_rng(5)
    qs = D.make_query_set(rng, [4, 8], a.queries) + D.make_moving_query_set(rng, [4, 8], a.queries)
    # the same corridors in the opposite order: start at the last goal, visit the goals backwards
    rev = [(g[-1], [], (g[:-1][::-1] + [s]), "reverse") for s, w, g, t in qs if t == "random"]
    qs = qs + rev
    t0 = time.time()
    with Pool(a.workers, initializer=D.init_worker, initargs=(env_cfg, {}, {})) as p:
        res = p.map(D.gen_queries_gt, [(int(rng.integers(1 << 31)), s, w, g, a.mc) for s, w, g, _ in qs], chunksize=1)
    fwd = [r for (_, _, _, t), r in zip(qs, res) if t == "random"]
    bwd = [r for (_, _, _, t), r in zip(qs, res) if t == "reverse"]
    tags = [t for (_, _, _, t), r in zip(qs, res) if r is not None and t != "reverse"]
    res = [r for (_, _, _, t), r in zip(qs, res) if r is not None and t != "reverse"]
    print(f"real physics {real or 'default'}, noise {a.noise}: {len(res)} queries x {a.mc} runs in {time.time() - t0:.0f}s")
    out = {}
    for st in ("rest", "moving"):
        for K in (4, 8):
            qq = [r for r, t in zip(res, tags) if (t == "moving") == (st == "moving") and r["K"] == K]
            cells = []
            for c in COSTS:
                R = np.array([x["R"][c]["mean"] for x in qq]); S = np.array([x["S"][c]["mean"] for x in qq])
                se = np.mean([x["R"][c]["se"] for x in qq])
                v = dict(spread=float(R.std() / abs(R.mean())), spread_over_se=float(R.std() / se), gap=float((R.mean() - S.mean()) / abs(S.mean())),
                         corr=float(np.corrcoef(R, S)[0, 1]))
                out[f"{st}|K={K}|{c}"] = v
                cells.append(f"{c[:5]} {100 * v['spread']:3.0f}% ({v['spread_over_se']:4.1f}x) gap {100 * v['gap']:+4.0f}% r {v['corr']:.2f}")
            fails = (np.mean([x["S"]["fail_rate"] for x in qq]), np.mean([x["R"]["fail_rate"] for x in qq]))
            print(f"  {st:6s} K={K} (segments missed sim {fails[0]:.3f} real {fails[1]:.3f}): " + " | ".join(cells), flush=True)
    for K in (4, 8):
        pairs = [(f, b) for f, b in zip(fwd, bwd) if f["K"] == K]
        cells = []
        for c in COSTS:
            d = np.array([f["R"][c]["mean"] - b["R"][c]["mean"] for f, b in pairs])
            m = np.mean([abs(f["R"][c]["mean"]) for f, b in pairs]); se = np.mean([np.hypot(f["R"][c]["se"], b["R"][c]["se"]) for f, b in pairs])
            ds = np.array([f["S"][c]["mean"] - b["S"][c]["mean"] for f, b in pairs])
            cells.append(f"{c[:5]} |fwd-rev| {100 * np.abs(d).mean() / m:3.0f}% ({np.abs(d).mean() / se:4.1f}x) sim-real corr {np.corrcoef(d, ds)[0, 1]:.2f}")
        print(f"  order  K={K}: same corridors, opposite order (real): " + " | ".join(cells), flush=True)
    out_dir = os.path.join(ROOT, "data", "checks")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(real=real, noise=a.noise, res=out), open(os.path.join(out_dir, f"scenes_{'_'.join(f'{k}{v}' for k, v in real.items()) or 'default'}_n{a.noise}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
