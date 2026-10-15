"""Step 1: datasets that do not depend on n or the model seed (the toy study's make_data.py for the car).

Outputs (data/, or $CARMAZE_DATA):
  queries_test.pkl          evaluation queries: 536 starting at rest (K = 1, 2, 4, 8 random + K=4 suffix groups, as the
                            toy) and 400 starting moving (after 1-3 real warm-up goals); ground truth J_R / J_S from
                            N_MC rollouts each
  sim_rollouts_test.pkl     N_ROLL sim rollouts of every query (raw sim, segment-wise correction, sim-backed, B2)
  sim_train.pkl             20000 simulation-only transitions: the segments of simulated routes of 8 goals from rest
  d{d}/chain_train.pkl      paired transitions along real chains, data seed d = 0, 1, 2 (subsets = first n transitions)
  d{d}/chain_val.pkl        50 paired validation transitions (early stopping), same collection
  x_norm.json               mean / std of the 8 state features over the simulation-only data (model input scaling)
  meta.json                 configs, seeds, sizes
Usage: python scripts/make_data.py [--workers 150]"""
import argparse, json, os, pickle, sys, time  # noqa: E401
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carmaze import data as D  # noqa: E402
from carmaze import maze as M  # noqa: E402
from carmaze.models import state_features  # noqa: E402

DATA = Path(os.environ.get("CARMAZE_DATA", ROOT / "data"))
SIM_DATA = os.environ.get("CARMAZE_SIM", "routes")
CFG = {k: json.loads((ROOT / "configs" / f"{k}.json").read_text()) for k in ("env", "policy", "segment")}
MASTER_SEED = 20261015
ROLLOUT_SEED, N_ROLL = 777, 30
CHAIN_SEED = 9090
N_SIM_ROUTES, N_MC, N_CHAIN_TRAIN, N_CHAIN_VAL, DATA_SEEDS = 2500, 200, 1000, 50, (0, 1, 2)


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="tiny sizes, to check the pipeline")
    a = ap.parse_args()
    if a.quick:
        N_SIM_ROUTES, N_MC, N_ROLL, N_CHAIN_TRAIN, N_CHAIN_VAL = 20, 3, 2, 40, 10
    t0 = time.time()
    ss = np.random.SeedSequence(MASTER_SEED).spawn(10)
    seed = lambda i: int(ss[i].generate_state(1)[0])
    from carmaze.car import CarWorld
    world = CarWorld("R")
    with Pool(a.workers, initializer=D.init_worker, initargs=(CFG["env"], CFG["policy"], CFG["segment"])) as pool:
        # simulation-only transitions: the segments of simulated routes of 8 goals from rest (default), or the toy's
        # random starts near the goals (CARMAZE_SIM=toy; the same number of transitions)
        rng = np.random.default_rng(seed(2))
        if SIM_DATA == "toy":
            kinds = rng.choice(list(D.SIM_MIX), size=8 * N_SIM_ROUTES, p=list(D.SIM_MIX.values()))
            jobs = [(int(rng.integers(1 << 31)), list(kinds[i:i + 100])) for i in range(0, len(kinds), 100)]
            sim_train = [r for rs in pool.map(D.gen_sim_toy, jobs) for r in rs]
        else:
            jobs = [(int(rng.integers(1 << 31)), *M.random_route(rng, 8)) for _ in range(N_SIM_ROUTES)]
            sim_train = [r for rs in pool.map(D.gen_sim_route, jobs, chunksize=4) for r in rs]
        dump(sim_train, DATA / "sim_train.pkl")
        X = state_features([r["x"] for r in sim_train] + [r["S"]["x_plus"] for r in sim_train])
        (DATA / "x_norm.json").write_text(json.dumps(dict(mean=X.mean(0).tolist(), std=X.std(0).tolist())))
        print(f"sim_train: {len(sim_train)} transitions, falls {np.mean([r['S']['fell'] for r in sim_train]):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        # evaluation queries with Monte Carlo ground truth
        rng = np.random.default_rng(seed(8))
        Ks = [1, 2, 4, 8]
        qs = D.make_query_set(rng, Ks, 3 if a.quick else 100, suffix_group_K=None if a.quick else 4) \
            + D.make_moving_query_set(rng, Ks, 3 if a.quick else 100)
        gt = pool.map(D.gen_queries_gt, [(int(rng.integers(1 << 31)), s, w, g, N_MC) for s, w, g, _ in qs], chunksize=1)
        q_test = []
        for q, (_, _, _, tag) in zip(gt, qs):
            if q is not None:
                q["tag"] = tag
                q_test.append(q)
        dump(q_test, DATA / "queries_test.pkl")
        print(f"queries: {sum(q['tag'] != 'moving' for q in q_test)} at rest + {sum(q['tag'] == 'moving' for q in q_test)} moving "
              f"({len(qs) - len(q_test)} dropped: fell in the warm-up); real fall rate per query "
              f"{np.mean([q['R']['fall_rate'] for q in q_test]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        # sim rollouts of the queries
        rng = np.random.default_rng(ROLLOUT_SEED)
        dump(pool.map(D.gen_sim_rollouts, [(int(rng.integers(1 << 31)), q["x1"], q["goals"], N_ROLL) for q in q_test], chunksize=1),
             DATA / "sim_rollouts_test.pkl")
        print(f"sim rollouts done ({time.time() - t0:.0f}s)", flush=True)
        # paired data along real chains; chains may not share (start, first 4 goals) with a rest-start K>=4 query
        forbidden = {(q["start"], tuple(q["goals"][:4])) for q in q_test if q["tag"] != "moving" and len(q["goals"]) >= 4}
        for d in DATA_SEEDS:
            rng = np.random.default_rng(CHAIN_SEED + 1000 * d)
            tr = [r for c in pool.map(D.gen_paired_chain, D.make_chain_specs(rng, N_CHAIN_TRAIN, forbidden, world)) for r in c][:N_CHAIN_TRAIN]
            va = [r for c in pool.map(D.gen_paired_chain, D.make_chain_specs(rng, N_CHAIN_VAL, forbidden, world)) for r in c][:N_CHAIN_VAL]
            dump(tr, DATA / f"d{d}" / "chain_train.pkl"); dump(va, DATA / f"d{d}" / "chain_val.pkl")
            print(f"d={d}: train {len(tr)} val {len(va)}; real falls {np.mean([r['R']['fell'] for r in tr]):.3f}, "
                  f"sim falls {np.mean([r['S']['fell'] for r in tr]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    meta = dict(configs=CFG, sim_data=SIM_DATA, master_seed=MASTER_SEED, rollout_seed=ROLLOUT_SEED, chain_seed=CHAIN_SEED,
                sizes=dict(sim_train=len(sim_train), n_test_queries=len(q_test), n_mc=N_MC, n_roll=N_ROLL,
                           chain_train=N_CHAIN_TRAIN, chain_val=N_CHAIN_VAL),
                data_seeds=list(DATA_SEEDS), seconds=round(time.time() - t0))
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
