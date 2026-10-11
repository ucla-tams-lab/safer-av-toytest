"""Step 1: datasets that do not depend on n or the model seed.

Outputs (data/):
  queries_test.pkl          536 test routes (K = 1, 2, 4, 8 random + K=4 suffix groups), ground truth J_R / J_S
                            from N_MC rollouts each
  sim_rollouts_test.pkl     N_ROLL sim rollouts of every test route (raw sim, outcome-only, sim-backed, B2, residual)
  sim_train.pkl             20000 one-segment sim transitions (near_goal / at_rest / harvest mixture)
  d{d}/chain_train.pkl      paired data along real chains, data seed d = 0, 1, 2 (subsets = first n transitions)
  d{d}/chain_val.pkl        50 paired validation transitions (early stopping), same collection
  meta.json                 configs, seeds, sizes
"""
import argparse, json, os, pickle, sys, time  # noqa: E401
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safer_av import data as D  # noqa: E402

DATA = ROOT / "data"
CFG = {k: json.loads((ROOT / "configs" / f"{k}.json").read_text()) for k in ("env", "policy", "segment")}
MASTER_SEED = 20261003            # sim_train and test queries
ROLLOUT_SEED, N_ROLL = 777, 30    # sim rollouts of the test routes
CHAIN_SEED = 9090                 # + 1000 * data seed
N_SIM, N_MC, N_CHAIN_TRAIN, N_CHAIN_VAL, DATA_SEEDS, CHUNK = 20000, 100, 1000, 50, (0, 1, 2), 50


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(96, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()
    t0 = time.time()
    ss = np.random.SeedSequence(MASTER_SEED).spawn(10)
    seed = lambda i: int(ss[i].generate_state(1)[0])
    with Pool(a.workers, initializer=D.init_worker, initargs=(CFG["env"], CFG["policy"], CFG["segment"])) as pool:
        # sim-only transitions
        rng = np.random.default_rng(seed(2))
        names, probs = list(D.MIX), np.array(list(D.MIX.values()))
        kinds = list(rng.choice(names, size=N_SIM, p=probs / probs.sum()))
        jobs = [(int(rng.integers(1 << 31)), len(kinds[i:i + CHUNK]), kinds[i:i + CHUNK], False, None) for i in range(0, N_SIM, CHUNK)]
        dump([r for rs in pool.map(D.gen_chunk, jobs) for r in rs], DATA / "sim_train.pkl")
        # test routes with Monte Carlo ground truth
        rng = np.random.default_rng(seed(8))
        q_test = D.make_query_set(rng, [1, 2, 4, 8], 100, suffix_group_K=4)
        D.make_query_set(rng, [1, 2, 4, 8], 50)  # an unused dev set was drawn here originally; kept so the random stream is unchanged
        gt = pool.map(D.gen_queries_gt, [(int(rng.integers(1 << 31)), s, g, N_MC) for s, g, _ in q_test])
        for q, (_, _, tag) in zip(gt, q_test):
            q["tag"] = tag
        dump(gt, DATA / "queries_test.pkl")
        # sim rollouts of the test routes
        rng = np.random.default_rng(ROLLOUT_SEED)
        dump(pool.map(D.gen_sim_rollouts, [(int(rng.integers(1 << 31)), q["start"], q["goals"], N_ROLL) for q in gt]), DATA / "sim_rollouts_test.pkl")
        # paired data along real chains; rest-start chains may not share (start, first 4 goals) with a K>=4 test route
        forbidden = {(x["start"], tuple(x["goals"][:4])) for x in gt if len(x["goals"]) >= 4}
        for d in DATA_SEEDS:
            rng = np.random.default_rng(CHAIN_SEED + 1000 * d)
            tr = [r for c in pool.map(D.gen_paired_chain, D.make_chain_specs(rng, N_CHAIN_TRAIN, forbidden)) for r in c][:N_CHAIN_TRAIN]
            va = [r for c in pool.map(D.gen_paired_chain, D.make_chain_specs(rng, N_CHAIN_VAL, forbidden)) for r in c][:N_CHAIN_VAL]
            dump(tr, DATA / f"d{d}" / "chain_train.pkl"); dump(va, DATA / f"d{d}" / "chain_val.pkl")
            sp = np.linalg.norm([r["x"][2:] for r in tr], axis=1)
            print(f"d={d}: train {len(tr)} val {len(va)}; start at rest {np.mean(sp < 1e-6):.2f}, median start speed {np.median(sp):.2f}, "
                  f"R timeout {np.mean([r['R']['fail'] for r in tr]):.3f}", flush=True)
    meta = dict(configs=CFG, master_seed=MASTER_SEED, rollout_seed=ROLLOUT_SEED, chain_seed=CHAIN_SEED, mix=D.MIX, r_near=D.R_NEAR, v_hi=D.V_HI,
                sizes=dict(sim_train=N_SIM, n_test_queries=len(gt), n_mc=N_MC, n_roll=N_ROLL, chain_train=N_CHAIN_TRAIN, chain_val=N_CHAIN_VAL),
                data_seeds=list(DATA_SEEDS), seconds=round(time.time() - t0))
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
