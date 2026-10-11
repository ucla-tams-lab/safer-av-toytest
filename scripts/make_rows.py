"""Step 2: data that depend on n (paired subset size) and the model seed, for every data seed.

Outputs (data/):
  d{d}/seedR_n{n}.pkl           4000 sim transitions started around the real states of chain_train[:n]
  rows/simtrain_s{s}.pkl        residual rows (suffix-sampled, with Monte Carlo sim values W_S) of sim_train
  d{d}/rows/paired_n{n}_s{s}.pkl, d{d}/rows/val_s{s}.pkl, d{d}/rows/seedR_n{n}_s{s}.pkl   the same for the paired / sim data
  d{d}/windows/train_n{n}.pkl, d{d}/windows/val.pkl                                     B1 / B2 windows with sim returns

Usage: python scripts/make_rows.py                         (everything: 3 data seeds x 5 sizes x 3 model seeds)
       python scripts/make_rows.py --data-seeds 0 --sizes 200 --model-seeds 0   (only what run.py 0 200 0 needs)
A subset produces exactly the same files as the full run.
"""
import argparse, json, math, os, pickle, sys, time  # noqa: E401
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safer_av import data as D  # noqa: E402
from safer_av.residual import attach_mc, make_rows  # noqa: E402
from safer_av.route_baselines import attach_js, windows  # noqa: E402

DATA = ROOT / "data"
CFG = json.loads((DATA / "meta.json").read_text())["configs"]
SIZES, MODEL_SEEDS, DATA_SEEDS, N_SEED = [200, 400, 600, 800, 1000], [0, 1, 2], [0, 1, 2], 4000
load = lambda p: pickle.load(open(p, "rb"))


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(obj, open(path, "wb"))


def mc_rows(pool, path, recs, paired, n_suf, seed):
    if not path.exists():
        t0 = time.time(); rows = attach_mc(pool, make_rows(recs, paired, n_suf, seed), paired, seed + 1)
        save(rows, path); print(f"  {path.relative_to(DATA)}: {len(rows)} rows {time.time() - t0:.0f}s", flush=True)


def skip_js(ws, rng, chunk=50):
    """Advance the window random stream exactly as attach_js would, without simulating (subset runs)."""
    for _ in range(math.ceil(len(ws) / chunk)):
        rng.integers(1 << 30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seeds", type=int, nargs="+", default=DATA_SEEDS)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--model-seeds", type=int, nargs="+", default=MODEL_SEEDS)
    ap.add_argument("--workers", type=int, default=max(1, min(300, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()
    sim_train = load(DATA / "sim_train.pkl")
    with Pool(a.workers, initializer=D.init_worker, initargs=(CFG["env"], CFG["policy"], CFG["segment"])) as pool:
        # B1 / B2 windows; one random stream over all data seeds and sizes, in this order
        rng = np.random.default_rng(13)
        for d in DATA_SEEDS:
            chain, val = load(DATA / f"d{d}" / "chain_train.pkl"), load(DATA / f"d{d}" / "chain_val.pkl")
            if d in a.data_seeds:
                save(attach_js(pool, windows(val), rng), DATA / f"d{d}" / "windows" / "val.pkl")
            else:
                skip_js(windows(val), rng)
            for n in SIZES:
                if d in a.data_seeds and n in a.sizes:
                    ws = attach_js(pool, windows(chain[:n]), rng); save(ws, DATA / f"d{d}" / "windows" / f"train_n{n}.pkl")
                    print(f"d={d} n={n}: {len(ws)} windows", flush=True)
                else:
                    skip_js(windows(chain[:n]), rng)
        # seed_R sim data around the real states of chain[:n]
        for d in a.data_seeds:
            chain = load(DATA / f"d{d}" / "chain_train.pkl")
            for n in a.sizes:
                seeds, sub = D.chain_seeds(chain[:n]), np.random.default_rng(777 + n + 100000 * d)
                jobs = [(int(sub.integers(1 << 31)), 50, ["seed_R"] * 50, False, seeds) for _ in range(N_SEED // 50)]
                save([r for c in pool.map(D.gen_chunk, jobs) for r in c], DATA / f"d{d}" / f"seedR_n{n}.pkl")
        # residual rows with Monte Carlo sim values
        for s in a.model_seeds:
            mc_rows(pool, DATA / "rows" / f"simtrain_s{s}.pkl", sim_train, False, 1, 200 + s)
        for d in a.data_seeds:
            chain, val = load(DATA / f"d{d}" / "chain_train.pkl"), load(DATA / f"d{d}" / "chain_val.pkl")
            for s in a.model_seeds:
                mc_rows(pool, DATA / f"d{d}" / "rows" / f"val_s{s}.pkl", val, True, 6, 100 + s)
                for n in a.sizes:
                    mc_rows(pool, DATA / f"d{d}" / "rows" / f"paired_n{n}_s{s}.pkl", chain[:n], True, 6, 300 + 10 * s + n)
                    mc_rows(pool, DATA / f"d{d}" / "rows" / f"seedR_n{n}_s{s}.pkl", load(DATA / f"d{d}" / f"seedR_n{n}.pkl"), False, 1, 400 + 10 * s + n)
    print("done")
