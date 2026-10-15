"""Step 2: data that depend on the paired subset size n, for every data seed (the toy's make_rows.py without the
residual variant's Monte Carlo rows).
Outputs (data/, or $CARMAZE_DATA):
  d{d}/seedR_n{n}.pkl                              4000 simulation-only transitions started near the real states of chain_train[:n]
  d{d}/windows/train_n{n}.pkl, d{d}/windows/val.pkl   B1 / B2 windows of the real chains with simulated returns
Usage: python scripts/make_rows.py [--data-seeds 0 1 2] [--sizes 200 400 600 800 1000] [--workers 150]"""
import argparse, json, os, pickle, sys, time  # noqa: E401
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carmaze import data as D  # noqa: E402
from carmaze.route_baselines import attach_js, windows  # noqa: E402

DATA = Path(os.environ.get("CARMAZE_DATA", ROOT / "data"))
CFG = json.loads((DATA / "meta.json").read_text())["configs"]
SIZES, DATA_SEEDS, N_SEED = [200, 400, 600, 800, 1000], [0, 1, 2], 4000
load = lambda p: pickle.load(open(p, "rb"))


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(obj, open(path, "wb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seeds", type=int, nargs="+", default=DATA_SEEDS)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--workers", type=int, default=150)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers, initializer=D.init_worker, initargs=(CFG["env"], CFG["policy"], CFG["segment"])) as pool:
        for d in a.data_seeds:
            chain, val = load(DATA / f"d{d}" / "chain_train.pkl"), load(DATA / f"d{d}" / "chain_val.pkl")
            rng = np.random.default_rng(13 + d)
            save(attach_js(pool, windows(val), rng), DATA / f"d{d}" / "windows" / "val.pkl")
            for n in a.sizes:
                ws = attach_js(pool, windows(chain[:n]), rng)
                save(ws, DATA / f"d{d}" / "windows" / f"train_n{n}.pkl")
                seeds, sub = D.chain_seeds(chain[:n]), np.random.default_rng(777 + n + 100000 * d)
                jobs = [(int(sub.integers(1 << 31)), 50, seeds) for _ in range(N_SEED // 50)]
                save([r for c in pool.map(D.gen_seed_R, jobs) for r in c], DATA / f"d{d}" / f"seedR_n{n}.pkl")
                print(f"d={d} n={n}: {len(ws)} windows, {N_SEED} seed_R transitions ({time.time() - t0:.0f}s)", flush=True)
    print("done")
