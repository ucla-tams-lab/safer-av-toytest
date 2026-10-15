"""Append paired transitions to the paired chains of every data seed, for paired sizes n above 1000. The first 1000
transitions and the validation chains stay as they are (new chains come from a separate seed), so every run with
n <= 1000 is unchanged. Then: python scripts/make_rows.py --sizes 1500 2000 and run.py with N = 1500, 2000.
Usage: python scripts/extend_chains.py [--total 2000] [--workers 150]"""
import argparse, json, os, pickle, sys, time  # noqa: E401
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carmaze import data as D  # noqa: E402
from carmaze.car import CarWorld  # noqa: E402

DATA = Path(os.environ.get("CARMAZE_DATA", ROOT / "data"))
CFG = json.loads((DATA / "meta.json").read_text())["configs"]
EXTRA_SEED = 9191

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=150)
    a = ap.parse_args()
    t0 = time.time()
    q_test = pickle.load(open(DATA / "queries_test.pkl", "rb"))
    forbidden = {(q["start"], tuple(q["goals"][:4])) for q in q_test if q["tag"] != "moving" and len(q["goals"]) >= 4}
    world = CarWorld("R")
    with Pool(a.workers, initializer=D.init_worker, initargs=(CFG["env"], CFG["policy"], CFG["segment"])) as pool:
        for d in (0, 1, 2):
            p = DATA / f"d{d}" / "chain_train.pkl"
            old = pickle.load(open(p, "rb"))[:1000]
            rng = np.random.default_rng(EXTRA_SEED + 1000 * d)
            new = [r for c in pool.map(D.gen_paired_chain, D.make_chain_specs(rng, a.total - len(old), forbidden, world)) for r in c]
            tr = (old + new)[:a.total]
            pickle.dump(tr, open(p, "wb"))
            print(f"d={d}: {len(old)} + {len(tr) - len(old)} = {len(tr)} paired transitions; real fails "
                  f"{np.mean([r['R']['fail'] for r in tr[len(old):]]):.3f} in the new ones ({time.time() - t0:.0f}s)", flush=True)
