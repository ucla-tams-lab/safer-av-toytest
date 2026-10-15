"""Variant: a separate network for every remaining-goal length (V_1..V_8 and h_1..h_8; target-only W_1..W_8) instead
of one network shared by all lengths. Same data, training budget per fit and evaluation as run.py; only the learned
values change. Usage: python scripts/run_perlength.py D N S
Out: outputs/results_perlength/res_d{d}_n{n}_s{s}.json (target_only, V_direct, V_simbacked; the run.py format)"""
import json, os, pickle, sys, time  # noqa: E401
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from carmaze.backward import predict_W, train_target_only  # noqa: E402
from carmaze.ours import direct, sim_backed, train_ours  # noqa: E402
from carmaze.segment import COSTS  # noqa: E402
from run import metrics  # noqa: E402

DATA = Path(os.environ.get("CARMAZE_DATA", ROOT / "data"))
OUT = Path(os.environ.get("CARMAZE_OUT", ROOT / "outputs" / "results_perlength"))
load = lambda p: pickle.load(open(p, "rb"))

if __name__ == "__main__":
    d, n, s = map(int, sys.argv[1:4])
    OUT.mkdir(parents=True, exist_ok=True); DD = DATA / f"d{d}"; t_all = time.time()
    q, ro = load(DATA / "queries_test.pkl"), load(DATA / "sim_rollouts_test.pkl")
    Ks = np.array([x["K"] for x in q]); x1s = [x["x1"] for x in q]; routes = [x["goals"] for x in q]
    starts = np.array(["moving" if x["tag"] == "moving" else "rest" for x in q])
    paired, val = load(DD / "chain_train.pkl")[:n], load(DD / "chain_val.pkl")
    sim = load(DATA / "sim_train.pkl") + load(DD / f"seedR_n{n}.pkl")
    res = {}
    for cost in COSTS:
        t0 = time.time()
        gt = np.array([x["R"][cost]["mean"] for x in q])
        P = {}
        W_to, _ = train_target_only(paired, val, cost, seed=s, per_length=True)
        P["target_only"] = predict_W(W_to, x1s, routes)
        V, h, _ = train_ours(paired, val, sim, cost, seed=s, per_length=True)
        P["V_direct"] = direct(V, x1s, routes)
        P["V_simbacked"] = np.array([sim_backed(V, h, rr, x["goals"], cost) for rr, x in zip(ro, q)])
        res[cost] = {k: metrics(v, gt, Ks, starts) for k, v in P.items()}
        print(f"[per-length d={d} n={n} s={s}] {cost}: K=8 MAE rest / moving " + " ".join(
            f"{k}={res[cost][k]['rest|K=8']['mae']:.3f}/{res[cost][k]['moving|K=8']['mae']:.3f}" for k in P) + f" ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(OUT / f"res_d{d}_n{n}_s{s}.json", "w"), indent=1)
    print(f"[per-length d={d} n={n} s={s}] done in {time.time() - t_all:.0f}s", flush=True)
