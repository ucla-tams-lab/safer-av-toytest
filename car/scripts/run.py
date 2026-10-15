"""Step 3: one run = (data seed d, paired size n, model seed s). Trains every method for every cost and evaluates
on the evaluation queries (rest-start and moving-start, reported separately). Usage: python scripts/run.py D N S

Methods
  baselines : raw_sim (raw simulation), outcome_only (segment-wise outcome correction, SCAPE-style),
              target_only (backward Bellman learning on real targets only), B1_mc_real (direct regression of real
              window returns), B2_route_scape (route-level SCAPE)
  ours      : V_direct (direct estimation, the main estimator), V_simbacked (simulation-backed estimation)
Out: outputs/results/res_d{d}_n{n}_s{s}.json (MAE / bias / RMSE / Spearman per start type and K), preds_*.json"""
import json, os, pickle, sys, time  # noqa: E401
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carmaze.backward import outcome_only_estimate, predict_W, train_outcome_calibrator, train_target_only  # noqa: E402
from carmaze.ours import direct, sim_backed, train_ours  # noqa: E402
from carmaze.route_baselines import b1_mc_real, b2_route_scape  # noqa: E402
from carmaze.segment import COSTS  # noqa: E402

DATA = Path(os.environ.get("CARMAZE_DATA", ROOT / "data"))
OUT = Path(os.environ.get("CARMAZE_OUT", ROOT / "outputs" / "results"))
load = lambda p: pickle.load(open(p, "rb"))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def metrics(pred, gt, Ks, starts):
    out = {}
    for st in ("rest", "moving"):
        for K in sorted(set(Ks)):
            m = (Ks == K) & (starts == st)
            if not m.any():
                continue
            e = pred[m] - gt[m]
            out[f"{st}|K={K}"] = dict(mae=float(np.abs(e).mean()), bias=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                                      spearman=spearman(pred[m], gt[m]) if m.sum() > 2 and np.std(gt[m]) > 0 else None)
    return out


if __name__ == "__main__":
    d, n, s = map(int, sys.argv[1:4])
    OUT.mkdir(parents=True, exist_ok=True); DD = DATA / f"d{d}"; t_all = time.time()
    q, ro = load(DATA / "queries_test.pkl"), load(DATA / "sim_rollouts_test.pkl")
    Ks = np.array([x["K"] for x in q]); x1s = [x["x1"] for x in q]; routes = [x["goals"] for x in q]
    starts = np.array(["moving" if x["tag"] == "moving" else "rest" for x in q])
    paired, val = load(DD / "chain_train.pkl")[:n], load(DD / "chain_val.pkl")
    sim = load(DATA / "sim_train.pkl") + load(DD / f"seedR_n{n}.pkl")
    w_tr, w_va = load(DD / "windows" / f"train_n{n}.pkl"), load(DD / "windows" / "val.pkl")
    res, preds = {}, {}
    for cost in COSTS:
        t0 = time.time()
        gt = np.array([x["R"][cost]["mean"] for x in q])
        P = {"raw_sim": np.array([np.mean([r["y"][cost].sum() for r in rr]) for rr in ro])}
        h_out, _ = train_outcome_calibrator(paired, val, cost, seed=s)
        P["outcome_only"] = np.array([outcome_only_estimate(h_out, rr, x["goals"], cost) for rr, x in zip(ro, q)])
        W_to, _ = train_target_only(paired, val, cost, seed=s)
        P["target_only"] = predict_W(W_to, x1s, routes)
        P["B1_mc_real"] = b1_mc_real(w_tr, w_va, cost, x1s, routes, seed=s)
        P["B2_route_scape"] = b2_route_scape(w_tr, w_va, cost, x1s, routes, P["raw_sim"], seed=s)
        V, h, _ = train_ours(paired, val, sim, cost, seed=s)
        P["V_direct"] = direct(V, x1s, routes)
        P["V_simbacked"] = np.array([sim_backed(V, h, rr, x["goals"], cost) for rr, x in zip(ro, q)])
        res[cost] = {k: metrics(v, gt, Ks, starts) for k, v in P.items()}
        preds[cost] = {k: v.tolist() for k, v in P.items()} | {"gt": gt.tolist()}
        print(f"[d={d} n={n} s={s}] {cost}: K=8 MAE rest / moving " + " ".join(
            f"{k}={res[cost][k]['rest|K=8']['mae']:.3f}/{res[cost][k]['moving|K=8']['mae']:.3f}" for k in P) + f" ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(OUT / f"res_d{d}_n{n}_s{s}.json", "w"), indent=1)
    json.dump(preds, open(OUT / f"preds_d{d}_n{n}_s{s}.json", "w"))
    print(f"[d={d} n={n} s={s}] done in {time.time() - t_all:.0f}s", flush=True)
