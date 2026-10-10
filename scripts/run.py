"""Step 3: one run = (data seed d, paired size n, model seed s). Trains every method for every cost and
evaluates on the 536 test routes. Usage: python scripts/run.py D N S

Methods
  baselines : raw_sim (raw simulation), outcome_only (segment-wise outcome correction, SCAPE-style),
              target_only (backward Bellman on real data only), B1_mc_real (direct regression of real window returns),
              B2_route_scape (route-level SCAPE)
  proposed  : W_direct (direct amortized estimator), W_simbacked (simulation-backed estimator),
              res_direct (residual variant W = W_S + D, direct)
Out: outputs/results/res_d{d}_n{n}_s{s}.json (MAE / bias / RMSE / Spearman per K), preds_d{d}_n{n}_s{s}.json
"""
import json, pickle, sys, time  # noqa: E401
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safer_av import maze as M  # noqa: E402
from safer_av.backward import outcome_only_estimate, predict_W, train_outcome_calibrator, train_target_only  # noqa: E402
from safer_av.ours import direct, sim_backed, train_ours  # noqa: E402
from safer_av.residual import predict_residual, train_residual  # noqa: E402
from safer_av.route_baselines import b1_mc_real, b2_route_scape  # noqa: E402
from safer_av.segment import COSTS  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs" / "results"
load = lambda p: pickle.load(open(p, "rb"))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def metrics(pred, gt, Ks):
    out = {}
    for K in sorted(set(Ks)):
        m = Ks == K
        e = pred[m] - gt[m]
        out[f"K={K}"] = dict(mae=float(np.abs(e).mean()), bias=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                             spearman=spearman(pred[m], gt[m]) if m.sum() > 2 and np.std(gt[m]) > 0 else None)
    return out


if __name__ == "__main__":
    d, n, s = map(int, sys.argv[1:4])
    OUT.mkdir(parents=True, exist_ok=True); DD = DATA / f"d{d}"; t_all = time.time()
    q, ro = load(DATA / "queries_test.pkl"), load(DATA / "sim_rollouts_test.pkl")
    Ks = np.array([x["K"] for x in q]); x1s = [np.r_[M.GOALS[x["start"]], 0, 0] for x in q]; routes = [x["goals"] for x in q]
    paired, val = load(DD / "chain_train.pkl")[:n], load(DD / "chain_val.pkl")
    sim = load(DATA / "sim_train.pkl") + load(DD / f"seedR_n{n}.pkl")
    p_rows, v_rows = load(DD / "rows" / f"paired_n{n}_s{s}.pkl"), load(DD / "rows" / f"val_s{s}.pkl")
    s_rows = load(DATA / "rows" / f"simtrain_s{s}.pkl") + load(DD / "rows" / f"seedR_n{n}_s{s}.pkl")
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
        W, h, _ = train_ours(paired, val, sim, cost, seed=s)
        P["W_direct"] = direct(W, x1s, routes)
        P["W_simbacked"] = np.array([sim_backed(W, h, rr, x["goals"], cost) for rr, x in zip(ro, q)])
        D_res, _ = train_residual(p_rows, v_rows, s_rows, cost, seed=s)
        P["res_direct"] = predict_residual(D_res, x1s, routes, P["raw_sim"])
        res[cost] = {k: metrics(v, gt, Ks) for k, v in P.items()}
        preds[cost] = {k: v.tolist() for k, v in P.items()} | {"gt": gt.tolist()}
        print(f"[d={d} n={n} s={s}] {cost}: K=8 MAE " + " ".join(f"{k}={res[cost][k]['K=8']['mae']:.3f}" for k in P) + f" ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(OUT / f"res_d{d}_n{n}_s{s}.json", "w"), indent=1)
    json.dump(preds, open(OUT / f"preds_d{d}_n{n}_s{s}.json", "w"))
    print(f"[d={d} n={n} s={s}] done in {time.time() - t_all:.0f}s", flush=True)
