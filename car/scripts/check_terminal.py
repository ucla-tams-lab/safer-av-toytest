"""How much of the real-sim gap is the terminal-state effect? (README: "Segment-wise correction has a floor")
(1) Paired transitions (same start, same goal): difference of the real and sim end states.
(2) Exact segment-wise correction: run the query in sim; at every simulated segment start x_k^S drive the REAL car for
    that goal and add up the real segment costs (J_seg). Then J_R - J_S = (J_seg - J_S) [local part, what any perfect
    segment-wise correction captures] + (J_R - J_seg) [terminal-state part: real starts later segments from other states].
Usage: python scripts/check_terminal.py"""
import json, os, pickle, sys  # noqa: E401
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carmaze import data as D  # noqa: E402
from carmaze import maze as M  # noqa: E402
from carmaze.segment import COSTS, run_route, run_segment  # noqa: E402

DATA = os.environ.get("CARMAZE_DATA", str(ROOT / "data"))
CFG = json.load(open(f"{DATA}/meta.json"))["configs"]
R_ROLL = 20

def job(args):
    seed, x1, goals = args
    rng = np.random.default_rng(seed)
    JS, JG = [], []
    for r in range(R_ROLL):
        D._W["S"].reseed(int(rng.integers(1 << 31)))
        X, infos = run_route(D._W["S"], D._W["pol"], x1, goals, D._W["cfg"])
        JS.append([sum(i[f"y_{c}"] for i in infos) for c in COSTS])
        tot = np.zeros(len(COSTS))
        for k, g in enumerate(goals):
            D._W["R"].reseed(int(rng.integers(1 << 31)))
            _, info = run_segment(D._W["R"], D._W["pol"], X[k], M.GOALS[g], D._W["cfg"])
            tot += [info[f"y_{c}"] for c in COSTS]
        JG.append(tot)
    return np.mean(JS, 0), np.mean(JG, 0)

if __name__ == "__main__":
    # (1) end-state differences on the paired transitions of data seed 0
    ch = pickle.load(open(f"{DATA}/d0/chain_train.pkl", "rb"))
    XR = np.array([r["R"]["x_plus"] for r in ch]); XS = np.array([r["S"]["x_plus"] for r in ch])
    dpos = np.linalg.norm(XR[:, :2] - XS[:, :2], axis=1)
    dyaw = np.degrees(np.abs((XR[:, 2] - XS[:, 2] + np.pi) % (2 * np.pi) - np.pi))
    print(f"(1) paired end states, real - sim ({len(ch)} transitions): position {dpos.mean():.2f} m (90%: {np.quantile(dpos, .9):.2f}), "
          f"speed {np.mean(XR[:,3]-XS[:,3]):+.2f} m/s (|.| {np.abs(XR[:,3]-XS[:,3]).mean():.2f}), heading {dyaw.mean():.1f} deg, "
          f"sideways speed |.| {np.abs(XR[:,4]-XS[:,4]).mean():.2f} m/s, turn rate |.| {np.abs(XR[:,5]-XS[:,5]).mean():.2f} rad/s; "
          f"end speed sim {XS[:,3].mean():.2f}, real {XR[:,3].mean():.2f} m/s")
    q = pickle.load(open(f"{DATA}/queries_test.pkl", "rb"))
    q = [x for x in q if x["K"] in (2, 4, 8) and x["tag"] in ("random", "moving")]
    rng = np.random.default_rng(0)
    with Pool(150, initializer=D.init_worker, initargs=(CFG["env"], CFG["policy"], CFG["segment"])) as p:
        out = p.map(job, [(int(rng.integers(1 << 31)), x["x1"], x["goals"]) for x in q], chunksize=1)
    print("(2) K, start: per cost  total gap J_R-J_S | local part (J_seg-J_S) | terminal-state part (J_R-J_seg) and its share | "
          "MAE of the exact segment-wise correction")
    for st in ("rest", "moving"):
        for K in (2, 4, 8):
            idx = [i for i, x in enumerate(q) if x["K"] == K and (x["tag"] == "moving") == (st == "moving")]
            cells = []
            for j, c in enumerate(COSTS):
                JR = np.array([q[i]["R"][c]["mean"] for i in idx]); JS = np.array([out[i][0][j] for i in idx]); JG = np.array([out[i][1][j] for i in idx])
                tot, loc, term = (JR - JS).mean(), (JG - JS).mean(), (JR - JG).mean()
                cells.append(f"{c[:5]} {tot:+.3f} | {loc:+.3f} | {term:+.3f} ({100 * term / tot:3.0f}%) | MAE {np.abs(JR - JG).mean():.3f}")
            print(f"  K={K} {st:6s} " + " || ".join(cells), flush=True)
