"""Sim vs real on the same goal sequences: car paths of several runs (different actuation-noise seeds) in each.
Usage: python scripts/plot_routes.py [--runs 5] [--noise 0.05] [--out figs/routes_sim_real.png]"""
import argparse, os, sys  # noqa: E401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from carmaze import maze as M  # noqa: E402
from carmaze.car import CarWorld  # noqa: E402
from carmaze.policy import CarPolicy  # noqa: E402
from carmaze.segment import run_route  # noqa: E402

# outer ring clockwise (downhill) | the same ring counter-clockwise (uphill) | inner cross
ROUTES = [("A", "BCFIHGDA"), ("A", "DGHIFCBA"), ("D", "EFIHEBAD")]


def start_state(world, start, first_goal):
    p = M.GOALS[start]
    w = M.waypoint(p, M.GOALS[first_goal])
    return world.rest_state(p, float(np.arctan2(w[1] - p[1], w[0] - p[0])))


def draw_maze(ax):
    for rc in M.WET_CELLS:
        x, y = M.cell_to_xy(rc)
        ax.add_patch(plt.Rectangle((x - M.CELL / 2, y - M.CELL / 2), M.CELL, M.CELL, color="#cfe6ff", lw=0, zorder=0))
    for r, c in M.wall_cells():
        x, y = M.cell_to_xy((r, c))
        ax.add_patch(plt.Rectangle((x - M.CELL / 2, y - M.CELL / 2), M.CELL, M.CELL, color="#cfcfcf", lw=0))
    for rc, d in M.SLOPE_CELLS.items():
        x, y = M.cell_to_xy(rc); dx, dy = M.SLOPE_DIR[d]
        ax.annotate("", xy=(x + 0.6 * dx, y + 0.6 * dy), xytext=(x - 0.6 * dx, y - 0.6 * dy),
                    arrowprops=dict(arrowstyle="-|>", color="#7a5c2e", lw=1.4), zorder=1)
    for k, p in M.GOALS.items():
        ax.add_patch(plt.Circle(p, 0.2 * M.CELL, fill=False, ls=":", color="k", lw=0.8))
        ax.text(p[0] + 0.45, p[1] + 0.45, k, fontsize=11, fontweight="bold")
    ax.set_xlim(-M.X_CENTER, M.X_CENTER); ax.set_ylim(-M.Y_CENTER, M.Y_CENTER); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, "figs", "routes_sim_real.png"))
    ap.add_argument("--accel", type=float, default=None)
    ap.add_argument("--decel", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--mu_wet", type=float, default=None)
    a = ap.parse_args()
    real = {k: v for k, v in (("accel", a.accel), ("decel", a.decel), ("mu", a.mu), ("mu_wet", a.mu_wet)) if v is not None}
    pol = CarPolicy()
    worlds = {"S": CarWorld("S", noise_std=a.noise), "R": CarWorld("R", noise_std=a.noise, **real)}
    fig, axes = plt.subplots(1, len(ROUTES), figsize=(6.4 * len(ROUTES), 5.6))
    for ax, (start, goals) in zip(axes, ROUTES):
        draw_maze(ax)
        stats = {}
        for kind, color in (("S", "#1f5fa8"), ("R", "#c0392b")):
            w = worlds[kind]
            tot = dict(time=[], fails=[], contact=[], safety=[])
            for m in range(a.runs):
                w.reseed(100 * m + (kind == "R"))
                _, infos = run_route(w, pol, start_state(w, start, goals[0]), list(goals), keep_traj=True)
                tr = np.concatenate([i["traj"] for i in infos])
                ax.plot(tr[:, 0], tr[:, 1], color=color, lw=1.0, alpha=0.6, label={"S": "sim", "R": "real"}[kind] if m == 0 else None)
                tot["time"].append(sum(i["tau"] for i in infos)); tot["fails"].append(sum(i["fail"] for i in infos))
                tot["contact"].append(sum(i["contact"] for i in infos)); tot["safety"].append(sum(i["y_safety"] for i in infos))
            stats[kind] = {k: float(np.mean(v)) for k, v in tot.items()}
        arrow = " \u2192 "
        title = f"{start}{arrow}{arrow.join(goals)}"
        for kind, lab in (("S", "sim"), ("R", "real")):
            st = stats[kind]
            title += f"\n{lab}: {st['time']:.1f} s, Safety {st['safety']:.2f}, {st['contact']:.2f} s wall contact"
        ax.set_title(title, fontsize=9)
        print(start, goals, stats, flush=True)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Car maze: the same goal sequences in sim (blue) and real (red), {a.runs} runs each; arrows = downhill, "
                 "light blue = wet (real only)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print("figure:", a.out)


if __name__ == "__main__":
    main()
