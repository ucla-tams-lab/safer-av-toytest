"""Two corners of the clockwise ring zoomed in, and the speed along it, in sim and real (5 runs each).
Usage: python scripts/plot_zoom.py  ->  figs/routes_zoom_speed.png"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from carmaze import maze as M
from carmaze.car import CarWorld
from carmaze.policy import CarPolicy
from carmaze.segment import run_route
from plot_routes import draw_maze, start_state

start, goals = "A", "BCFIHGDA"                       # outer ring clockwise: downhill all the way
pol = CarPolicy()
runs = {}
for kind in ("S", "R"):
    w = CarWorld(kind, noise_std=0.05)
    runs[kind] = []
    for m in range(5):
        w.reseed(100 * m + (kind == "R"))
        _, infos = run_route(w, pol, start_state(w, start, goals[0]), list(goals), keep_traj=True)
        tr = np.concatenate([infos[0]["traj"]] + [i["traj"][1:] for i in infos[1:]])     # drop repeated boundary points
        ends = np.cumsum([len(infos[0]["traj"])] + [len(i["traj"]) - 1 for i in infos[1:]]) - 1
        runs[kind].append((tr, ends, [i["y_safety"] for i in infos]))
COL = {"S": "#1f5fa8", "R": "#c0392b"}
LAB = {"S": "sim", "R": "real"}
fig = plt.figure(figsize=(17, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
for k, (name, title) in enumerate([("C", "Corner C: after the downhill top row, wet in real"),
                                   ("I", "Corner I: after the downhill right column")]):
    ax = fig.add_subplot(gs[0, k])
    draw_maze(ax)
    for t in ax.texts:
        t.set_clip_on(True)
    for kind in ("S", "R"):
        for j, (tr, _, _) in enumerate(runs[kind]):
            ax.plot(tr[:, 0], tr[:, 1], color=COL[kind], lw=1.6, alpha=0.7, label=LAB[kind] if j == 0 else None)
    cx, cy = M.GOALS[name]
    ax.set_xlim(cx - 4.6, cx + 2.0); ax.set_ylim(cy - 4.6 if name == "C" else cy - 2.0, cy + 2.0 if name == "C" else cy + 4.6)
    ax.set_title(title, fontsize=10)
    if k == 0:
        ax.legend(loc="lower left", fontsize=9)
ax = fig.add_subplot(gs[0, 2])
for kind in ("S", "R"):
    for j, (tr, ends, _) in enumerate(runs[kind]):
        step = np.linalg.norm(np.diff(tr, axis=0), axis=1)
        dist = np.r_[0, np.cumsum(step)]; v = np.r_[0, step / 0.05]
        ax.plot(dist, v, color=COL[kind], lw=1.2, alpha=0.7, label=LAB[kind] if j == 0 else None)
tr, ends, _ = runs["S"][0]
dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr, axis=0), axis=1))]
for g, e in zip(goals, ends):
    ax.axvline(dist[e], color="k", lw=0.6, ls=":")
    ax.text(dist[e], 5.6, g, ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("distance driven [m]"); ax.set_ylabel("speed [m/s]"); ax.set_ylim(0, 6); ax.grid(alpha=.25)
ax.set_title("Speed along A > B > C > F > I > H > G > D > A (dotted: goals reached)", fontsize=10)
ax.legend(loc="lower right", fontsize=9)
sS = np.mean([sum(r[2]) for r in runs["S"]]); sR = np.mean([sum(r[2]) for r in runs["R"]])
fig.suptitle(f"Clockwise ring (downhill), 5 runs each. Both reach a junction at about 4.3 m/s; in the turn the real car brakes less hard and has less grip, "
             f"so it slows down later and swings wide.  Safety over the 8 goals: sim {sS:.2f}, real {sR:.2f}", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = str(Path(__file__).resolve().parents[1] / "figs" / "routes_zoom_speed.png")
fig.savefig(out, dpi=120, facecolor="white")
print(out)
for kind in ("S", "R"):
    tr, ends, ys = runs[kind][0]
    step = np.linalg.norm(np.diff(tr, axis=0), axis=1); v = np.r_[0, step / 0.05]
    print(LAB[kind], "speed at each goal:", " ".join(f"{g}:{v[e]:.1f}" for g, e in zip(goals, ends)), "| per-goal Safety:", " ".join(f"{y:.2f}" for y in ys))
