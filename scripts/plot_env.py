"""Figure of the environment for the README: the same goal sequences driven in simulation and in "real".
Out: outputs/report/fig_env.png"""
import json, sys  # noqa: E401
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safer_av import maze as M  # noqa: E402
from safer_av.envs import make_env  # noqa: E402
from safer_av.policy import PDPolicy  # noqa: E402
from safer_av.segment import SegmentConfig, clearance, run_segment  # noqa: E402

CFG = {k: json.loads((ROOT / "configs" / f"{k}.json").read_text()) for k in ("env", "policy", "segment")}
ROUTES = [("H", list("GDEBADGH")), ("A", list("BCFIHEDG"))]   # two K=8 test sequences, start at rest
pol, cfg = PDPolicy(CFG["policy"]["kp"], CFG["policy"]["kd"]), SegmentConfig(**CFG["segment"])
envs = {"sim": make_env("S", **CFG["env"]["S"]), "real": make_env("R", **CFG["env"]["R"])}   # noise-free for the picture

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (start, goals) in zip(axes, ROUTES):
    for r in range(M.N_ROWS):
        for c in range(M.N_COLS):
            if M.MAZE_MAP[r][c]:
                ax.add_patch(plt.Rectangle(M.cell_to_xy((r, c)) - .5, 1, 1, color="#8b6b4a"))
    for k, xy in M.GOALS.items():
        ax.add_patch(plt.Circle(xy, 0.2, color="#bbb", alpha=.6)); ax.text(xy[0] + .14, xy[1] + .14, k, fontsize=11, weight="bold", color="#444")
    safety = {}
    for name, col in (("sim", "tab:blue"), ("real", "tab:red")):
        x, trs, safety[name] = np.r_[M.GOALS[start], 0, 0], [], 0.0
        for g in goals:
            x, info = run_segment(envs[name], pol, x, M.GOALS[g], cfg, keep_traj=True)
            trs.append(info["traj"]); safety[name] += info["y_safety"]
        tr = np.concatenate(trs)
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=2, label=name)
    near = np.array([clearance(p[:2]) < 0.1 for p in tr])
    ax.scatter(tr[near, 0], tr[near, 1], s=12, color="black", zorder=5, label="real within 0.1 of a wall")
    ax.plot(*M.GOALS[start], "g*", ms=18, mec="k", zorder=6, ls="none", label="start (at rest)")
    ax.set_title(" → ".join([start] + goals) + f"\nSafety cost: sim {safety['sim']:.1f}, real {safety['real']:.1f}", fontsize=12)
    ax.set_xlim(-3.7, 3.7); ax.set_ylim(-2.7, 2.7); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=11, frameon=False)
fig.suptitle("Same policy, same goals: simulation vs \"real\" (weaker force and brakes)", fontsize=13)
fig.tight_layout(rect=[0, 0.07, 1, 0.95])
out = ROOT / "outputs" / "report" / "fig_env.png"; out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, facecolor="white"); print(out)
