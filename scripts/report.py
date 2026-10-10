"""Step 4: aggregate the 45 runs (3 data seeds x 3 model seeds x 5 n), draw the figures, write the results page.
Out: outputs/report/{summary.json, wins.json, fig_*.png, report.html}"""
import base64, json, pickle, sys  # noqa: E401
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
from safer_av.segment import COSTS, SegmentConfig, clearance, run_segment  # noqa: E402

DATA, RES, OUT = ROOT / "data", ROOT / "outputs" / "results", ROOT / "outputs" / "report"
OUT.mkdir(parents=True, exist_ok=True)
N, SEEDS, DSEEDS, KS = [200, 400, 600, 800, 1000], [0, 1, 2], [0, 1, 2], ["K=1", "K=2", "K=4", "K=8"]
BASE = [("raw_sim", "Raw sim"), ("outcome_only", "Seg. outcome"), ("target_only", "Target-only"), ("B1_mc_real", "B1 MC-real"),
        ("B2_route_scape", "B2 route-SCAPE")]
OURS = [("W_direct", "Direct"), ("W_simbacked", "Sim-backed"), ("res_direct", "Residual direct")]
METHODS = BASE + OURS
LABEL = dict(METHODS)
CNAME = dict(safety=("Safety", "minimum wall clearance"), expo=("Exposure", "integrated safety-margin intrusion"),
             ttc=("TTC", "time to collision"), path=("Path", "path length"), time=("Time", "arrival time"))
ORDER = ["safety", "expo", "ttc", "path", "time"]
load = lambda p: pickle.load(open(p, "rb"))

# ---------------- aggregate ----------------
R = {(d, n, s): json.loads((RES / f"res_d{d}_n{n}_s{s}.json").read_text()) for d in DSEEDS for n in N for s in SEEDS}
S = {}
for c in COSTS:
    for K in KS:
        for n in N:
            to = np.array([R[(d, n, s)][c]["target_only"][K]["mae"] for d in DSEEDS for s in SEEDS])
            for m, _ in METHODS:
                v = np.array([R[(d, n, s)][c][m][K]["mae"] for d in DSEEDS for s in SEEDS])
                b = np.array([R[(d, n, s)][c][m][K]["bias"] for d in DSEEDS for s in SEEDS])
                sp = np.array([R[(d, n, s)][c][m][K]["spearman"] for d in DSEEDS for s in SEEDS], dtype=float)
                S[f"{c}|{K}|{n}|{m}"] = dict(mae=float(v.mean()), std=float(v.std()), bias=float(b.mean()), spearman=float(np.nanmean(sp)),
                                             wins_vs_TO=int((v < to).sum()), runs=len(v))
WINS = {c: {m: sum(S[f"{c}|K=8|{n}|{m}"]["wins_vs_TO"] for n in N) for m, _ in METHODS if m != "target_only"} for c in COSTS}
json.dump(S, open(OUT / "summary.json", "w"), indent=1); json.dump(WINS, open(OUT / "wins.json", "w"), indent=1)
g = lambda c, K, n, m: S[f"{c}|{K}|{n}|{m}"]["mae"]
TOT = lambda m: sum(WINS[c][m] for c in COSTS)

# ---------------- facts from the data ----------------
q = load(DATA / "queries_test.pkl"); chain0 = load(DATA / "d0" / "chain_train.pkl")
sel8 = [x for x in q if x["K"] == 8 and x["tag"] == "random"]
gap = {}
for c in COSTS:
    Js, Jr = np.array([x["S"][c]["mean"] for x in sel8]), np.array([x["R"][c]["mean"] for x in sel8])
    ra, rb = np.argsort(np.argsort(Js)), np.argsort(np.argsort(Jr))
    gap[c] = dict(ratio=float(Jr.mean() / Js.mean()), rank=float(np.corrcoef(ra, rb)[0, 1]), JR=float(Jr.mean()))
arr_R = np.median([np.linalg.norm(r["R"]["x_plus"][2:]) for r in chain0 if not r["R"]["fail"]])
arr_S = np.median([np.linalg.norm(r["S"]["x_plus"][2:]) for r in chain0 if not r["S"]["fail"]])
start_sp = np.linalg.norm([r["x"][2:] for r in chain0], axis=1)
path_gap = float(np.mean([abs(r["R"]["y_path"] - r["S"]["y_path"]) for r in chain0]))
win_len = {}
for n in (200, 1000):
    c8 = [sum(len(w["s"]) == 8 for w in load(DATA / f"d{d}" / "windows" / f"train_n{n}.pkl")) for d in DSEEDS]
    win_len[n] = (min(c8), max(c8))
rest_max = max((len(w["s"]) for d in DSEEDS for w in load(DATA / f"d{d}" / "windows" / "train_n1000.pkl")
                if np.linalg.norm(w["x"][2:]) < 1e-9), default=0)

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = dict(target_only="#6b6b6b", W_direct="#7aa6d8", W_simbacked="#1f5fa8", res_direct="#c96b2c")
TITLE = dict(safety="Safety (min wall clearance)", expo="Exposure (integrated intrusion)", ttc="TTC (time-to-collision)", path="Path length", time="Time")
fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
for ax, c in zip(axes, ORDER):
    for m in ["target_only", "W_direct", "W_simbacked", "res_direct"]:
        ax.errorbar(N, [g(c, "K=8", n, m) for n in N], yerr=[S[f"{c}|K=8|{n}|{m}"]["std"] for n in N], color=COL[m],
                    ls="--" if m == "target_only" else "-", marker="o", ms=4, lw=1.4, capsize=2, label=LABEL[m])
    ax.set_title(TITLE[c]); ax.set_xlabel("paired transitions n"); ax.set_xticks(N); ax.set_ylim(bottom=0); ax.grid(alpha=.25)
axes[0].set_ylabel("K=8 MAE (mean ± std, 9 runs)"); axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_mae_vs_n.png", dpi=130, facecolor="white"); plt.close(fig)

fig, axes = plt.subplots(1, 5, figsize=(21, 4))
for ax, c in zip(axes, ORDER):
    Js, Jr = np.array([x["S"][c]["mean"] for x in sel8]), np.array([x["R"][c]["mean"] for x in sel8])
    ax.scatter(Js, Jr, s=12, color="#1f5fa8", alpha=.7)
    lim = [min(Js.min(), Jr.min()), max(Js.max(), Jr.max())]; ax.plot(lim, lim, color="#999", lw=1, ls="--")
    ax.set_title(f"{TITLE[c]}\nreal/sim ×{gap[c]['ratio']:.2f}, rank corr {gap[c]['rank']:.2f}", fontsize=10)
    ax.set_xlabel("simulation  J_S"); ax.grid(alpha=.25)
axes[0].set_ylabel("real  J_R  (K=8 sequences)")
fig.tight_layout(); fig.savefig(OUT / "fig_gap_structure.png", dpi=130, facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(start_sp, bins=np.linspace(0, 5.2, 53), color="#1f5fa8", alpha=.75, density=True, label="start states of paired transitions (data seed 0)")
for v, lab, col in ((arr_R, "real arrival", "#8a3b12"), (arr_S, "sim arrival", "#555")):
    ax.axvline(v, color=col, lw=1.5, ls="--"); ax.text(v + .05, ax.get_ylim()[1] * .85, f"{lab}\n≈ {v:.1f}", color=col, fontsize=9)
ax.set_xlabel("start-state speed"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_coverage.png", dpi=130, facecolor="white"); plt.close(fig)

CFG = json.loads((DATA / "meta.json").read_text())["configs"]
pol, cfg = PDPolicy(CFG["policy"]["kp"], CFG["policy"]["kd"]), SegmentConfig(**CFG["segment"])
envs = dict(S=make_env("S", **CFG["env"]["S"]), R=make_env("R", **CFG["env"]["R"]))
fig, axes = plt.subplots(2, 3, figsize=(21, 11))
for ax, x in zip(axes.ravel(), sel8[:6]):
    for r in range(M.N_ROWS):
        for c in range(M.N_COLS):
            if M.MAZE_MAP[r][c]:
                ax.add_patch(plt.Rectangle(M.cell_to_xy((r, c)) - .5, 1, 1, color="#8b6b4a"))
    for k, xy in M.GOALS.items():
        ax.add_patch(plt.Circle(xy, 0.2, color="#bbb", alpha=.5)); ax.text(xy[0] + .13, xy[1] + .13, k, fontsize=10, weight="bold", color="#555")
    for e, col in (("S", "tab:blue"), ("R", "tab:red")):
        xx, trs, ys, yp = np.r_[M.GOALS[x["start"]], 0, 0], [], 0.0, 0.0
        for gg in x["goals"]:
            xx, info = run_segment(envs[e], pol, xx, M.GOALS[gg], cfg, keep_traj=True)
            trs.append(info["traj"]); ys += info["y_safety"]; yp += info["y_path"]
        tr = np.concatenate(trs)
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.6, label=f"{'sim' if e == 'S' else 'real'}: safety {ys:.2f}, path {yp:.1f}")
    close = np.array([clearance(p[:2]) < 0.1 for p in tr])
    ax.scatter(tr[close, 0], tr[close, 1], s=14, color="black", zorder=5, label="real within 0.1 of a wall")
    ax.plot(*M.GOALS[x["start"]], "g*", ms=18, mec="k", zorder=6)
    ax.set_title(" → ".join([x["start"]] + x["goals"]), fontsize=11)
    ax.set_xlim(-3.7, 3.7); ax.set_ylim(-2.7, 2.7); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower center", fontsize=8.5, framealpha=.9)
fig.suptitle("Six K=8 test sequences (one noise-free rollout each): simulation (blue) vs real (red, β=0.85, brake=0.4). "
             "Green star = start at rest. Black dots = real closer than 0.1 to a wall.", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0); fig.savefig(OUT / "fig_routes_K8.png", dpi=95, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
img = lambda p: "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()
pct = lambda c, n, m="W_direct": f"{100 * (1 - g(c, 'K=8', n, m) / g(c, 'K=8', n, 'target_only')):.0f}%"
BEST = ' class="best"'


def best_cell(vals):
    b = min(range(len(vals)), key=lambda i: vals[i])
    return lambda i: BEST if i == b else ""


def k8_table(c):
    ms = [("target_only", "Target-only"), ("B1_mc_real", "B1"), ("B2_route_scape", "B2")] + OURS
    rows = []
    for n in N:
        vals = [g(c, "K=8", n, m) for m, _ in ms]; cls = best_cell(vals)
        rows.append(f"<tr><th scope=\"row\">{n}</th>" + "".join(
            f"<td{cls(i)}>{v:.3f}<span class=\"sd\">±{S[f'{c}|K=8|{n}|{m}']['std']:.3f}</span></td>" for i, ((m, _), v) in enumerate(zip(ms, vals))) + "</tr>")
    head = "".join(f"<th>{lab}</th>" for _, lab in ms)
    sp = lambda m, n: f"{S[f'{c}|K=8|{n}|{m}']['spearman']:.2f}"
    return (f"<div class=\"tablewrap\"><table class=\"num\"><thead><tr><th scope=\"col\">n</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p class=\"note\">For reference: raw simulation {g(c, 'K=8', 200, 'raw_sim'):.3f}; segment-wise outcome correction {g(c, 'K=8', 200, 'outcome_only'):.3f} (n=200) / "
            f"{g(c, 'K=8', 1000, 'outcome_only'):.3f} (n=1000). Spearman rank correlation over sequences (n=200 / 1000): target-only {sp('target_only', 200)} / {sp('target_only', 1000)}, "
            f"direct {sp('W_direct', 200)} / {sp('W_direct', 1000)}, simulation-backed {sp('W_simbacked', 200)} / {sp('W_simbacked', 1000)}, "
            f"residual direct {sp('res_direct', 200)} / {sp('res_direct', 1000)}, raw simulation {sp('raw_sim', 200)}.</p>")


def wins_table():
    ms = OURS + [("outcome_only", "Seg. outcome"), ("B1_mc_real", "B1 MC-real"), ("B2_route_scape", "B2 route-SCAPE")]
    rows = []
    for c in ORDER:
        vals = [WINS[c][m] for m, _ in ms]; mx = max(vals)
        rows.append(f"<tr><th scope=\"row\">{CNAME[c][0]} <span class=\"ko\">{CNAME[c][1]}</span></th>" + "".join(
            f"<td{BEST if v == mx else ''}>{v}<span class=\"sd\">/45</span></td>" for v in vals) + "</tr>")
    rows.append("<tr><th scope=\"row\">Total</th>" + "".join(f"<td>{TOT(m)}<span class=\"sd\">/225</span></td>" for m, _ in ms) + "</tr>")
    head = "".join(f"<th>{lab}</th>" for _, lab in ms)
    return f"<div class=\"tablewrap\"><table class=\"num\"><thead><tr><th scope=\"col\">Cost</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def kwise_table(c):
    rows = []
    for n in (200, 1000):
        for K in KS:
            vals = [g(c, K, n, m) for m, _ in METHODS]; cls = best_cell(vals)
            rows.append(f"<tr><th scope=\"row\">n={n} · {K}</th>" + "".join(f"<td{cls(i)}>{v:.3f}</td>" for i, v in enumerate(vals)) + "</tr>")
    head = "".join(f"<th>{lab}</th>" for _, lab in METHODS)
    return f"<div class=\"tablewrap\"><table class=\"num\"><thead><tr><th scope=\"col\">MAE</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def best_table():
    rows = []
    for c in ORDER:
        cells = []
        for n in N:
            m = min(OURS + [("target_only", "")], key=lambda t: g(c, "K=8", n, t[0]))[0]
            cells.append(f"<td>{LABEL[m]}</td>")
        rows.append(f"<tr><th scope=\"row\">{CNAME[c][0]}</th><td class=\"mono\">×{gap[c]['ratio']:.2f}</td><td class=\"mono\">{gap[c]['rank']:.2f}</td>{''.join(cells)}</tr>")
    return ("<div class=\"tablewrap\"><table><thead><tr><th>Cost</th><th>real / sim</th><th>rank corr.</th>"
            + "".join(f"<th>n={n}</th>" for n in N) + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


ratio = lambda c, m, ref: [g(c, "K=8", n, m) / g(c, "K=8", n, ref) for n in N]
oo = ratio("safety", "outcome_only", "target_only") + ratio("expo", "outcome_only", "target_only")
bb = [r for c in COSTS for m in ("B1_mc_real", "B2_route_scape") for r in ratio(c, m, "W_direct")]
k1_raw = [CNAME[c][0] for c in ORDER if all(min(METHODS, key=lambda t: g(c, "K=1", n, t[0]))[0] == "raw_sim" for n in N)]
CSS = (ROOT / "scripts" / "report.css").read_text()

html = f"""<title>safer-av toy study</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<p class="eyebrow">safer-av · PointMaze Sim2Sim</p>
<h1>Terminal-State-Aware Long-Horizon Sim-to-Real Evaluation: toy study</h1>
<p class="sub">From short real–simulation paired transitions, we estimate the real cost of a fixed policy over a long goal sequence.
Numbers are for the 100 test sequences of length K=8 (536 test sequences in total), averaged over 3 data seeds × 3 model seeds = 9 runs (mean ± std).
All learned methods use the same real data and the same training budget.</p>

<div class="lede">
<b>Main findings</b>
<ul>
<li><strong>Terminal-state correction is needed.</strong> Segment-wise outcome correction (SCAPE-style), which corrects only the per-segment
cost, has {min(oo):.0f}–{max(oo):.0f}× the K=8 error of target-only on Safety and Exposure, and beats target-only in {TOT('outcome_only')} of 225 comparisons
(5 costs × 5 values of n × 9 runs).</li>
<li><strong>The Bellman decomposition is needed.</strong> Learning long-sequence costs directly from the same real rollouts (B1, B2) gives
{min(bb):.1f}–{max(bb):.0f}× the K=8 error of the direct estimator. B1 beats target-only in {TOT('B1_mc_real')} of 225 comparisons, B2 in {TOT('B2_route_scape')}.</li>
<li><strong>Calibrated simulation data improve on real data alone.</strong> Against target-only (the same backward Bellman learning on real data only),
the direct estimator wins {TOT('W_direct')} of 225 comparisons, residual direct {TOT('res_direct')}, and the simulation-backed estimator {TOT('W_simbacked')}.
With n=200 paired transitions, the direct estimator reduces the target-only error by {pct('safety', 200)} (Safety), {pct('expo', 200)} (Exposure),
{pct('ttc', 200)} (TTC) and {pct('path', 200)} (Path).</li>
<li><strong>The best estimator depends on the cost</strong> (§5). The direct estimator is the most stable across costs and is the main one.</li>
<li><strong>At K=1 no correction is needed.</strong> Every test sequence starts at rest at a goal, and a single segment from rest is almost the same in
real and simulation. At K=1, raw simulation has the lowest error ({', '.join(k1_raw) if k1_raw else 'no cost'}; §4, per-K tables).</li>
</ul>
</div>

<h2>1. Problem and method</h2>
<p>A fixed goal-conditioned policy executes a goal sequence of K local goals; we estimate its real performance J<sub>R</sub>. Simulation can be
run freely, but the real system is observed only through short segments. Segment-wise outcome correction adds the per-segment cost
mismatch. However, the real execution ends each segment in a different state (position and velocity) than the simulation, and that terminal
state changes the cost of all remaining goals.</p>
<p>Backward Bellman calibration learns the real continuation value W<sub>θ</sub>(x; g<sub>k:K</sub>) of the remaining-goal sequence, for remaining-goal
lengths 1→8. Paired transitions provide paired Bellman targets that continue from the real terminal state; simulated transitions are converted to
real targets by the calibration model h<sub>φ</sub>(x, g<sub>k:K</sub>, Z<sub>S</sub>), which adds training data.</p>
<p><b>Premise.</b> The simulator is a fixed black box. Its parameters cannot be tuned to imitate the real system, and the real–simulation difference
is not assumed to be expressible through simulator parameters. System identification and learned real-dynamics models are therefore out of scope;
the only operation is to correct simulation outcomes toward the real system.</p>
<pre class="flow">paired transition :  Ẑ_R = y_R + W_prev(x_R⁺; g_k+1:K),   Ẑ_S = y_S + W_prev(x_S⁺; g_k+1:K)
calibration model :  h_φ(x, g_k:K, Ẑ_S) → Ẑ_R                     (fit on paired transitions)
sim transition    :  Z̃_R = h_φ(x, g_k:K, y_S + W_prev(x_S⁺; g_k+1:K))
continuation value:  W_θ(x; g_k:K) ← Ẑ_R (paired) and Z̃_R (simulation)   (each length L refits all lengths 1..L)
training rows     :  75% paired targets, 25% corrected simulation targets</pre>
<div class="grid2">
<div class="card"><h3>Proposed estimators <span class="tag">ours</span></h3>
<p><b>Direct</b> (direct amortized estimator): Ĵ<sub>R</sub><sup>direct</sup> = Ŵ<sub>1</sub>(x<sub>1</sub>; g<sub>1:K</sub>). No simulation per query.<br>
<b>Sim-backed</b> (simulation-backed estimator): target-sequence simulation rollouts plus the calibrated correction h<sub>φ</sub> − Ẑ<sub>S</sub> at every segment
(finite-horizon sim-to-real decomposition).<br>
<b>Residual direct</b>: W<sub>R</sub> = W<sub>S</sub> + D, with W<sub>S</sub> computed by simulation Monte Carlo and only the residual D learned backward.
Estimate Ĵ<sub>S</sub> + D̂(x<sub>1</sub>; g<sub>1:K</sub>).</p></div>
<div class="card"><h3>Baselines <span class="tag rust">baselines</span></h3>
<p><b>Raw sim</b>: mean simulated cost Ĵ<sub>S</sub>.<br>
<b>Seg. outcome</b> (segment-wise outcome correction, J<sub>seg</sub>): a learned correction h<sub>out</sub>(x, g, y<sub>S</sub>) → y<sub>R</sub> summed along
simulation rollouts. SCAPE-style; ignores the terminal-state mismatch.<br>
<b>Target-only</b>: the same backward Bellman learning of W<sub>θ</sub> on the real side of the paired data only, without simulation data. The strongest baseline.<br>
<b>B1 MC-real</b>: direct regression of the observed real return over every window (start, length) of the same real rollouts. No Bellman decomposition and no goal-sequence reuse.<br>
<b>B2 route-level SCAPE</b>: h(x, g, J<sub>S</sub>) → observed real return on the same windows, with J<sub>S</sub> from simulating the window's goals. SCAPE applied to whole sequences.</p>
<p class="note">All learned methods use the same n paired transitions plus 50 separate real validation transitions, and the same budget
(MLP 3×256; stop after 3,000 gradient steps without validation improvement, at most 30,000; best validation checkpoint).</p></div>
</div>

<h2>2. Environment</h2>
<div class="grid2">
<div class="card"><h3>Maze and policy</h3>
<p>A 9×7-cell maze on PointMaze (MuJoCo ball) with 9 local goals A–I fixed at the junctions. A valid goal sequence moves only between adjacent
goals and never returns to the previous goal. The policy is a PD controller (kp 2.0, kd 0.4) aiming at the end of the current straight corridor of the
shortest path, and sees only the current goal. A segment ends when the ball enters the radius-0.2 ball around the goal; the velocity carries over.</p></div>
<div class="card"><h3>Real vs. simulation</h3>
<p>The real system transmits less force (β = 0.85) and brakes much more weakly (brake = 0.4: only 40% of commands opposing the velocity).
It therefore reaches junctions at a speed of about {arr_R:.1f} (simulation about {arr_S:.1f}) and swings wide toward the outer wall in the next turn.
The policy is the same in both. This is a Sim2Sim study: the "real" system is the simulator with changed dynamics.</p></div>
</div>
<div class="tablewrap"><table>
<thead><tr><th>Cost</th><th>Per-segment definition</th><th>What it measures</th></tr></thead>
<tbody>
<tr><td><b>Safety</b></td><td class="mono">max(0, 0.4 − min wall clearance) / 0.4</td><td>the single most dangerous moment (0.4 = corridor centre)</td></tr>
<tr><td><b>Exposure</b></td><td class="mono">∫ max(0, 0.4 − clearance) / 0.4 dt</td><td>how deeply and how long the safety margin is intruded</td></tr>
<tr><td><b>TTC</b></td><td class="mono">max(0, 0.5 − min TTC) / 0.5</td><td>seconds until wall contact at the current velocity (ray cast)</td></tr>
<tr><td><b>Path</b></td><td class="mono">path length of the segment</td><td>wider turns are longer</td></tr>
<tr><td><b>Time</b></td><td class="mono">τ / 5 s</td><td>reported for completeness</td></tr>
</tbody></table></div>
<p class="note">A sequence cost is the sum of its segment costs. A segment that does not arrive within 5 s adds 5 to Path and Time.</p>
<figure><div class="plate"><img src="{img(OUT / 'fig_routes_K8.png')}" alt="Simulation and real trajectories on six K=8 test sequences"></div>
<figcaption>Six K=8 test sequences. Blue is simulation, red is real, black dots mark where the real ball is within 0.1 of a wall. Almost all risk occurs on the
outer wall right after a turn; on straight corridors simulation and real nearly overlap.</figcaption></figure>

<h2>3. Data</h2>
<div class="grid2">
<div class="card"><h3>Paired transitions from real rollouts</h3>
<p>The real system executes random goal sequences of length 2–8 without reset (real chains). At every segment, the simulator is set to the current real
state (position and velocity) and executes the same goal once. Half of the chains start at rest at a goal, half at a moving state near a goal. A chain that
starts at rest may not share its first 4 goals with a K≥4 test sequence (no leakage). The n paired transitions are the first n of the collection;
50 validation transitions are collected the same way.</p></div>
<div class="card"><h3>Simulation data and test sequences</h3>
<p><b>sim_train (20,000)</b>: random position and velocity (0–5) near a goal 60%, simulated arrival states 30%, at rest at a goal 10%; one segment each.<br>
<b>seedR (4,000)</b>: one simulated segment from near the real states of the n paired transitions (small noise), where Ŵ is evaluated most.<br>
<b>Test sequences</b>: 100 random sequences each for K = 1, 2, 4, 8 plus 136 K=4 sequences grouped by first edge, all starting at rest.
J<sub>R</sub> is the mean over 100 real rollouts.</p></div>
</div>
<figure><div class="plate"><img src="{img(OUT / 'fig_coverage.png')}" alt="Speed distribution of the start states of the paired transitions"></div>
<figcaption>Start-state speeds of the paired transitions (data seed 0). Backward learning evaluates Ŵ at real terminal states at every length, so those states
must be in the training data, otherwise extrapolation errors accumulate. With real chains, {100 * np.mean((start_sp > 3.5) & (start_sp < 4.2)):.0f}% of the start
states lie near the real arrival speed (3.5–4.2).</figcaption></figure>

<h2>4. Results</h2>
<figure><div class="plate"><img src="{img(OUT / 'fig_mae_vs_n.png')}" alt="K=8 MAE versus the number of paired transitions for each cost"></div>
<figcaption>K=8 MAE versus the number of paired transitions n (mean ± std over 9 runs). The dashed grey line is target-only.</figcaption></figure>
<h3>Comparisons won against target-only (K=8; 5 values of n × 9 runs = 45 per cost)</h3>
{wins_table()}
"""
for c in ORDER:
    html += f"<h3>{CNAME[c][0]} <span class=\"ko\">{CNAME[c][1]}</span> · K=8 MAE</h3>{k8_table(c)}"
html += f"""
<h3>Comparison by sequence length K</h3>
<p>B1 and B2 can only learn a length-L sequence from a real window of length L under the same real budget. Windows of length 8 number
{win_len[200][0]}–{win_len[200][1]} per data seed at n=200 and only {win_len[1000][0]}–{win_len[1000][1]} at n=1000. Because of the leakage rule, real windows that start
at rest have length at most {rest_max}, while every test sequence starts at rest. K=1 is a single straight segment from rest to an adjacent goal; simulation and
real are almost identical there, and uncorrected raw simulation is often the most accurate.</p>
"""
for c in ORDER:
    html += f"<h3>{CNAME[c][0]}</h3>{kwise_table(c)}"
html += f"""
<h2>5. Why the best estimator depends on the cost</h2>
<figure><div class="plate"><img src="{img(OUT / 'fig_gap_structure.png')}" alt="Simulation versus real sequence cost for each cost"></div>
<figcaption>Simulation cost J<sub>S</sub> versus real cost J<sub>R</sub> on the K=8 test sequences. Safety and Exposure are several times larger in real but keep the
ranking of sequences; Path differs little but the ranking is heavily reshuffled.</figcaption></figure>
<h3>Structure of the real–simulation gap and the best estimator per n (K=8 MAE; target-only and the three proposed estimators)</h3>
{best_table()}
<p>The simulation-backed estimator corrects along simulation rollouts with a calibration model that learns the map from simulation values to real values.
When the gap is large but structured (Safety, Exposure: ranking preserved), this map is easy to learn. When the gap is small, the situation differs:
the real–simulation path-length difference of one segment is {path_gap:.2f} on average, while the real path length of a K=8 sequence is {gap['path']['JR']:.1f}
on average. Handling two large quantities (Ŵ and the simulation cost) separately makes small relative errors large compared with the difference.
The residual variant computes the large common part W<sub>S</sub> by simulation and learns only the small difference D, which avoids this. (This is an
interpretation; no error-decomposition experiment is included in this study.)</p>

<h2>6. Limitations</h2>
<ul>
<li>The study is Sim2Sim (real = simulation with changed dynamics). The black-box simulator premise has not been tested on a physical robot or a high-fidelity simulator.</li>
<li>The 75% share of paired targets was chosen by looking at test results (a small sweep over 9%, 25%, 50%, 75%). It should be selected on validation data.</li>
<li>The residual variant computes W<sub>S</sub> by multi-segment simulation rollouts from arbitrary states during training, which departs from the short-simulation-data setting.</li>
<li>Training and test use the same maze and the same 9 goals; generalization to new mazes or goal locations has not been tested.</li>
<li>No single estimator is best on every cost (§5).</li>
<li>Paired transitions are collected along real rollouts of several consecutive segments (real chains); this collection assumption should be stated.</li>
<li>The dynamics are a PointMaze ball; vehicle dynamics (heading, turning radius) have not been studied.</li>
</ul>
<p class="note">Reproduction: <span class="mono">scripts/run_all.sh</span> (data → 45 runs → this page). Method and theory: <span class="mono">METHODOLOGY.md</span>.</p>
</div>
"""
(OUT / "report.html").write_text(html)
print(json.dumps(WINS, indent=1)); print("report written", OUT / "report.html")
