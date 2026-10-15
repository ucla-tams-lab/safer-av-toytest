"""Step 4: aggregate the 45 runs (3 data seeds x 5 n x 3 model seeds) into mean errors over the 9 runs at each n, draw
the main figure and write a short results page. Out: outputs/report/{summary.json, fig_mae_vs_n.png, report.html}"""
import base64, json, os, pickle, sys  # noqa: E401
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carmaze.segment import COSTS  # noqa: E402

DATA = Path(os.environ.get("CARMAZE_DATA", ROOT / "data"))
RES = Path(os.environ.get("CARMAZE_OUT", ROOT / "outputs" / "results"))
OUT = Path(os.environ.get("CARMAZE_REPORT", ROOT / "outputs" / "report"))
OUT.mkdir(parents=True, exist_ok=True)
N, SEEDS, DSEEDS, KS, STARTS = [200, 400, 600, 800, 1000], [0, 1, 2], [0, 1, 2], ["K=1", "K=2", "K=4", "K=8"], ["rest", "moving"]
BASE = [("raw_sim", "Raw sim"), ("outcome_only", "Seg. outcome"), ("target_only", "Target-only"), ("B1_mc_real", "B1"),
        ("B2_route_scape", "B2")]
OURS = [("V_direct", "Direct"), ("V_simbacked", "Sim-backed")]
METHODS = BASE + OURS
LABEL = dict(METHODS)
CNAME = dict(safety="Safety", expo="Exposure", ttc="TTC", path="Path", time="Time")
ORDER = ["safety", "expo", "ttc", "path", "time"]

R = {(d, n, s): json.loads(p.read_text()) for d in DSEEDS for n in N + [1500, 2000] for s in SEEDS
     if (p := RES / f"res_d{d}_n{n}_s{s}.json").exists()}
N = sorted({n for (_, n, _) in R if all((d, n, s) in R for d in DSEEDS for s in SEEDS)})   # sizes with all 9 runs
S = {}
for c in COSTS:
    for st in STARTS:
        for K in KS:
            for n in N:
                for m, _ in METHODS:
                    runs = [R[(d, n, s)][c][m][f"{st}|{K}"] for d in DSEEDS for s in SEEDS if (d, n, s) in R]
                    if not runs:
                        continue
                    arr = lambda k: np.array([r[k] if r[k] is not None else np.nan for r in runs], dtype=float)
                    S[f"{c}|{st}|{K}|{n}|{m}"] = dict(mae=float(arr("mae").mean()), mae_std=float(arr("mae").std()),
                                                     bias=float(arr("bias").mean()), rmse=float(arr("rmse").mean()),
                                                     spearman=float(np.nanmean(arr("spearman"))), runs=len(runs))

# average view: error of the mean cost over all K=8 queries of a start type, relative to the true mean
q = pickle.load(open(DATA / "queries_test.pkl", "rb"))
Kq = np.array([x["K"] for x in q]); Tq = np.array(["moving" if x["tag"] == "moving" else "rest" for x in q])
AV = {}
for (d, n, s) in R:
    P = json.loads((RES / f"preds_d{d}_n{n}_s{s}.json").read_text())
    for c in COSTS:
        gt = np.array(P[c]["gt"])
        for st in STARTS:
            msk = (Kq == 8) & (Tq == st)
            for m, _ in METHODS:
                AV.setdefault(f"{c}|{st}|{n}|{m}", []).append(abs(np.mean(np.array(P[c][m])[msk]) - gt[msk].mean()) / gt[msk].mean())
AV = {k: float(np.mean(v)) for k, v in AV.items()}
K8 = {st: [x for x, k, t in zip(q, Kq, Tq) if k == 8 and t == st] for st in STARTS}
GAP = {f"{c}|{st}": np.mean([x["R"][c]["mean"] - x["S"][c]["mean"] for x in K8[st]]) / np.mean([x["S"][c]["mean"] for x in K8[st]])
       for c in COSTS for st in STARTS}
json.dump(dict(per_scene=S, mean_error=AV), open(OUT / "summary.json", "w"), indent=1)

# ---------------- the setting in one picture: true real vs true sim cost of every K=8 query ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
for ax, c in zip(axes, ["safety", "expo", "ttc", "path", "time"]):
    for st, col in (("rest", "#9a9a9a"), ("moving", "#1f5fa8")):
        r = np.array([x["R"][c]["mean"] for x in K8[st]]); sm = np.array([x["S"][c]["mean"] for x in K8[st]])
        ax.scatter(sm, r, s=10, alpha=0.7, color=col, lw=0, label=f"{st} start (r = {np.corrcoef(sm, r)[0, 1]:.2f})")
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls=":"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title({"safety": "Safety", "expo": "Exposure", "ttc": "TTC", "path": "Path", "time": "Time"}[c])
    ax.set_xlabel("sim cost"); ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.grid(alpha=.25)
axes[0].set_ylabel("real cost")
fig.tight_layout(); fig.savefig(OUT / "fig_gap.png", dpi=110, facecolor="white"); plt.close(fig)
g = lambda c, st, K, n, m: S[f"{c}|{st}|{K}|{n}|{m}"]["mae"]

# ---------------- main figure: K=8 MAE vs n, rest and moving starts ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = dict(raw_sim="#9a9a9a", outcome_only="#b8860b", target_only="#6b6b6b", B1_mc_real="#8e6fbf", V_direct="#1f5fa8", V_simbacked="#7aa6d8")
fig, axes = plt.subplots(2, 5, figsize=(21, 7.6))
for row, st in enumerate(STARTS):
    for ax, c in zip(axes[row], ORDER):
        ax.axhline(g(c, st, "K=8", N[0], "raw_sim"), color=COL["raw_sim"], ls=":", lw=1.4, label="Raw sim")
        for m in ["outcome_only", "target_only", "B1_mc_real", "V_direct", "V_simbacked"]:
            ax.errorbar(N, [g(c, st, "K=8", n, m) for n in N], yerr=[S[f"{c}|{st}|K=8|{n}|{m}"]["mae_std"] for n in N], color=COL[m],
                        ls="--" if m in ("target_only", "outcome_only", "B1_mc_real") else "-", marker="o", ms=4, lw=1.4, capsize=2, label=LABEL[m])
        ax.set_title(f"{CNAME[c]} ({st} start)"); ax.set_xticks(N); ax.set_ylim(bottom=0); ax.grid(alpha=.25)
        if row == 1:
            ax.set_xlabel("paired transitions n")
    axes[row, 0].set_ylabel("K=8 MAE (mean over 9 runs, ± std)")
axes[0, 0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_mae_vs_n.png", dpi=120, facecolor="white"); plt.close(fig)

# ---------------- slide figure: K=8 MAE vs n with rest and moving starts pooled (100 + 100 sequences, so the mean of
# the two start types' MAE), without the methods that are off scale (raw sim, B1, B2) ----------------
def pooled(c, n, m):
    v = [(R[(d, n, s)][c][m]["rest|K=8"]["mae"] + R[(d, n, s)][c][m]["moving|K=8"]["mae"]) / 2 for d in DSEEDS for s in SEEDS]
    return float(np.mean(v)), float(np.std(v))


fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
for ax, c in zip(axes, ORDER):
    for m in ["outcome_only", "target_only", "V_direct", "V_simbacked"]:
        mu, sd = zip(*[pooled(c, n, m) for n in N])
        ax.errorbar(N, mu, yerr=sd, color=COL[m], ls="--" if m in ("target_only", "outcome_only") else "-", marker="o", ms=4, lw=1.6,
                    capsize=2, label=LABEL[m])
    ax.set_title(f"{CNAME[c]}   (raw sim {pooled(c, N[0], 'raw_sim')[0]:.2f}, off scale)", fontsize=10)
    ax.set_xticks(N); ax.tick_params(axis="x", labelsize=8); ax.set_ylim(bottom=0); ax.grid(alpha=.25); ax.set_xlabel("paired transitions n")
axes[0].set_ylabel("K=8 MAE (mean over 9 runs, ± std)"); axes[0].legend(fontsize=9, frameon=False)
fig.suptitle("Error of the estimated real cost of 8-goal sequences (200 sequences: 100 starting at rest, 100 starting while driving)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig_mae_vs_n_pooled.png", dpi=130, facecolor="white"); plt.close(fig)

# ---------------- per-scene picture: estimated vs true K=8 cost of every moving-start query, one run ----------------
SC_RUN, SC_METHODS = (0, 1000, 0), [("raw_sim", "Raw sim"), ("outcome_only", "Seg. outcome"), ("target_only", "Target-only"), ("V_direct", "Direct")]
if (RES / "preds_d{}_n{}_s{}.json".format(*SC_RUN)).exists():
    P = json.loads((RES / "preds_d{}_n{}_s{}.json".format(*SC_RUN)).read_text())
    msk = (Kq == 8) & (Tq == "moving")
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.3))
    for ax, c in zip(axes, ORDER):
        gt = np.array(P[c]["gt"])[msk]
        lo, hi = min(gt.min(), min(np.array(P[c][m])[msk].min() for m, _ in SC_METHODS)), max(gt.max(), max(np.array(P[c][m])[msk].max() for m, _ in SC_METHODS))
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls=":")
        for m, lab in SC_METHODS:
            ax.scatter(gt, np.array(P[c][m])[msk], s=9, alpha=0.6, color=COL[m], label=lab, lw=0)
        ax.set_title(CNAME[c]); ax.set_xlabel("true real cost"); ax.grid(alpha=.25)
    axes[0].set_ylabel("estimate"); axes[0].legend(fontsize=8, frameon=False, markerscale=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_scenes.png", dpi=110, facecolor="white"); plt.close(fig)

# ---------------- numbers for the key results (K=8) ----------------
BEST = ' class="best"'
rel = lambda a, b: 100 * (1 - a / b)                    # % lower than b (negative: higher)
best_base = lambda c, st, n: min(g(c, st, "K=8", n, m) for m, _ in BASE)
vs_best = {(c, st): [rel(g(c, st, "K=8", n, "V_direct"), best_base(c, st, n)) for n in N] for c in ORDER for st in STARTS}
vs_to = {(c, st): [rel(g(c, st, "K=8", n, "V_direct"), g(c, st, "K=8", n, "target_only")) for n in N] for c in ORDER for st in STARTS}
vs_raw = {(c, st): [rel(g(c, st, "K=8", n, "V_direct"), g(c, st, "K=8", n, "raw_sim")) for n in N] for c in ORDER for st in STARTS}
sb_vs = {(c, st): [rel(g(c, st, "K=8", n, "V_simbacked"), g(c, st, "K=8", n, "V_direct")) for n in N] for c in ORDER for st in STARTS}
sp = lambda c, st, m, n=1000: S[f"{c}|{st}|K=8|{n}|{m}"]["spearman"]


def change(vals):
    """Range of '% lower' values over n, in words."""
    lo, hi = min(vals), max(vals)
    if lo >= 0:
        return f"{lo:.0f}–{hi:.0f}% lower" if round(lo) != round(hi) else f"{lo:.0f}% lower"
    if hi <= 0:
        return f"{-hi:.0f}–{-lo:.0f}% higher" if round(lo) != round(hi) else f"{-lo:.0f}% higher"
    return f"{-lo:.0f}% higher to {hi:.0f}% lower"


def per_cost(d, st):
    return "; ".join(f"{CNAME[c]} {change(d[(c, st)])}" for c in ORDER)


rng = lambda vals: f"{min(vals):.2f}–{max(vals):.2f}"
def not_best(c):
    """(start, n, baseline) wherever a baseline has a lower K=8 error than Direct."""
    out = []
    for st in STARTS:
        for n in N:
            vals = {m: g(c, st, "K=8", n, m) for m, _ in BASE}
            b = min(vals, key=vals.get)
            if vals[b] < g(c, st, "K=8", n, "V_direct"):
                out.append((st, n, b))
    return out


def ns(lst):
    return ", ".join(str(n) for n in lst)


best_all = [c for c in ORDER if not not_best(c)]
others = []
for c in ORDER:
    nb = not_best(c)
    if nb:
        by = {}
        for st, n, b in nb:
            by.setdefault(b, {}).setdefault(st, []).append(n)
        txt = "; ".join(f"{LABEL[b]} is lower at " + " and ".join(f"n = {ns(v)} ({st} start)" for st, v in d.items()) for b, d in by.items())
        others.append(f"{CNAME[c]}: {txt}")
mult = lambda m, cs: [g(c, st, "K=8", n, m) / g(c, st, "K=8", n, "V_direct") for c in cs for st in STARTS for n in N]
key = []
if best_all:
    key.append(f"<li><strong>Direct's error is lower than every baseline's on {', '.join(CNAME[c] for c in best_all)}</strong> at every n and for both "
               f"start types: {min(v for c in best_all for st in STARTS for v in vs_best[(c, st)]):.0f}–"
               f"{max(v for c in best_all for st in STARTS for v in vs_best[(c, st)]):.0f}% below the best baseline.</li>")
if others:
    key.append("<li><strong>Elsewhere a baseline is lower in some cells.</strong> " + ". ".join(others) + ".</li>")
key += [
    f"<li><strong>Against target-only</strong> (the same learning on real data only), Direct's error is: rest start {per_cost(vs_to, 'rest')}; "
    f"moving start {per_cost(vs_to, 'moving')}.</li>",
    f"<li><strong>Raw simulation</strong> has {min(mult('raw_sim', ORDER)):.0f}–{max(mult('raw_sim', ORDER)):.0f}× Direct's error (it misses the "
    f"sim–real gap). <strong>B1 and B2</strong>, which learn whole-sequence costs from the same real drives, have "
    f"{min(mult('B1_mc_real', ORDER) + mult('B2_route_scape', ORDER)):.1f}–{max(mult('B1_mc_real', ORDER) + mult('B2_route_scape', ORDER)):.0f}× Direct's error.</li>",
    f"<li><strong>Ranking the scenes</strong> (Spearman correlation between estimated and true K=8 costs at n = 1000, range over costs and "
    f"start types): Direct {rng([sp(c, st, 'V_direct') for c in ORDER for st in STARTS])}, target-only "
    f"{rng([sp(c, st, 'target_only') for c in ORDER for st in STARTS])}, raw sim {rng([sp(c, st, 'raw_sim') for c in ORDER for st in STARTS])}.</li>",
    f"<li><strong>Simulation-backed estimation</strong> (equal to Direct in expectation) is {change([v for c in ORDER for st in STARTS for v in sb_vs[(c, st)]])} "
    f"than Direct across costs, start types and n.</li>",
]


def k8_table(n=1000):
    rows = []
    for c in ORDER:
        for st in STARTS:
            vals = [g(c, st, "K=8", n, m) for m, _ in METHODS]; b = int(np.argmin(vals))
            rows.append(f"<tr><th scope=\"row\">{CNAME[c]} · {st}</th>" + "".join(f"<td{BEST if i == b else ''}>{v:.3f}</td>" for i, v in enumerate(vals)) + "</tr>")
    head = "".join(f"<th>{lab}</th>" for _, lab in METHODS)
    return f"<div class=\"tablewrap\"><table class=\"num\"><thead><tr><th scope=\"col\">K=8 MAE, n={n}</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def k_table(c, st, n=1000):
    ms = [("raw_sim", "Raw sim"), ("outcome_only", "Seg. outcome"), ("target_only", "Target-only")] + OURS
    rows = []
    for K in KS:
        vals = [g(c, st, K, n, m) for m, _ in ms]; b = int(np.argmin(vals))
        rows.append(f"<tr><th scope=\"row\">{K}</th>" + "".join(f"<td{BEST if i == b else ''}>{v:.3f}</td>" for i, v in enumerate(vals)) + "</tr>")
    head = "".join(f"<th>{lab}</th>" for _, lab in ms)
    return f"<div class=\"tablewrap\"><table class=\"num\"><thead><tr><th scope=\"col\">{CNAME[c]}, {st} start, n={n}</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def avg_table(n=1000):
    ms = [("raw_sim", "Raw sim"), ("outcome_only", "Seg. outcome"), ("target_only", "Target-only"), ("V_direct", "Direct")]
    rows = []
    for c in ORDER:
        for st in STARTS:
            vals = [100 * AV[f"{c}|{st}|{n}|{m}"] for m, _ in ms]; b = int(np.argmin(vals))
            rows.append(f"<tr><th scope=\"row\">{CNAME[c]} · {st}</th><td>{100 * GAP[f'{c}|{st}']:+.0f}%</td>"
                        + "".join(f"<td{BEST if i == b else ''}>{v:.1f}%</td>" for i, v in enumerate(vals)) + "</tr>")
    head = "".join(f"<th>{lab}</th>" for _, lab in ms)
    return (f"<div class=\"tablewrap\"><table class=\"num\"><thead><tr><th scope=\"col\">K=8, n={n}</th><th>real vs sim</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


img = lambda p: "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()
CSS = (ROOT / "scripts" / "report.css").read_text()
html = f"""<title>Car maze study</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<p class="eyebrow">safer-av · car maze Sim2Sim</p>
<h1>Terminal-State-Aware Long-Horizon Sim-to-Real Evaluation: car maze</h1>
<p class="sub">The toy study's maze, goals, costs and methods with a small car (1/10 scale) instead of a ball. A fixed controller
drives through K goals. "Real" is the simulator with a weaker motor and brakes, less tyre grip and five wet patches; the downhill
slopes are the same in both. Each method estimates the real cost of 936 evaluation goal sequences: 536 start at rest on a goal, 400
start while driving (where the real car is after 1–3 earlier goals). All numbers are averaged over 9 runs (3 data seeds × 3 model seeds).</p>

<figure><div class="plate"><img src="{img(ROOT / 'figs' / 'routes_sim_real.png')}" alt="Sim and real car paths on three goal sequences"></div>
<figcaption>The same goal sequences in sim (blue) and real (red), 5 runs each. Arrows: downhill. Light blue: wet (real only). Both reach a
junction at about the same speed; in the turn the real car brakes less hard and has less grip, so it swings wide.</figcaption></figure>

<figure><div class="plate"><img src="{img(OUT / 'fig_gap.png')}" alt="True real versus true simulated cost of every K=8 goal sequence"></div>
<figcaption>True real cost (mean of 200 real runs) versus simulated cost (200 sim runs) of every evaluation goal sequence with K = 8.
The costs differ a lot between sequences; real is higher than sim and the gap changes from sequence to sequence, but the two are
related (r: correlation across sequences). Dotted: real = sim.</figcaption></figure>

<div class="lede">
<b>Key results (K = 8, mean over 9 runs)</b>
<ul>
{''.join(key)}
</ul>
</div>

<h2>Per-scene error</h2>
<p class="note">Mean absolute error between the estimated and the true real cost of each goal sequence. Raw sim: uncorrected simulation.
Seg. outcome: segment-wise outcome correction (SCAPE-style). Target-only: backward Bellman learning on real data only. B1: direct
regression of real returns. B2: route-level SCAPE. Direct: our main estimator. Sim-backed: our second evaluation mode (equal to Direct
in expectation). Lowest error highlighted.</p>
{k8_table()}
<figure><div class="plate"><img src="{img(OUT / 'fig_mae_vs_n.png')}" alt="K=8 MAE versus the number of paired transitions for each cost"></div>
<figcaption>K=8 MAE versus the number of paired transitions n (mean over 9 runs; bars: std across runs). Dotted: raw sim. Top: rest
start; bottom: moving start.</figcaption></figure>

<figure><div class="plate"><img src="{img(OUT / 'fig_scenes.png')}" alt="Estimated versus true K=8 cost of every moving-start goal sequence"></div>
<figcaption>Estimated versus true real cost of each of the 100 moving-start goal sequences with K = 8, one run (data seed 0, model seed 0,
n = 1000). Points on the dotted line are exact.</figcaption></figure>

<h2>Error by sequence length K</h2>
{k_table('safety', 'moving')}

<h2>Average over all goal sequences</h2>
<p>If only the average cost over all K=8 sequences is wanted (average-performance evaluation), the error of that average, relative to
the true average, is:</p>
{avg_table()}

<h2>Limitations</h2>
<ul>
<li>Sim2Sim study (real = the simulator with changed physics) in one maze with 9 fixed goals and a hand-written controller.</li>
<li>Path and Time differ little between sim and real here (see "real vs sim" above), so raw simulation is already close on them.</li>
<li>The real physics and the controller were chosen with checks on separate queries (cost differences between scenes, sim–real gap
and correlation, one-segment learnability) before any method was run; see README.md.</li>
</ul>
<p class="note">Per-run metrics (MAE, bias, RMSE, Spearman for every K and n): <span class="mono">outputs/results/</span>; means:
<span class="mono">outputs/report/summary.json</span>.</p>
</div>
"""
(OUT / "report.html").write_text(html)
for c in ORDER:
    for st in STARTS:
        print(f"{CNAME[c]:8s} {st:6s} vs best {change(vs_best[(c, st)]):22s} vs target-only {change(vs_to[(c, st)]):22s} "
              f"vs raw {change(vs_raw[(c, st)]):22s} Spearman D/raw/TO {sp(c, st, 'V_direct'):.2f}/{sp(c, st, 'raw_sim'):.2f}/"
              f"{sp(c, st, 'target_only'):.2f} | mean err D/raw {100 * AV[f'{c}|{st}|1000|V_direct']:.1f}/{100 * AV[f'{c}|{st}|1000|raw_sim']:.1f}%")
print("report written", OUT / "report.html", f"({len(R)} runs)")
