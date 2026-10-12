# safer-av: Terminal-State-Aware Long-Horizon Sim-to-Real Evaluation (toy study)

Given a few short real–simulation paired transitions and many short simulated transitions, we estimate the **real
performance of a fixed goal-conditioned policy over a long sequence of goals**. Unlike segment-wise
outcome correction (SCAPE-style), backward Bellman calibration also corrects how the real–simulation terminal-state
mismatch changes the cost of all remaining goals.

- Method and theory: [METHODOLOGY.md](METHODOLOGY.md)
- Results page: `outputs/report/report.html` (a single file with embedded figures; download it and open it in a browser,
  since GitHub shows HTML files as source)

![Simulation vs real](outputs/report/fig_env.png)

The same policy drives the same goal sequence in simulation (blue) and in "real" (red). Real reaches the junctions
faster and swings wide into the outer wall in turns (black dots), so it ends each segment in a different state than the
simulation, and that changes the cost of the following goals.

## Setting

| Item | Setting |
|---|---|
| Environment | PointMaze (MuJoCo ball) in a 9×7-cell maze with 9 goals A–I at the junctions. A goal sequence moves between adjacent goals and never turns back |
| Policy | A PD controller that follows the shortest path to the current goal. When the ball reaches the goal (radius 0.2), the next goal starts; the ball keeps its velocity |
| Real vs. simulation | Sim2Sim. "Real" is the same simulator with weaker force (β = 0.85) and much weaker brakes (brake = 0.4). It reaches junctions faster and swings wide in turns |
| Paired data | The real system drives random goal sequences (2–8 goals). At every segment, the simulator is set to the same state as the real system and drives to the same goal. n = 200–1000 segments, plus 50 for validation |
| Simulation-only data | 24,000 one-segment simulations from many random states (position and velocity) around the goals |
| Test sequences | 536 sequences of K = 1, 2, 4, 8 goals, all starting at rest. Ground truth: mean of 100 real runs |
| Training | Same small MLP and the same training budget for every method |
| Repetitions | 3 data seeds × 3 model seeds = 9 runs for each n |

**Compared methods**

| Type | Name | What it does |
|---|---|---|
| baseline | Raw sim | simulation cost, no correction |
| baseline | Segment-wise outcome correction | SCAPE-style: corrects the cost of each segment along the simulated run; ignores that the real run ends a segment in a different state |
| baseline | Target-only | our backward learning, but with real data only |
| baseline | B1 MC-real | learns the real cost of whole goal sequences directly from the real runs |
| baseline | B2 route-level SCAPE | like B1, with the simulated cost of the sequence as an extra input |
| ours | Direct | reads the learned real value at the start state (no simulation at test time) |
| ours | Sim-backed | runs the test sequence in simulation and corrects every simulated segment |
| ours | Residual direct | simulated cost of the sequence plus a learned real–simulation difference |

### How our three estimators work

The real system ends a segment in a different state than the simulator (for example, faster), and that changes the
cost of all later goals. All three estimators learn this effect from the paired segments.

**Direct.** A model `W(state, remaining goals)` predicts the real cost of finishing the remaining goals from a state.
It is trained backward, from the last goal to the first: the target for a paired segment is "its real cost + W at the
state where the real run ended". A small correction model `h` turns simulation-only segments into extra real-like
targets. At test time we only read the model at the start: `J_real ≈ W(start, all goals)`. No simulation is needed.

**Sim-backed.** Run the test sequence in the simulator (30 runs). At every simulated segment, `h` estimates how much
more the real system would cost from that state, including the effect of ending the segment in a different place.
The estimate is the simulated cost plus these corrections: `J_real ≈ J_sim + Σ_segments (real − sim correction)`.

**Residual direct.** Like Direct, but the model learns only the difference between real and simulation:
`W_real = W_sim + D`. `W_sim` is computed by running the simulator; only the small difference `D` is learned (same
backward training). Estimate: `J_real ≈ J_sim + D(start, all goals)`. This helps when the difference is small compared
with the cost itself, e.g. Path: cost ≈ 20, real–sim difference ≈ 1.

## Evaluation

Each method estimates the real cost of every test sequence, separately for each of
five costs. A segment cost is computed from the trajectory of one segment (`safer_av/segment.py`); the sequence cost is
the sum over its K segments.

| Cost | Per-segment definition | What it measures |
|---|---|---|
| Safety | max(0, 0.4 − minimum wall clearance) / 0.4 | the most dangerous moment of the segment (0.4 = corridor centre) |
| Exposure | ∫ max(0, 0.4 − wall clearance) / 0.4 dt | how deeply and how long the safety margin is intruded |
| TTC | max(0, 0.5 − minimum time to collision) / 0.5 | time until wall contact at the current velocity (ray cast) |
| Path | path length of the segment | wider turns are longer |
| Time | segment duration / 5 s | arrival time |

Path and Time add 5 when a segment does not reach its goal within 5 s.

Error metrics, computed over the test sequences of each length K ∈ {1, 2, 4, 8} against the real cost from 100 real
rollouts per sequence:

| Metric | Meaning |
|---|---|
| MAE | mean absolute error of the estimate (headline: K = 8) |
| Bias | mean signed error (estimate − truth) |
| RMSE | root mean squared error |
| Spearman | rank correlation between estimates and truth over sequences (how well candidate routes are ranked) |

Every metric is reported per number of paired transitions n ∈ {200, 400, 600, 800, 1000} and averaged over the 9 runs
(3 data seeds × 3 model seeds).

## Results (K = 8)

K=8 MAE, mean over 9 runs (lower is better; best in bold). Means of every metric for every K and n are in
`outputs/report/summary.json`, per-run values in `outputs/results/`.

| Cost | n | Raw sim | Seg. outcome | Target-only | B1 | B2 | Direct | Sim-backed | Residual direct |
|---|---|---|---|---|---|---|---|---|---|
| Safety | 200 | 3.224 | 2.468 | 0.244 | 0.652 | 0.518 | 0.109 | **0.098** | 0.150 |
| Safety | 1000 | 3.224 | 2.578 | 0.110 | 0.530 | 0.400 | 0.059 | 0.067 | **0.057** |
| Exposure | 200 | 1.011 | 0.759 | 0.066 | 0.211 | 0.172 | 0.034 | **0.033** | 0.053 |
| Exposure | 1000 | 1.011 | 0.752 | 0.025 | 0.163 | 0.138 | **0.017** | 0.024 | 0.020 |
| TTC | 200 | 1.897 | 0.867 | 0.141 | 0.390 | 0.250 | **0.076** | 0.077 | 0.094 |
| TTC | 1000 | 1.897 | 0.919 | 0.080 | 0.247 | 0.172 | **0.038** | 0.055 | 0.041 |
| Path | 200 | 1.074 | 0.799 | 0.105 | 1.081 | 0.423 | 0.067 | 0.105 | **0.062** |
| Path | 1000 | 1.074 | 0.820 | 0.094 | 0.712 | 0.277 | 0.056 | 0.131 | **0.034** |
| Time | 200 | 0.035 | 0.013 | 0.007 | 0.044 | 0.029 | 0.005 | 0.006 | **0.004** |
| Time | 1000 | 0.035 | 0.018 | 0.008 | 0.031 | 0.017 | 0.005 | 0.009 | **0.003** |

- For K = 2, 4 and 8, the direct estimator has a lower MAE than every baseline for all five costs and all five values of n.
- At K = 1, raw simulation is the most accurate for Safety, Exposure and Path: every test sequence starts at rest, and a
  single segment from rest is almost the same in real and simulation, so no correction is needed.
- The simulation-backed estimator is worse than target-only on Path and Time at larger n; no single estimator is best
  on every cost.
- The 75% share of paired targets in the training rows of W was chosen by looking at test results (a small sweep).

## Running

**Install.** Python 3.11 and `pip install -r requirements.txt` (choose the torch build that matches your CUDA version).
A GPU is strongly recommended for step 3; CPU-only training works but is very slow.

**Included results.** The 45 per-run metric files (`outputs/results/res_*.json`) and the results page with its figures
(`outputs/report/`) are part of the repository, so the results can be read without running anything.

**Pipeline.**

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/make_data.py` | `data/`: test sequences with ground truth, their simulation rollouts, simulation-only data (sim_train), real chains per data seed |
| 2 | `scripts/make_rows.py` | `data/`: simulated transitions near the real states (seedR), Monte Carlo rows for the residual variant, B1/B2 windows |
| 3 | `scripts/run.py D N S` | `outputs/results/res_d{D}_n{N}_s{S}.json`: 8 methods × 5 costs, MAE / bias / RMSE / Spearman per K |
| 4 | `scripts/report.py` | `outputs/report/`: summary.json (means over the 9 runs), fig_mae_vs_n.png, report.html |
| – | `scripts/plot_env.py` | `outputs/report/fig_env.png`: the environment figure above |

**Quick check: one configuration** (data seed 0, n = 200, model seed 0). This regenerates only the data that run needs
and compares the result with the included file.

```bash
python scripts/make_data.py --workers 16
python scripts/make_rows.py --data-seeds 0 --sizes 200 --model-seeds 0 --workers 16
python scripts/run.py 0 200 0
git diff --stat outputs/results    # no output = the included numbers were reproduced exactly
```

**Full reproduction** (all 45 runs and the results page):

```bash
GPUS="0 1" JOBS=8 bash scripts/run_all.sh
```

`GPUS` lists the GPU ids used round-robin and `JOBS` the number of runs executed at the same time.

All random seeds are fixed. On the machine used for the reported results (x86-64, NVIDIA RTX PRO 6000 Blackwell,
CUDA 12.8), regenerating the data and rerunning all 45 runs reproduced every file and every metric exactly. Other
hardware or library versions can change the numbers slightly through floating-point differences.

## Terms and code

| Term (METHODOLOGY) | Code |
|---|---|
| remaining goals (remaining-goal sequence) | `suffix`; without the current goal: `tail` |
| value model W (real continuation value) | `W` in `safer_av/ours.py` |
| correction model h (calibration model) | `h` in `safer_av/ours.py` |
| paired targets, real / sim (paired Bellman targets) | `ZR`, `ZS` |
| segment cost, end state of a segment | `y_<cost>`, `x_plus` in each transition record |
| Direct, Sim-backed | `direct`, `sim_backed` (result keys `W_direct`, `W_simbacked`) |
| Residual direct: difference D, simulated value W_sim | `D`, `WS_S` / `WS_R` in `safer_av/residual.py` (result key `res_direct`) |

| File | Content |
|---|---|
| `safer_av/maze.py`, `policy.py`, `envs.py` | maze and goal graph, policy, real / simulation dynamics |
| `safer_av/segment.py`, `route.py` | one segment and its costs; a goal sequence executed without reset |
| `safer_av/data.py` | start-state samplers, simulation-only data, test ground truth, real chains |
| `safer_av/models.py` | input encoding, MLP regressor, training budget |
| `safer_av/backward.py` | goal-sequence reuse, target-only, segment-wise outcome correction |
| `safer_av/ours.py` | value model W and correction model h; Direct and Sim-backed |
| `safer_av/residual.py` | Residual direct (W_sim by running the simulator, learned D) |
| `safer_av/route_baselines.py` | B1, B2 |
