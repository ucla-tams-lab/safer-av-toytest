# safer-av: Terminal-State-Aware Long-Horizon Sim-to-Real Evaluation (toy study)

Given a few short real–simulation paired transitions and many short simulated transitions, we estimate the **real
performance of a fixed goal-conditioned policy over a long goal sequence** $\mathbf g_{1:K}$. Unlike segment-wise
outcome correction (SCAPE-style), backward Bellman calibration also corrects how the real–simulation terminal-state
mismatch changes the cost of all remaining goals.

- Method and theory: [METHODOLOGY.md](METHODOLOGY.md)
- Results page: `outputs/report/report.html` (a single file with embedded figures; download it and open it in a browser,
  since GitHub shows HTML files as source)

## Setting

| Item | Setting |
|---|---|
| Environment | PointMaze (MuJoCo ball) with a 9×7-cell maze; 9 local goals A–I at the junctions. A valid goal sequence is a non-backtracking walk between adjacent goals |
| Policy $\pi$ | PD controller (kp 2.0, kd 0.4) toward the end of the current straight corridor of the shortest path; sees only the current goal. A segment ends on entering the radius-0.2 ball around the goal (timeout 5 s); the terminal state keeps its velocity |
| Real vs. simulation | Sim2Sim: the "real" system is the simulator with weaker force transmission (β = 0.85) and much weaker braking (brake = 0.4: only 40% of commands opposing the velocity). Both have actuation noise 0.02 (`configs/env.json`) |
| Segment costs | Safety, Exposure, TTC, Path, Time (see Evaluation); the sequence cost is their sum ($\gamma=1$) |
| Test sequences | 100 random sequences each for K = 1, 2, 4, 8, plus 136 K=4 sequences grouped by first edge (536 in total). All start at rest at a goal. $J_\mathrm R$ is the mean over 100 real rollouts |
| $\mathcal D_\mathrm{pair}$ | paired transitions extracted from real rollouts along random goal sequences of length 2–8 (real chains); at every segment the simulator is set to the current real state and executes the same goal. n = 200, 400, 600, 800, 1000, plus 50 for validation |
| $\mathcal D_\mathrm S$ | 20,000 one-segment simulated transitions plus 4,000 started near the real states of the n paired transitions |
| Training | MLP 3×256, AdamW (lr 1e-3). Same budget for every method: stop after 3,000 gradient steps without validation improvement, at most 30,000 steps. Paired targets make up 75% of the training rows of $W_\theta$ |
| Repetitions | 3 data seeds × 3 model seeds = 9 runs per n; 45 runs in total |

**Compared methods**

| Type | Name | Description |
|---|---|---|
| baseline | Raw simulation | $\widehat J_\mathrm S$: mean simulated cost of the target sequence |
| baseline | Segment-wise outcome correction | $J_\mathrm{seg}$ (SCAPE-style): a learned per-segment correction $h_\mathrm{out}(x,g,y^\mathrm S)\to y^\mathrm R$ summed along simulation rollouts; ignores the terminal-state mismatch |
| baseline | Target-only | backward Bellman learning of $W_\theta$ on the real side of $\mathcal D_\mathrm{pair}$ only (no simulation data) |
| baseline | B1 MC-real | direct regression of observed real returns over every window of the same real rollouts (no Bellman decomposition) |
| baseline | B2 route-level SCAPE | $h(x,\mathbf g, J_\mathrm S)\to$ observed real return on the same windows |
| proposed | Direct | $\widehat J_\mathrm R^\mathrm{direct}=\widehat W_1(x_1;\mathbf g_{1:K})$ (direct amortized estimator) |
| proposed | Simulation-backed | $\widehat J_\mathrm R^\mathrm{sim}$: target-sequence simulation rollouts plus the calibrated correction $h_\phi-\widehat Z_\mathrm S$ per segment |
| proposed | Residual direct | $W_\mathrm R=W_\mathrm S+D$ with $W_\mathrm S$ from simulation Monte Carlo; only $D$ is learned; estimate $\widehat J_\mathrm S+\widehat D(x_1;\mathbf g_{1:K})$ (METHODOLOGY §7) |

## Evaluation

Each method estimates the real sequence cost $J_\mathrm R(\mathbf g_{1:K})$ of every test sequence, separately for each of
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

Error metrics, computed over the test sequences of each length K ∈ {1, 2, 4, 8} against $J_\mathrm R$ from 100 real
rollouts per sequence:

| Metric | Meaning |
|---|---|
| MAE | mean absolute error of the estimate (headline: K = 8) |
| Bias | mean signed error (estimate − truth) |
| RMSE | root mean squared error |
| Spearman | rank correlation between estimates and truth over sequences (how well candidate routes are ranked) |
| Wins vs. target-only | number of runs in which a method has a lower K=8 MAE than target-only, the strongest baseline |

Every metric is reported per number of paired transitions n ∈ {200, 400, 600, 800, 1000} as mean ± std over 9 runs.

## Results (K = 8)

K=8 MAE, mean over 9 runs (lower is better; best in bold). Full tables, standard deviations, Spearman and per-K results
are on the results page.

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

Comparisons won against target-only at K=8 (5 costs × 5 values of n × 9 runs = 225): direct 201, residual direct 213, simulation-backed 162, segment-wise outcome correction 7, B1 0, B2 1.

- For K = 2, 4 and 8, the direct estimator has a lower MAE than every baseline for all five costs and all five values of n.
- At K = 1, raw simulation is the most accurate for Safety, Exposure and Path: every test sequence starts at rest, and a
  single segment from rest is almost the same in real and simulation, so no correction is needed.
- The simulation-backed estimator is worse than target-only on Path and Time at larger n; no single estimator is best
  on every cost.
- The 75% share of paired targets in the training rows of $W_\theta$ was chosen by looking at test results (a small sweep).

## Running

**Install.** Python 3.11 and `pip install -r requirements.txt` (choose the torch build that matches your CUDA version).
A GPU is strongly recommended for step 3; CPU-only training works but is very slow.

**Included results.** The 45 per-run metric files (`outputs/results/res_*.json`) and the results page with its figures
(`outputs/report/`) are part of the repository, so the results can be read without running anything.

**Pipeline.**

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/make_data.py` | `data/`: test sequences with ground truth, their simulation rollouts, $\mathcal D_\mathrm S$ (sim_train), real chains per data seed |
| 2 | `scripts/make_rows.py` | `data/`: simulated transitions near the real states (seedR), Monte Carlo rows for the residual variant, B1/B2 windows |
| 3 | `scripts/run.py D N S` | `outputs/results/res_d{D}_n{N}_s{S}.json`: 8 methods × 5 costs, MAE / bias / RMSE / Spearman per K |
| 4 | `scripts/report.py` | `outputs/report/`: summary.json, wins.json, figures, report.html |

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
| remaining-goal sequence $\mathbf g_{k:K}$ | `suffix` / `s`; its tail $\mathbf g_{k+1:K}$ is `tail` |
| real continuation value $W_\theta$ | `W` in `safer_av/ours.py` |
| calibration model $h_\phi$ | `h` in `safer_av/ours.py` |
| paired Bellman targets $\widehat Z_\mathrm R$, $\widehat Z_\mathrm S$ | `ZR`, `ZS` |
| segment cost $y$, terminal state $x_{k+1}$ | `y_<cost>`, `x_plus` in each transition record |
| $\widehat J_\mathrm R^\mathrm{direct}$, $\widehat J_\mathrm R^\mathrm{sim}$ | `direct`, `sim_backed` (results keys `W_direct`, `W_simbacked`) |
| residual $D$, $W_\mathrm S$ | `D`, `WS_S` / `WS_R` in `safer_av/residual.py` (results key `res_direct`) |

| File | Content |
|---|---|
| `safer_av/maze.py`, `policy.py`, `envs.py` | maze and goal graph, policy, real / simulation dynamics |
| `safer_av/segment.py`, `route.py` | one segment and its costs; a goal sequence executed without reset |
| `safer_av/data.py` | start-state samplers, $\mathcal D_\mathrm S$, test ground truth, real chains |
| `safer_av/models.py` | input encoding, MLP regressor, training budget |
| `safer_av/backward.py` | goal-sequence reuse, target-only, segment-wise outcome correction |
| `safer_av/ours.py` | backward Bellman calibration ($W_\theta$, $h_\phi$), direct and simulation-backed estimators |
| `safer_av/residual.py` | residual variant (Monte Carlo $W_\mathrm S$, learned $D$) |
| `safer_av/route_baselines.py` | B1, B2 |
