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

Notation: a test sequence starts at state $x_1$ and visits goals $\mathbf g_{1:K}=(g_1,\dots,g_K)$. For one segment,
$y^\mathrm R, y^\mathrm S$ are its real and simulated costs and $x^\mathrm R_+, x^\mathrm S_+$ the states where the real and
the simulated run end it. $\mathbf g_{k:K}$ are the goals still to visit.

**Two learned models (used by Direct and Sim-backed).**

- $W_\theta(x;\mathbf g_{k:K})$: the expected **real** cost of visiting the remaining goals $\mathbf g_{k:K}$ from state $x$.
  It is learned backward: first with one remaining goal, then two, and so on up to eight. For a paired segment the
  training target is this segment's real cost plus the already learned value of the rest, starting from where the real
  run ended:

  $$\widehat Z_\mathrm R = y^\mathrm R + \widehat W(x^\mathrm R_+;\mathbf g_{k+1:K}).$$

- $h_\phi(x,\mathbf g_{k:K},z^\mathrm S)$: a calibration model that maps the same quantity computed from the simulated
  run, $\widehat Z_\mathrm S = y^\mathrm S + \widehat W(x^\mathrm S_+;\mathbf g_{k+1:K})$, to the real one $\widehat Z_\mathrm R$.
  It is fit on paired segments. Simulation-only segments have no real run, so $h_\phi$ turns their $\widehat Z_\mathrm S$
  into a corrected real target $\widetilde Z_\mathrm R = h_\phi(x,\mathbf g_{k:K},\widehat Z_\mathrm S)$, which is extra
  training data for $W_\theta$ (75% of the training rows are paired, 25% corrected simulation rows).

**1. Direct.** Read the learned value at the start state:

$$\widehat J_\mathrm R^\mathrm{direct} = \widehat W(x_1;\mathbf g_{1:K}).$$

**2. Sim-backed: run the sequence in simulation, then correct every segment.** Run the whole test sequence $M=30$
times in simulation from $x_1$. For run $j$ and segment $k$, with simulated cost $Y^\mathrm S_{j,k}$ and states
$X^\mathrm S_{j,k}\to X^\mathrm S_{j,k+1}$:

$$\widehat Z^\mathrm S_{j,k} = Y^\mathrm S_{j,k} + \widehat W(X^\mathrm S_{j,k+1};\mathbf g_{k+1:K}),\qquad
\widehat\Delta_{j,k} = h_\phi\big(X^\mathrm S_{j,k},\mathbf g_{k:K},\widehat Z^\mathrm S_{j,k}\big) - \widehat Z^\mathrm S_{j,k},$$

$$\widehat J_\mathrm R^\mathrm{sim} = \underbrace{\frac1M\sum_{j=1}^{M}\sum_{k=1}^{K} Y^\mathrm S_{j,k}}_{\text{simulated cost}}
+ \underbrace{\sum_{k=1}^{K}\frac1M\sum_{j=1}^{M}\widehat\Delta_{j,k}}_{\text{correction per segment}}.$$

$\widehat\Delta_{j,k}$ is how much more (or less) the real system would cost than the simulation from the same state
$X^\mathrm S_{j,k}$, including the effect of ending the segment in a different state. Summing these corrections along
simulated runs gives exactly the real–simulation gap when the models are exact (sim-to-real decomposition,
METHODOLOGY §5).

**3. Residual direct: learn only the real–simulation difference.** Write the real value as the simulated value plus a
difference, $W_\mathrm R = W_\mathrm S + D$. $W_\mathrm S(x;\mathbf g)$ is not learned: it is computed by running the
simulator from $x$ through the goals $\mathbf g$. Only $D$ is learned, backward like $W_\theta$, with the target

$$D(x;\mathbf g_{k:K}) \leftarrow \underbrace{(y^\mathrm R - y^\mathrm S)}_{\text{cost difference of this segment}}
+ \underbrace{\big[W_\mathrm S(x^\mathrm R_+;\mathbf g_{k+1:K}) - W_\mathrm S(x^\mathrm S_+;\mathbf g_{k+1:K})\big]}_{\text{effect of the different end state, measured in simulation}}
+ \underbrace{\widehat D(x^\mathrm R_+;\mathbf g_{k+1:K})}_{\text{learned difference of the rest}}.$$

Simulation-only segments enter through a calibration model, as above. The estimate is the simulated cost of the test
sequence plus the learned difference:

$$\widehat J_\mathrm R^\mathrm{res} = \widehat J_\mathrm S + \widehat D(x_1;\mathbf g_{1:K}).$$

This helps when the difference is much smaller than the cost itself. For Path, the K=8 real cost is about 20.5 and
the real–simulation difference about 1; learning a value of 20 with 0.3% error already costs 0.06, while learning a
difference of 1 does not have this problem. The price is that $W_\mathrm S$ needs many multi-segment simulation runs
during training.

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
| 4 | `scripts/report.py` | `outputs/report/`: summary.json (means over the 9 runs), fig_mae_vs_n.png, report.html |

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
