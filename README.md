# safer-av: Terminal-State-Aware Long-Horizon Sim-to-Real Evaluation (toy study)

Given a few short real–simulation paired transitions and many short simulated transitions, we estimate the **real
performance of a fixed goal-conditioned policy over a long goal sequence** $\mathbf g_{1:K}$. Unlike segment-wise
outcome correction (SCAPE-style), backward Bellman calibration also corrects how the real–simulation terminal-state
mismatch changes the cost of all remaining goals.

- Method and theory: [METHODOLOGY.md](METHODOLOGY.md)
- Results page: `outputs/report/report.html` (written by `scripts/report.py`)

## Setting

| Item | Setting |
|---|---|
| Environment | PointMaze (MuJoCo ball) with a 9×7-cell maze; 9 local goals A–I at the junctions. A valid goal sequence is a non-backtracking walk between adjacent goals |
| Policy $\pi$ | PD controller (kp 2.0, kd 0.4) toward the end of the current straight corridor of the shortest path; sees only the current goal. A segment ends on entering the radius-0.2 ball around the goal (timeout 5 s); the terminal state keeps its velocity |
| Real vs. simulation | Sim2Sim: the "real" system is the simulator with weaker force transmission (β = 0.85) and much weaker braking (brake = 0.4: only 40% of commands opposing the velocity). Both have actuation noise 0.02 (`configs/env.json`) |
| Segment costs | Safety, Exposure, TTC, Path, Time (`safer_av/segment.py`); the sequence cost is their sum ($\gamma=1$) |
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

## Running

Environment: Python 3.11, torch 2.11, gymnasium 1.3.0, mujoco 3.14.0, numpy 2.4, matplotlib. gymnasium-robotics is
installed in editable mode from `third_party/Gymnasium-Robotics` (commit 4d1ebec, version 1.4.2):
`pip install -e third_party/Gymnasium-Robotics`.

```bash
GPUS="0 1" bash scripts/run_all.sh
```

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/make_data.py` | `data/`: test sequences with ground truth, their simulation rollouts, $\mathcal D_\mathrm S$ (sim_train), real chains per data seed |
| 2 | `scripts/make_rows.py` | `data/`: simulated transitions near the real states (seedR), Monte Carlo rows for the residual variant, B1/B2 windows |
| 3 | `scripts/run.py D N S` | `outputs/results/res_d{D}_n{N}_s{S}.json`: 8 methods × 5 costs, MAE / bias / RMSE / Spearman per K |
| 4 | `scripts/report.py` | `outputs/report/`: summary.json, wins.json, figures, report.html |

Steps 1 and 2 take about 20 minutes on a many-core machine. One `run.py` takes about 1–1.5 hours on a GPU when the 45
runs share GPUs. All seeds are fixed, so regenerating the data reproduces the same files.

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
