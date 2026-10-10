# Terminal-State-Aware Long-Horizon Sim-to-Real Evaluation

We consider long-horizon policy evaluation with limited short real–simulation paired transitions and abundant short
simulated transitions. Correcting only the simulated segment outcomes is insufficient because real and simulated
executions may terminate in different states, which changes the performance of all remaining goals. We propose
**backward Bellman calibration**, which reuses local transitions across valid remaining-goal sequences. The learned
models support both simulation-free direct prediction and a simulation-backed estimator justified by a finite-horizon
simulation lemma. Target-sequence simulation rollouts are generated at evaluation time.

Sections 1–6 describe the method. Section 7 describes the residual variant that is also evaluated in the toy study, and
Section 8 states how the method is instantiated in this repository.

---

## 1. Problem setting

An evaluation query is a goal sequence $\mathbf g_{1:K}=(g_1,\dots,g_K)$ of $K$ local goals executed by a fixed
goal-conditioned policy $\pi$. The sequence length and ordering may vary across queries. For environment
$e\in\{\mathrm R,\mathrm S\}$ (real, simulation), segment $k$ begins from state $X_k^e$ and produces a scalar cost
$Y_k^e$ and terminal state $X_{k+1}^e$:

$$
(Y_k^e, X_{k+1}^e) \sim K_e^\pi(\cdot,\cdot \mid X_k^e, g_k).
$$

Both environments start from the same fixed initial state $x_1$. The long-horizon performance is

$$
J_e^\pi(\mathbf g_{1:K}) := \mathbb E_e^\pi\Big[\sum_{k=1}^{K}\gamma^{k-1}Y_k^e\Big],\qquad \gamma\in(0,1],
$$

and the objective is to estimate the real performance $J_\mathrm R^\pi(\mathbf g_{1:K})$.

**Short paired data.** A limited dataset
$\mathcal D_\mathrm{pair}=\{(x_{i,k}, g_i, y_{i,k}^\mathrm R, x_{i,k+1}^\mathrm R, y_{i,k}^\mathrm S, x_{i,k+1}^\mathrm S)\}_{i=1}^{n}$,
where the real and simulated executions start from the same state and execute the same local goal.

**Short simulation data.** A larger dataset of local simulated transitions
$\mathcal D_\mathrm S=\{(x_{j,k}^\mathrm S, g_j, y_{j,k}^\mathrm S, x_{j,k+1}^\mathrm S)\}_{j=1}^{m}$, $m\gg n$.

Here $i$ and $j$ index short-transition samples. When a transition is reused at position $k$ of a goal sequence, $k$
denotes its assigned position and $k+1$ its terminal state. Short transitions may be collected independently or
extracted from longer rollouts; they need not form connected trajectories for training.

**Goal-sequence reuse.** A local transition for goal $g_k$ can be used to train values for multiple valid
remaining-goal sequences $(g_k,\dots,g_K)$: its observed outcome and terminal state are combined with the previously
learned value for the selected suffix $(g_{k+1},\dots,g_K)$. Reusing a transition with different suffixes does not
create additional independent observations. A valid goal sequence must be **sequentially feasible**: after executing
$g_k$, the next goal $g_{k+1}$ must be meaningful and executable from the resulting state.

## 2. Why local outcome correction is insufficient

A one-shot method such as SCAPE corrects the outcome mismatch $Y^\mathrm R-Y^\mathrm S$. A long-horizon task has an
additional **terminal-state mismatch** $X_{k+1}^\mathrm R\neq X_{k+1}^\mathrm S$, which matters because the terminal
state determines the difficulty of all remaining goals. With $\mu_e(x,g):=\mathbb E_e[Y^e\mid X^e=x,g]$, a
**segment-wise outcome correction** estimates

$$
J_\mathrm{seg}^\pi = J_\mathrm S^\pi + \sum_{k=1}^{K}\gamma^{k-1}\,\mathbb E_{X_k\sim d_{\mathrm S,k}}\big[\mu_\mathrm R(X_k,g_k)-\mu_\mathrm S(X_k,g_k)\big],
$$

where $d_{\mathrm S,k}$ is the state distribution at segment $k$ of the long simulation rollout. Even with perfect local
outcome correction, this follows the simulated state sequence and ignores that real execution may reach a different
terminal state and therefore a different future cost. Directly predicting $X_{k+1}^\mathrm R$ from
$X_{k+1}^\mathrm S$ instead requires predicting a potentially high-dimensional, multimodal state and propagating the
prediction forward repeatedly.

## 3. Backward Bellman calibration

### 3.1 Real continuation value

For the remaining goal sequence $\mathbf g_{k:K}=(g_k,\dots,g_K)$, the **real continuation value** is

$$
W_{\mathrm R,k}^\pi(x;\mathbf g_{k:K}) := \mathbb E_\mathrm R^\pi\Big[\sum_{\ell=k}^{K}\gamma^{\ell-k}Y_\ell^\mathrm R \,\Big|\, X_k^\mathrm R=x\Big],\qquad W_{\mathrm R,K+1}^\pi=0,
$$

the expected discounted cost of executing all remaining goals in the real world from state $x$. A shared approximation
$W_\theta(x;\mathbf g_{k:K})\approx W_{\mathrm R,k}^\pi(x;\mathbf g_{k:K})$ is learned. The remaining goal sequence is
an input because the same state may be safe for one continuation and risky for another.

### 3.2 Paired Bellman targets

Given an already learned $\widehat W_{k+1}$, a paired transition whose current goal is $g_k$ gives

$$
\widehat Z_{i,\mathrm R}^{(k)} := y_{i,k}^\mathrm R + \gamma\,\widehat W_{k+1}(x_{i,k+1}^\mathrm R;\mathbf g_{k+1:K}),\qquad
\widehat Z_{i,\mathrm S}^{(k)} := y_{i,k}^\mathrm S + \gamma\,\widehat W_{k+1}(x_{i,k+1}^\mathrm S;\mathbf g_{k+1:K}).
$$

Both targets evaluate the remaining goals under real-world dynamics; they differ in how the current segment is
executed. Their difference is

$$
\widehat Z_{i,\mathrm R}^{(k)}-\widehat Z_{i,\mathrm S}^{(k)} =
\underbrace{y_{i,k}^\mathrm R-y_{i,k}^\mathrm S}_{\text{current outcome mismatch}}
+\gamma\underbrace{\big[\widehat W_{k+1}(x_{i,k+1}^\mathrm R)-\widehat W_{k+1}(x_{i,k+1}^\mathrm S)\big]}_{\text{future effect of terminal-state mismatch}} .
$$

The method does not reconstruct the real terminal state; it converts the terminal-state mismatch into its scalar effect
on the remaining real-world performance.

### 3.3 Calibration model

The **calibration model**

$$
h_\phi(x,\mathbf g_{k:K},z^\mathrm S)\approx \mathbb E\big[Z^\mathrm R \mid X=x,\ \mathbf G=\mathbf g_{k:K},\ Z^\mathrm S=z^\mathrm S\big]
$$

is learned from the paired targets $(\widehat Z_{i,\mathrm S}^{(k)},\widehat Z_{i,\mathrm R}^{(k)})$. For a short
simulated transition in $\mathcal D_\mathrm S$, choose a valid remaining-goal sequence $\mathbf g_{k:K}$ with $g_k=g_j$
and compute

$$
\widehat Z_{j,\mathrm S}^{(k)} := y_{j,k}^\mathrm S + \gamma\,\widehat W_{k+1}(x_{j,k+1}^\mathrm S;\mathbf g_{k+1:K}),\qquad
\widetilde Z_{j,\mathrm R}^{(k)} := h_\phi\big(x_{j,k}^\mathrm S,\mathbf g_{k:K},\widehat Z_{j,\mathrm S}^{(k)}\big).
$$

$W_\theta$ is trained on real targets from paired data and corrected targets from simulation:
$W_\theta(x_{i,k};\mathbf g_{k:K})\leftarrow\widehat Z_{i,\mathrm R}^{(k)}$ (paired data) and
$W_\theta(x_{j,k}^\mathrm S;\mathbf g_{k:K})\leftarrow\widetilde Z_{j,\mathrm R}^{(k)}$ (simulation data).

### 3.4 Training over the remaining-goal length

Training starts with one-goal sequences ($\widehat W_{K+1}=0$) and increases the remaining-goal length
$L=1,2,\dots,L_\max$. At each length, paired and simulated short transitions are reused with valid remaining-goal
sequences of that length, and the value for the shorter suffix is held fixed while the longer value is trained. The
sequence-conditioned models $W_\theta$ and $h_\phi$ are shared across goal orderings. Dataset splits are made at the
original transition level, before suffix augmentation.

## 4. Two evaluation modes

**Direct amortized estimator.**
$\widehat J_\mathrm R^\mathrm{direct}(x_1,\mathbf g_{1:K}) := \widehat W_1(x_1;\mathbf g_{1:K})$.
No new simulation rollout is needed per query; simulation is used during offline training and amortized into
$W_\theta$. Its accuracy depends on whether $W_\theta$ generalizes to the new initial state and goal sequence.

**Simulation-backed estimator.** Generate $M$ long simulation rollouts of the target sequence from $x_1$, with average
cost $\widehat J_\mathrm S^\pi=\frac1M\sum_{j}\sum_k\gamma^{k-1}Y_{j,k}^\mathrm S$. For every simulated segment,

$$
\widehat Z_{j,\mathrm S}^{(k)} := Y_{j,k}^\mathrm S+\gamma\,\widehat W_{k+1}(X_{j,k+1}^\mathrm S;\mathbf g_{k+1:K}),\qquad
\widehat\Delta_{j,k} := h_\phi\big(X_{j,k}^\mathrm S,\mathbf g_{k:K},\widehat Z_{j,\mathrm S}^{(k)}\big)-\widehat Z_{j,\mathrm S}^{(k)},
$$

$$
\widehat J_\mathrm R^\mathrm{sim} = \widehat J_\mathrm S^\pi + \sum_{k=1}^{K}\gamma^{k-1}\frac1M\sum_{j=1}^{M}\widehat\Delta_{j,k}.
$$

| Estimator | New target-route simulation | Main limitation |
|---|---|---|
| $\widehat J_\mathrm R^\mathrm{direct}$ | Not required | Value generalization |
| $\widehat J_\mathrm R^\mathrm{sim}$ | Required | Simulation cost |

## 5. Theoretical interpretation

### 5.1 Finite-horizon sim-to-real decomposition

Fix a target sequence $\mathbf g_{1:K}$ (the result is route-wise). The **local Bellman discrepancy** is

$$
\Delta_k(x) := \mathbb E_\mathrm R\big[Y_k^\mathrm R+\gamma W_{\mathrm R,k+1}^\pi(X_{k+1}^\mathrm R)\mid X_k^\mathrm R=x,g_k\big]
-\mathbb E_\mathrm S\big[Y_k^\mathrm S+\gamma W_{\mathrm R,k+1}^\pi(X_{k+1}^\mathrm S)\mid X_k^\mathrm S=x,g_k\big].
$$

**Proposition.** If real and simulated executions have the same initial-state distribution, then

$$
J_\mathrm R^\pi-J_\mathrm S^\pi=\sum_{k=1}^{K}\gamma^{k-1}\,\mathbb E_{X_k\sim d_{\mathrm S,k}}\big[\Delta_k(X_k)\big].
$$

*Proof sketch.* Let $G_k:=W_{\mathrm R,k}^\pi-W_{\mathrm S,k}^\pi$ and introduce the hybrid target
$H_k(x):=\mathbb E_\mathrm S[Y_k^\mathrm S+\gamma W_{\mathrm R,k+1}^\pi(X_{k+1}^\mathrm S)\mid x,g_k]$ (simulated
current outcome and next-state distribution, real continuation value). Then $W_{\mathrm R,k}^\pi-H_k=\Delta_k$ and
$H_k-W_{\mathrm S,k}^\pi=\gamma\,\mathbb E_\mathrm S[G_{k+1}(X_{k+1}^\mathrm S)\mid x,g_k]$, so
$G_k(x)=\Delta_k(x)+\gamma\,\mathbb E_\mathrm S[G_{k+1}(X_{k+1}^\mathrm S)\mid x,g_k]$. Averaging over
$d_{\mathrm S,k}$, unrolling to $G_{K+1}=0$, and using $d_{\mathrm R,1}=d_{\mathrm S,1}$ gives the result. $\square$

**Interpretation.** At each segment, the remaining real–simulation gap splits into the local mismatch
$\Delta_k(X_k)$ and the gap remaining after the *simulated* transition, $\gamma G_{k+1}(X_{k+1}^\mathrm S)$. Expanding
the second part follows the simulated trajectory, so the correction is averaged under $d_{\mathrm S,k}$ rather than
$d_{\mathrm R,k}$. Short paired transitions are used to learn $\Delta_k$; long simulation rollouts provide samples from
$d_{\mathrm S,k}$. This is the justification of the simulation-backed estimator.

### 5.2 Segment-wise correction as a special case

With the outcome-only discrepancy $\Delta_k^\mathrm{out}(x):=\mu_\mathrm R(x,g_k)-\mu_\mathrm S(x,g_k)$,

$$
\Delta_k(x)=\Delta_k^\mathrm{out}(x)+\gamma\Big\{\mathbb E_\mathrm R\big[W_{\mathrm R,k+1}^\pi(X_{k+1}^\mathrm R)\mid x,g_k\big]-\mathbb E_\mathrm S\big[W_{\mathrm R,k+1}^\pi(X_{k+1}^\mathrm S)\mid x,g_k\big]\Big\}.
$$

Segment-wise outcome correction is the special case that removes the second term, so

**proposed method = segment-wise outcome correction + terminal-state continuation correction.**

## 6. Required assumptions and scope

We evaluate a fixed policy using a segment-level Markov state and additive costs. Goal sequences must be sequentially
feasible. Reusing a short transition with different remaining-goal sequences assumes that the current segment's
transition law depends only on its starting state and current goal; if the policy's current actions depend on the full
plan, arbitrary suffix reuse is not valid.

Real and simulated states must share a representation on which $W_\theta$ can be evaluated. The paired data must
adequately cover the states visited in target simulation rollouts and the terminal states used to learn continuation
values, and the paired calibration relationship must transfer to those states. Both estimators rely on the learned
models generalizing to the states and goal suffixes used at evaluation. The simulation-backed estimator additionally
requires the target sequence to be executable in simulation; the direct estimator must predict the full-sequence value
from the initial state.

The simulator is treated as a fixed black box: its parameters cannot be tuned to imitate the real system, and the
real–simulation difference is not assumed to be expressible through simulator parameters. System identification and
learned real-dynamics or next-state models are therefore outside the scope; the only operation is to correct
simulation outcomes toward the real system.

## 7. Residual variant (also evaluated in the toy study)

The real continuation value is split into a simulation part and a residual,
$W_{\mathrm R}(x;\mathbf g_{k:K}) = W_{\mathrm S}(x;\mathbf g_{k:K}) + D(x;\mathbf g_{k:K})$, where $W_\mathrm S$ is
computed by running the simulator from $x$ along the remaining goals (Monte Carlo) and only $D$ is learned backward:

$$
D(x_{i,k};\mathbf g_{k:K}) \leftarrow \underbrace{(y_{i,k}^\mathrm R-y_{i,k}^\mathrm S)}_{\text{current outcome mismatch}}
+\gamma\underbrace{\big[W_\mathrm S(x_{i,k+1}^\mathrm R)-W_\mathrm S(x_{i,k+1}^\mathrm S)\big]}_{\text{terminal-state mismatch, simulation value}}
+\gamma\,\widehat D_{k+1}(x_{i,k+1}^\mathrm R;\mathbf g_{k+1:K}).
$$

Simulation data enter through a calibration model $h_D(x,\mathbf g_{k:K},z^\mathrm S)$ with
$z^\mathrm S=y^\mathrm S+\gamma W_\mathrm S(x^\mathrm S_{+})$, as in Section 3.3. The direct residual estimate is
$\widehat J_\mathrm R = \widehat J_\mathrm S^\pi + \widehat D(x_1;\mathbf g_{1:K})$. Because the large common part
$W_\mathrm S$ is not learned, this form helps when the real–simulation difference is small relative to the value itself
(e.g. path length). It requires multi-segment simulation rollouts from arbitrary states during training, which departs
from the short-simulation-data setting of Section 1.

## 8. Instantiation in the toy study

| Element | Toy study (PointMaze Sim2Sim) |
|---|---|
| Local goals | 9 junction goals A–I; a valid goal sequence is a non-backtracking walk on the goal graph |
| Policy $\pi$ | PD controller toward the end of the current straight corridor of the shortest path; sees only the current goal |
| Segment | ends when the ball enters the radius-0.2 ball around the goal (or after 5 s); the terminal state keeps its velocity |
| Costs $Y_k$, $\gamma$ | Safety, Exposure, TTC, Path, Time (each evaluated separately); $\gamma=1$ |
| $\mathcal D_\mathrm{pair}$ | paired transitions extracted from real rollouts along random goal sequences (real chains): at every segment the simulator is set to the current real state and executes the same goal |
| $\mathcal D_\mathrm S$ | 20,000 one-segment simulated transitions plus 4,000 started near the real states of $\mathcal D_\mathrm{pair}$ |
| Reuse | 6 sampled remaining-goal sequences per paired transition and length, 2 per simulated transition; $L_\max=8$ |
| Training rows of $W_\theta$ | paired targets $\widehat Z_\mathrm R$ make up 75% of the rows, corrected simulation targets $\widetilde Z_\mathrm R$ 25% |
| Simulation-backed estimator | $M=30$ target-sequence simulation rollouts |
| Models | MLP 3×256 for $W_\theta$, $h_\phi$ and all learned baselines, same gradient-step budget for every method |

**Compared methods.** Baselines: raw simulation $\widehat J_\mathrm S$; segment-wise outcome correction $J_\mathrm{seg}$
(SCAPE-style); target-only (the same backward Bellman learning on the real side of $\mathcal D_\mathrm{pair}$ only, no
simulation); B1 MC-real (direct regression of observed real returns of every window of the same real rollouts, no
Bellman decomposition); B2 route-level SCAPE ($h(x,\mathbf g,J_\mathrm S)\to J_\mathrm R$ on the same windows).
Proposed: $\widehat J_\mathrm R^\mathrm{direct}$, $\widehat J_\mathrm R^\mathrm{sim}$, and the residual direct estimate
(Section 7).

**Metrics.** Error against $J_\mathrm R$ from 100 real rollouts per test sequence: MAE, bias and Spearman rank
correlation over test sequences, reported per sequence length $K$ and number of paired transitions $n$.
