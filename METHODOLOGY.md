# Terminal-State-Aware Long-Horizon Sim-to-Real Evaluation

Reliable policy evaluation in real-world settings is essential for determining whether a policy can operate safely
under a given scenario. Simulation-augmented policy evaluation combines limited real-world data with scalable
simulation-based testing to improve performance estimation. However, extending this approach to long horizons involving
multiple goal sequences remains challenging. Real-world rollouts that cover diverse, extended goal sequences are
difficult to obtain, and the sim-to-real gap can compound over successive goals as the robot's state diverges between
simulation and the real world. Consequently, simulation rollouts cannot be used directly to estimate real-world
performance over long sequences.

To address this challenge, we propose **backward Bellman calibration**, a method for estimating performance over
increasingly long goal sequences by reusing estimates for their future goal suffixes. We further introduce a real-sim
calibration procedure that corrects simulation-based Bellman targets using paired real-world and simulated data.
Together, these methods enable simulation data to support the estimation of real-world policy performance over long
goal sequences.

Sections 1–3 describe the method. Section 4 describes a residual variant that we also tried in the toy study (it is not
part of the method), and Section 5 states how the method is instantiated in this repository.

---

## 1. Problem formulation

### 1.1 Evaluation query and goal-conditioned policy

Let $\mathcal G$ be a predefined set of local goals. We evaluate a fixed policy $\pi$ from an initial observation $x_1$
over a valid sequence of $K$ goals drawn from $\mathcal G$,

$$
g_{1:K}=(g_1,\ldots,g_K),\qquad g_k\in\mathcal G.
$$

The number and order of goals may vary across evaluation queries. While executing goal $g_k$, the policy selects each
action from its current observation $x$ according to $\pi(a\mid x,g_k)$.

### 1.2 Real and simulated segment transitions

Executing each goal defines a segment. Let $e\in\lbrace \mathrm R,\mathrm S\rbrace$ denote the real world or
simulation, and let $X_k^e$ be the observation at the start of segment $k$. The actions selected by $\pi$ during this
segment produce a scalar cost $Y_k^e$ and a terminal observation $X_{k+1}^e$:

$$
(Y_k^e,X_{k+1}^e)\sim P_e^\pi(\cdot\mid X_k^e,g_k).
$$

Here, $Y_k^e$ measures performance during the execution of $g_k$, such as a collision or traffic-rule violation cost.
The terminal observation $X_{k+1}^e$ becomes the initial observation for the next goal. We assume that real and
simulated observations share a representation and that $X_k^e$ contains the information needed to characterize the
segment transition under $\pi$.

### 1.3 Evaluation objective and available data

For an evaluation query $(x_1,g_{1:K})$, our objective is to estimate the expected cumulative real-world cost

$$
J_\mathrm R^\pi(x_1,g_{1:K}) := \mathbb E_\mathrm R^\pi\Big[\sum_{k=1}^{K}\gamma^{k-1}Y_k^\mathrm R \thinspace\Big|\thinspace X_1^\mathrm R=x_1,\ g_{1:K}\Big],
$$

where $\gamma\in(0,1]$ is the discount factor per segment.

We have a limited set of paired real and simulated segment transitions,

$$
\mathcal D_\mathrm p=\big\lbrace (x_i^\mathrm p,g_i^\mathrm p,y_i^\mathrm R,x_{i,\mathrm{end}}^\mathrm R,y_i^\mathrm S,x_{i,\mathrm{end}}^\mathrm S)\big\rbrace_{i=1}^{n}.
$$

Here, $i$ indexes the $n$ paired transitions, and the superscript $\mathrm p$ labels their shared starting observations
and goals. In each pair, the real and simulated segments start from the same observation $x_i^\mathrm p$ and execute
the same goal $g_i^\mathrm p$. These pairs need not be collected as independent, single-goal rollouts: a real-world
rollout may execute a sequence of goals, with a simulated segment started from a recorded real-world observation at
each selected goal boundary.

We also have a larger set of simulation-only segment transitions,

$$
\mathcal D_\mathrm s=\big\lbrace (x_j^\mathrm s,g_j^\mathrm s,y_j^\mathrm S,x_{j,\mathrm{end}}^\mathrm S)\big\rbrace_{j=1}^{m},\qquad m\gg n.
$$

Here, $j$ indexes the $m$ simulation-only transitions, and the superscript $\mathrm s$ labels their starting
observations and goals. These transitions may likewise be obtained from simulation rollouts that execute sequences of
goals. The subscript $\mathrm{end}$ denotes the terminal observation of a segment. For segments extracted from the same
rollout, we also retain their rollout membership and order.

## 2. Backward Bellman calibration

### 2.1 Real continuation value function

For a remaining goal sequence $g_{k:K}=(g_k,\ldots,g_K)$, define the real continuation value function

$$
V_\mathrm R^\pi(x;g_{k:K}) := \mathbb E_\mathrm R^\pi\Big[\sum_{\ell=k}^{K}\gamma^{\ell-k}Y_\ell^\mathrm R \thinspace\Big|\thinspace X_k^\mathrm R=x,\ g_{k:K}\Big].
$$

It is the expected cost of executing the remaining goals in the real world from observation $x$. For an empty goal
suffix, define $V_\mathrm R^\pi(x;\emptyset)=0$. In particular, the evaluation objective satisfies

$$
J_\mathrm R^\pi(x_1,g_{1:K})=V_\mathrm R^\pi(x_1;g_{1:K}).
$$

We learn a goal-sequence-conditioned approximation $V_\theta(x;g_{k:K})$ to this value function, with
$V_\theta(x;\emptyset)=0$. Training starts with one-goal sequences and proceeds to longer sequences, so the value for
$g_{k+1:K}$ has already been learned when constructing targets for $g_{k:K}$.

### 2.2 Paired Bellman targets

Select a paired transition
$(x_i^\mathrm p,g_i^\mathrm p,y_i^\mathrm R,x_{i,\mathrm{end}}^\mathrm R,y_i^\mathrm S,x_{i,\mathrm{end}}^\mathrm S)\in\mathcal D_\mathrm p$
and a valid remaining goal sequence $g_{k:K}$ with $g_i^\mathrm p=g_k$. Using the value function learned for the
shorter suffix $g_{k+1:K}$, define

$$
Z_{i,\mathrm R}^{(k)} := y_i^\mathrm R+\gamma V_\theta(x_{i,\mathrm{end}}^\mathrm R;g_{k+1:K}),\qquad
Z_{i,\mathrm S}^{(k)} := y_i^\mathrm S+\gamma V_\theta(x_{i,\mathrm{end}}^\mathrm S;g_{k+1:K}).
$$

The shorter-suffix outputs of $V_\theta$ are held fixed when these targets are constructed. The real target
$Z_{i,\mathrm R}^{(k)}$ combines the observed real cost of the current goal with the estimated real-world cost of the
remaining goals. It serves as a training target for $V_\theta(x_i^\mathrm p;g_{k:K})$. The simulated target
$Z_{i,\mathrm S}^{(k)}$ instead uses the observed simulated cost and estimates real-world continuation from the
simulated terminal observation.

The difference between the targets is

$$
Z_{i,\mathrm R}^{(k)}-Z_{i,\mathrm S}^{(k)} = y_i^\mathrm R-y_i^\mathrm S
+\gamma\big[V_\theta(x_{i,\mathrm{end}}^\mathrm R;g_{k+1:K})-V_\theta(x_{i,\mathrm{end}}^\mathrm S;g_{k+1:K})\big].
$$

The first term is the difference in cost while executing the current goal. The second captures how the difference in
terminal observations affects the estimated real-world cost of the remaining goals.

Because the shorter-suffix value function is already learned, $Z_{i,\mathrm S}^{(k)}$ can also be computed for a
simulation-only transition. We therefore use the paired targets to train a calibrator that predicts the real target
from the starting observation, remaining goals, and simulated target. The calibrated targets then provide additional
training data for the longer-suffix value function.

### 2.3 Calibration and value-function training

Using paired transitions, we train a calibrator $h_\phi$ to predict $Z_{i,\mathrm R}^{(k)}$ from the shared starting
observation $x_i^\mathrm p$, the remaining goals $g_{k:K}$, and the simulated target $Z_{i,\mathrm S}^{(k)}$:

$$
h_\phi\big(x_i^\mathrm p,g_{k:K},Z_{i,\mathrm S}^{(k)}\big)\approx Z_{i,\mathrm R}^{(k)}.
$$

For a simulation-only transition indexed by $j$ in $\mathcal D_\mathrm s$ with $g_j^\mathrm s=g_k$, compute

$$
Z_{j,\mathrm S}^{(k)} := y_j^\mathrm S+\gamma V_\theta(x_{j,\mathrm{end}}^\mathrm S;g_{k+1:K}),\qquad
Z_{j,\mathrm{cal}}^{(k)} := h_\phi\big(x_j^\mathrm s,g_{k:K},Z_{j,\mathrm S}^{(k)}\big).
$$

Here, $Z_{j,\mathrm{cal}}^{(k)}$ is the calibrated target for a simulation-only transition. We train $V_\theta$ with the
real targets from paired transitions and the calibrated targets from simulation-only transitions:

$$
V_\theta(x_i^\mathrm p;g_{k:K})\approx Z_{i,\mathrm R}^{(k)},\qquad
V_\theta(x_j^\mathrm s;g_{k:K})\approx Z_{j,\mathrm{cal}}^{(k)}.
$$

Thus, the value function for $g_{k:K}$ is learned using the previously learned value function for $g_{k+1:K}$. The
newly learned values are then used to construct targets for longer goal sequences.

## 3. Two evaluation modes

### 3.1 Direct estimation using the value function

Given an evaluation query $(x_1,g_{1:K})$, the direct estimator is

$$
J_\mathrm{direct}^\pi(x_1,g_{1:K}) := V_\theta(x_1;g_{1:K}).
$$

It estimates the expected real-world cost without simulating the evaluation query.

### 3.2 Simulation-backed estimation

When the evaluation query can be executed in simulation, we run $\pi$ from $x_1$ under the same goal sequence
$g_{1:K}$. Let $(x_k^\mathrm S,y_k^\mathrm S,x_{k+1}^\mathrm S)$ be the observed result of simulated segment $k$, with
$x_1^\mathrm S=x_1$. For each segment, compute

$$
Z_{k,\mathrm S} := y_k^\mathrm S+\gamma V_\theta(x_{k+1}^\mathrm S;g_{k+1:K}),\qquad
Z_{k,\mathrm{cal}} := h_\phi(x_k^\mathrm S,g_{k:K},Z_{k,\mathrm S}).
$$

The simulation-backed estimate from this rollout is

$$
J_\mathrm{SB}^\pi(x_1,g_{1:K}) := \sum_{k=1}^{K}\gamma^{k-1}\big(y_k^\mathrm S+Z_{k,\mathrm{cal}}-Z_{k,\mathrm S}\big).
$$

When multiple simulated rollouts are available, we average the resulting estimates.

### 3.3 Relation to direct estimation

Suppose the mean calibrated target at each simulated observation matches the learned value function:

$$
\mathbb E_\mathrm S^\pi\big[Z_{k,\mathrm{cal}}\thinspace\big|\thinspace X_k^\mathrm S=x,\ g_{k:K}\big]=V_\theta(x;g_{k:K}).
$$

Since $Z_{k,\mathrm S}=Y_k^\mathrm S+\gamma V_\theta(X_{k+1}^\mathrm S;g_{k+1:K})$, taking the expectation of the
simulation-backed estimator gives

$$
\mathbb E_\mathrm S^\pi\big[J_\mathrm{SB}^\pi(x_1,g_{1:K})\big]
=\mathbb E_\mathrm S^\pi\Big[\sum_{k=1}^{K}\gamma^{k-1}\big(V_\theta(X_k^\mathrm S;g_{k:K})-\gamma V_\theta(X_{k+1}^\mathrm S;g_{k+1:K})\big)\Big]
=V_\theta(x_1;g_{1:K})=J_\mathrm{direct}^\pi(x_1,g_{1:K}),
$$

where the terms cancel because $X_1^\mathrm S=x_1$ and $V_\theta(x;\emptyset)=0$. The simulation-backed estimator was
essentially equivalent to the direct estimator in expectation and offered no clear advantage.

## 4. Residual variant (tried in the toy study; not part of the method)

We also tried splitting the real continuation value into a simulation part and a learned residual,
$V_\mathrm R^\pi(x;g_{k:K})=V_\mathrm S^\pi(x;g_{k:K})+D(x;g_{k:K})$. Here $V_\mathrm S^\pi$ is computed by running the
simulator from $x$ along the remaining goals (Monte Carlo), and only $D$ is learned backward, from the paired
transitions:

$$
D(x_i^\mathrm p;g_{k:K}) \leftarrow \underbrace{(y_i^\mathrm R-y_i^\mathrm S)}_{\text{current-goal cost difference}}
+\gamma\underbrace{\big[V_\mathrm S^\pi(x_{i,\mathrm{end}}^\mathrm R;g_{k+1:K})-V_\mathrm S^\pi(x_{i,\mathrm{end}}^\mathrm S;g_{k+1:K})\big]}_{\text{terminal-observation difference, simulated value}}
+\gamma\thinspace \widehat D(x_{i,\mathrm{end}}^\mathrm R;g_{k+1:K}).
$$

Simulation-only transitions enter through a calibrator $h_D(x,g_{k:K},z^\mathrm S)$ with
$z^\mathrm S=y^\mathrm S+\gamma V_\mathrm S^\pi(x_\mathrm{end}^\mathrm S;g_{k+1:K})$, as in Section 2.3. The estimate is
the simulated cost of the evaluation query plus the learned residual, $\widehat J_\mathrm S^\pi(x_1,g_{1:K})+\widehat D(x_1;g_{1:K})$.
Unlike the method above, this variant needs multi-segment simulation rollouts from arbitrary observations during
training, which is outside the data setting of Section 1.3.

## 5. Instantiation in the toy study

| Element | Toy study (PointMaze Sim2Sim) |
|---|---|
| Local goals $\mathcal G$ | 9 junction goals A–I; a valid goal sequence is a non-backtracking walk on the goal graph |
| Policy $\pi$ | PD controller toward the end of the current straight corridor of the shortest path; it sees only the current goal |
| Segment | ends when the ball enters the radius-0.2 ball around the goal (or after 5 s); the terminal observation keeps its velocity |
| Costs $Y_k$, $\gamma$ | Safety, Exposure, TTC, Path, Time (each evaluated separately); $\gamma=1$ |
| $\mathcal D_\mathrm p$ | real rollouts along random goal sequences; at every goal boundary the simulator is set to the recorded real observation and executes the same goal |
| $\mathcal D_\mathrm s$ | 20,000 one-segment simulated transitions plus 4,000 started near the real observations of $\mathcal D_\mathrm p$ |
| Remaining goal sequences | 6 sampled per paired transition and length, 2 per simulation-only transition; longest length 8 |
| Training rows of $V_\theta$ | real targets $Z_\mathrm R$ make up 75% of the rows, calibrated targets $Z_\mathrm{cal}$ 25% |
| Simulation-backed estimation | 30 simulated rollouts of every evaluation query |
| Models | MLP 3×256 for $V_\theta$, $h_\phi$ and all learned baselines, same gradient-step budget for every method |

**Compared methods.** Baselines: raw simulation (the simulated cost of the evaluation query); segment-wise outcome
correction (SCAPE-style: corrects the cost of each simulated segment but not the effect of the terminal observation on
later goals); target-only (backward Bellman learning of $V_\theta$ on the real targets of $\mathcal D_\mathrm p$ only, no
simulation); B1 MC-real (regression of the observed real costs of every window of the same real rollouts, no Bellman
decomposition); B2 route-level SCAPE ($h(x,g,J_\mathrm S)\to J_\mathrm R$ on the same windows). Ours: direct estimation
(the main estimator) and simulation-backed estimation. Also tried: the residual variant (Section 4).

**Metrics.** Error against $J_\mathrm R$ from 100 real rollouts per evaluation query: MAE, bias, RMSE and Spearman rank
correlation over queries, reported per number of goals $K$ and number of paired transitions $n$.
