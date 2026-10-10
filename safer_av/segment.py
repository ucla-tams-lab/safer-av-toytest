"""Segment runner and per-segment costs.

A segment runs pi(., g) from state x until |p - g| <= eps (arrival) or t_max (timeout).
The terminal state keeps its velocity (no stopping at the goal).

Costs of one segment (all are recorded):
    safety : max(0, d_safe - min wall clearance) / d_safe                   in [0, 1]
    expo   : integral of max(0, d_expo - clearance) / d_expo dt              [s]
    ttc    : max(0, ttc_ref - min time-to-collision) / ttc_ref               in [0, 1]
    path   : path length travelled  (+ lam_fail if timed out)
    time   : tau / t_max            (+ lam_fail if timed out)

Wall contact time (MuJoCo ball-wall contacts) is recorded for diagnostics. Ball and blocks each have
margin 0.002, so the contact constraint switches on at distance < 0.004; thresholding on dist <= 0 misses it.
"""
from dataclasses import dataclass

import numpy as np

COSTS = ("safety", "path", "time", "ttc", "expo")


@dataclass(frozen=True)
class SegmentConfig:
    eps: float = 0.2        # arrival radius
    t_max: float = 5.0      # timeout [s]
    d_safe: float = 0.4     # safety cost: clearance below which a segment is unsafe (0.4 = corridor centre)
    ttc_ref: float = 0.5    # ttc cost: TTC below which a segment is unsafe [s]
    d_expo: float = 0.4     # expo cost: clearance margin whose intrusion is integrated over time
    lam_fail: float = 5.0   # timeout penalty added to path and time


class Clearance:
    """Distance from the ball surface to the nearest wall box (vectorized over all blocks)."""

    def __init__(self):
        from . import maze as M
        self.centers = np.array([M.cell_to_xy((r, c)) for r in range(M.N_ROWS) for c in range(M.N_COLS)
                                 if M.MAZE_MAP[r][c] == 1])
        self.half, self.radius = 0.5 * M.CELL, M.BALL_RADIUS

    def __call__(self, p):
        d = np.abs(p[None, :] - self.centers) - self.half
        sdf = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
        return float(sdf.min() - self.radius)


_CLEARANCE = None


def time_to_collision(p, v):
    """Seconds until the ball (radius r) hits a wall box if it keeps velocity v (slab test vs boxes inflated by r)."""
    global _CLEARANCE
    if _CLEARANCE is None:
        _CLEARANCE = Clearance()
    C, h = _CLEARANCE.centers, _CLEARANCE.half + _CLEARANCE.radius
    if np.linalg.norm(v) < 1e-9:
        return np.inf
    lo = np.empty_like(C); hi = np.empty_like(C)
    for a in range(2):
        if abs(v[a]) > 1e-12:
            t1 = (C[:, a] - h - p[a]) / v[a]; t2 = (C[:, a] + h - p[a]) / v[a]
            lo[:, a] = np.minimum(t1, t2); hi[:, a] = np.maximum(t1, t2)
        else:  # ray parallel to this axis: inside the slab for all t, or never
            inside = (p[a] > C[:, a] - h) & (p[a] < C[:, a] + h)
            lo[:, a] = np.where(inside, -np.inf, np.inf); hi[:, a] = np.where(inside, np.inf, -np.inf)
    t_in, t_out = lo.max(axis=1), hi.min(axis=1)
    hit = (t_in <= t_out) & (t_out > 0)
    return float(np.maximum(t_in[hit], 0.0).min()) if hit.any() else np.inf


def clearance(p):
    global _CLEARANCE
    if _CLEARANCE is None:
        _CLEARANCE = Clearance()
    return _CLEARANCE(p)


class ContactMeter:
    """Whether the ball is in contact (penetration) with any wall block this step."""

    def __init__(self, model):
        self.ball = model.geom("particle_geom").id
        self.walls = {i for i in range(model.ngeom) if model.geom(i).name.startswith("block_")}

    def in_contact(self, data):
        for i in range(data.ncon):
            c = data.contact[i]
            other = c.geom2 if c.geom1 == self.ball else c.geom1 if c.geom2 == self.ball else None
            if other is not None and other in self.walls:
                return True
        return False


def crossing_fraction(p0, p1, g, eps):
    """Smallest s in (0, 1] with |p0 + s (p1 - p0) - g| = eps, given p0 outside and p1 inside."""
    d, f = p1 - p0, p0 - g
    a, b, c = d @ d, 2 * f @ d, f @ f - eps ** 2
    if a == 0:
        return 1.0
    s = (-b - np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    return float(np.clip(s, 1e-12, 1.0))


def run_segment(env, policy, x, g, cfg=SegmentConfig(), keep_traj=False):
    """Returns (x_plus, info); info holds y_<cost> for every cost in COSTS. `env` exposes set_state/step/model/data/dt."""
    meter = ContactMeter(env.model)
    g = np.asarray(g, dtype=np.float64)
    n_max = int(round(cfg.t_max / env.dt))
    env.set_state(x)
    x = np.asarray(x, dtype=np.float64).copy()
    traj, contact_steps, arrived, t, frac = [x.copy()], 0, False, 0, 1.0
    min_clear, path = clearance(x[:2]), 0.0
    min_ttc, expo_steps = np.inf, 0
    if np.linalg.norm(x[:2] - g) <= cfg.eps:  # already at the goal: empty segment
        arrived, frac = True, 0.0
    else:
        for t in range(1, n_max + 1):
            p_prev = x[:2].copy()
            x = env.step(policy(x, g))
            path += float(np.linalg.norm(x[:2] - p_prev))
            contact_steps += meter.in_contact(env.data)
            c_now = clearance(x[:2])
            min_clear = min(min_clear, c_now)
            min_ttc = min(min_ttc, time_to_collision(x[:2], x[2:]))
            expo_steps += max(0.0, cfg.d_expo - c_now) / cfg.d_expo
            if keep_traj:
                traj.append(x.copy())
            if np.linalg.norm(x[:2] - g) <= cfg.eps:
                arrived, frac = True, crossing_fraction(p_prev, x[:2], g, cfg.eps)
                break
    # tau is continuous: the eps-crossing time is interpolated within the last step
    # (linear motion over one dt). x_plus is the actual simulator state at the end of that step.
    tau = (t - 1 + frac) * env.dt if arrived and t > 0 else t * env.dt
    fail = not arrived
    info = dict(tau=tau, contact=contact_steps * env.dt, fail=fail, steps=t, min_clear=min_clear, min_ttc=min_ttc,
                y_safety=max(0.0, cfg.d_safe - min_clear) / cfg.d_safe,
                y_expo=expo_steps * env.dt,
                y_ttc=max(0.0, cfg.ttc_ref - min_ttc) / cfg.ttc_ref,
                y_path=path + cfg.lam_fail * float(fail),
                y_time=tau / cfg.t_max + cfg.lam_fail * float(fail))
    if keep_traj:
        info["traj"] = np.array(traj)
    return x, info
