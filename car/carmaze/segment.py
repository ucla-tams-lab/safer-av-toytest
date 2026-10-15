"""Segment runner and per-segment costs: the toy study's definitions (the toy's safer_av/segment.py) with distances in
cells (1 cell = 3 m), so the constants are the toy's: arrival radius 0.2 cell (0.6 m), d_safe = d_expo = 0.4 cell (about
the clearance at the corridor centre), time-to-collision reference 0.5 s, timeout 5 s, +5 on Path and Time when the
goal is not reached. Clearance and time-to-collision use the car centre with its body radius (0.2 m).
    safety : max(0, d_safe - min wall clearance) / d_safe                in [0, 1]
    expo   : integral of max(0, d_expo - clearance) / d_expo dt           [s]
    ttc    : max(0, ttc_ref - min time-to-collision) / ttc_ref            in [0, 1]
    path   : path length in cells   (+ lam_fail if the goal is not reached)
    time   : tau / t_max            (+ lam_fail if the goal is not reached)"""
from dataclasses import dataclass

import numpy as np

from . import maze as M
from .car import RADIUS

COSTS = ("safety", "path", "time", "ttc", "expo")


@dataclass(frozen=True)
class SegmentConfig:
    eps: float = 0.2
    t_max: float = 5.0
    d_safe: float = 0.4
    ttc_ref: float = 0.5
    d_expo: float = 0.4
    lam_fail: float = 5.0


_C, _H = (a / M.CELL for a in M.wall_boxes())
_R = RADIUS / M.CELL


def clearance(p):
    """Distance from the car body to the nearest wall box [cells]; p in metres."""
    d = np.abs(p[None, :] / M.CELL - _C) - _H
    sdf = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
    return float(sdf.min() - _R)


def velocity(x):
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([x[3] * c - x[4] * s, x[3] * s + x[4] * c])


def time_to_collision(p, v):
    p, v = p / M.CELL, v / M.CELL
    if np.linalg.norm(v) < 1e-9:
        return np.inf
    C, H = _C, _H + _R
    lo, hi = np.empty_like(C), np.empty_like(C)
    for a in range(2):
        h = H[:, a]
        if abs(v[a]) > 1e-12:
            t1, t2 = (C[:, a] - h - p[a]) / v[a], (C[:, a] + h - p[a]) / v[a]
            lo[:, a], hi[:, a] = np.minimum(t1, t2), np.maximum(t1, t2)
        else:
            inside = (p[a] > C[:, a] - h) & (p[a] < C[:, a] + h)
            lo[:, a], hi[:, a] = np.where(inside, -np.inf, np.inf), np.where(inside, np.inf, -np.inf)
    t_in, t_out = lo.max(axis=1), hi.min(axis=1)
    hit = (t_in <= t_out) & (t_out > 0)
    return float(np.maximum(t_in[hit], 0.0).min()) if hit.any() else np.inf


def crossing_fraction(p0, p1, g, eps):
    d, f = p1 - p0, p0 - g
    a, b, c = d @ d, 2 * f @ d, f @ f - eps ** 2
    if a == 0:
        return 1.0
    s = (-b - np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    return float(np.clip(s, 1e-12, 1.0))


def run_segment(world, policy, x, g, cfg=SegmentConfig(), keep_traj=False):
    """Returns (x_plus, info); info holds y_<cost> for every cost in COSTS."""
    g = np.asarray(g, dtype=np.float64)
    eps = cfg.eps * M.CELL
    n_max = int(round(cfg.t_max / world.dt))
    world.set_state(x)
    x = np.asarray(x, dtype=np.float64).copy()
    traj, contact_steps, arrived, t, frac = [x[:2].copy()], 0, False, 0, 1.0
    min_clear, path, min_ttc, expo_steps = clearance(x[:2]), 0.0, np.inf, 0.0
    if np.linalg.norm(x[:2] - g) <= eps:
        arrived, frac = True, 0.0
    else:
        for t in range(1, n_max + 1):
            p_prev = x[:2].copy()
            x = world.step(policy(x, g))
            path += float(np.linalg.norm(x[:2] - p_prev)) / M.CELL
            contact_steps += world.in_contact
            c_now = clearance(x[:2])
            min_clear = min(min_clear, c_now)
            min_ttc = min(min_ttc, time_to_collision(x[:2], velocity(x)))
            expo_steps += max(0.0, cfg.d_expo - c_now) / cfg.d_expo
            if keep_traj:
                traj.append(x[:2].copy())
            if np.linalg.norm(x[:2] - g) <= eps:
                arrived, frac = True, crossing_fraction(p_prev, x[:2], g, eps)
                break
    tau = (t - 1 + frac) * world.dt if arrived and t > 0 else t * world.dt
    fail = not arrived
    info = dict(tau=tau, contact=contact_steps * world.dt, fail=fail, fell=False, steps=t, min_clear=min_clear, min_ttc=min_ttc,
                y_safety=max(0.0, cfg.d_safe - min_clear) / cfg.d_safe,
                y_expo=expo_steps * world.dt,
                y_ttc=max(0.0, cfg.ttc_ref - min_ttc) / cfg.ttc_ref,
                y_path=path + cfg.lam_fail * float(fail),
                y_time=tau / cfg.t_max + cfg.lam_fail * float(fail))
    if keep_traj:
        info["traj"] = np.array(traj)
    return x, info


def run_route(world, policy, x1, goals, cfg=SegmentConfig(), keep_traj=False):
    x = np.asarray(x1, dtype=np.float64).copy()
    states, infos = [x.copy()], []
    for name in goals:
        x, info = run_segment(world, policy, x, M.GOALS[name], cfg, keep_traj)
        states.append(x.copy()); infos.append(info)
    return states, infos
