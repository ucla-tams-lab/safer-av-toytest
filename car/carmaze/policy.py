"""The fixed goal-conditioned policy: pure pursuit along the shortest path to the current goal + a speed plan made for
the nominal (sim) car on flat ground.

pi(x, g) depends only on the current state x and the current goal g (it never sees later goals):
  path       corridor centre lines of the shortest cell path to the goal, every corner rounded by an arc of radius
             R_ARC (so a car that tracks it stays in the middle of the corridor; it starts at the car)
  steering   pure pursuit to the point L_d = 0.6 + 0.2 v [m] ahead along the path (the goal if closer)
  speed      v_max on straights; plans to be at v_turn where each corner arc starts and at v_goal at the goal (it does not
             stop at goals), braking at a_plan; while turning it also slows down so that speed^2 x pure-pursuit
             curvature <= a_lat; a proportional controller (gain k_v) tracks the planned speed
The turn towards the next goal starts only after the goal is reached (at about 4.3 m/s); the controller then asks for
hard braking, which the sim car delivers (7 m/s^2) and the real car does not (3.5 m/s^2, less grip), so the real car
swings wide."""
from dataclasses import dataclass

import numpy as np

from . import maze as M
from .car import P

R_ARC = 1.0


def path_points(p, g):
    """Dense polyline from p along the shortest cell path to g, corners rounded by arcs. Also returns the arc-length
    positions (from p) where each arc starts (the corners the speed plan slows down for)."""
    start, goal = M.xy_to_cell(p), M.xy_to_cell(g)
    if not M.is_free(start):
        start = min(M.free_cells(), key=lambda rc: np.linalg.norm(M.cell_to_xy(rc) - p))
    cells = M._bfs_path(start, goal)
    pts, corners = [np.asarray(p, dtype=np.float64)], []
    for i in range(1, len(cells) - 1):
        d0 = np.array([cells[i][1] - cells[i - 1][1], cells[i - 1][0] - cells[i][0]], dtype=float)   # (dx, dy) in xy
        d1 = np.array([cells[i + 1][1] - cells[i][1], cells[i][0] - cells[i + 1][0]], dtype=float)
        if np.array_equal(d0, d1):
            continue
        c = M.cell_to_xy(cells[i])
        a0, a1 = c - R_ARC * d0, c + R_ARC * d1
        corners.append(a0)
        centre = a0 + R_ARC * d1
        th0, th1 = np.arctan2(*(a0 - centre)[::-1]), np.arctan2(*(a1 - centre)[::-1])
        dth = (th1 - th0 + np.pi) % (2 * np.pi) - np.pi
        for t in np.linspace(0, 1, 7):
            pts.append(centre + R_ARC * np.array([np.cos(th0 + t * dth), np.sin(th0 + t * dth)]))
    pts.append(np.asarray(g, dtype=np.float64))
    return pts, corners


@dataclass(frozen=True)
class CarPolicy:
    v_max: float = 5.0
    v_turn: float = 2.5
    v_goal: float = 3.0
    a_plan: float = 4.0
    k_v: float = 6.0
    k_delta: float = 8.0
    a_lat: float = 6.0          # planned lateral acceleration for turning (0: no turn speed limit)
    v_min: float = 1.0          # the turn speed limit never asks for less than this

    def __call__(self, x, g):
        p, psi, v, delta = x[:2], x[2], x[3], x[6]
        g = np.asarray(g, dtype=np.float64)
        pts, corners = path_points(p, g)
        Ld = 0.6 + 0.2 * v
        target, left, run, v_cap, ci = pts[-1], Ld, 0.0, self.v_max, 0
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            seg = float(np.linalg.norm(b - a))
            if ci < len(corners) and np.allclose(a, corners[ci]):          # an arc starts here: slow down for it
                v_cap = min(v_cap, np.sqrt(self.v_turn ** 2 + 2 * self.a_plan * max(run - 0.2, 0.0)))
                ci += 1
            if left is not None and seg >= left:
                target, left = a + (b - a) * (left / max(seg, 1e-9)), None
            elif left is not None:
                left -= seg
            run += seg
        v_cap = min(v_cap, np.sqrt(self.v_goal ** 2 + 2 * self.a_plan * max(run - 0.2, 0.0)))
        alpha = np.arctan2(target[1] - p[1], target[0] - p[0]) - psi
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        L = P["lf"] + P["lr"]
        if self.a_lat > 0:                       # slow down while turning: v^2 x pure-pursuit curvature <= a_lat
            kappa = abs(2 * np.sin(alpha)) / max(np.linalg.norm(target - p), 0.3)
            v_cap = min(v_cap, max(np.sqrt(self.a_lat / max(kappa, 1e-6)), self.v_min))
        delta_cmd = np.clip(np.arctan2(2 * L * np.sin(alpha), max(np.linalg.norm(target - p), 0.3)), -P["s_max"], P["s_max"])
        steer_vel = np.clip(self.k_delta * (delta_cmd - delta), -P["sv_max"], P["sv_max"])
        accel = np.clip(self.k_v * (v_cap - v), -P["a_max"], P["a_max"])
        return np.array([steer_vel, accel])
