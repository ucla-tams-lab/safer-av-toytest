"""Custom PointMaze map, named goals, goal graph, and exact state control.

We build the official PointMaze env only to get the compiled MuJoCo model
(walls + ball) and then drive its inner `point_env` directly. This bypasses
PointMaze's own goal sampling / reset noise / goal-update logic, which we do
not use: segments, goals and costs are defined by us.
"""
from collections import deque

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

# 9 (wide) x 7 (tall) cells, cell size 1.0. A 3x3 grid of junctions
# (rows 1,3,5 x cols 1,4,7) joined by 1-cell-wide corridors with right angles.
# Arriving at a junction, the next goal may be straight ahead or a 90° turn,
# which is what makes arrival velocity matter for the next segment.
MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]
CELL = 1.0
N_ROWS, N_COLS = len(MAZE_MAP), len(MAZE_MAP[0])
X_CENTER, Y_CENTER = N_COLS / 2 * CELL, N_ROWS / 2 * CELL

# Named goals at junction cells (row, col):
#   A B C      (top row)
#   D E F
#   G H I      (bottom row)
GOAL_CELLS = {
    "A": (1, 1), "B": (1, 4), "C": (1, 7),
    "D": (3, 1), "E": (3, 4), "F": (3, 7),
    "G": (5, 1), "H": (5, 4), "I": (5, 7),
}
# Valid consecutive goals: junctions joined by a straight corridor.
GOAL_EDGES = [
    ("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"), ("G", "H"), ("H", "I"),
    ("A", "D"), ("D", "G"), ("B", "E"), ("E", "H"), ("C", "F"), ("F", "I"),
]
BALL_RADIUS = 0.1
V_MAX = 5.0  # per-axis velocity clip inside PointEnv.step


def cell_to_xy(rc):
    r, c = rc
    return np.array([(c + 0.5) * CELL - X_CENTER, Y_CENTER - (r + 0.5) * CELL])


def xy_to_cell(xy):
    r = int(np.floor((Y_CENTER - xy[1]) / CELL))
    c = int(np.floor((xy[0] + X_CENTER) / CELL))
    return r, c


def is_free(rc):
    r, c = rc
    return 0 <= r < N_ROWS and 0 <= c < N_COLS and MAZE_MAP[r][c] == 0


GOALS = {k: cell_to_xy(rc) for k, rc in GOAL_CELLS.items()}
NEIGHBORS = {k: sorted({b for a, b in GOAL_EDGES if a == k} | {a for a, b in GOAL_EDGES if b == k})
             for k in GOALS}


def free_cells():
    return [(r, c) for r in range(N_ROWS) for c in range(N_COLS) if MAZE_MAP[r][c] == 0]


def bfs_next_cell(start_rc, goal_rc):
    """Next cell on a 4-connected shortest path from start to goal (start itself if equal)."""
    if start_rc == goal_rc:
        return goal_rc
    prev = {start_rc: None}
    q = deque([start_rc])
    while q:
        cur = q.popleft()
        if cur == goal_rc:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if is_free(nxt) and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if goal_rc not in prev:
        raise ValueError(f"goal {goal_rc} unreachable from {start_rc}")
    node = goal_rc
    while prev[node] != start_rc:
        node = prev[node]
    return node


class MazeSim:
    """Thin handle around PointMaze's MuJoCo model with exact state get/set."""

    def __init__(self):
        self._gym_env = gym.make("PointMaze_UMaze-v3", maze_map=MAZE_MAP,
                                 continuing_task=True, reset_target=False)
        self._gym_env.reset(seed=0)
        self.point_env = self._gym_env.unwrapped.point_env
        self.model = self.point_env.model
        self.data = self.point_env.data
        self.dt = float(self.point_env.dt)
        m = self._gym_env.unwrapped.maze
        assert (m.map_length, m.map_width) == (N_ROWS, N_COLS)
        assert np.isclose(m.x_map_center, X_CENTER) and np.isclose(m.y_map_center, Y_CENTER)

    def set_state(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.point_env.set_state(x[:2].copy(), x[2:].copy())  # calls mj_forward

    def get_state(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).copy()

    def step(self, u):
        """One control step (dt) with simulator input u in [-1,1]^2."""
        obs, *_ = self.point_env.step(np.asarray(u, dtype=np.float64))
        return obs.copy()

    def close(self):
        self._gym_env.close()
