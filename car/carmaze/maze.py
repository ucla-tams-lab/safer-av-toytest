"""The toy study's 3x3-junction maze at 1/10-car scale (one cell = 3 m: corridors 3 m wide), shortest-path waypoints
and random goal sequences (same layout, goals, goal graph and waypoint rule as the toy's safer_av/maze.py).
Scene features make the costs depend on the goal sequence and on the order:
  wet      cells with a slippery surface in real only (sim assumes the normal surface everywhere)
  slope    corridor cells that run downhill one way (the same in sim and real): the car gains speed going down, so the
           same corridor costs differently in the two directions and the speed carried into the next turn depends on
           where the car came from"""
from collections import deque
from functools import lru_cache

import numpy as np


def _grid(rows, cols):
    """Junction rows and columns (cell indices) -> maze map (1 = wall), goal cells (row-major names A, B, ...), edges."""
    m = [[1] * (cols[-1] + 2) for _ in range(rows[-1] + 2)]
    for r in rows:
        for c in range(cols[0], cols[-1] + 1):
            m[r][c] = 0
    for c in cols:
        for r in range(rows[0], rows[-1] + 1):
            m[r][c] = 0
    nm = lambda i, j: chr(ord("A") + i * len(cols) + j)
    goals = {nm(i, j): (r, c) for i, r in enumerate(rows) for j, c in enumerate(cols)}
    edges = [(nm(i, j), nm(i, j + 1)) for i in range(len(rows)) for j in range(len(cols) - 1)]
    edges += [(nm(i, j), nm(i + 1, j)) for i in range(len(rows) - 1) for j in range(len(cols))]
    return m, goals, edges


MAZE_MAP, GOAL_CELLS, GOAL_EDGES = _grid([1, 3, 5], [1, 4, 7])
# wet at the junctions E, C, G and on the corridor H-I
WET_CELLS = [(3, 4), (1, 7), (5, 1), (5, 5), (5, 6)]
# outer ring downhill clockwise (A-B-C-F-I-H-G-D-A), inner cross downhill east (D-E-F) and south (B-E-H)
SLOPE_CELLS = {(1, 2): "E", (1, 3): "E", (1, 5): "E", (1, 6): "E", (2, 7): "S", (4, 7): "S",
               (5, 5): "W", (5, 6): "W", (5, 2): "W", (5, 3): "W", (4, 1): "N", (2, 1): "N",
               (2, 4): "S", (4, 4): "S", (3, 2): "E", (3, 3): "E", (3, 5): "E", (3, 6): "E"}
SLOPE_DEG = 6.0
SLOPE_DIR = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}
CELL = 3.0
N_ROWS, N_COLS = len(MAZE_MAP), len(MAZE_MAP[0])
X_CENTER, Y_CENTER = N_COLS / 2 * CELL, N_ROWS / 2 * CELL


def cell_to_xy(rc):
    r, c = rc
    return np.array([(c + 0.5) * CELL - X_CENTER, Y_CENTER - (r + 0.5) * CELL])


def xy_to_cell(xy):
    return int(np.floor((Y_CENTER - xy[1]) / CELL)), int(np.floor((xy[0] + X_CENTER) / CELL))


def is_free(rc):
    r, c = rc
    return 0 <= r < N_ROWS and 0 <= c < N_COLS and MAZE_MAP[r][c] == 0


def free_cells():
    return [(r, c) for r in range(N_ROWS) for c in range(N_COLS) if MAZE_MAP[r][c] == 0]


def wall_cells():
    return [(r, c) for r in range(N_ROWS) for c in range(N_COLS) if MAZE_MAP[r][c] == 1]


GOALS = {k: cell_to_xy(rc) for k, rc in GOAL_CELLS.items()}
NEIGHBORS = {k: sorted({b for a, b in GOAL_EDGES if a == k} | {a for a, b in GOAL_EDGES if b == k}) for k in GOALS}


def bfs_next_cell(start_rc, goal_rc):
    """Next cell on a 4-connected shortest path from start to goal (start itself if equal)."""
    if start_rc == goal_rc:
        return goal_rc
    prev, q = {start_rc: None}, deque([start_rc])
    while q:
        cur = q.popleft()
        if cur == goal_rc:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if is_free(nxt) and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    node = goal_rc
    while prev[node] != start_rc:
        node = prev[node]
    return node


@lru_cache(maxsize=None)
def _bfs_path(start_rc, goal_rc):
    path = [start_rc]
    while path[-1] != goal_rc:
        path.append(bfs_next_cell(path[-1], goal_rc))
    return tuple(path)


def waypoint(p, g):
    """End of the current straight corridor along the shortest path (the goal itself when it is on that line)."""
    start_rc, goal_rc = xy_to_cell(p), xy_to_cell(g)
    if not is_free(start_rc):                # torso centre over a wall cell (leaning on a wall): nearest free cell
        start_rc = min(free_cells(), key=lambda rc: np.linalg.norm(cell_to_xy(rc) - p))
    path = _bfs_path(start_rc, goal_rc)
    if len(path) == 1:
        return np.asarray(g, dtype=np.float64)
    d = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    i = 1
    while i + 1 < len(path) and (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) == d:
        i += 1
    return np.asarray(g, dtype=np.float64) if path[i] == goal_rc else cell_to_xy(path[i])


def random_route(rng, K, start=None):
    """Non-backtracking random walk of K goals on the goal graph. Returns (start_name, [g_1, ..., g_K])."""
    start = start if start is not None else list(GOALS)[rng.integers(len(GOALS))]
    cur, prev, seq = start, None, []
    for _ in range(K):
        nbrs = [n for n in NEIGHBORS[cur] if n != prev]
        nxt = nbrs[rng.integers(len(nbrs))]
        seq.append(nxt)
        prev, cur = cur, nxt
    return start, seq


def wall_boxes():
    """Every wall box as arrays of centres and half sizes [m]."""
    boxes = [(tuple(cell_to_xy(rc)), (CELL / 2, CELL / 2)) for rc in wall_cells()]
    return np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])
