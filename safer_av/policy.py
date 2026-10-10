"""Fixed goal-conditioned policy: grid shortest path -> look-ahead waypoint -> PD.

pi(x, g) depends only on the current state x=(px,py,vx,vy) and the CURRENT goal
g=(gx,gy). It never sees future goals, so a segment collected for goal g can be
reused with any suffix.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import maze as M


@lru_cache(maxsize=None)
def _bfs_path(start_rc, goal_rc):
    path = [start_rc]
    while path[-1] != goal_rc:
        path.append(M.bfs_next_cell(path[-1], goal_rc))
    return tuple(path)


def waypoint(p, g):
    """Farthest cell center reachable in a straight line along the shortest path.

    Aiming at the end of the current straight corridor (instead of the adjacent
    cell) gives smooth motion and makes the PD brake before each turn. When the
    goal cell is on that straight line, the waypoint is the goal itself.
    """
    start_rc, goal_rc = M.xy_to_cell(p), M.xy_to_cell(g)
    if not M.is_free(start_rc):  # numerically inside a wall (soft contact): use nearest free cell
        start_rc = min(M.free_cells(), key=lambda rc: np.linalg.norm(M.cell_to_xy(rc) - p))
    path = _bfs_path(start_rc, goal_rc)
    if len(path) == 1:
        return np.asarray(g, dtype=np.float64)
    d = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    i = 1
    while i + 1 < len(path) and (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) == d:
        i += 1
    if path[i] == goal_rc:
        return np.asarray(g, dtype=np.float64)
    return M.cell_to_xy(path[i])


@dataclass(frozen=True)
class PDPolicy:
    kp: float
    kd: float

    def __call__(self, x, g):
        x = np.asarray(x, dtype=np.float64)
        p, v = x[:2], x[2:]
        w = waypoint(p, g)
        return np.clip(self.kp * (w - p) - self.kd * v, -1.0, 1.0)
