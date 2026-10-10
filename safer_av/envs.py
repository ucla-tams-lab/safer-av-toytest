"""Sim (S) and real (R) environments: the same maze, different actuation.

Policy command a -> simulator input u:
    1. brake (R only): the component of a along the current velocity is scaled by `brake` when it
       opposes the motion (1.0 = full braking). Weaker brakes act like a smaller D gain.
    2. u = beta * a + eps           (beta: force transmission, eps: Gaussian actuation noise)
    3. |u_x|, |u_y| <= 1            (PointMaze force limit)
S uses beta = brake = 1. Everything is a function of the current (p, v) and the command, so the state stays Markov.
"""
import numpy as np

from .maze import MazeSim

BASE_DAMPING = 1.0  # PointMaze joint damping


class ActuationEnv:
    """MazeSim + action transform. Same interface as MazeSim (set_state/get_state/step)."""

    def __init__(self, beta=1.0, brake=1.0, noise_std=0.0, seed=0):
        self.sim = MazeSim()
        self.beta, self.brake, self.noise_std = float(beta), float(brake), float(noise_std)
        self.sim.model.dof_damping[:2] = BASE_DAMPING
        self.rng = np.random.default_rng(seed)
        self.model, self.data, self.dt = self.sim.model, self.sim.data, self.sim.dt

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def transform(self, a, v):
        a = np.asarray(a, dtype=np.float64)
        if self.brake != 1.0:
            speed = np.linalg.norm(v)
            if speed > 1e-6:
                e = v / speed
                a_par = (a @ e) * e
                a_perp = a - a_par
                if a @ e < 0:  # braking component
                    a_par = self.brake * a_par
                a = a_par + a_perp
        u = self.beta * a
        if self.noise_std > 0:
            u = u + self.rng.normal(0.0, self.noise_std, size=2)
        return np.clip(u, -1.0, 1.0)

    def set_state(self, x):
        self.sim.set_state(x)

    def get_state(self):
        return self.sim.get_state()

    def step(self, a):
        v = self.sim.data.qvel[:2].copy()
        return self.sim.step(self.transform(a, v))

    def close(self):
        self.sim.close()


def make_env(kind, beta=1.0, brake=1.0, noise_std=0.0, seed=0):
    """kind 'S': nominal dynamics (only noise applies). kind 'R': target dynamics."""
    if kind == "S":
        return ActuationEnv(noise_std=noise_std, seed=seed)
    if kind == "R":
        return ActuationEnv(beta=beta, brake=brake, noise_std=noise_std, seed=seed)
    raise ValueError(kind)
