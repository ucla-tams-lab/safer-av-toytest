"""A 1/10-scale car (F1TENTH mass, size and tyre stiffness) in the maze: dynamic bicycle model with saturating tyres,
exact state control.

State x = (p_x, p_y, yaw psi, longitudinal speed u, lateral speed w, yaw rate r, steering angle delta) -- 7 numbers; the
controller has no memory, so x is the full state. Inputs: steering velocity [rad/s] and longitudinal acceleration
command [m/s^2]; RK4 at 100 Hz, 5 steps per control step (20 Hz).
Tyres: lateral force = cornering stiffness x slip angle, clipped at mu x axle load (the car skids beyond it). Below
0.5 m/s the kinematic bicycle model is used.
Walls: when the body (radius 0.2 m) touches a wall it is pushed back out, the velocity component into the wall is removed,
the car is turned along the wall and keeps 70% of the remaining speed; the contact time is recorded.
Slopes (maze.SLOPE_CELLS, same in sim and real) add g sin(slope) along the downhill direction to the car's acceleration.

Sim (S) and real (R), as the toy study's "weaker force and brakes", for a car:
    accel     largest acceleration the motor delivers [m/s^2] (S 9.51, R 4: weaker motor)
    decel     largest deceleration the brakes deliver [m/s^2] (S 7, R 3.5: weaker brakes; the controller plans with 4)
    mu        tyre friction on the dry surface (S 1.0, R 0.8: less grip)
    mu_wet    tyre friction on the wet cells (maze.WET_CELLS; sim does not know them: S 1.0, R 0.5)
Both get the same Gaussian actuation noise (noise_std on steering velocity [rad/s] and 4 x noise_std on acceleration)."""
import numpy as np

from . import maze as M

P = dict(m=3.74, I=0.04712, lf=0.15875, lr=0.17145, C_Sf=4.718, C_Sr=5.4562, s_max=0.4189, sv_max=3.2, a_max=9.51,
         v_max=10.0)
G = 9.81
RADIUS = 0.2                     # body radius for walls and costs [m]
FZF = P["m"] * G * P["lr"] / (P["lf"] + P["lr"])
FZR = P["m"] * G * P["lf"] / (P["lf"] + P["lr"])
CF, CR = P["C_Sf"] * FZF, P["C_Sr"] * FZR          # cornering stiffness [N/rad]
PHYSICS = {"S": dict(accel=9.51, decel=7.0, mu=1.0, mu_wet=1.0), "R": dict(accel=4.0, decel=3.5, mu=0.8, mu_wet=0.5)}
_WALL_C, _WALL_H = M.wall_boxes()


def wall_distance(p):
    """Signed distance from point p to the nearest wall box and the outward unit normal there."""
    d = np.abs(p[None, :] - _WALL_C) - _WALL_H
    sdf = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
    i = int(np.argmin(sdf))
    q = p - _WALL_C[i]
    n = q - np.clip(q, -_WALL_H[i], _WALL_H[i])
    if np.linalg.norm(n) < 1e-9:
        k = int(np.argmax(np.abs(q) - _WALL_H[i]))
        n = np.zeros(2); n[k] = np.sign(q[k]) or 1.0
    return float(sdf[i]), n / np.linalg.norm(n)


def _deriv(x, u, mu, slope):
    psi, vu, vw, r, delta = x[2], x[3], x[4], x[5], x[6]
    sv, a = u
    c, s = np.cos(psi), np.sin(psi)
    ax_slope, ay_slope = slope @ np.array([c, s]), slope @ np.array([-s, c])      # downhill push in the body frame
    lf, lr, m, I = P["lf"], P["lr"], P["m"], P["I"]
    if vu < 0.5:                                    # kinematic bicycle at low speed
        lwb = lf + lr
        rk = vu / lwb * np.tan(delta)
        return np.array([vu * c, vu * s, rk, a + ax_slope, 0.0, 0.0, sv])
    af = delta - np.arctan2(vw + lf * r, vu)
    ar = -np.arctan2(vw - lr * r, vu)
    Ff = np.clip(CF * af, -mu * FZF, mu * FZF)
    Fr = np.clip(CR * ar, -mu * FZR, mu * FZR)
    du = a + ax_slope + vw * r - Ff * np.sin(delta) / m
    dw = (Ff * np.cos(delta) + Fr) / m - vu * r + ay_slope
    dr = (lf * Ff * np.cos(delta) - lr * Fr) / I
    return np.array([vu * c - vw * s, vu * s + vw * c, r, du, dw, dr, sv])


class CarWorld:
    def __init__(self, kind="S", noise_std=0.0, seed=0, **physics):
        p = dict(PHYSICS[kind], **physics)
        self.accel, self.decel, self.mu, self.mu_wet = p["accel"], p["decel"], p["mu"], p["mu_wet"]
        self.noise_std, self.rng = float(noise_std), np.random.default_rng(seed)
        self.dt, self.substeps = 0.05, 5
        self.x = np.zeros(7)
        self.in_contact = False
        self.slope = {rc: G * np.sin(np.radians(M.SLOPE_DEG)) * np.array(M.SLOPE_DIR[d]) for rc, d in M.SLOPE_CELLS.items()}
        self.wet = set(map(tuple, M.WET_CELLS))

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def rest_state(self, xy, yaw):
        return np.array([xy[0], xy[1], yaw, 0.0, 0.0, 0.0, 0.0])

    def set_state(self, x):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.in_contact = False

    def get_state(self):
        return self.x.copy()

    def step(self, u):
        """u = (steering velocity, acceleration) commands; returns the new state."""
        u = np.asarray(u, dtype=np.float64).copy()
        if self.noise_std > 0:
            u = u + self.rng.normal(0.0, self.noise_std, 2) * np.array([1.0, 4.0])
        x = self.x
        u[0] = np.clip(u[0], -P["sv_max"], P["sv_max"])
        if u[0] > 0 and x[6] >= P["s_max"] or u[0] < 0 and x[6] <= -P["s_max"]:
            u[0] = 0.0
        u[1] = np.clip(u[1], -self.decel, self.accel)
        rc = M.xy_to_cell(x[:2])
        mu = self.mu_wet if rc in self.wet else self.mu
        slope = self.slope.get(rc, np.zeros(2))
        h = self.dt / self.substeps
        self.in_contact = False
        for _ in range(self.substeps):
            k1 = _deriv(x, u, mu, slope); k2 = _deriv(x + h / 2 * k1, u, mu, slope)
            k3 = _deriv(x + h / 2 * k2, u, mu, slope); k4 = _deriv(x + h * k3, u, mu, slope)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x[3] = np.clip(x[3], 0.0, P["v_max"])  # no reversing
            x[6] = np.clip(x[6], -P["s_max"], P["s_max"])
            if x[3] < 0.5:
                x[4] = 0.0; x[5] = x[3] / (P["lf"] + P["lr"]) * np.tan(x[6])
            dist, n = wall_distance(x[:2])
            if dist < RADIUS:                      # scrape: pushed out, slides along the wall, loses speed
                x[:2] += (RADIUS - dist) * n
                c, s = np.cos(x[2]), np.sin(x[2])
                vw = np.array([x[3] * c - x[4] * s, x[3] * s + x[4] * c])
                vn = vw @ n
                if vn < 0:
                    vt = vw - vn * n
                    sp = float(np.linalg.norm(vt))
                    if sp > 1e-6:
                        x[2] = np.arctan2(vt[1], vt[0])
                    x[3], x[4], x[5] = 0.7 * sp, 0.0, 0.0
                self.in_contact = True
        self.x = x
        return x.copy()
