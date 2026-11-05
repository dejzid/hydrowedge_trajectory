"""Swimmer trajectories from the equations of motion, the equations of motion of the paper.

    rho'   = v0 sin(phi) cos(delta) + V_r,          z' = v0 cos(phi),
    beta'  = (v0 sin(phi) sin(delta) + V_theta) / rho,
    delta' = Omega_z - beta'.

Two orientations are supported (the mixed case is not needed):
  - axial dipole, phi = 0 (e = e_z). delta drops out: V_r and V_theta do not
    depend on it, and Omega_z only spins the swimmer about its own axis.
  - transverse dipole, phi = pi/2 (e = cos(delta) e_rho + sin(delta) e_beta),
    moving in the cross-section z = const.

The wall-induced velocities scale as V = sigma f(beta, delta) / rho^2 and
Omega_z = sigma f_omega(beta, delta) / rho^3, so the solver is evaluated once per
wedge, at rho = sigma = 1, on a grid in beta (Profile). The dipole flow is
quadratic in the orientation, hence for the transverse dipole
    f = f_cc(beta) cos^2(delta) + f_sc(beta) sin(delta) cos(delta) + f_ss(beta) sin^2(delta),
with f_cc, f_ss, f_sc obtained from delta = 0, pi/2, pi/4.

Near a wall, f ~ 1/sin^p(alpha - |beta|) with p = 2 (velocities) or 3 (Omega_z).
The profile interpolates the regular functions g = f sin^p(alpha - |beta|), whose
values at the wall are the single-wall results for a dipole inclined by t to the
wall at distance d (Berke et al. 2008, checked in tests/test_single_wall.py):
normal velocity -3/(8 d^2) (1 - 3 sin^2 t), tangential 3/(8 d^2) sin(2t),
Omega = 3/(8 d^3) sin(t) cos(t). At the upper wall t = -delta, so (WALL_LIMIT)
    g_r -> -3/4 sc,   g_theta -> 3/8 cc - 3/4 ss,   g_omega -> -3/8 sc.

Optional soft wall repulsion (not part of the equations of motion; off unless given). Each wall adds
a velocity along its normal into the fluid,
    u(d) = U ((d_r/d)^n - 1)  for d < d_r,  0 otherwise,
with d the distance to that wall. It is a pure force (no torque), so the heading
beta + delta is unchanged by it. It lets a swimmer come closer than one swimmer size
and shows whether it slides along a wall or is held there; d_min then only guards
against penetration and must be below d_r.
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from ._tabulate import POWER, g_at

DIPOLES = ("axial", "transverse")

# Symmetry under the mirror beta -> -beta, delta -> -delta: V_r is even, V_theta and
# Omega_z are odd; the basis function sin(delta) cos(delta) is odd in delta.
PARITY = {"r": 1, "theta": -1, "omega": -1, "1": 1, "cc": 1, "sc": -1, "ss": 1}

WALL_LIMIT = {
    ("r", "1"): 0.0, ("theta", "1"): 0.375,
    ("r", "cc"): 0.0, ("r", "sc"): -0.75, ("r", "ss"): 0.0,
    ("theta", "cc"): 0.375, ("theta", "sc"): 0.0, ("theta", "ss"): -0.75,
    ("omega", "cc"): 0.0, ("omega", "sc"): -0.375, ("omega", "ss"): 0.0,
}

# Closest approach to the wall (alpha - |beta|, in rad) at which the solver is
# evaluated; from there to the wall the spline runs to WALL_LIMIT. Transverse
# evaluations near the wall are ~15x more expensive, hence the larger gap.
GAP_MIN = {"axial": 0.05, "transverse": 0.1}
N_GRID = 20

# Worker processes for the tabulation (each needs ~100 MB).
WORKERS = min(8, os.cpu_count() or 1)


@dataclass
class Profile:
    """Tabulated g(beta) on [0, alpha] for one dipole and wedge; see module doc."""

    dipole: str
    alpha: float
    alpha_label: str
    gap_min: float
    beta: np.ndarray
    g: dict  # (quantity, basis) -> array over beta

    def __post_init__(self):
        # Splines on [0, alpha]; negative beta follows from PARITY. g has a kink at
        # beta = 0 because sin^p(alpha - |beta|) does, so even components are clamped
        # to their exact one-sided slope there: g'(0+) = -p g(0) cot(alpha).
        self._spline = {}
        for (q, b), vals in self.g.items():
            if PARITY[q] * PARITY[b] > 0:
                slope = -POWER[q] * vals[0] / np.tan(self.alpha)
                bc = ((1, slope), "not-a-knot")
            else:
                bc = "not-a-knot"
            self._spline[q, b] = CubicSpline(self.beta, vals, bc_type=bc)

    @classmethod
    def compute(cls, dipole, alpha, alpha_label=None, n=N_GRID, gap_min=None, workers=None):
        """Tabulate with the solver: n points uniform in beta on [0, alpha - 2 gap_min],
        one at alpha - gap_min, and WALL_LIMIT at beta = alpha.

        Pass alpha as an mpmath number at mp.dps >= 30 (e.g. mp.pi/3).
        """
        if dipole not in DIPOLES:
            raise ValueError(f"dipole must be one of {DIPOLES}")
        gap_min = GAP_MIN[dipole] if gap_min is None else gap_min
        with mp.workdps(30):
            alpha = +mp.mpf(alpha)
            betas = [(alpha - 2 * gap_min) * k / (n - 1) for k in range(n)] + [alpha - gap_min]
            jobs = [(dipole, mp.nstr(alpha, 40), mp.nstr(b, 40)) for b in betas]
        # Workers inherit the environment: one BLAS thread each keeps them small.
        saved = os.environ.get("OPENBLAS_NUM_THREADS")
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        try:
            with ProcessPoolExecutor(max_workers=workers or WORKERS) as pool:
                rows = list(pool.map(g_at, jobs))
        finally:
            if saved is None:
                del os.environ["OPENBLAS_NUM_THREADS"]
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = saved
        g = {
            key: np.array([row[key] for row in rows] + [WALL_LIMIT[key]])
            for key in rows[0]
        }
        beta = np.array([float(b) for b in betas] + [float(alpha)])
        return cls(dipole, float(alpha), alpha_label or mp.nstr(alpha, 17), gap_min, beta, g)

    def g_at(self, beta):
        """{(quantity, basis): g} from the splines, for 0 <= beta <= alpha."""
        return {key: s(beta) for key, s in self._spline.items()}

    def f(self, beta, delta=0.0):
        """(f_r, f_theta, f_omega): V_r = sigma f_r / rho^2, V_theta = sigma f_theta / rho^2,
        Omega_z = sigma f_omega / rho^3. f_omega is 0 for the axial dipole."""
        b = abs(beta)
        sign = 1.0 if beta >= 0 else -1.0
        c, s = np.cos(delta), np.sin(delta)
        weight = {"1": 1.0, "cc": c * c, "sc": s * c, "ss": s * s}
        gap = np.sin(self.alpha - b)
        out = {"r": 0.0, "theta": 0.0, "omega": 0.0}
        for (q, basis), spline in self._spline.items():
            parity = sign if PARITY[q] * PARITY[basis] < 0 else 1.0
            out[q] += parity * weight[basis] * float(spline(b)) / gap ** POWER[q]
        return out["r"], out["theta"], out["omega"]

    def save(self, path):
        keys = list(self.g)
        with open(path, "w", newline="", encoding="ascii") as fh:
            fh.write(
                f"# dipole={self.dipole} alpha={self.alpha_label} alpha_value={self.alpha!r} "
                f"gap_min={self.gap_min!r}\n"
            )
            w = csv.writer(fh)
            w.writerow(["beta"] + [f"g_{q}_{b}" for q, b in keys])
            for i, beta in enumerate(self.beta):
                w.writerow([repr(float(beta))] + [repr(float(self.g[k][i])) for k in keys])

    @classmethod
    def load(cls, path):
        with open(path, newline="", encoding="ascii") as fh:
            meta = dict(item.split("=", 1) for item in fh.readline()[1:].split())
            rows = list(csv.DictReader(fh))
        g = {}
        for col in rows[0]:
            if col.startswith("g_"):
                _, q, b = col.split("_")
                g[q, b] = np.array([float(r[col]) for r in rows])
        beta = np.array([float(r["beta"]) for r in rows])
        return cls(meta["dipole"], float(meta["alpha_value"]), meta["alpha"],
                   float(meta["gap_min"]), beta, g)


@dataclass
class Trajectory:
    dipole: str
    alpha: float
    t: np.ndarray
    rho: np.ndarray
    beta: np.ndarray
    delta: np.ndarray  # NaN for the axial dipole (orientation is e_z)
    z: np.ndarray
    reason: str  # "wall", "apex", "far" or "t_max"

    @property
    def x(self):
        return self.rho * np.cos(self.beta)

    @property
    def y(self):
        return self.rho * np.sin(self.beta)

    @property
    def wall_distance(self):
        return self.rho * np.sin(self.alpha - np.abs(self.beta))

    @property
    def heading(self):
        """Orientation angle in the cross-section, measured from the x-axis (transverse)."""
        return self.beta + self.delta

    def state(self, t):
        """(rho, beta, delta, z) interpolated at times t (within the trajectory)."""
        out = np.array([self._sol.sol(ti) for ti in np.atleast_1d(t)])
        if self.dipole == "axial":
            out[:, 2] = np.nan
        return out


def repulsion_velocity(rho, beta, alpha, strength, range_, power):
    """(u_r, u_theta) of the soft repulsion from both walls (see module doc)."""
    u_r = u_t = 0.0
    # Upper wall: inward normal (sin a, -cos a), i.e. e_r part sin(a - b), e_theta part
    # -cos(a - b); the lower wall is its mirror image. A wall is a half-plane: beyond
    # gap = pi/2 its nearest point is the edge, at distance rho along e_r.
    for side in (1, -1):
        gap = min(alpha - side * beta, np.pi / 2)
        d = rho * np.sin(gap)
        if d < range_:
            u = strength * ((range_ / d) ** power - 1)
            u_r += u * np.sin(gap)
            u_t -= side * u * np.cos(gap)
    return u_r, u_t


def trajectory(profile, rho0, beta0, delta0=0.0, sigma=1.0, v0=1.0, t_max=1e4,
               d_min=1.0, rho_min=1.0, rho_max=1e3, n_out=2000, repulsion=None):
    """Integrate the equations of motion from (rho0, beta0, delta0, z = 0).

    Stops when the distance to the nearest wall, rho sin(alpha - |beta|), reaches
    d_min ("wall"), rho reaches rho_min ("apex") or rho_max ("far"), or at t_max.
    The far-field model is only valid well away from the walls, so d_min should be
    at least the swimmer size. delta0 is ignored for the axial dipole.

    repulsion: None (off) or dict(strength=U, range=d_r, power=n), the soft wall
    repulsion of the module doc; d_min must then be below d_r.
    """
    alpha = profile.alpha
    if not abs(beta0) < alpha or rho0 * np.sin(alpha - abs(beta0)) <= d_min or rho0 <= rho_min:
        raise ValueError(f"start (rho0={rho0}, beta0={beta0}) is not beyond d_min and rho_min")
    if repulsion is not None:
        rep = (float(repulsion["strength"]), float(repulsion["range"]), float(repulsion["power"]))
        if not (rep[0] > 0 and rep[1] > d_min and rep[2] > 0):
            raise ValueError(f"repulsion needs strength > 0, power > 0 and range > d_min, got {repulsion}")
    transverse_ = profile.dipole == "transverse"

    def rhs(t, y):
        rho, beta, delta, _ = y
        f_r, f_t, f_o = profile.f(beta, delta)
        V_r, V_t = sigma * f_r / rho**2, sigma * f_t / rho**2
        if repulsion is not None:
            u_r, u_t = repulsion_velocity(rho, beta, alpha, *rep)
            V_r, V_t = V_r + u_r, V_t + u_t
        if not transverse_:
            return [V_r, V_t / rho, 0.0, v0]
        beta_dot = (v0 * np.sin(delta) + V_t) / rho
        return [v0 * np.cos(delta) + V_r, beta_dot, sigma * f_o / rho**3 - beta_dot, 0.0]

    def wall(t, y):
        return y[0] * np.sin(alpha - abs(y[1])) - d_min

    def apex(t, y):
        return y[0] - rho_min

    def far(t, y):
        return y[0] - rho_max

    events = [wall, apex, far]
    for e in events:
        e.terminal = True
    y0 = [rho0, beta0, delta0 if transverse_ else 0.0, 0.0]
    sol = solve_ivp(rhs, (0, t_max), y0, method="DOP853", events=events,
                    rtol=1e-10, atol=1e-12, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    reason = "t_max"
    for name, te in zip(("wall", "apex", "far"), sol.t_events):
        if len(te):
            reason = name
    t = np.linspace(0, sol.t[-1], n_out)
    rho, beta, delta, z = sol.sol(t)
    if not transverse_:
        delta = np.full_like(t, np.nan)
    tr = Trajectory(profile.dipole, alpha, t, rho, beta, delta, z, reason)
    tr._sol = sol
    return tr


__all__ = ["DIPOLES", "GAP_MIN", "N_GRID", "Profile", "Trajectory", "repulsion_velocity", "trajectory"]
