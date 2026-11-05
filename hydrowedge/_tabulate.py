"""Solver evaluations for the trajectory profiles (run in worker processes).

Kept free of numpy/scipy so that each worker process stays small.
"""

import warnings

import mpmath as mp

from . import axial, transverse
from .common import NU_MAX

QUANTITIES = {"axial": ("r", "theta"), "transverse": ("r", "theta", "omega")}
BASES = {"axial": ("1",), "transverse": ("cc", "sc", "ss")}
POWER = {"r": 2, "theta": 2, "omega": 3}

# Relative accuracy accepted for the tabulated profile (double precision is
# ample for trajectories; the solver's own goal is 1e-25).
_PROFILE_TOL = 1e-12


def nu_max(gap):
    """Truncation needed at angular distance `gap` from the wall.

    The integrands decay like nu^3 exp(-2 gap nu); 30/gap leaves a tail < ~1e-15.
    """
    return max(NU_MAX, int(30 / gap) + 1)


def velocities(dipole, alpha, beta, delta=None):
    """(V_r, V_theta, Omega_z) at rho = sigma = 1, as floats."""
    n = nu_max(float(alpha - abs(beta)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if dipole == "axial":
            v = axial.velocities(alpha, beta, rho=1, nu_max=n)
        else:
            v = transverse.velocities(alpha, beta, delta, rho=1, nu_max=n)
    for w in caught:
        # Truncation warnings quote the achieved relative accuracy as their last number.
        msg = str(w.message)
        if "truncation" not in msg or float(msg.split("~")[-1]) > _PROFILE_TOL:
            warnings.warn(msg, stacklevel=2)
    return v.as_floats()


def g_at(args):
    """g = f sin^p(alpha - beta) for every (quantity, basis) at one beta >= 0.

    args = (dipole, alpha, beta) with the angles as strings (read at 30 digits).
    Transverse: f = f_cc cos^2 + f_sc sin cos + f_ss sin^2 in delta, from delta = 0, pi/2, pi/4.
    """
    dipole, alpha, beta = args
    with mp.workdps(30):
        alpha, beta = mp.mpf(alpha), mp.mpf(beta)
        s = float(mp.sin(alpha - beta))
        if dipole == "axial":
            f = {"1": velocities(dipole, alpha, beta)}
        else:
            v0, v90, v45 = (velocities(dipole, alpha, beta, d) for d in (0, mp.pi / 2, mp.pi / 4))
            f = {"cc": v0, "ss": v90, "sc": tuple(2 * c - a - b for a, b, c in zip(v0, v90, v45))}
    q_index = {"r": 0, "theta": 1, "omega": 2}
    return {
        (q, b): f[b][q_index[q]] * s ** POWER[q]
        for q in QUANTITIES[dipole] for b in BASES[dipole]
    }
