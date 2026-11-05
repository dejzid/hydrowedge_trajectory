"""Independent reference for alpha = pi/2: a force dipole next to a single plane wall.

A wedge with half-angle alpha = pi/2 is the half-space x > 0 bounded by the wall
x = 0. Its exact solution is Blake's (1971) image system, which is used here as
an independent check of the wedge formulas.

Blake's formula is written for a wall at X3 = 0 with the fluid in X3 > 0. We use
Blake coordinates (X1, X2, X3) = (y, z, x) of the wedge frame (a cyclic, hence
right-handed, relabelling). With sigma = 1, the free-space dipole is
v = (e . grad_0) G^inf . e, with G^inf = I/s + s s / s^3 (the paper, Sec. 2.1).
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp

x = sp.symbols("x1:4", real=True)
y1, y2, h = sp.symbols("y1 y2 h", real=True)
e = sp.symbols("e1:4", real=True)
Y = (y1, y2, h)


def _oseen(d):
    s = sp.sqrt(sum(di**2 for di in d))
    return sp.Matrix(3, 3, lambda i, j: sp.KroneckerDelta(i, j) / s + d[i] * d[j] / s**3)


@lru_cache(maxsize=None)
def blake_tensors():
    """Free-space Oseen tensor and Blake's image tensor (both without 1/(8 pi eta))."""
    r = [x[i] - Y[i] for i in range(3)]
    R = [x[0] - y1, x[1] - y2, x[2] + h]
    Rn = sp.sqrt(sum(Ri**2 for Ri in R))

    G_img = -_oseen(R)
    for i in range(3):
        phi = h * R[i] / Rn**3 - (sp.KroneckerDelta(i, 2) / Rn + R[i] * R[2] / Rn**3)
        dphi = [sp.diff(phi, x[k]) for k in range(3)]  # d/dR_k at fixed h
        for j in range(3):
            # 2h (delta_{j,alpha} delta_{alpha,k} - delta_{j3} delta_{3k}) d/dR_k phi_i
            G_img[i, j] += 2 * h * (dphi[j] if j < 2 else -dphi[2])
    return _oseen(r), G_img


@lru_cache(maxsize=None)
def _dipole_functions():
    _, G_img = blake_tensors()
    u = G_img * sp.Matrix(e)  # image flow of a unit Stokeslet along e
    # Dipole: derivative with respect to the source position along e.
    uD = sp.Matrix([sum(e[k] * sp.diff(u[i], Y[k]) for k in range(3)) for i in range(3)])
    curl = sp.Matrix([
        sp.diff(uD[2], x[1]) - sp.diff(uD[1], x[2]),
        sp.diff(uD[0], x[2]) - sp.diff(uD[2], x[0]),
        sp.diff(uD[1], x[0]) - sp.diff(uD[0], x[1]),
    ]) / 2
    args = (*x, y1, y2, h, *e)
    return sp.lambdify(args, uD, "mpmath"), sp.lambdify(args, curl, "mpmath")


def wall_velocities(beta, rho, e_wedge):
    """V_r, V_theta, Omega_z for a dipole with orientation e_wedge (wedge Cartesian x, y, z)
    at (rho, beta, 0), next to the wall x = 0 (i.e. a wedge with alpha = pi/2)."""
    uD, curl = _dipole_functions()
    pos = (rho * mp.sin(beta), mp.mpf(0), rho * mp.cos(beta))  # Blake (X1, X2, X3)
    eb = (e_wedge[1], e_wedge[2], e_wedge[0])
    args = (*pos, *pos, *eb)
    U = uD(*args)
    W = curl(*args)
    V_x, V_y = U[2], U[0]
    V_r = V_x * mp.cos(beta) + V_y * mp.sin(beta)
    V_theta = -V_x * mp.sin(beta) + V_y * mp.cos(beta)
    return V_r, V_theta, W[1]


def total_flow_at(point, source, e_vec):
    """Full Stokeslet flow (free space + image) at `point`; used to check no-slip."""
    G0, G_img = blake_tensors()
    G = (G0 + G_img) * sp.Matrix(e_vec)
    subs = dict(zip(x, point)) | {y1: source[0], y2: source[1], h: source[2]}
    return [sp.N(gi.subs(subs), 30) for gi in G]
