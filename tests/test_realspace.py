"""Independent validation for ANY wedge angle via the real-space flow (see realspace.py).

1. No-slip: free-space dipole + complementary flow must vanish on theta = +-alpha.
   This validates every coefficient (Lambda, H, Z, ... incl. all b, p, c, q terms).
2. Swimmer: the complementary flow at (rho, beta, 0) and Omega_z = -(curl phi)_z
   must equal the solver's V_r, V_theta, Omega_z. This validates the velocity
   formulas V_r^(n), V_theta^(n), Omega_z^(n) (A, B, C coefficients).
"""

from functools import lru_cache

import mpmath as mp
import pytest

from hydrowedge import axial, transverse
from realspace import ComplementaryFlow, conical_derivatives, free_dipole

pytestmark = pytest.mark.slow

RHO = mp.mpf("0.5")

# (alpha, beta, delta): the requested cases, plus one generic, non-symmetric case
CASES = {
    "pi2_0": (mp.mpf(1) / 2, mp.mpf(0), mp.mpf(1) / 12),
    "pi2_pi6": (mp.mpf(1) / 2, mp.mpf(1) / 6, mp.mpf(1) / 12),
    "pi3_0": (mp.mpf(1) / 3, mp.mpf(0), mp.mpf(1) / 12),
    "pi3_pi6": (mp.mpf(1) / 3, mp.mpf(1) / 6, mp.mpf(1) / 12),
    "generic": (mp.mpf("0.37"), mp.mpf("-0.067"), mp.mpf("0.13")),
}  # all angles in units of pi


def _angles(case):
    a, b, d = CASES[case]
    return a * mp.pi, b * mp.pi, d * mp.pi


@lru_cache(maxsize=None)
def _flow(kind, case):
    alpha, beta, delta = _angles(case)
    return ComplementaryFlow(alpha, beta, RHO, delta=None if kind == "axial" else delta)


def _orientation(kind, beta, delta):
    if kind == "axial":
        return (0, 0, 1)
    return (mp.cos(beta + delta), mp.sin(beta + delta), 0)


# --- the helper itself ------------------------------------------------------------


def test_kernel_reproduces_one_over_s():
    """1/s = int_0^inf ch((pi - |theta - beta|) nu) K_{i nu}(r, z) dnu (FKL transform of 1/s)."""
    beta = mp.pi / 6
    # The integrand decays like exp(-|theta - beta| nu); keep |theta - beta| >~ 0.4.
    for r, th, z in [(0.7, 0.9, 0.3), (0.3, -1.0, 0.1), (0.45, 1.2, 0.05)]:
        r, th, z = mp.mpf(r), mp.mpf(th), mp.mpf(z)
        mu = (RHO**2 + r**2 + z**2) / (2 * RHO * r)
        f = lambda nu: (
            mp.cosh((mp.pi - abs(th - beta)) * nu) / mp.cosh(mp.pi * nu)
            * conical_derivatives(nu, mu)[0] / mp.sqrt(RHO * r)
        )
        I = mp.quad(f, [0, 1, 2.5, 5, 10, 20, 40, 80, 160])
        s = mp.sqrt(r**2 + RHO**2 + z**2 - 2 * RHO * r * mp.cos(th - beta))
        assert abs(I - 1 / s) < 1e-12


def test_conical_derivatives_match_numerical_differentiation():
    for nu, mu in [(0.3, 1.2), (2.5, 1.05), (7.0, 2.0)]:
        nu, mu = mp.mpf(nu), mp.mpf(mu)
        exact = conical_derivatives(nu, mu)
        f = lambda m: conical_derivatives(nu, m)[0]
        for n in range(4):
            assert abs(exact[n] - mp.diff(f, mu, n)) < 1e-15 * (1 + abs(exact[n]))


# --- no-slip on both wedge faces --------------------------------------------------

WALL_POINTS = [  # (r, face sign, z)
    (0.6, +1, 0.2),
    (0.3, -1, 0.0),
    (1.1, +1, 0.5),
    (0.45, -1, 0.35),
]


@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("point", WALL_POINTS, ids=lambda p: f"r={p[0]},face={p[1]:+d},z={p[2]}")
def test_no_slip_on_wedge_faces(kind, case, point):
    alpha, beta, delta = _angles(case)
    r, face, z = mp.mpf(point[0]), point[1], mp.mpf(point[2])
    theta = face * alpha
    v_comp = _flow(kind, case).at(r, theta, z)[:3]
    v_free = free_dipole(
        (r * mp.cos(theta), r * mp.sin(theta), z),
        (RHO * mp.cos(beta), RHO * mp.sin(beta), 0),
        _orientation(kind, beta, delta),
    )
    assert mp.norm(v_free) > 1e-3  # the check is not trivially satisfied
    for vc, vf in zip(v_comp, v_free):
        assert abs(vc + vf) < 1e-15


# --- swimmer velocities from the real-space flow ------------------------------------


@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("case", list(CASES))
def test_swimmer_velocity_from_real_space_flow(kind, case):
    alpha, beta, delta = _angles(case)
    v_x, v_y, v_z, omega_z = _flow(kind, case).at(RHO, beta, 0)
    V_r = v_x * mp.cos(beta) + v_y * mp.sin(beta)
    V_theta = -v_x * mp.sin(beta) + v_y * mp.cos(beta)

    if kind == "axial":
        ref = axial.velocities(alpha, beta, RHO)
    else:
        ref = transverse.velocities(alpha, beta, delta, RHO)

    assert abs(v_z) < 1e-20  # reflection symmetry z -> -z
    assert abs(V_r - ref.V_r) < 1e-18
    assert abs(V_theta - ref.V_theta) < 1e-18
    assert abs(omega_z - ref.Omega_z) < 1e-18
