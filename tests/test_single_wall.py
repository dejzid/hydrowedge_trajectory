"""alpha = pi/2: the wedge degenerates to a single plane wall -> compare with Blake (1971)."""

import mpmath as mp
import pytest

from blake import total_flow_at, wall_velocities
from hydrowedge import axial, transverse

RHO = mp.mpf("0.5")
TOL = mp.mpf("1e-20")
BETAS = [0, mp.pi / 6, -mp.pi / 5, mp.pi / 3]
RHOS = [mp.mpf("0.5"), mp.mpf("1.7")]
DELTAS = [mp.pi / 12, mp.pi / 3, -mp.pi / 5, mp.pi / 2]


def test_blake_reference_satisfies_no_slip():
    """Sanity check of the reference itself: Stokeslet + images vanish on the wall X3 = 0."""
    source = (mp.mpf("0.3"), mp.mpf("-0.2"), mp.mpf("0.7"))
    e_vec = (mp.mpf("0.48"), mp.mpf("0.6"), mp.mpf("0.64"))
    for point in [(1.1, 0.4, 0), (-0.5, 2.0, 0), (0.3, -0.2, 0)]:
        u = total_flow_at(point, source, e_vec)
        assert max(abs(c) for c in u) < 1e-25


@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("beta", BETAS)
def test_axial_matches_blake(beta, rho):
    beta = mp.mpf(beta)
    got = axial.velocities(mp.pi / 2, beta, rho)
    ref = wall_velocities(beta, rho, (0, 0, 1))
    for g, r in zip((got.V_r, got.V_theta, got.Omega_z), ref):
        assert abs(g - r) < TOL


@pytest.mark.parametrize("beta", BETAS)
def test_axial_closed_form(beta):
    """Dipole parallel to a wall: V = -3 sigma / (8 h^2) along the wall normal, Omega = 0."""
    beta = mp.mpf(beta)
    got = axial.velocities(mp.pi / 2, beta, RHO)
    h = RHO * mp.cos(beta)
    V_n = -3 / (8 * h**2)
    assert abs(got.V_r - V_n * mp.cos(beta)) < TOL
    assert abs(got.V_theta + V_n * mp.sin(beta)) < TOL
    assert abs(got.Omega_z) < TOL


@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("delta", DELTAS)
def test_transverse_matches_blake(beta, delta, rho):
    beta, delta = mp.mpf(beta), mp.mpf(delta)
    got = transverse.velocities(mp.pi / 2, beta, delta, rho)
    e_wedge = (mp.cos(beta + delta), mp.sin(beta + delta), 0)
    ref = wall_velocities(beta, rho, e_wedge)
    for g, r in zip((got.V_r, got.V_theta, got.Omega_z), ref):
        assert abs(g - r) < TOL


@pytest.mark.parametrize("delta", DELTAS)
def test_transverse_closed_form_on_bisector(delta):
    """Berke et al. (2008) point dipole at distance h, inclined by angle t to the wall:
    normal velocity -3/(8h^2) (1 - 3 sin^2 t), tangential 3/(8h^2) sin(2t),
    |Omega| = 3/(8h^3) sin(t) cos(t). At beta = 0 the wall normal is e_rho, so t = pi/2 - delta."""
    delta = mp.mpf(delta)
    got = transverse.velocities(mp.pi / 2, 0, delta, RHO)
    h, t = RHO, mp.pi / 2 - delta
    assert abs(got.V_r - (-3 / (8 * h**2) * (1 - 3 * mp.sin(t) ** 2))) < TOL
    assert abs(got.V_theta - 3 / (8 * h**2) * mp.sin(2 * t)) < TOL
    assert abs(got.Omega_z - 3 / (8 * h**3) * mp.sin(t) * mp.cos(t)) < TOL
