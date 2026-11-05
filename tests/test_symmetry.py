"""Symmetry and truncation checks, valid for any wedge angle (used here at alpha = pi/3)."""

import mpmath as mp
import pytest

from hydrowedge import axial, transverse

RHO = mp.mpf("0.5")
TOL = mp.mpf("1e-20")


@pytest.mark.parametrize("alpha", [mp.pi / 3, mp.pi / 4])
def test_axial_mirror_symmetry(alpha):
    """Reflection y -> -y: V_r is even in beta, V_theta and Omega_z are odd."""
    beta = mp.pi / 6
    p = axial.velocities(alpha, beta, RHO)
    m = axial.velocities(alpha, -beta, RHO)
    assert abs(p.V_r - m.V_r) < TOL
    assert abs(p.V_theta + m.V_theta) < TOL
    assert abs(p.Omega_z + m.Omega_z) < TOL


@pytest.mark.parametrize("alpha", [mp.pi / 3, mp.pi / 4])
def test_axial_on_bisector(alpha):
    """At beta = 0 the axial dipole only moves radially."""
    v = axial.velocities(alpha, 0, RHO)
    assert abs(v.V_theta) < TOL
    assert abs(v.Omega_z) < TOL


@pytest.mark.parametrize("alpha", [mp.pi / 3, mp.pi / 4])
def test_transverse_mirror_symmetry(alpha):
    """Reflection y -> -y maps (beta, delta) -> (-beta, -delta)."""
    beta, delta = mp.pi / 6, mp.pi / 12
    p = transverse.velocities(alpha, beta, delta, RHO)
    m = transverse.velocities(alpha, -beta, -delta, RHO)
    assert abs(p.V_r - m.V_r) < TOL
    assert abs(p.V_theta + m.V_theta) < TOL
    assert abs(p.Omega_z + m.Omega_z) < TOL


def test_transverse_head_tail_symmetry():
    """A force dipole is unchanged by e -> -e, i.e. delta -> delta + pi."""
    alpha, beta, delta = mp.pi / 3, mp.pi / 6, mp.pi / 12
    a = transverse.velocities(alpha, beta, delta, RHO)
    b = transverse.velocities(alpha, beta, delta + mp.pi, RHO)
    for x, y in zip((a.V_r, a.V_theta, a.Omega_z), (b.V_r, b.V_theta, b.Omega_z)):
        assert abs(x - y) < TOL


def test_transverse_radial_dipole_on_bisector():
    """beta = 0, delta = 0: dipole along the bisector -> purely radial motion."""
    v = transverse.velocities(mp.pi / 3, 0, 0, RHO)
    assert abs(v.V_theta) < TOL
    assert abs(v.Omega_z) < TOL


def test_rho_scaling():
    """V ~ 1/rho^2 and Omega ~ 1/rho^3 (the integrands do not depend on rho)."""
    alpha, beta, delta = mp.pi / 3, mp.pi / 6, mp.pi / 12
    a = transverse.velocities(alpha, beta, delta, RHO)
    b = transverse.velocities(alpha, beta, delta, 2 * RHO)
    assert abs(a.V_r - 4 * b.V_r) < TOL
    assert abs(a.V_theta - 4 * b.V_theta) < TOL
    assert abs(a.Omega_z - 8 * b.Omega_z) < TOL


@pytest.mark.parametrize("module", ["axial", "transverse"])
def test_truncation_nu_max_100_is_converged(module):
    """Cutting the integrals at nu = 60 instead of 100 changes nothing at 1e-20."""
    alpha, beta, delta = mp.pi / 3, mp.pi / 6, mp.pi / 12
    if module == "axial":
        a = axial.velocities(alpha, beta, RHO, nu_max=100)
        b = axial.velocities(alpha, beta, RHO, nu_max=60)
    else:
        a = transverse.velocities(alpha, beta, delta, RHO, nu_max=100)
        b = transverse.velocities(alpha, beta, delta, RHO, nu_max=60)
    for x, y in zip((a.V_r, a.V_theta, a.Omega_z), (b.V_r, b.V_theta, b.Omega_z)):
        assert abs(x - y) < TOL
