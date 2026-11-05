"""Numerical robustness: precision, quadrature, small-nu regularity, truncation."""

import mpmath as mp
import pytest

import hydrowedge.common as common
from hydrowedge import axial, transverse

RHO = mp.mpf("0.5")
A3, B6, D12 = mp.pi / 3, mp.pi / 6, mp.pi / 12


def _integrands(kind, alpha, beta, nu):
    if kind == "axial":
        return axial.integrands(alpha, beta, nu)
    return transverse.integrands(alpha, beta, D12, nu)


@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("nu", ["1e-9", "0.01", "0.8", "4", "15", "40", "75", "100"])
def test_adaptive_precision_matches_1500_digits(kind, nu):
    """The nu-dependent working precision gives the integrand to >= 25 digits."""
    for alpha, beta in [(mp.pi / 2, B6), (A3, B6), (A3, mp.mpf(0))]:
        with mp.workdps(common._dps_at(mp.mpf(nu))):
            got = _integrands(kind, +alpha, +beta, mp.mpf(nu))
        with mp.workdps(1500):
            ref = _integrands(kind, +alpha, +beta, mp.mpf(nu))
        scale = max(abs(x) for x in ref)
        for g, r in zip(got, ref):
            assert abs(g - r) <= mp.mpf("1e-25") * scale


def test_float64_precision_is_insufficient():
    """Documents why mpmath is needed: at fixed 16 digits the integrand is wrong at nu = 40."""
    with mp.workdps(16):
        low = transverse.integrands(+A3, +B6, +D12, mp.mpf(40))
    with mp.workdps(600):
        ref = transverse.integrands(+A3, +B6, +D12, mp.mpf(40))
    assert max(abs(l - r) / abs(r) for l, r in zip(low, ref)) > 1


@pytest.mark.parametrize("kind", ["axial", "transverse"])
def test_independent_of_base_precision(kind, monkeypatch):
    def run():
        if kind == "axial":
            return axial.velocities(A3, B6, RHO)
        return transverse.velocities(A3, B6, D12, RHO)

    a = run()
    monkeypatch.setattr(common, "BASE_DPS", 45)
    b = run()
    for x, y in zip((a.V_r, a.V_theta, a.Omega_z), (b.V_r, b.V_theta, b.Omega_z)):
        assert abs(x - y) < 1e-25


def _velocities_tanh_sinh(kind, alpha, beta):
    """Independent integration of the solver's integrands: tanh-sinh, different break
    points, plus extra break points at nu ~ |cos alpha| to resolve the narrow feature
    near nu = 0 that appears when alpha is close to (but not equal to) pi/2."""

    def f(nu, i):
        with mp.workdps(common._dps_at(nu)):
            v = _integrands(kind, +alpha, +beta, mp.mpf(nu))[i]
        return +v

    pts = [0, 0.6, 1.9, 4, 9, 17, 30, 55, 100]
    eps = abs(mp.cos(alpha))
    if eps > 1e-20:
        pts += [eps * k for k in (0.1, 1, 10, 100) if eps * k < 0.6]
    pts = sorted(set(pts))
    I = [mp.quad(lambda nu: f(nu, i), pts, method="tanh-sinh") for i in range(3)]
    return (-I[0] / (16 * RHO**2), I[1] / (8 * RHO**2), I[2] / (16 * RHO**3))


def _solver(kind, alpha, beta):
    v = (axial.velocities(alpha, beta, RHO) if kind == "axial"
         else transverse.velocities(alpha, beta, D12, RHO))
    return v.V_r, v.V_theta, v.Omega_z


@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("alpha,beta", [(A3, B6), (mp.pi / 2, B6)])
def test_independent_quadrature(kind, alpha, beta):
    got = _velocities_tanh_sinh(kind, alpha, beta)
    for g, r in zip(got, _solver(kind, alpha, beta)):
        assert abs(g - r) < 1e-24


@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("alpha", [mp.pi / 2, A3])
def test_integrand_regular_at_small_nu(kind, alpha):
    """Finite limit nu -> 0 (no 1/nu or log divergence)."""
    vals = []
    for nu in ("1e-6", "1e-10", "1e-14"):
        with mp.workdps(common._dps_at(mp.mpf(nu))):
            vals.append(_integrands(kind, +alpha, +B6, mp.mpf(nu)))
    for i in range(3):
        assert abs(vals[0][i] - vals[2][i]) < 1e-4 * (1 + abs(vals[2][i]))
        assert abs(vals[1][i] - vals[2][i]) < 1e-8 * (1 + abs(vals[2][i]))


@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("alpha,beta", [(mp.pi / 2, 0), (mp.pi / 2, B6), (A3, 0), (A3, B6)])
def test_tail_beyond_nu_max_is_negligible(kind, alpha, beta):
    """The integrand decays like exp(-2 (alpha - |beta|) nu); for nu >= 100 it is < 1e-30."""
    beta = mp.mpf(beta)
    for nu in (100, 120, 150):
        with mp.workdps(common._dps_at(nu)):
            v = _integrands(kind, +alpha, +beta, mp.mpf(nu))
        assert max(abs(x) for x in v) < 1e-30


def test_sigma_is_linear_prefactor():
    a = transverse.velocities(A3, B6, D12, RHO, sigma=1)
    b = transverse.velocities(A3, B6, D12, RHO, sigma=-2.5)
    for x, y in zip((a.V_r, a.V_theta, a.Omega_z), (b.V_r, b.V_theta, b.Omega_z)):
        assert abs(-2.5 * x - y) < 1e-25


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["axial", "transverse"])
def test_formulas_smooth_as_alpha_tends_to_pi_over_2(kind):
    """The formulas have no singularity as alpha -> pi/2 (tan/cot-free form of Z_j): with a
    resolving quadrature the one-sided second difference is O(eps^2).
    One-sided because the model is stated for alpha in (0, pi/2] (the paper, Sec. 2)."""
    eps = mp.mpf("1e-4")
    f0, f1, f2 = (_velocities_tanh_sinh(kind, mp.pi / 2 - s * eps, B6) for s in (0, 1, 2))
    for i in range(3):
        assert abs(f0[i] - 2 * f1[i] + f2[i]) < 100 * eps**2
        assert abs(f0[i] - f1[i]) < 100 * eps


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["axial", "transverse"])
def test_solver_accurate_near_pi_over_2(kind):
    """For alpha close to pi/2 the integrand has a feature of width |cos alpha| near
    nu = 0; common._breaks adds break points to resolve it."""
    alpha = mp.pi / 2 - mp.mpf("1e-5")
    for g, r in zip(_solver(kind, alpha, B6), _velocities_tanh_sinh(kind, alpha, B6)):
        assert abs(g - r) < 1e-15
