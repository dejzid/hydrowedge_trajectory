"""Coefficient-level checks against relations stated in the paper (fast, any alpha).

The w-potential boundary condition is algebraic in FKL space. Using
T{1/s} = (2 pi / nu) csh(pi nu) ch((pi - |theta - beta|) nu) K_{i nu}(|k| rho):

  axial:       Phi_w = 1/(rho s)
               -> ch((pi-|t-b|)nu) + Lambda_w sh(t nu) + Lambda_w^dag ch(t nu) = 0
  transverse:  Phi_w = (1/rho + cos(d) D0) 1/s,  D0 = cos(d) d_rho + sin(d)/rho d_beta
               1/rho part:  ch(.) + sin(d)cos(d) d_beta ch(.) + Lambda_w sh + Lambda_w^dag ch = 0
               d_rho part:  cos^2(d) ch(.) + H_w sh + H_w^dag ch = 0
at theta = t = +-alpha.
"""

import mpmath as mp
import pytest

from hydrowedge import axial, transverse

ALPHAS = [mp.pi / 2, mp.pi / 3, mp.mpf("0.37") * mp.pi, mp.pi / 8]
BETAS = [mp.mpf(0), mp.pi / 6 * mp.mpf("0.4"), -mp.mpf("0.1")]
NUS = ["1e-6", "0.3", "1.7", "6.5", "25"]
DELTAS = [mp.pi / 12, -mp.mpf("1.1"), mp.mpf("2.5")]


def _chf(theta, beta, nu):
    return mp.cosh((mp.pi - abs(theta - beta)) * nu)


def _d_beta_chf(theta, beta, nu):
    return nu * mp.sinh((mp.pi - abs(theta - beta)) * nu) * mp.sign(theta - beta)


def _rel(x, scale):
    return abs(x) / (1 + abs(scale))


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("nu", NUS)
def test_axial_w_boundary_condition(alpha, beta, nu):
    with mp.workdps(60):
        nu = mp.mpf(nu)
        c = axial.coefficients(+alpha, +beta, nu)
        for t in (alpha, -alpha):
            free = _chf(t, beta, nu)
            res = free + c["Lambda_w"] * mp.sinh(t * nu) + c["Lambda_w_dag"] * mp.cosh(t * nu)
            assert _rel(res, free) < 1e-40


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("nu", NUS)
def test_transverse_w_boundary_conditions(alpha, beta, delta, nu):
    with mp.workdps(60):
        nu = mp.mpf(nu)
        c = transverse.coefficients(+alpha, +beta, +delta, nu)
        sd, cd = mp.sin(delta), mp.cos(delta)
        for t in (alpha, -alpha):
            sh, ch = mp.sinh(t * nu), mp.cosh(t * nu)
            free1 = _chf(t, beta, nu) + sd * cd * _d_beta_chf(t, beta, nu)
            res1 = free1 + c["Lambda_w"] * sh + c["Lambda_w_dag"] * ch
            free2 = cd**2 * _chf(t, beta, nu)
            res2 = free2 + c["H_w"] * sh + c["H_w_dag"] * ch
            assert _rel(res1, free1) < 1e-40
            assert _rel(res2, free2) < 1e-40


@pytest.mark.parametrize("alpha", [mp.pi / 3, mp.mpf("0.37") * mp.pi, mp.pi / 8])
@pytest.mark.parametrize("nu", NUS)
def test_dagger_relations(alpha, nu):
    """Z_x^dag = -tan(a) th(a nu) Z_y and Z_y^dag = -cot(a) th(a nu) Z_x (both dipoles)."""
    with mp.workdps(60):
        nu, beta, delta = mp.mpf(nu), mp.pi / 7, mp.pi / 12
        th = mp.tanh(alpha * nu)
        for c in (axial.coefficients(+alpha, beta, nu),
                  transverse.coefficients(+alpha, beta, delta, nu)):
            assert _rel(c["Z_x_dag"] + mp.tan(alpha) * th * c["Z_y"], c["Z_x_dag"]) < 1e-40
            assert _rel(c["Z_y_dag"] + mp.cot(alpha) * th * c["Z_x"], c["Z_y_dag"]) < 1e-40


def test_axial_Z_formulas_from_paper():
    """Z_x = -Gamma_- sin(a) ch(a nu) sh(pi nu),  Z_x^dag = Gamma_+ sin(a) sh(a nu) sh(pi nu)."""
    with mp.workdps(60):
        a, b, nu = mp.pi / 3, mp.pi / 7, mp.mpf("1.3")
        Dp = (mp.sinh(2*a*nu) - nu*mp.sin(2*a)) * (mp.cosh(2*a*nu) + mp.cos(2*a))
        Dm = (mp.sinh(2*a*nu) + nu*mp.sin(2*a)) * (mp.cosh(2*a*nu) - mp.cos(2*a))
        t1 = mp.sin(a + b) * mp.sinh((a - b) * nu)
        t2 = mp.sin(a - b) * mp.sinh((a + b) * nu)
        Gp, Gm = 2 * (t1 + t2) / Dp, 2 * (t1 - t2) / Dm
        c = axial.coefficients(a, b, nu)
        assert _rel(c["Z_x"] + Gm * mp.sin(a) * mp.cosh(a*nu) * mp.sinh(mp.pi*nu), c["Z_x"]) < 1e-40
        assert _rel(c["Z_x_dag"] - Gp * mp.sin(a) * mp.sinh(a*nu) * mp.sinh(mp.pi*nu), c["Z_x"]) < 1e-40


def test_transverse_Z_reduce_to_axial_pattern():
    """Transverse Z_j equal minus the axial Z_j times cos^2(delta) (same Gamma structure)."""
    with mp.workdps(60):
        a, b, d, nu = mp.pi / 3, mp.pi / 7, mp.mpf("0.4"), mp.mpf("2.2")
        ca = axial.coefficients(a, b, nu)
        ct = transverse.coefficients(a, b, d, nu)
        for k in ("Z_x", "Z_y", "Z_x_dag", "Z_y_dag"):
            assert _rel(ct[k] + mp.cos(d) ** 2 * ca[k], ca[k]) < 1e-40


@pytest.mark.parametrize("module", ["axial", "transverse"])
def test_coefficients_finite_at_small_nu(module):
    """Guards against a csch/coth mix-up (it gives a 1/nu blow-up)."""
    with mp.workdps(60):
        a, b, d = mp.pi / 3, mp.pi / 6, mp.pi / 12
        for nu in ("1e-4", "1e-8", "1e-12"):
            nu = mp.mpf(nu)
            c = axial.coefficients(a, b, nu) if module == "axial" else transverse.coefficients(a, b, d, nu)
            assert abs(c["Lambda_w_dag"]) < 10
            assert abs(c["Lambda_w"]) < 10
