"""Tests for the code-review findings (written before the fixes; they fail until fixed).

Contracts the fixes are expected to satisfy:
  - an integral that is not converged (tail at nu_max not negligible, or quadrature
    error estimate too large) raises RuntimeError instead of returning a number;
  - parameters outside 0 < alpha <= pi/2, |beta| < alpha raise ValueError;
  - calling at a working precision below common.BASE_DPS emits a UserWarning.
"""

import gc
import importlib
import warnings

import mpmath as mp
import pytest

import hydrowedge.common as common
from hydrowedge import axial, transverse

RHO = mp.mpf("0.5")
B6, D12 = mp.pi / 6, mp.pi / 12


def _run(kind, alpha, beta, **kw):
    if kind == "axial":
        return axial.velocities(alpha, beta, RHO, **kw)
    return transverse.velocities(alpha, beta, D12, RHO, **kw)


def _vals(v):
    return v.V_r, v.V_theta, v.Omega_z


# --- Finding 1: fixed truncation nu_max = 100 ----------------------------------------
# The integrand decays like exp(-2 (alpha - |beta|) nu). Near a wall or in a narrow
# wedge it is still O(1) at nu = 100 (Omega_z moves from -9.2 to -0.16 when nu_max
# goes from 100 to 200 for the first case below).

@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("alpha,beta", [
    (mp.pi / 3, mp.pi / 3 - mp.mpf("0.05")),     # near the upper wall
    (mp.pi / 3, -(mp.pi / 3 - mp.mpf("0.05"))),  # near the lower wall
    (mp.mpf("0.1"), mp.mpf(0)),                  # narrow wedge
])
def test_unconverged_truncation_raises(kind, alpha, beta):
    with pytest.raises(RuntimeError):
        _run(kind, alpha, beta)


# --- Finding 2: quadrature error estimate ignored ------------------------------------

@pytest.mark.parametrize("kind", ["axial", "transverse"])
def test_unresolved_quadrature_raises(kind, monkeypatch):
    """A single Gauss-Legendre panel on [0, 100] has error estimate ~1e-14 >> 1e-30."""
    monkeypatch.setattr(common, "_BREAKS", (0, 100))
    with pytest.raises(RuntimeError):
        _run(kind, mp.pi / 3, B6)


# --- Finding 3: run_cases.py angles built at import-time precision -------------------

def test_run_cases_angles_are_exact_to_30_digits():
    """Importing run_cases at the default 15 digits must not round pi/3, pi/6, pi/12."""
    import run_cases

    with mp.workdps(15):
        rc = importlib.reload(run_cases)

    def angle(v):  # accepts a multiple of pi (current layout) or a callable
        return v() if callable(v) else v * mp.pi

    exact = {"0": mp.mpf(0), "pi/2": mp.pi / 2, "pi/3": mp.pi / 3,
             "pi/6": mp.pi / 6, "pi/12": mp.pi / 12}
    for label, v in list(rc.ALPHAS) + list(rc.BETAS) + [rc.DELTA]:
        assert abs(angle(v) - exact[label]) < mp.mpf("1e-28"), label


# --- Finding 4: alpha close to pi/2 ---------------------------------------------------
# Reference: tanh-sinh with |cos alpha|-dependent break points (test_numerics.
# _velocities_tanh_sinh); a Gauss-Legendre run with the same extra break points agrees
# to ~1e-26. The current solver is off by ~1e-6.

NEAR_PI2 = {
    "axial": (
        mp.mpf("-1.73207033898220058556705"),
        mp.mpf("1.000024718178804249386034"),
        mp.mpf("-0.00000487146260691483082342402"),
    ),
    "transverse": (
        mp.mpf("1.866028832032508557994733"),
        mp.mpf("1.232083527221281901692382"),
        mp.mpf("2.309448122334632146583335"),
    ),
}


@pytest.mark.parametrize("kind", ["axial", "transverse"])
def test_accurate_close_to_pi_over_2(kind):
    got = _run(kind, mp.pi / 2 - mp.mpf("1e-5"), B6)
    for g, r in zip(_vals(got), NEAR_PI2[kind]):
        assert abs(g - r) < mp.mpf("1e-20")


# --- Finding 5: parameter validation --------------------------------------------------

@pytest.mark.parametrize("kind", ["axial", "transverse"])
@pytest.mark.parametrize("alpha,beta", [
    (mp.pi / 3, mp.pi / 3 + mp.mpf("0.1")),  # swimmer outside the wedge
    (mp.pi / 3, mp.pi / 3),                  # swimmer on the wall
    (mp.pi / 3, -mp.pi / 3),
    (mp.mpf(2), mp.mpf("0.3")),              # alpha > pi/2
    (mp.mpf(0), mp.mpf(0)),
    (mp.mpf(-1), mp.mpf(0)),
])
def test_invalid_geometry_raises(kind, alpha, beta):
    with pytest.raises(ValueError):
        _run(kind, alpha, beta)


# --- Finding 6: inputs converted at the caller's precision ----------------------------

@pytest.mark.parametrize("kind", ["axial", "transverse"])
def test_low_caller_precision_warns(kind):
    with mp.workdps(15):
        with pytest.warns(UserWarning):
            _run(kind, mp.pi / 3, B6)


@pytest.mark.parametrize("kind", ["axial", "transverse"])
def test_no_warning_at_base_precision(kind):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(kind, mp.pi / 3, B6)


# --- Finding 7: repeated sh/ch(2 alpha nu), sh/ch(4 alpha nu) -------------------------

def test_transverse_integrand_hyperbolic_call_count(monkeypatch):
    """Currently 72 sinh/cosh calls per integrand, 44 of them recomputing the same four
    values in _bc_sums/_pq_sums. After hoisting them at most ~28 remain."""
    calls = [0]
    sinh, cosh = mp.sinh, mp.cosh

    def counted(f):
        def g(x):
            calls[0] += 1
            return f(x)
        return g

    monkeypatch.setattr(mp, "sinh", counted(sinh))
    monkeypatch.setattr(mp, "cosh", counted(cosh))
    with mp.workdps(common._dps_at(50)):
        transverse.integrands(mp.pi / 3, B6, D12, mp.mpf(50))
    assert calls[0] <= 32


# --- Finding 10: results CSV opened without closing -----------------------------------

def test_results_csv_check_closes_file():
    import test_regression

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test_regression.test_results_file_matches_solver()
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
