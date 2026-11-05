"""Shared machinery for the wedge force-dipole solvers.

Notation follows the paper (the reference for every formula):
    sh, ch, th  -> sinh, cosh, tanh
    csh = 1/sh,  sch = 1/ch

All integrands are evaluated with mpmath. The individual terms grow like
exp(c*nu) while the integrand itself decays, so cancellation is severe at
large nu; the working precision is therefore raised with nu (see `_dps_at`).
"""

import warnings
from dataclasses import dataclass

import mpmath as mp

# Precision (decimal digits) of the returned results.
BASE_DPS = 30

# Default truncation of the nu-integrals.
NU_MAX = 100

# Break points for the piecewise quadrature on [0, nu_max].
_BREAKS = (0, 0.25, 1, 2.5, 5, 10, 20, 35, 50, 70, 100)

# For alpha near pi/2: extra break points 0.1 eps * 3^k (eps = |cos alpha|) up to _NARROW.
_NARROW = mp.mpf("0.25")

# Truncation tail (relative to the result) above which a RuntimeError is raised:
# the result would not even be good to double precision.
_TAIL_RAISE = mp.mpf("1e-15")


def _accuracy_goal():
    """Relative accuracy expected of the returned results."""
    return mp.mpf(10) ** (5 - BASE_DPS)


@dataclass(frozen=True)
class Velocities:
    """Wall-induced swimming velocities of a force dipole at (rho, beta, 0)."""

    V_r: mp.mpf
    V_theta: mp.mpf
    Omega_z: mp.mpf

    def as_floats(self):
        return float(self.V_r), float(self.V_theta), float(self.Omega_z)


def _dps_at(nu):
    """Working precision needed to evaluate the integrand at `nu`.

    Large nu: terms of size up to ~exp(4*pi*nu) cancel down to a result of
    order one, i.e. ~5.5 digits are lost per unit of nu.
    Small nu: removable 0/0 behaviour (e.g. Delta_+ ~ nu^3 at alpha = pi/2)
    costs a few digits per decade of nu.
    """
    nu = float(nu)
    extra_large = int(5.6 * nu)
    extra_small = int(3 * max(0.0, -mp.log10(nu))) if nu > 0 else 0
    return BASE_DPS + 10 + extra_large + extra_small


def deltas(alpha, nu):
    """Denominators Delta_+ and Delta_- (the paper, axial section).

    Delta_pm = (sh(2 alpha nu) -+ nu sin(2 alpha)) (ch(2 alpha nu) +- cos(2 alpha)).
    The second factors are rewritten with exact identities,
        ch(2x) + cos(2a) = 2 sh(x)^2 + 2 cos(a)^2,
        ch(2x) - cos(2a) = 2 sh(x)^2 + 2 sin(a)^2,
    to avoid the 0/0 cancellation at small nu when alpha = pi/2.
    """
    sh_a = mp.sinh(alpha * nu)
    delta_plus = (mp.sinh(2 * alpha * nu) - nu * mp.sin(2 * alpha)) * (
        2 * sh_a**2 + 2 * mp.cos(alpha) ** 2
    )
    delta_minus = (mp.sinh(2 * alpha * nu) + nu * mp.sin(2 * alpha)) * (
        2 * sh_a**2 + 2 * mp.sin(alpha) ** 2
    )
    return delta_plus, delta_minus


def gammas(alpha, beta, nu, delta_plus, delta_minus):
    """Gamma_+ and Gamma_- (the paper, axial section)."""
    term1 = mp.sin(alpha + beta) * mp.sinh((alpha - beta) * nu)
    term2 = mp.sin(alpha - beta) * mp.sinh((alpha + beta) * nu)
    return 2 * (term1 + term2) / delta_plus, 2 * (term1 - term2) / delta_minus


def assemble_integrands(kernel, beta, nu):
    """The three nu-integrands (radial, azimuthal, rotational), without prefactors.

    `kernel` = (V_r1, V_r2, V_theta1, V_theta2, Omega_z1, Omega_z2); each pair is
    combined as (X1 sh(beta nu) + X2 ch(beta nu)) sch(pi nu).
    """
    V_r1, V_r2, V_t1, V_t2, O_1, O_2 = kernel
    shb, chb = mp.sinh(beta * nu), mp.cosh(beta * nu)
    sch_pi = 1 / mp.cosh(mp.pi * nu)
    return (
        (V_r1 * shb + V_r2 * chb) * sch_pi,
        (V_t1 * shb + V_t2 * chb) * sch_pi,
        (O_1 * shb + O_2 * chb) * sch_pi,
    )


def _breaks(alpha, nu_max):
    """Quadrature break points on [0, nu_max].

    For alpha close to (not equal to) pi/2 the integrand has a feature of width
    |cos alpha| near nu = 0 with a slowly decaying shoulder; it is resolved by a
    geometric ladder of extra break points from |cos alpha|/10 up to _NARROW.
    """
    pts = {b for b in _BREAKS if b < nu_max}
    eps = abs(mp.cos(alpha))
    if mp.mpf("1e-20") < eps < _NARROW:
        x = eps / 10
        while x < _NARROW:
            pts.add(x)
            x *= 3
    return sorted(pts) + [nu_max]


def integrate_velocities(integrands, alpha, beta, rho, sigma=1, nu_max=NU_MAX):
    """Assemble V_r, V_theta, Omega_z from the nu-integrands.

    `integrands(nu)` returns the three full integrands (see `assemble_integrands`),
    built from the coefficients (V_r1, V_r2, V_theta1, V_theta2, Omega_z1, Omega_z2)
    in the notation of the paper, Sec. "Swimmer hydrodynamics":

        V_r     = -sigma/(16 rho^2) int (V_r1 sh(beta nu) + V_r2 ch(beta nu)) sch(pi nu) dnu
        V_theta =  sigma/( 8 rho^2) int (V_theta1 sh + V_theta2 ch) sch(pi nu) dnu
        Omega_z =  sigma/(16 rho^3) int (Omega_z1 sh + Omega_z2 ch) sch(pi nu) dnu

    Raises RuntimeError if the quadrature error estimate exceeds the accuracy goal, or
    if the neglected tail beyond nu_max (the integrand decays like
    exp(-2 (alpha - |beta|) nu)) exceeds 1e-15 relative; warns if that tail exceeds
    the accuracy goal.
    """
    with mp.workdps(BASE_DPS):
        cache = {}

        def full(nu):
            key = mp.mpf(nu)
            if key not in cache:
                with mp.workdps(_dps_at(nu)):
                    vals = integrands(mp.mpf(nu))
                cache[key] = tuple(+v for v in vals)
            return cache[key]

        breaks = _breaks(alpha, nu_max)
        goal = _accuracy_goal()
        results = []
        for i, name in enumerate(("V_r", "V_theta", "Omega_z")):
            value, err = mp.quad(
                lambda x: full(x)[i], breaks, method="gauss-legendre", error=True
            )
            scale = max(1, abs(value))
            if err > goal * scale:
                raise RuntimeError(
                    f"{name}: nu-quadrature not converged (error estimate {mp.nstr(err, 3)})"
                )
            tail = abs(full(nu_max)[i]) / (2 * (alpha - abs(beta)))
            if tail > _TAIL_RAISE * scale:
                raise RuntimeError(
                    f"{name}: integral not converged at nu_max = {nu_max} (tail "
                    f"~{mp.nstr(tail, 3)}); the swimmer is too close to a wall"
                )
            if tail > goal * scale:
                warnings.warn(
                    f"{name}: truncation at nu_max = {nu_max} limits the accuracy to "
                    f"~{mp.nstr(tail / scale, 3)}",
                    stacklevel=3,
                )
            results.append(value)
        I_r, I_t, I_z = results

        rho = mp.mpf(rho)
        sigma = mp.mpf(sigma)
        return Velocities(
            V_r=-sigma / (16 * rho**2) * I_r,
            V_theta=sigma / (8 * rho**2) * I_t,
            Omega_z=sigma / (16 * rho**3) * I_z,
        )


def to_mpf(*values):
    """Convert user inputs (floats, ints, mpf or strings) to mpf.

    Strings and ints are converted at BASE_DPS (or the current precision, if higher).
    Other values (e.g. mp.pi/3) have already been rounded at the caller's precision;
    a UserWarning is emitted if that is below BASE_DPS.
    """
    if mp.mp.dps < BASE_DPS and not all(isinstance(v, (str, int)) for v in values):
        warnings.warn(
            f"inputs given at mp.dps = {mp.mp.dps} < {BASE_DPS}: results are only "
            f"accurate to ~{mp.mp.dps} digits; set mp.mp.dps = {BASE_DPS} or pass strings",
            stacklevel=3,
        )
    with mp.workdps(max(mp.mp.dps, BASE_DPS)):
        return tuple(+mp.mpf(v) for v in values)


def check_geometry(alpha, beta):
    """Require 0 < alpha <= pi/2 (the paper, Sec. 2) and the swimmer inside the wedge."""
    if not 0 < alpha <= mp.pi / 2 * (1 + mp.eps):
        raise ValueError(f"wedge half-angle alpha = {mp.nstr(alpha, 10)} not in (0, pi/2]")
    if not abs(beta) < alpha:
        raise ValueError(
            f"swimmer angle beta = {mp.nstr(beta, 10)} not inside the wedge |beta| < alpha"
        )
