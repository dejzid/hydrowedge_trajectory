"""Independent real-space reconstruction of the complementary flow (any wedge angle).

the paper, Sec. "Solution in real space": each complementary potential is

    phi_j(r, theta, z) = int_0^inf Psi_j(theta, nu) K_{i nu}(r, z) dnu,
    Psi_j = psi1/rho + psi2 d/drho - psi3 d/dz - psi4 rho d^2/dz^2,
    (psi1, psi2, psi3, psi4) = (Lambda, H, Delta, Z)_j sh(theta nu)
                             + (Lambda, H, Delta, Z)_j^dagger ch(theta nu),
    K_{i nu}(r, z) = sch(pi nu) P_{i nu - 1/2}(mu) / sqrt(rho r),
    mu = (rho^2 + r^2 + z^2) / (2 rho r),

with phi_x, phi_y, phi_z Cartesian components, and the velocity follows from the
Papkovich-Neuber form  v = grad(r . phi + rho phi_w) - 2 phi.

Only the coefficients (Lambda, H, Delta, Z) are taken from the solver; the
velocity formulas V_r^(n), V_theta^(n), Omega_z^(n) are NOT used. This gives
  * a no-slip test: free-space dipole + complementary flow = 0 on theta = +-alpha,
    which checks every coefficient for any alpha;
  * a swimmer test: the complementary flow (and half its curl) at the swimmer
    position must equal the solver's V_r, V_theta, Omega_z, which checks the
    velocity formulas for any alpha.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp

from hydrowedge import axial, transverse
from hydrowedge.common import _breaks, _dps_at

# Derivative orders (d_r, d_z, d_rho) of the kernel that are needed: the four
# kernel operators K, d_rho K, d_z K, d_z^2 K and their r- and z-derivatives.
_OPS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0)]
_ORDERS = sorted({(a + i, b + j, c) for a, b, c in _OPS for i, j in [(0, 0), (1, 0), (0, 1)]})


@lru_cache(maxsize=None)
def _kernel_derivatives():
    """Lambdified d_r^a d_z^b d_rho^c [P(mu) / sqrt(rho r)] in terms of P^(n)(mu)."""
    r, z, rho = sp.symbols("r z rho", positive=True)
    Pn = sp.symbols("P0:5")  # P0 = P(mu), Pn = n-th mu-derivative
    mu = (rho**2 + r**2 + z**2) / (2 * rho * r)

    def total_diff(expr, var):
        # chain rule: d/dvar = partial/dvar + sum_n P_{n+1} dmu/dvar partial/dP_n
        mu_var = sp.diff(mu, var)
        return sp.diff(expr, var) + sum(
            Pn[n + 1] * mu_var * sp.diff(expr, Pn[n]) for n in range(4)
        )

    g = Pn[0] / sp.sqrt(rho * r)
    out = {}
    for a, b, c in _ORDERS:
        expr = g
        for var, n in ((r, a), (z, b), (rho, c)):
            for _ in range(n):
                expr = total_diff(expr, var)
        assert not expr.has(Pn[4])
        out[(a, b, c)] = sp.lambdify((r, z, rho, *Pn[:4]), expr, "mpmath")
    return out


def conical_derivatives(nu, mu):
    """P^(n)_{i nu - 1/2}(mu), n = 0..3, from P = 2F1(1/2 - i nu, 1/2 + i nu; 1; (1 - mu)/2)."""
    a = mp.mpf(1) / 2 + 1j * nu
    b = mp.conj(a)
    x = (1 - mu) / 2
    vals = []
    for n in range(4):
        pref = (-mp.mpf(1) / 2) ** n * mp.rf(a, n) * mp.rf(b, n) / mp.factorial(n)
        vals.append(mp.re(pref * mp.hyp2f1(a + n, b + n, 1 + n, x)))
    return vals


def _axial_coeffs(alpha, beta, nu):
    c = axial.coefficients(alpha, beta, nu)
    # Delta_z, Delta_z^dagger from the paper (not needed by the solver itself).
    D = mp.sinh((mp.pi - alpha) * nu) * mp.sinh(beta * nu) / mp.sinh(alpha * nu)
    Dd = mp.cosh((mp.pi - alpha) * nu) * mp.cosh(beta * nu) / mp.cosh(alpha * nu)
    zero = (0, 0)
    return {
        "x": (zero, zero, zero, (c["Z_x"], c["Z_x_dag"])),
        "y": (zero, zero, zero, (c["Z_y"], c["Z_y_dag"])),
        "z": (zero, zero, (D, Dd), zero),
        "w": ((c["Lambda_w"], c["Lambda_w_dag"]), zero, zero, zero),
    }


def _transverse_coeffs(alpha, beta, delta, nu):
    c = transverse.coefficients(alpha, beta, delta, nu)
    zero = (0, 0)
    pair = lambda k: (c[k], c[k + "_dag"])
    return {
        "x": (pair("Lambda_x"), pair("H_x"), zero, pair("Z_x")),
        "y": (pair("Lambda_y"), pair("H_y"), zero, pair("Z_y")),
        "z": (zero, zero, zero, zero),
        "w": (pair("Lambda_w"), pair("H_w"), zero, zero),
    }


class ComplementaryFlow:
    """Complementary flow of an axial (delta=None) or transverse dipole."""

    def __init__(self, alpha, beta, rho, delta=None, nu_max=100, dps=25):
        self.alpha, self.beta, self.delta = mp.mpf(alpha), mp.mpf(beta), delta
        self.rho = mp.mpf(rho)
        self.nu_max = nu_max
        self.dps = dps
        self._psi_cache = {}

    def _psi(self, theta, nu):
        """(psi_k, d psi_k / d theta) * sch(pi nu) for k = 1..4 and j in x, y, z, w."""
        key = (theta, nu)
        if key not in self._psi_cache:
            with mp.workdps(_dps_at(nu)):
                nu_h = mp.mpf(nu)
                if self.delta is None:
                    coeffs = _axial_coeffs(self.alpha, self.beta, nu_h)
                else:
                    coeffs = _transverse_coeffs(self.alpha, self.beta, mp.mpf(self.delta), nu_h)
                sh, ch = mp.sinh(theta * nu_h), mp.cosh(theta * nu_h)
                sch = 1 / mp.cosh(mp.pi * nu_h)
                res = {}
                for j, cs in coeffs.items():
                    res[j] = [
                        ((A * sh + Ad * ch) * sch, nu_h * (A * ch + Ad * sh) * sch)
                        for A, Ad in cs
                    ]
            self._psi_cache[key] = {
                j: [(+p, +dp) for p, dp in v] for j, v in res.items()
            }
        return self._psi_cache[key]

    def _integrand(self, r, theta, z, nu):
        """nu-integrand of (v_x, v_y, v_z, Omega_z) at the field point (r, theta, z)."""
        rho = self.rho
        kd = _kernel_derivatives()
        mu = (rho**2 + r**2 + z**2) / (2 * rho * r)
        P = conical_derivatives(nu, mu)
        Kd = {o: f(r, z, rho, *P) for o, f in kd.items()}

        # kernel operators: K/rho, d_rho K, -d_z K, -rho d_z^2 K, and their r-, z-derivatives
        def ops(a, b):
            return (
                Kd[(a, b, 0)] / rho,
                Kd[(a, b, 1)],
                -Kd[(a, b + 1, 0)],
                -rho * Kd[(a, b + 2, 0)],
            )

        O, Or, Oz = ops(0, 0), ops(1, 0), ops(0, 1)
        psi = self._psi(theta, nu)
        ct, st = mp.cos(theta), mp.sin(theta)
        phi, grad = {}, {}
        for j in "xywz":
            p = psi[j]
            f = sum(p[k][0] * O[k] for k in range(4))
            f_r = sum(p[k][0] * Or[k] for k in range(4))
            f_z = sum(p[k][0] * Oz[k] for k in range(4))
            f_t = sum(p[k][1] * O[k] for k in range(4))
            phi[j] = f
            grad[j] = (ct * f_r - st / r * f_t, st * f_r + ct / r * f_t, f_z)

        x, y = r * ct, r * st
        v = [
            -phi["xyz"[i]] + x * grad["x"][i] + y * grad["y"][i] + z * grad["z"][i]
            + rho * grad["w"][i]
            for i in range(3)
        ]
        # Omega = 1/2 curl(grad Pi - 2 phi) = -curl phi
        omega_z = -(grad["y"][0] - grad["x"][1])
        return v + [omega_z]

    def at(self, r, theta, z):
        """(v_x, v_y, v_z, Omega_z) of the complementary flow at (r, theta, z)."""
        r, theta, z = mp.mpf(r), mp.mpf(theta), mp.mpf(z)
        with mp.workdps(self.dps):
            cache = {}

            def f(nu, i):
                if nu not in cache:
                    cache[nu] = self._integrand(r, theta, z, nu)
                return cache[nu][i]

            breaks = _breaks(self.alpha, self.nu_max)
            return [
                +mp.quad(lambda nu: f(nu, i), breaks, method="gauss-legendre")
                for i in range(4)
            ]


def free_dipole(point_xyz, source_xyz, e):
    """Free-space dipole flow sigma/s^2 (3 (e . s_hat)^2 - 1) s_hat with sigma = 1."""
    s = [p - q for p, q in zip(point_xyz, source_xyz)]
    sn = mp.sqrt(sum(c**2 for c in s))
    sh = [c / sn for c in s]
    es = sum(a * b for a, b in zip(e, sh))
    return [(3 * es**2 - 1) * c / sn**2 for c in sh]
