"""Axial force dipole (orientation e = e_z, parallel to the wedge edge).

Every formula is transcribed from the paper:
  - Sec. "Complementary solution in FKL space / Axial force dipole"
    for Lambda_w, Lambda_w^dagger, Gamma_pm, Delta_pm, Z_x, Z_x^dagger;
  - Sec. "Swimmer hydrodynamics / Axial dipole" for V_r^(n), V_theta^(n),
    Omega_z^(n).
"""

import mpmath as mp

from .common import (
    NU_MAX,
    Velocities,
    assemble_integrands,
    check_geometry,
    deltas,
    gammas,
    integrate_velocities,
    to_mpf,
)


def coefficients(alpha, beta, nu):
    """Wedge coefficients of the axial dipole at spectral parameter nu.

    Returns a dict with Lambda_w, Lambda_w_dag, Z_x, Z_x_dag, Z_y, Z_y_dag.
    """
    sh, ch = mp.sinh, mp.cosh

    # Lambda_w = -Delta_z,  Lambda_w^dagger = -Delta_z^dagger, with
    # Delta_z = sh((pi-alpha)nu) sh(beta nu) csh(alpha nu),
    # Delta_z^dagger = ch((pi-alpha)nu) ch(beta nu) sch(alpha nu).
    Lambda_w = -sh((mp.pi - alpha) * nu) * sh(beta * nu) / sh(alpha * nu)
    Lambda_w_dag = -ch((mp.pi - alpha) * nu) * ch(beta * nu) / ch(alpha * nu)

    delta_plus, delta_minus = deltas(alpha, nu)
    gamma_plus, gamma_minus = gammas(alpha, beta, nu, delta_plus, delta_minus)

    pi_nu = mp.pi * nu
    Z_x = -gamma_minus * mp.sin(alpha) * ch(alpha * nu) * sh(pi_nu)
    Z_x_dag = gamma_plus * mp.sin(alpha) * sh(alpha * nu) * sh(pi_nu)

    # From  Z_x^dagger = -tan(alpha) th(alpha nu) Z_y  and
    #       Z_y^dagger = -cot(alpha) th(alpha nu) Z_x,
    # written without tan/cot so that alpha = pi/2 is regular:
    Z_y = -gamma_plus * mp.cos(alpha) * ch(alpha * nu) * sh(pi_nu)
    Z_y_dag = gamma_minus * mp.cos(alpha) * sh(alpha * nu) * sh(pi_nu)

    return dict(
        Lambda_w=Lambda_w,
        Lambda_w_dag=Lambda_w_dag,
        Z_x=Z_x,
        Z_x_dag=Z_x_dag,
        Z_y=Z_y,
        Z_y_dag=Z_y_dag,
    )


def kernel_coefficients(alpha, beta, nu):
    """V_r^(1,2), V_theta^(1,2), Omega_z^(1,2) of the axial dipole."""
    c = coefficients(alpha, beta, nu)
    Lw, Lwd = c["Lambda_w"], c["Lambda_w_dag"]
    Zx, Zxd, Zy, Zyd = c["Z_x"], c["Z_x_dag"], c["Z_y"], c["Z_y_dag"]
    cb, sb = mp.cos(beta), mp.sin(beta)
    f = 4 * nu**2 + 1

    V_r1 = 8 * Lw + 5 * f * (Zx * cb + Zy * sb)
    V_r2 = 8 * Lwd + 5 * f * (Zxd * cb + Zyd * sb)

    V_theta1 = 8 * nu * Lwd + f * ((nu * Zxd - Zy) * cb + (nu * Zyd + Zx) * sb)
    V_theta2 = 8 * nu * Lw + f * ((nu * Zx - Zyd) * cb + (nu * Zy + Zxd) * sb)

    Omega_z1 = f * ((2 * nu * Zxd + 3 * Zy) * cb + sb * (2 * nu * Zyd - 3 * Zx))
    Omega_z2 = f * ((2 * nu * Zx + 3 * Zyd) * cb + sb * (2 * nu * Zy - 3 * Zxd))

    return V_r1, V_r2, V_theta1, V_theta2, Omega_z1, Omega_z2


def integrands(alpha, beta, nu):
    """The three nu-integrands (radial, azimuthal, rotational), without prefactors."""
    return assemble_integrands(kernel_coefficients(alpha, beta, nu), beta, nu)


def velocities(alpha, beta, rho, sigma=1, nu_max=NU_MAX):
    """V_r, V_theta, Omega_z of an axial dipole at (rho, beta) in a wedge of half-angle alpha.

    Pass alpha and beta as mpmath numbers built at mp.dps >= 30 (e.g. mp.pi/3) or as
    strings for full accuracy.
    """
    alpha, beta = to_mpf(alpha, beta)
    check_geometry(alpha, beta)
    return integrate_velocities(
        lambda nu: integrands(alpha, beta, nu), alpha, beta, rho, sigma=sigma, nu_max=nu_max
    )


__all__ = ["coefficients", "kernel_coefficients", "integrands", "velocities", "Velocities"]
