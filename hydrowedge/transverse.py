"""Transverse force dipole (orientation e = cos(delta) e_rho + sin(delta) e_beta).

Every formula is transcribed from the paper:
  - Sec. "Complementary solution in FKL space / Transverse force dipole"
    for Z_j, Lambda_w, H_w, Lambda_j, H_j (and daggers);
  - Appendix A for b_{j,n}, p_{j,n}, c_{j,n}, q_{j,n} (and daggers);
  - Sec. "Swimmer hydrodynamics / Transverse dipole" for A, B, C coefficients.
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


def _bc_sums(n0, n1, n2, n3, n4, hyp):
    """X_0 + X_1 sh(2 a nu) + X_2 ch(2 a nu) + X_3 sh(4 a nu) + X_4 ch(4 a nu)  (Eq. B_j_C_j).

    hyp = (sh(2 a nu), ch(2 a nu), sh(4 a nu), ch(4 a nu)).
    """
    sh2, ch2, sh4, ch4 = hyp
    return n0 + n1 * sh2 + n2 * ch2 + n3 * sh4 + n4 * ch4


def _pq_sums(n0, n1, n2, hyp):
    """X_0 + X_1 sh(2 a nu) + X_2 ch(2 a nu)  (Eq. P_j_Q_j)."""
    sh2, ch2 = hyp[:2]
    return n0 + n1 * sh2 + n2 * ch2


def coefficients(alpha, beta, delta, nu):
    """All wedge coefficients of the transverse dipole at spectral parameter nu.

    Returns a dict with keys Z_x, Z_y, Lambda_x, Lambda_y, H_x, H_y, Lambda_w, H_w
    and their '_dag' counterparts.
    """
    sh, ch = mp.sinh, mp.cosh
    sin, cos = mp.sin, mp.cos
    a, b, d = alpha, beta, delta

    shb, chb = sh(b * nu), ch(b * nu)
    shp, chp = sh(mp.pi * nu), ch(mp.pi * nu)
    thp = mp.tanh(mp.pi * nu)
    sa, ca = sin(a), cos(a)
    s2a, c2a = sin(2 * a), cos(2 * a)
    sd, cd = sin(d), cos(d)
    sb, cb = sin(b), cos(b)
    s2d, c2d = sin(2 * d), cos(2 * d)

    # Abbreviations of Appendix A
    s_p, s_m = sin(b + d), sin(b - d)
    c_p, c_m = cos(b + d), cos(b - d)

    delta_plus, delta_minus = deltas(a, nu)
    gamma_plus, gamma_minus = gammas(a, b, nu, delta_plus, delta_minus)

    # --- Z coefficients -------------------------------------------------------
    Z_x = gamma_minus * cd**2 * sa * ch(a * nu) * shp
    Z_y = gamma_plus * cd**2 * ca * ch(a * nu) * shp
    # Z_x^dagger = -tan(a) th(a nu) Z_y,  Z_y^dagger = -cot(a) th(a nu) Z_x,
    # written without tan/cot so that alpha = pi/2 is regular:
    Z_x_dag = -gamma_plus * cd**2 * sa * sh(a * nu) * shp
    Z_y_dag = -gamma_minus * cd**2 * ca * sh(a * nu) * shp

    # --- w-component ----------------------------------------------------------
    sdcd = sd * cd
    Lambda_w = -(shb + nu * sdcd * chb) * sh((mp.pi - a) * nu) / sh(a * nu)
    Lambda_w_dag = -(chb + nu * sdcd * shb) * ch((mp.pi - a) * nu) / ch(a * nu)
    H_w = -(cd**2) * shb * sh((mp.pi - a) * nu) / sh(a * nu)
    H_w_dag = -(cd**2) * chb * ch((mp.pi - a) * nu) / ch(a * nu)

    # --- b_{j,n} --------------------------------------------------------------
    bx0_br = (4 * c2a - 3) * cd * cb - sd * sb
    by0_br = (4 * c2a + 3) * cd * sb - sd * cb
    b_x0 = -sd / 2 * chb * shp * bx0_br
    b_x0_dag = sd / 2 * shb * shp * bx0_br
    b_y0 = sd / 2 * chb * shp * by0_br
    b_y0_dag = -sd / 2 * shb * shp * by0_br

    b_x1 = sd * (c_p * c2a * chb * chp - s_m * s2a * shb * shp)
    b_x1_dag = sd * (c_p * c2a * shb * chp - s_m * s2a * chb * shp)
    b_y1 = -sd * (s_p * c2a * chb * chp + c_m * s2a * shb * shp)
    b_y1_dag = -sd * (s_p * c2a * shb * chp + c_m * s2a * chb * shp)

    b_x2 = 2 * s2d * cb * sa**2 * chb * shp
    b_x2_dag = 2 * s2d * cb * sa**2 * shb * shp
    b_y2 = 2 * s2d * sb * ca**2 * chb * shp
    b_y2_dag = 2 * s2d * sb * ca**2 * shb * shp

    b_x3 = -c_p * sd / 2 * chb * chp
    b_x3_dag = c_p * sd / 2 * shb * chp
    b_y3 = -s_p * sd / 2 * chb * chp
    b_y3_dag = s_p * sd / 2 * shb * chp

    b_x4, b_x4_dag = -thp * b_x3, -thp * b_x3_dag
    b_y4, b_y4_dag = -thp * b_y3, -thp * b_y3_dag

    # --- p_{j,n} --------------------------------------------------------------
    p_x0 = 2 * sa * (
        c_m * cd * sa * shb * shp
        + sd * c2a * (c_p * ca * chb * chp + s_p * sa * shb * shp)
    )
    p_x0_dag = -2 * sa * (
        c_m * cd * sa * chb * shp
        + sd * c2a * (c_p * ca * shb * chp + s_p * sa * chb * shp)
    )
    p_y0 = 2 * ca * (
        s_m * cd * ca * shb * shp
        + sd * c2a * (s_p * sa * chb * chp + c_p * ca * shb * shp)
    )
    p_y0_dag = -2 * ca * (
        s_m * cd * ca * chb * shp
        + sd * c2a * (s_p * sa * shb * chp + c_p * ca * chb * shp)
    )

    p_x1 = -s_m * cd * s2a * chb * shp
    p_x1_dag = -s_m * cd * s2a * shb * shp
    p_y1 = -c_m * cd * s2a * chb * shp
    p_y1_dag = -c_m * cd * s2a * shb * shp

    p_x2 = 2 * sa * (c2d * cb * sa * shb * shp - c_p * sd * ca * chb * chp)
    p_x2_dag = 2 * sa * (c2d * cb * sa * chb * shp - c_p * sd * ca * shb * chp)
    p_y2 = 2 * ca * (c2d * sb * ca * shb * shp + s_p * sd * sa * chb * chp)
    p_y2_dag = 2 * ca * (c2d * sb * ca * shp * chb + s_p * sd * sa * shb * chp)

    # --- c_{j,n} --------------------------------------------------------------
    cx0_br = cb * ((4 * c2a - 3) * c2d + 1) - s2d * sb
    cy0_br = sb * ((4 * c2a + 3) * c2d - 1) - s2d * cb
    c_x0 = -shb * shp / 4 * cx0_br
    c_x0_dag = chb * shp / 4 * cx0_br
    c_y0 = shb * shp / 4 * cy0_br
    c_y0_dag = -chb * shp / 4 * cy0_br

    c_x1 = c_p * cd * c2a * shb * chp + c_m * sd * s2a * chb * shp
    c_x1_dag = c_p * cd * c2a * chb * chp + c_m * sd * s2a * shb * shp
    c_y1 = -s_p * cd * c2a * shb * chp - s_m * sd * s2a * chb * shp
    c_y1_dag = -s_p * cd * c2a * chb * chp - s_m * sd * s2a * shb * shp

    c_x2 = 2 * c2d * cb * sa**2 * shb * shp
    c_x2_dag = 2 * c2d * cb * sa**2 * chb * shp
    c_y2 = 2 * c2d * sb * ca**2 * shb * shp
    c_y2_dag = 2 * c2d * sb * ca**2 * chb * shp

    c_x3 = -c_p * cd / 2 * shb * chp
    c_x3_dag = c_p * cd / 2 * chb * chp
    c_y3 = -s_p * cd / 2 * shb * chp
    c_y3_dag = s_p * cd / 2 * chb * chp

    c_x4, c_x4_dag = -thp * c_x3, -thp * c_x3_dag
    c_y4, c_y4_dag = -thp * c_y3, -thp * c_y3_dag

    # --- q_{j,n} --------------------------------------------------------------
    q_x0 = 2 * cd * sa * (
        c_p * ca * c2a * shb * chp + sa * chb * shp * (s_p * c2a + s_m)
    )
    q_x0_dag = -2 * sa * cd * (
        c_p * ca * c2a * chb * chp + sa * shb * shp * (s_p * c2a + s_m)
    )
    q_y0 = 2 * cd * ca * (
        s_p * sa * c2a * shb * chp + ca * chb * shp * (c_p * c2a - c_m)
    )
    q_y0_dag = -2 * cd * ca * (
        s_p * sa * c2a * chb * chp + ca * shb * shp * (c_p * c2a - c_m)
    )

    q_x1 = c_m * cd * s2a * shb * shp
    q_x1_dag = c_m * cd * s2a * chb * shp
    q_y1 = -s_m * cd * s2a * shb * shp
    q_y1_dag = -s_m * cd * s2a * chb * shp

    q_x2 = -2 * cd * sa * (c_p * ca * shb * chp + 2 * sd * cb * sa * chb * shp)
    q_x2_dag = -2 * cd * sa * (c_p * ca * chb * chp + 2 * sd * cb * sa * shb * shp)
    q_y2 = 2 * cd * ca * (s_p * sa * shb * chp - 2 * sd * sb * ca * chb * shp)
    q_y2_dag = 2 * cd * ca * (s_p * sa * chb * chp - 2 * sd * sb * ca * shb * shp)

    # --- composite B, C, P, Q -------------------------------------------------
    hyp = (sh(2 * a * nu), ch(2 * a * nu), sh(4 * a * nu), ch(4 * a * nu))
    B_x = _bc_sums(b_x0, b_x1, b_x2, b_x3, b_x4, hyp)
    B_y = _bc_sums(b_y0, b_y1, b_y2, b_y3, b_y4, hyp)
    B_x_dag = _bc_sums(b_x0_dag, b_x1_dag, b_x2_dag, b_x3_dag, b_x4_dag, hyp)
    B_y_dag = _bc_sums(b_y0_dag, b_y1_dag, b_y2_dag, b_y3_dag, b_y4_dag, hyp)
    C_x = _bc_sums(c_x0, c_x1, c_x2, c_x3, c_x4, hyp)
    C_y = _bc_sums(c_y0, c_y1, c_y2, c_y3, c_y4, hyp)
    C_x_dag = _bc_sums(c_x0_dag, c_x1_dag, c_x2_dag, c_x3_dag, c_x4_dag, hyp)
    C_y_dag = _bc_sums(c_y0_dag, c_y1_dag, c_y2_dag, c_y3_dag, c_y4_dag, hyp)

    P_x = _pq_sums(p_x0, p_x1, p_x2, hyp)
    P_y = _pq_sums(p_y0, p_y1, p_y2, hyp)
    P_x_dag = _pq_sums(p_x0_dag, p_x1_dag, p_x2_dag, hyp)
    P_y_dag = _pq_sums(p_y0_dag, p_y1_dag, p_y2_dag, hyp)
    Q_x = _pq_sums(q_x0, q_x1, q_x2, hyp)
    Q_y = _pq_sums(q_y0, q_y1, q_y2, hyp)
    Q_x_dag = _pq_sums(q_x0_dag, q_x1_dag, q_x2_dag, hyp)
    Q_y_dag = _pq_sums(q_y0_dag, q_y1_dag, q_y2_dag, hyp)

    # --- Lambda_j, H_j (Eq. coeffs_appendix) ----------------------------------
    # delta_sigma = Delta_- for (x, non-dagger) and (y, dagger),
    #             = Delta_+ for (y, non-dagger) and (x, dagger).
    Lambda_x = nu / delta_minus * (B_x + nu * P_x)
    Lambda_y = nu / delta_plus * (B_y + nu * P_y)
    Lambda_x_dag = nu / delta_plus * (B_x_dag + nu * P_x_dag)
    Lambda_y_dag = nu / delta_minus * (B_y_dag + nu * P_y_dag)
    H_x = (C_x + nu * Q_x) / delta_minus
    H_y = (C_y + nu * Q_y) / delta_plus
    H_x_dag = (C_x_dag + nu * Q_x_dag) / delta_plus
    H_y_dag = (C_y_dag + nu * Q_y_dag) / delta_minus

    return dict(
        Z_x=Z_x, Z_y=Z_y, Z_x_dag=Z_x_dag, Z_y_dag=Z_y_dag,
        Lambda_w=Lambda_w, Lambda_w_dag=Lambda_w_dag, H_w=H_w, H_w_dag=H_w_dag,
        Lambda_x=Lambda_x, Lambda_y=Lambda_y,
        Lambda_x_dag=Lambda_x_dag, Lambda_y_dag=Lambda_y_dag,
        H_x=H_x, H_y=H_y, H_x_dag=H_x_dag, H_y_dag=H_y_dag,
    )


def kernel_coefficients(alpha, beta, delta, nu):
    """V_r^(1,2), V_theta^(1,2), Omega_z^(1,2) of the transverse dipole."""
    c = coefficients(alpha, beta, delta, nu)
    Lx, Ly, Lxd, Lyd = c["Lambda_x"], c["Lambda_y"], c["Lambda_x_dag"], c["Lambda_y_dag"]
    Hx, Hy, Hxd, Hyd = c["H_x"], c["H_y"], c["H_x_dag"], c["H_y_dag"]
    Zx, Zy, Zxd, Zyd = c["Z_x"], c["Z_y"], c["Z_x_dag"], c["Z_y_dag"]
    Lw, Lwd, Hw, Hwd = c["Lambda_w"], c["Lambda_w_dag"], c["H_w"], c["H_w_dag"]
    n = nu

    # Radial
    A_r1 = 24 * Ly + 5 * (4 * n**2 + 1) * Zy - 2 * (4 * n**2 + 7) * Hy
    B_r1 = 24 * Lx + 5 * (4 * n**2 + 1) * Zx - 2 * (4 * n**2 + 7) * Hx
    A_r2 = 24 * Lyd + 5 * (4 * n**2 + 1) * Zyd - 2 * (4 * n**2 + 7) * Hyd
    B_r2 = 24 * Lxd + 5 * (4 * n**2 + 1) * Zxd - 2 * (4 * n**2 + 7) * Hxd
    C_r1 = 8 * Lw - 2 * (4 * n**2 + 3) * Hw
    C_r2 = 8 * Lwd - 2 * (4 * n**2 + 3) * Hwd

    # Azimuthal
    A_t1 = (4 * n**3 * Zyd + 4 * n**2 * Zx + n * (8 * Lyd + Zyd - 4 * Hyd)
            + 8 * Lx + Zx - 4 * Hx)
    B_t1 = (4 * n**3 * Zxd - 4 * n**2 * Zy + n * (8 * Lxd + Zxd - 4 * Hxd)
            - 8 * Ly - Zy + 4 * Hy)
    A_t2 = (4 * n**3 * Zy + 4 * n**2 * Zxd + n * (8 * Ly + Zy - 4 * Hy)
            + 8 * Lxd + Zxd - 4 * Hxd)
    B_t2 = (4 * n**3 * Zx - 4 * n**2 * Zyd + n * (8 * Lx + Zx - 4 * Hx)
            - 8 * Lyd - Zyd + 4 * Hyd)
    C_t1 = 4 * n * (2 * Lwd - Hwd)
    C_t2 = 4 * n * (2 * Lw - Hw)

    # Rotational
    A_z1 = (8 * n**3 * Zyd + 4 * n**2 * (2 * Hx - 3 * Zx)
            + 2 * n * (8 * Lyd + Zyd - 4 * Hyd) + 6 * Hx - 3 * Zx - 8 * Lx)
    B_z1 = (8 * n**3 * Zxd + 4 * n**2 * (3 * Zy - 2 * Hy)
            + 2 * n * (8 * Lxd + Zxd - 4 * Hxd) + 8 * Ly + 3 * Zy - 6 * Hy)
    A_z2 = (8 * n**3 * Zy + 4 * n**2 * (2 * Hxd - 3 * Zxd)
            + 2 * n * (8 * Ly + Zy - 4 * Hy) + 6 * Hxd - 3 * Zxd - 8 * Lxd)
    B_z2 = (8 * n**3 * Zx + 4 * n**2 * (3 * Zyd - 2 * Hyd)
            + 2 * n * (8 * Lx + Zx - 4 * Hx) + 8 * Lyd + 3 * Zyd - 6 * Hyd)

    sb, cb = mp.sin(beta), mp.cos(beta)
    V_r1 = A_r1 * sb + B_r1 * cb + C_r1
    V_r2 = A_r2 * sb + B_r2 * cb + C_r2
    V_theta1 = A_t1 * sb + B_t1 * cb + C_t1
    V_theta2 = A_t2 * sb + B_t2 * cb + C_t2
    Omega_z1 = A_z1 * sb + B_z1 * cb
    Omega_z2 = A_z2 * sb + B_z2 * cb
    return V_r1, V_r2, V_theta1, V_theta2, Omega_z1, Omega_z2


def integrands(alpha, beta, delta, nu):
    """The three nu-integrands (radial, azimuthal, rotational), without prefactors."""
    return assemble_integrands(kernel_coefficients(alpha, beta, delta, nu), beta, nu)


def velocities(alpha, beta, delta, rho, sigma=1, nu_max=NU_MAX):
    """V_r, V_theta, Omega_z of a transverse dipole at (rho, beta) in a wedge of half-angle alpha.

    delta is the in-plane orientation angle measured from e_rho.
    Pass alpha, beta, delta as mpmath numbers built at mp.dps >= 30 (e.g. mp.pi/3) or
    as strings for full accuracy.
    """
    alpha, beta, delta = to_mpf(alpha, beta, delta)
    check_geometry(alpha, beta)
    return integrate_velocities(
        lambda nu: integrands(alpha, beta, delta, nu),
        alpha, beta, rho, sigma=sigma, nu_max=nu_max,
    )


__all__ = ["coefficients", "kernel_coefficients", "integrands", "velocities", "Velocities"]
