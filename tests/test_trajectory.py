"""Trajectories (hydrowedge.trajectory) and the case runner (run_trajectories.py).

The planar-wall limit alpha = pi/2 has closed forms (Berke et al. 2008, checked
against the solver in test_single_wall.py): for a dipole inclined by t to the wall
at height h, the wall-normal velocity is -3/(8 h^2) (1 - 3 sin^2 t), the tangential
one 3/(8 h^2) sin(2t) and Omega = 3/(8 h^3) sin(t) cos(t). For the axial dipole
(t = 0) the wall is reached (h = d_min) at t = 8 (h0^3 - d_min^3) / (9 sigma).
"""

import csv
import json

import mpmath as mp
import numpy as np
import pytest

import run_trajectories as runner
from hydrowedge import _tabulate, transverse
from hydrowedge.trajectory import WALL_LIMIT, Profile, trajectory


def _planar_axial_profile():
    b = np.linspace(0, np.pi / 2, 60)
    g = {("r", "1"): -0.375 * np.cos(b), ("theta", "1"): 0.375 * np.sin(b)}
    return Profile("axial", np.pi / 2, "pi/2", 0.05, b, g)


def _berke_planar(beta, psi):
    """(V_r, V_theta, Omega_z) at rho = 1, alpha = pi/2, for lab heading psi = beta + delta.

    Wall x = 0, normal n = e_x into the fluid, tau = e_y: sin t = cos psi, cos t = sin psi.
    """
    h = np.cos(beta)
    st, ct = np.cos(psi), np.sin(psi)
    V_n = -3 / (8 * h**2) * (1 - 3 * st**2)
    V_t = 3 / (8 * h**2) * 2 * st * ct
    return (V_n * np.cos(beta) + V_t * np.sin(beta),
            -V_n * np.sin(beta) + V_t * np.cos(beta),
            3 / (8 * h**3) * st * ct)


@pytest.mark.parametrize("beta", ["0.3", "1.2"])
def test_axial_tabulation_planar_wall(beta):
    """The tabulated quantity f = V(rho = sigma = 1) agrees with the closed form."""
    b = mp.mpf(beta)
    f_r, f_t, _ = _tabulate.velocities("axial", mp.pi / 2, b)
    h = float(mp.cos(b))
    assert f_r == pytest.approx(-0.375 / h, rel=1e-12)
    assert f_t == pytest.approx(0.375 * float(mp.sin(b)) / h**2, rel=1e-12)


def test_transverse_delta_decomposition_is_exact():
    """f(delta) = f_cc cos^2 + f_sc sin cos + f_ss sin^2, from delta = 0, pi/2, pi/4."""
    alpha, beta, delta = mp.pi / 3, mp.mpf("0.4"), mp.mpf("2.2")
    g = _tabulate.g_at(("transverse", mp.nstr(alpha, 40), "0.4"))
    s = float(mp.sin(alpha - beta))
    c, sn = float(mp.cos(delta)), float(mp.sin(delta))
    w = {"cc": c * c, "sc": sn * c, "ss": sn * sn}
    v = transverse.velocities(alpha, beta, delta, rho=1).as_floats()
    for k, q in enumerate(("r", "theta", "omega")):
        f = sum(w[b] * g[q, b] for b in w) / s ** _tabulate.POWER[q]
        assert f == pytest.approx(v[k], rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("delta", [0.0, 0.4, np.pi / 2, 2.0])
def test_wall_limits_match_single_wall(delta):
    """WALL_LIMIT = lim g at the upper wall, from the closed forms at alpha = pi/2."""
    gap = 1e-7
    beta = np.pi / 2 - gap
    f = _berke_planar(beta, beta + delta)
    c, s = np.cos(delta), np.sin(delta)
    w = {"cc": c * c, "sc": s * c, "ss": s * s}
    for k, q in enumerate(("r", "theta", "omega")):
        g_wall = sum(w[b] * WALL_LIMIT[q, b] for b in w)
        assert f[k] * np.sin(gap) ** _tabulate.POWER[q] == pytest.approx(g_wall, abs=1e-6)


@pytest.mark.parametrize("beta0", [0.0, 0.7, -1.2])
def test_planar_wall_axial_trajectory(beta0):
    p = _planar_axial_profile()
    rho0, sigma, d_min = 10.0, 2.0, 1.0
    tr = trajectory(p, rho0, beta0, sigma=sigma, v0=0.5, d_min=d_min, t_max=1e6)
    h0 = rho0 * np.cos(beta0)
    assert tr.reason == "wall"
    assert np.allclose(tr.y, tr.y[0], atol=1e-6)  # no motion parallel to the wall
    assert tr.t[-1] == pytest.approx(8 * (h0**3 - d_min**3) / (9 * sigma), rel=1e-5)
    assert tr.z[-1] == pytest.approx(0.5 * tr.t[-1])
    assert np.isnan(tr.delta).all()


def test_axial_bisector_rays():
    p = _planar_axial_profile()
    assert trajectory(p, 10.0, 0.0, sigma=-1, rho_max=30, t_max=1e5).reason == "far"
    q = Profile("axial", np.pi / 3, "pi/3", 0.05, p.beta * 2 / 3, p.g)  # any profile, g_theta(0) = 0
    tr = trajectory(q, 10.0, 0.0, sigma=1, d_min=0.1)
    assert tr.reason == "apex" and np.allclose(tr.beta, 0)


def test_transverse_free_swimming_is_straight():
    """sigma = 0: the equations of motion must give straight-line motion at speed v0 with fixed heading."""
    b = np.linspace(0, np.pi / 3, 30)
    g = {(q, k): np.ones_like(b) for q in ("r", "theta", "omega") for k in ("cc", "sc", "ss")}
    p = Profile("transverse", np.pi / 3, "pi/3", 0.1, b, g)
    tr = trajectory(p, 10.0, 0.2, 1.1, sigma=0.0, v0=0.7, t_max=4.0)
    psi = 0.2 + 1.1
    assert tr.reason == "t_max"
    assert np.allclose(tr.heading, psi, atol=1e-9)
    assert tr.x[-1] - tr.x[0] == pytest.approx(0.7 * 4 * np.cos(psi), rel=1e-8)
    assert tr.y[-1] - tr.y[0] == pytest.approx(0.7 * 4 * np.sin(psi), rel=1e-8)
    assert np.all(tr.z == 0)


def test_start_inside_cutoff_rejected():
    with pytest.raises(ValueError):
        trajectory(_planar_axial_profile(), 10.0, 1.5, d_min=1.0)  # h0 = 0.71


def test_profile_save_load_roundtrip(tmp_path):
    p = _planar_axial_profile()
    p.save(tmp_path / "p.csv")
    q = Profile.load(tmp_path / "p.csv")
    assert (q.dipole, q.alpha, q.alpha_label, q.gap_min) == (p.dipole, p.alpha, p.alpha_label, p.gap_min)
    assert np.allclose(q.f(0.4), p.f(0.4), rtol=0, atol=1e-15)


def test_runner_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "cases")
    monkeypatch.setattr(runner, "PROFILES", tmp_path / "profiles")
    (tmp_path / "profiles").mkdir()
    _planar_axial_profile().save(tmp_path / "profiles" / "axial_alpha=pi_2_n20_gap0.05.csv")
    case = {"name": "demo", "dipole": "axial", "alpha": "pi/2", "sigma": [1, -1],
            "rho_max": 20, "checkpoints": [10], "starts": [{"rho0": 10, "beta0": "pi/6"}]}
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(case))

    runner.main([str(path)])
    out = tmp_path / "cases" / "demo"
    files = {f.name: f.stat().st_mtime_ns for f in out.iterdir()}
    assert {"case.json", "summary.csv", "checkpoints.csv", "trajectories.csv", "paths.png"} <= set(files)

    runner.main([str(path)])  # same parameters: kept as they are
    assert {f.name: f.stat().st_mtime_ns for f in out.iterdir()} == files

    path.write_text(json.dumps({**case, "sigma": 2}))
    with pytest.raises(SystemExit):
        runner.main([str(path)])  # other parameters under the same name: refused
    assert {f.name: f.stat().st_mtime_ns for f in out.iterdir()} == files


def test_wall_relative_starts():
    alpha = mp.pi / 3
    with mp.workdps(30):
        rho0, beta0, delta0 = runner.resolve_start({"rho0": 12, "d0": 2, "along": "outward"}, alpha)
        assert rho0 * np.sin(float(alpha) - beta0) == pytest.approx(2)
        assert beta0 + delta0 == pytest.approx(float(alpha))  # parallel to the upper wall
        _, b, d = runner.resolve_start({"rho0": 12, "d0": 2, "along": "apexward", "tilt": "pi/12"}, alpha)
        assert b + d == pytest.approx(float(alpha + mp.pi - mp.pi / 12))
        _, b, d = runner.resolve_start(
            {"rho0": 12, "d0": 2, "wall": "lower", "along": "outward", "tilt": "pi/12"}, alpha)
        assert b == pytest.approx(-beta0)  # mirror image of the upper-wall start
        assert b + d == pytest.approx(-float(alpha + mp.pi / 12))


def test_scan_without_hydrodynamics(tmp_path, monkeypatch):
    """g = 0: straight paths. Heading outward from the upper wall with tilt > 0 hits it,
    tilt < 0 points into the wedge's opening (|heading| < alpha) and escapes."""
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "cases")
    monkeypatch.setattr(runner, "PROFILES", tmp_path / "profiles")
    (tmp_path / "profiles").mkdir()
    b = np.linspace(0, np.pi / 3, 30)
    g = {(q, k): np.zeros_like(b) for q in ("r", "theta", "omega") for k in ("cc", "sc", "ss")}
    Profile("transverse", np.pi / 3, "pi/3", 0.1, b, g).save(
        tmp_path / "profiles" / "transverse_alpha=pi_3_n20_gap0.1.csv")
    case = {"kind": "scan", "dipole": "transverse", "alpha": "pi/3", "sigma": 1, "rho_max": 60,
            "t_max": 1000, "start": {"rho0": 12, "d0": 2, "along": "outward"},
            "x": {"param": "tilt", "from": "-pi/4", "to": "pi/4", "n": 5},
            "y": {"param": "sigma", "from": -1, "to": 1, "n": 2}}
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(case))
    runner.main([str(path)])
    rows = list(csv.DictReader(open(tmp_path / "cases" / "scan" / "scan.csv")))
    for r in rows:
        tilt = float(r["tilt"])
        if abs(tilt) > 1e-9:
            assert r["outcome"] == ("same wall" if tilt > 0 else "escapes")
    assert (tmp_path / "cases" / "scan" / "diagram.pdf").exists()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:.*truncation")  # reference values good to ~1e-16
def test_axial_spline_matches_solver_off_nodes():
    alpha = mp.pi / 3
    p = Profile.compute("axial", alpha)  # the grid used by run_trajectories.py
    for beta in ["0.013", "0.4", "0.93"]:
        b = mp.mpf(beta)
        v = _tabulate.velocities("axial", alpha, b)
        f_r, f_t, _ = p.f(float(b))
        scale = abs(v[0]) + abs(v[1])
        assert f_r == pytest.approx(v[0], abs=1e-5 * scale)
        assert f_t == pytest.approx(v[1], abs=1e-5 * scale)


@pytest.mark.slow
def test_transverse_spline_matches_solver():
    """Off the grid (0.5 rad from the wall) and between the last grid point and the
    wall (0.05 rad). Reference: g = f sin^p(gap) from transverse.velocities at rho = 1,
    nu_max = 601 (evaluated 2026-10-24; ~3 min, hence stored)."""
    p = Profile.compute("transverse", mp.pi / 3)
    ref = {
        0.5: {0.0: (0.254091, 0.253983, 0.165183), np.pi / 2: (-0.0455818, -0.575167, -0.156056),
              np.pi / 4: (-0.237415, 0.174602, -0.0749422)},
        0.05: {0.0: (0.0187947, 0.373596, 0.0187215), np.pi / 2: (-4.74797e-5, -0.748127, -0.0187191),
               np.pi / 4: (-0.365982, -0.149825, -0.186508)},
    }
    for gap, by_delta in ref.items():
        tol = 2e-6 if gap > 0.1 else 1e-4
        for delta, g_ref in by_delta.items():
            f = p.f(p.alpha - gap, delta)
            g = [fk * np.sin(gap) ** pw for fk, pw in zip(f, (2, 2, 3))]
            assert g == pytest.approx(g_ref, abs=tol)


def test_repulsion_off_is_unchanged():
    """repulsion=None must reproduce the model of the equations of motion exactly."""
    p = _planar_axial_profile()
    a = trajectory(p, 10.0, 0.3, sigma=1, d_min=1.0)
    b = trajectory(p, 10.0, 0.3, sigma=1, d_min=1.0, repulsion=None)
    assert a.reason == b.reason and np.array_equal(a.rho, b.rho) and np.array_equal(a.beta, b.beta)


def test_repulsion_balances_swimming_into_a_wall():
    """sigma = 0, heading straight into the planar wall: the swimmer stops where
    U ((d_r/d)^n - 1) = v0, keeps its heading and does not slide."""
    b = np.linspace(0, np.pi / 2, 30)
    g = {(q, k): np.ones_like(b) for q in ("r", "theta", "omega") for k in ("cc", "sc", "ss")}
    p = Profile("transverse", np.pi / 2, "pi/2", 0.1, b, g)
    rep = dict(strength=0.5, range=1.0, power=4)
    x0, y0 = 5.0, 8.0  # the wall x = 0, far from the edge at the origin
    beta0 = np.arctan2(y0, x0)
    tr = trajectory(p, np.hypot(x0, y0), beta0, np.pi - beta0, sigma=0.0, d_min=0.1, t_max=50.0,
                    repulsion=rep)
    d_eq = (1 + 1 / 0.5) ** -0.25
    assert tr.reason == "t_max"
    assert tr.x[-1] == pytest.approx(d_eq, rel=1e-6) and tr.y[-1] == pytest.approx(y0, rel=1e-9)
    assert np.allclose(tr.heading, np.pi, atol=1e-9)


def test_repulsion_from_both_walls_is_mirror_symmetric():
    from hydrowedge.trajectory import repulsion_velocity
    up = repulsion_velocity(2.0, 0.4, np.pi / 3, 1.0, 1.5, 4)
    lo = repulsion_velocity(2.0, -0.4, np.pi / 3, 1.0, 1.5, 4)
    assert up[0] == pytest.approx(lo[0]) and up[1] == pytest.approx(-lo[1])
    assert up[1] < 0  # pushed away from the upper wall, towards smaller beta


def test_repulsion_needs_range_beyond_d_min():
    with pytest.raises(ValueError):
        trajectory(_planar_axial_profile(), 10.0, 0.3, d_min=1.0,
                   repulsion=dict(strength=1, range=1.0, power=4))
