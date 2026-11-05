"""Pinned values of the verification cases (rho = 0.5, sigma = 1, delta = pi/12).

alpha = pi/2 values are exact single-wall (Blake) results; alpha = pi/3 values were
produced by the solver and validated by tests/test_realspace.py. Any change of
these numbers means a change of the physics and must be investigated.
"""

import mpmath as mp
import pytest

from hydrowedge import axial, transverse

RHO = mp.mpf("0.5")
S3 = mp.sqrt(3)

# (alpha/pi, beta/pi) -> (V_r, V_theta, Omega_z)
AXIAL = {
    ("1/2", "0"): (-mp.mpf("1.5"), 0, 0),
    ("1/2", "1/6"): (-S3, 1, 0),
    ("1/3", "0"): (mp.mpf("-2.927451409084466283707"), 0, 0),
    ("1/3", "1/6"): (
        mp.mpf("-3.534438568960136405403"),
        mp.mpf("5.040592771963362335915"),
        mp.mpf("-0.6546504653868677010876"),
    ),
}
TRANSVERSE = {
    ("1/2", "0"): (
        mp.mpf("1.5") * (3 * mp.cos(mp.pi / 12) ** 2 - 1),  # -3/(8h^2) (1 - 3 cos^2 delta)
        mp.mpf("0.75"),
        mp.mpf("0.75"),
    ),
    ("1/2", "1/6"): (1 + S3 / 2, S3 - mp.mpf("0.5"), 4 / S3),
    ("1/3", "0"): (
        mp.mpf("3.809523819269801753128"),
        mp.mpf("1.71060841765526173809"),
        mp.mpf("0.7909899671162215554885"),
    ),
    ("1/3", "1/6"): (
        mp.mpf("1.329532544652432030579"),
        mp.mpf("5.830731422884942904176"),
        mp.mpf("7.228632848063349513824"),
    ),
}


def _angle(frac):
    num, _, den = frac.partition("/")
    return mp.pi * mp.mpf(num) / mp.mpf(den or 1)


def _check(got, expected):
    for g, e in zip((got.V_r, got.V_theta, got.Omega_z), expected):
        assert abs(g - e) < mp.mpf("1e-20") * (1 + abs(e))


@pytest.mark.parametrize("key", list(AXIAL))
def test_axial_values(key):
    _check(axial.velocities(_angle(key[0]), _angle(key[1]), RHO), AXIAL[key])


@pytest.mark.parametrize("key", list(TRANSVERSE))
def test_transverse_values(key):
    _check(
        transverse.velocities(_angle(key[0]), _angle(key[1]), mp.pi / 12, RHO),
        TRANSVERSE[key],
    )


def test_results_file_matches_solver():
    """results/velocities.csv agrees with the pinned values."""
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "velocities.csv"
    if not path.exists():
        pytest.skip("run run_cases.py first")
    table = {"axial": AXIAL, "transverse": TRANSVERSE}
    with path.open(newline="", encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    for row in rows:
        key = (row["alpha"].replace("pi", "1"), row["beta"].replace("pi", "1"))  # "pi/3" -> "1/3"
        expected = table[row["dipole"]][key]
        for col, e in zip(("V_r", "V_theta", "Omega_z"), expected):
            assert abs(mp.mpf(row[col]) - e) < 1e-13 * (1 + abs(e))
