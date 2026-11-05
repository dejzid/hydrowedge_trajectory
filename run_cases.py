"""Compute V_r, V_theta, Omega_z for the verification cases and write them to results/.

Cases: alpha in {pi/2, pi/3}, beta in {0, pi/6}, rho = 0.5, sigma = 1, nu_max = 100;
transverse dipole with delta = pi/12.

Usage:  python run_cases.py
"""

import csv
from pathlib import Path

import mpmath as mp

from hydrowedge import NU_MAX, axial, transverse

RHO = "0.5"
DIGITS = 15

# (label, angle). The angles are built when called, i.e. inside workdps(30) below,
# not at import time at mpmath's default 15 digits.
ALPHAS = [("pi/2", lambda: mp.pi / 2), ("pi/3", lambda: mp.pi / 3)]
BETAS = [("0", lambda: mp.mpf(0)), ("pi/6", lambda: mp.pi / 6)]
DELTA = ("pi/12", lambda: mp.pi / 12)


def main():
    rows = []
    with mp.workdps(30):
        for a_lbl, a in ALPHAS:
            for b_lbl, b in BETAS:
                v = axial.velocities(a(), b(), RHO, nu_max=NU_MAX)
                rows.append(("axial", a_lbl, b_lbl, "-", v))
                v = transverse.velocities(a(), b(), DELTA[1](), RHO, nu_max=NU_MAX)
                rows.append(("transverse", a_lbl, b_lbl, DELTA[0], v))

    def fmt(x):
        # Values below 1e-25 are zero to working precision.
        return "0" if abs(x) < mp.mpf("1e-25") else mp.nstr(x, DIGITS)

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    header = ["dipole", "alpha", "beta", "delta", "rho", "V_r", "V_theta", "Omega_z"]
    table = [
        [d, a, b, dl, RHO, fmt(v.V_r), fmt(v.V_theta), fmt(v.Omega_z)]
        for d, a, b, dl, v in rows
    ]

    with open(out / "velocities.csv", "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)

    widths = [max(len(str(r[i])) for r in [header] + table) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    md = [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    md += [line(r) for r in table]
    text = "\n".join(md)
    (out / "velocities.md").write_text(
        f"sigma = 1, nu_max = {NU_MAX}, {DIGITS} significant digits\n\n{text}\n",
        encoding="ascii",
    )
    print(text)


if __name__ == "__main__":
    main()
