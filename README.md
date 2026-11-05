# hydrowedge — force-dipole microswimmers in a wedge

Numerical code accompanying the paper

> A. Wills, D. Zidanek, M. J. Mitchell, A. M. Menzel and A. Daddi-Moussa-Ider,
> *Hydrodynamics of active self-propulsion under wedge-shaped confinements* (2026).
> <!-- Add journal and doi once published. -->

It evaluates the wall-induced velocities V_r, V_θ and rotation rate Ω_z of a force-dipole
swimmer inside a wedge of half-angle α (axial and transverse dipole orientations), and
integrates the swimmer's equations of motion to give the trajectories shown in the paper.

If you use this code, please cite the paper and the archived software version
(see [Citation](#citation)).

## Installation

Python 3.11 or newer (the published results were computed with Python 3.14 and the exact
package versions in `requirements.txt`).

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows
# .venv/bin/python -m pip install -r requirements.txt       # Linux / macOS
```

The paper figures are typeset with LaTeX (matplotlib `text.usetex`), so a LaTeX
installation (TeX Live or MiKTeX, with `dvipng`) is needed for `make_paper_figures.py`.
Nothing else needs LaTeX.

## Reproducing the paper figures

The repository already contains the results behind every figure, so the figures can be
redrawn directly:

```
python make_paper_figures.py --force     # results/paper/fig_*.png
```

To recompute the trajectories themselves, delete the case folders and run the cases
again (results are never overwritten, see `cases/README.md`):

```
python run_trajectories.py cases/axial_pi3_paper_round.json cases/transverse_pi3_paper_theta_s10.json
python make_paper_figures.py --force
```

This takes about a minute with the cached velocity profiles in `results/profiles/`.
Deleting those as well recomputes the profiles from the full solution (~1 min axial,
~1.5 min transverse, on 8 processes).

| figure | case file | parameters |
|---|---|---|
| `fig_axial_{pusher,puller}_s1.png` | `axial_pi3_paper_round.json` | α = π/3, σ/(v₀ℓ²) = ±1, ρ₀ = 10ℓ, β₀ = 0, ±π/12, ±π/6, ±π/4 |
| `fig_transverse_{pusher,puller}_s10.png` | `transverse_pi3_paper_theta_s10.json` | α = π/3, σ/(v₀ℓ²) = ±10, ρ₀ = 10ℓ, β₀ = π/6, ϑ = −3π/4, −π/2, …, π |

`results/paper/README.md` summarises what the figures show.

## Usage

```python
import mpmath as mp
from hydrowedge import axial, transverse
mp.mp.dps = 30                                   # so that mp.pi/3 is accurate
axial.velocities(mp.pi/3, mp.pi/6, rho=0.5)
transverse.velocities(mp.pi/3, mp.pi/6, delta=mp.pi/12, rho=0.5)

from hydrowedge.trajectory import Profile, trajectory
p = Profile.compute("transverse", mp.pi/3, "pi/3")   # ~1.5 min; or Profile.load(csv)
tr = trajectory(p, rho0=12, beta0=0.8, delta0=0.25, sigma=3, v0=1, d_min=1)
tr.reason, tr.x, tr.y, tr.heading                    # "wall" / "apex" / "far" / "t_max"
```
(`Profile.compute` uses worker processes, so call it under `if __name__ == "__main__":`
in a script.)

Trajectories are normally run from JSON case files: `python run_trajectories.py
cases/<name>.json` writes `results/cases/<name>/` (paths, end states, checkpoint tables
and a quick plot). `cases/README.md` describes the case format and the outputs.

## Model and units

- Lengths are in units of the swimmer size ℓ, velocities in units of the swimming speed
  v₀ (both 1 in the code), so the dipole strength enters as σ/(v₀ℓ²); σ > 0 is a pusher,
  σ < 0 a puller.
- Axial dipole (along the edge): ρ' = V_r, β' = V_θ/ρ, z' = v₀. The path in the
  cross-section depends only on α, the sign of σ and the start; the distance along the
  edge scales as v₀ρ₀³/σ.
- Transverse dipole (in the cross-section): ρ' = v₀ cos δ + V_r,
  β' = (v₀ sin δ + V_θ)/ρ, δ' = Ω_z − β', with heading ϑ = β + δ from the x-axis.
- V = σ f(β, δ)/ρ² and Ω_z = σ f_ω/ρ³, so f is tabulated once per (dipole, α) with the
  full solution (`Profile`) and splined; the dependence on δ is exactly quadratic. Near a
  wall the spline uses the exact single-wall limit (Berke et al. 2008). Spline vs direct
  solution: ~1e-6 on the grid range, ~1e-4 between the last grid point and the wall.
- The far-field description fails close to a wall, so a trajectory stops at wall
  distance `d_min` (default one swimmer size), at `rho_min`, `rho_max` or `t_max`.
- An optional soft wall repulsion (off by default, not used in the paper) can be enabled
  per case; see `cases/README.md`.

## Repository layout

- `hydrowedge/` — the solver: `axial.py`, `transverse.py` (coefficients and integrands),
  `common.py` (shared functions, adaptive-precision quadrature), `trajectory.py`
  (equations of motion), `_tabulate.py` (profile tabulation).
- `run_trajectories.py`, `cases/` — trajectory cases (JSON) and their runner.
- `make_paper_figures.py` — the paper figures, from stored case results.
- `run_cases.py` — velocities at eight reference configurations
  (`results/velocities.{csv,md}`).
- `results/` — `profiles/` (cached velocity profiles), `cases/<name>/` (one folder per
  case), `paper/` (figures and their summary), `velocities.{csv,md}`.
- `tests/` — see below.

## Numerics

The ν-integrals contain terms of size ~exp(4πν) that cancel, so the integrands are
evaluated in mpmath with working precision growing with ν (plain float64 is wrong for
ν > ~30). Safeguards:
- The model is stated for α in (0, π/2]; `velocities()` raises `ValueError` outside it
  or if the swimmer is not inside the wedge (|β| < α).
- The integrand decays like exp(−2(α − |β|)ν). If the tail beyond `nu_max` exceeds 1e-15
  (swimmer close to a wall, narrow wedge), or the quadrature error estimate exceeds 1e-25,
  `RuntimeError` is raised instead of returning a wrong number; a tail between 1e-25 and
  1e-15 gives a `UserWarning`. Close to a wall, pass `nu_max ~ 30/(α − |β|)` (as
  `Profile.compute` does).
- Calling with `mp.dps` below 30 (with mpf/float angles) gives a `UserWarning`; string
  angles are always converted at 30 digits.

## Tests

`python -m pytest -q` (~8 min); `python -m pytest -q -m "not slow"` for a faster subset.

| file | what it checks |
|---|---|
| `test_realspace.py` (slow) | rebuilds the full flow in real space from the coefficients: no-slip on both walls, and flow and vorticity at the swimmer equal to the solver's V_r, V_θ, Ω_z |
| `test_single_wall.py` | α = π/2 against Blake's (1971) image solution and the closed forms of Berke et al. (2008) |
| `test_coefficients.py` | boundary conditions of the potentials, dagger relations, Z formulas |
| `test_symmetry.py` | mirror and head–tail symmetry, ρ scaling, truncation `nu_max` = 60 vs 100 |
| `test_numerics.py` | adaptive precision vs 1500 digits, quadrature re-integration, small-ν regularity, tail size |
| `test_regression.py` | pinned values of the reference configurations and `results/velocities.csv` |
| `test_trajectory.py` | planar-wall limits and closed-form paths, exact δ-decomposition, free swimming, spline vs solver, wall repulsion, the case runner |

## Citation

Please cite the paper (above) and the version of this software you used:

> D. Zidanek, hydrowedge (version 1.0.0), Zenodo (2026). <!-- Add the Zenodo DOI after the release. -->

`CITATION.cff` holds the same information in machine-readable form (GitHub shows it as
"Cite this repository").

## License

MIT; see `LICENSE`.
