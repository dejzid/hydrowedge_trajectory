# Trajectory cases

Each `*.json` file here is one case: one dipole type and one wedge, with either a set of
trajectories (`"kind": "paths"`, the default) or a state diagram over two parameters
(`"kind": "scan"`). Run with

```
.venv/Scripts/python run_trajectories.py cases/<name>.json   # or --all
```

Results go to `results/cases/<name>/` (the name defaults to the file name) and are
never overwritten. Running a case again with the same parameters keeps the existing
results. Changing the parameters but keeping the name is refused, so give a changed
case a new name.

## Units
Lengths are in units of the swimmer size ℓ (the trajectory stops at wall distance
`d_min = 1`) and velocities in units of the swimming speed (`v0 = 1`). sigma is then
σ/(v0 ℓ²); sigma > 0 is a pusher and sigma < 0 a puller. For E. coli (Drescher et al.
2011), σ = p/(8πη) ≈ 17 µm³/s, v0 ≈ 22 µm/s and ℓ ≈ 2 µm give sigma ≈ 0.2.

## Fields (both kinds)
| field | required | meaning |
|---|---|---|
| `dipole` | yes | `"axial"` (φ = 0, swims along the edge) or `"transverse"` (φ = π/2, swims in the cross-section) |
| `alpha` | yes | wedge half-angle, number or expression in `pi` (`"pi/3"`) |
| `sigma` | yes for paths | dipole strength; for paths a number or a list (one figure panel per value) |
| `v0` | no (1) | swimming speed |
| `d_min` | no (1) | stop at this wall distance (the far-field model is invalid closer) |
| `rho_min`, `rho_max` | no (1, 100) | stop at these distances from the edge |
| `t_max` | no (1e4) | stop at this time |
| `repulsion` | no (off) | soft wall repulsion, `{"strength": U, "range": d_r, "power": n}`: each wall adds a velocity U((d_r/d)^n - 1) along its normal for d < d_r (no torque). Not part of the equations of motion; it lets a swimmer come closer than `d_min` = 1 and shows whether it slides along a wall or is held there. `d_min` must then be below d_r (it only guards against penetration) |
| `name`, `description` | no | free text; `name` sets the results folder |

## A start point
Angles may be numbers or expressions such as `"pi/3 - asin(1.5/12)"`.
- Position: `rho0` with `beta0`, or `rho0` with `d0` = distance to a wall (`"wall"`:
  `"upper"`, the default, or `"lower"`).
- Orientation (transverse only; the axial dipole ignores it), one of:
  - `delta0`: angle from e_ρ (as in the equations of motion);
  - `heading`: swimming direction from the x-axis (= β + δ);
  - `along` (`"outward"` or `"apexward"`, parallel to the wall) with `tilt` (rotation
    towards the wall, positive into it).

## Kind "paths"
| field | meaning |
|---|---|
| `starts` | list of start points |
| `checkpoints` | times at which the state is tabulated (for cross-checks, e.g. in Mathematica) |
| `view`, `arrows` | radius of the plotted region; orientation arrows per transverse path (8) |

Output: `case.json` (the case as run, the profile file used, the code version and the
date), `summary.csv` (start, end state and why the run stopped: `wall`, `apex`, `far`,
`t_max`), `checkpoints.csv` and `checkpoints.md` (state at the checkpoint times and at
the end), `trajectories.csv` (500 points uniform in time plus 500 uniform in path length, per path), `paths.png`/`.pdf`.

## Kind "scan" (state diagram)
| field | meaning |
|---|---|
| `start` | the base start point |
| `x`, `y` | `{"param", "from", "to", "n"}`: the scanned parameters, `sigma`, `v0` or any start-point key (`tilt`, `d0`, `rho0`, `heading`, ...) |

Outcomes: `escapes` (reaches `rho_max`), `same wall` / `opposite wall` (wall contact,
relative to the side the swimmer started on), `apex`, `t_max`.
Output: `case.json`, `scan.csv` (one row per grid point), `diagram.png`/`.pdf`.
