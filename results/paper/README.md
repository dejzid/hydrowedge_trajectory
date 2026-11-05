# Paper figures

Single-panel trajectory figures, drawn by `make_paper_figures.py` from the case results
in `results/cases/` (case files in `cases/`). Lengths are in units of the swimmer size ℓ,
velocities in units of the swimming speed v₀; σ/(v₀ℓ²) > 0 is a pusher, < 0 a puller.
Each path starts at the dot and stops at one swimmer size from a wall (the limit of the
far-field description), or where it leaves the plotted region.

All panels are 3.375 in wide (one REVTeX column) at 600 dpi, with the same axes width and
scale; include them at natural size, or all at the same width, so that the fonts and
lengths match between figures.

## Axial dipole (`fig_axial_{pusher,puller}_s1.png`)

Case `axial_pi3_paper_round`: α = π/3, σ/(v₀ℓ²) = ±1, ρ₀ = 10ℓ, start angles
β₀ = 0, ±π/12, ±π/6, ±π/4. The swimmer moves along the edge at speed v₀ while drifting
in the cross-section shown.
- Pushers drift to the nearer wall (to the apex on the bisector). The distance travelled
  along the edge before wall contact is 14.5ℓ (β₀ = ±π/4), 109ℓ (±π/6), 295ℓ (±π/12)
  and 455ℓ (β₀ = 0).
- Pullers are pushed away from both walls and converge towards the bisector.
- The cross-section path does not depend on |σ| or v₀; the distance along the edge
  scales as v₀ρ₀³/σ.

## Transverse dipole (`fig_transverse_{pusher,puller}_s10.png`)

Case `transverse_pi3_paper_theta_s10`: α = π/3, σ/(v₀ℓ²) = ±10, start at ρ₀ = 10ℓ,
β₀ = π/6 (5ℓ from the upper wall), headings ϑ = β + δ = −3π/4, −π/2, …, π from the
x-axis. The vertical range (±15ℓ) shows every wall contact.
- Pushers are pulled into the walls and turned towards them: ϑ = ±π/2 bend strongly
  before contact, ϑ = 3π/4 and π hook into the upper wall, and ϑ = −3π/4, heading for the
  lower wall near the apex, crosses to the upper wall.
- Pullers are pushed away from walls they approach head-on (ϑ = ±π/2 lean away and turn
  into the wall only at the end) and reach walls they face quickly and almost straight
  (ϑ = 3π/4 and π at t = 2.1 and 2.5 ℓ/v₀, against 10.1 and 7.0 ℓ/v₀ for the pusher).
- ϑ = 0 and ±π/4 escape for both; only their direction drifts slightly, in opposite
  senses for pusher and puller.
