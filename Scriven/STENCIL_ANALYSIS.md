# Stencil Anisotropy Analysis

## What this script does

`analyze_stencil_anisotropy.py` answers one question: **why does the
temperature gradient show a four-fold pattern on a structured mesh, and can we
fix it?**

It uses a synthetic 2D model — no simulation data is needed. An analytical
temperature field is placed on a fake Cartesian grid, and the least-squares
gradient stencil (exactly as implemented in T-Flows) is applied at 360 angles
around a bubble interface. Any angular variation in the computed gradient is
purely a stencil artifact.

## The analytical temperature field

Two functions define the known answer:

- `temperature_scriven(y, z)` — a spherically symmetric temperature field:
  T = T_sat inside the bubble, smoothly rising to T_inf outside
- `analytical_gradient(y, z)` — the exact mathematical derivative dT/dy, dT/dz

Since the field is spherically symmetric, the true gradient has the **same
magnitude at every angle**. Any angular variation = stencil error.

## The least-squares gradient (what T-Flows does)

`compute_lsq_gradient_2d()` mimics `Calculate_Grad_Matrix_With_Front.f90`:

1. For each neighbour, compute the displacement vector **d** = (neighbour − cell centre)
2. Build the **G matrix**: G = Σ w · d dᵀ (sum of weighted outer products)
3. Build the **right-hand side**: b = Σ w · Δφ · d (where Δφ = T_neighbour − T_cell)
4. Solve: ∇T = G⁻¹ · b

The weight `w` can be:
- `none` (w = 1): standard unweighted, what T-Flows uses
- `idw` (w = 1/|d|): inverse distance weighting
- `idw2` (w = 1/|d|²): inverse distance squared

## The front modification — where the anisotropy comes from

`analyze_cell_gradient()` is the core function. For each cell just outside the
bubble, it lists the 4 face neighbours (up, down, left, right) and checks: does
the line from cell centre to neighbour cross the bubble interface?

- **No crossing** → use the full displacement vector to the neighbour centre,
  with T = T(neighbour) from the analytical field
- **Crossing** → **shorten** the displacement vector to the intersection point,
  and set T = T_sat

This shortening is necessary — without it, the gradient samples across the
phase discontinuity and gets completely wrong values (up to 80% error). But the
shortening changes the G matrix differently depending on the cell's angular
position:

- **Cell at 0°** (directly above the bubble): **one** face is cut by the
  interface, producing one significantly shortened vector
- **Cell at 45°** (on the diagonal): **two** faces are cut, each shortened
  moderately

These different stencil geometries produce different G matrices, different
condition numbers, and different gradient magnitudes — that is the four-fold
pattern.

## The intersection test

`circle_face_intersection()` is a simple line-circle intersection. Given a line
segment (cell centre → neighbour centre), it finds where it crosses the bubble
interface (a circle of radius R). Returns the intersection point, or `None`.

## The main loop

`plot_stencil_analysis()` (line 210):

1. Place 360 cells in a ring just outside the bubble (at r = R + 0.7h)
2. For each cell, compute the gradient under **5 configurations**:

| Configuration | Front modified? | Weighting | Extra neighbours? |
|---------------|:-:|:-:|:-:|
| Standard LSQ (no front) | No | none | No |
| Front-modified (current) | Yes | none | No |
| Front + IDW | Yes | 1/\|d\| | No |
| Front + IDW² | Yes | 1/\|d\|² | No |
| Front + extended stencil | Yes | none | Yes (4 diagonal neighbours) |

3. Normalize each computed |∇T| by the analytical value
4. Plot as a polar diagram (ratio = 1.0 means perfect)

## Output figures

### `stencil_polar_comparison.png` (paper Figure 13)

Polar plot of |∇T|_computed / |∇T|_analytical for three panels:
- **(a) Standard vs front-modified**: the standard gradient (blue dashed) drops
  to ~0.2 at some angles (80% error from sampling across the phase boundary);
  the front-modified (red) stays within a few percent of 1.0 but shows a
  four-fold wobble
- **(b) Weighting schemes**: IDW (green) and IDW² (purple) compared to
  front-modified — IDW actually *worsens* the anisotropy because it amplifies
  the shortened vectors
- **(c) Extended stencil**: adding diagonal neighbours (orange) partially
  restores isotropy by diluting the modified faces

The dotted circle at 1.0 marks the ideal (isotropic) result.

### `stencil_geometry.png` (paper Figure 12)

Three panels showing the actual stencil for cells at 0°, 22.5°, and 45°:
- Blue arrows = full displacement vectors (faces not cut by the interface)
- Red arrows = shortened vectors (stopped at the interface intersection)
- This visually explains why different angles get different stencils

### `stencil_error_comparison.png` (diagnostic, not in paper)

- Line plot of relative error vs angle (in degrees)
- Bar chart comparing min/max ratio and RMS error across all 5 configurations

### `stencil_condition_number.png` (diagnostic, not in paper)

- Condition number κ(G) vs angle around the interface
- Standard: κ = 1 everywhere (Cartesian mesh = perfectly conditioned)
- Front-modified: κ peaks at 0°, 90°, 180°, 270° where one vector is
  shortened the most (κ ≈ 3)

## Why this matters

On a **structured mesh**, the four-fold gradient error is *coherent* — it
always peaks at the same 45° angles relative to the mesh. This means the error
accumulates systematically over time, producing the diamond-shaped bubble
deformation observed in simulations.

On a **polyhedral mesh**, the face orientations are random, so the gradient
errors are also random at each angle. Random errors tend to cancel rather than
accumulate, which is why the polyhedral mesh preserves spherical bubble shape.

## Usage

```
cd Scriven
python3 analyze_stencil_anisotropy.py
```

Requires only `numpy` and `matplotlib`.

---

# Erf Temperature Model (Appendix C)

## What this script does

`compute_erf_model.py` computes all gradient ratios cited in Appendix C of the
paper. It uses a 1D erf temperature profile to evaluate the standard and
front-modified gradients against the true interface gradient — no simulation
data is needed.

## The model

The temperature profile near the interface is modelled as:

    T(x) = T_sat + ΔT · erf(x / δ_T)

where δ_T is the **Scriven-consistent** thermal penetration depth, obtained by
matching the erf derivative at x=0 to the analytical Scriven interface gradient
from the Stefan condition:

    g_true = ρ_v · h_lv · dR/dt / k_l
    δ_T = 2·ΔT / (√π · g_true)

This gives δ_T ≈ 10–17 μm for t = 0.5–1.5 ms — approximately half the
pure-diffusion estimate 2√(α_l·t) because radial outflow compresses the
boundary layer.

## Key results

| Quantity | Range | Paper reference |
|----------|-------|-----------------|
| g_std / g_true | 0.66–0.70 | "underestimates by ~30%" |
| g_fm / g_true (analytical T_P) | 0.91–0.96 | "0.93–0.95" |
| g_fm / g_true (heat-sink limit) | 1.11–1.17 | "1.14–1.16" |
| g_fm / g_std (boost at d/h=0.44) | 1.68 | Eq. 6 |
| Boost range (d/h = 0.3–0.7) | 1.34–1.83 | "1.34–1.83" |

## Mesh refinement

The script also shows that g_fm_hs/g_true remains nearly constant (~1.15)
across mesh refinements (75³ to 300³), because the boost factor depends only on
d/h (not on h itself), and finer meshes bring T_P closer to T_sat, making the
heat-sink limit more representative.

## Usage

```
cd Scriven
python3 compute_erf_model.py
```

Requires `numpy` and `scipy`.
