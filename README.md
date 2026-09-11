# Validation Cases and Analysis Scripts for Boiling on Unstructured Meshes

Case setup files, reference data, and the post-processing scripts behind every
figure and table of:

> **Sharp-interface VOF method for phase-change simulations on unstructured
> meshes**
>
> Jan Kren, Bojan Ničeno, Yohei Sato
> *Laboratory for Simulation and Modelling, Paul Scherrer Institute*
>
> Submitted to *Computers & Fluids*.

All cases run with the [T-Flows](https://github.com/DelNov/T-Flows) CFD solver.
The public T-Flows release carries the base solver; the phase-change extension
used here is available from the authors on request.

> **Repository scope.** This repository contains *code and setup*, not the raw
> simulation output. The PVTU/front trees behind the three-dimensional cases run
> to tens of gigabytes and are not distributed here. Every script that needs them
> is marked **P** below and reads its input through an environment variable
> (see [Environment variables](#environment-variables)). Scripts marked **S** are
> self-contained and reproduce their figure from `numpy`/`scipy` alone.

- [`FIGURES.md`](FIGURES.md) — every figure and table in the paper, with the
  script that produces it.
- [`RUNS.md`](RUNS.md) — the Scriven run matrix: which simulation each reported
  number comes from.
- [`Scriven/STENCIL_ANALYSIS.md`](Scriven/STENCIL_ANALYSIS.md) — walkthrough of
  the synthetic stencil-anisotropy analysis (Figures 12 and B.1).

## Cases

### Stefan problem (`Stefan/`)

One-dimensional evaporation of a planar interface in a superheated liquid. The
liquid is at rest and heat conduction drives the phase boundary forward.

- **Domain**: 10 mm x 1 mm x 1 mm, 200 cells in the axial direction
- **Properties**: water-steam at 1 bar (rho_l = 958.4 kg/m3, rho_v = 0.597 kg/m3)
- **Time step**: 5 ms, 1800 steps (9 s physical time)
- **Analytical solution**: `stefans_solution.dat`, `interface_position.dat`
- **Reference**: Stefan (1891)

### Sucking problem (`Sucking/`)

One-dimensional evaporation with a moving liquid, extending the Stefan problem
to include convective transport. The interface advances into the superheated
liquid while vapour is generated behind it.

- **Domain**: 8 mm x 1 mm x 1 mm, 200 cells in the axial direction
- **Properties**: water-steam at 1 bar
- **Time step**: 25 us, 20 000 steps (0.5 s physical time)
- **Analytical solution**: `interface_position.exa`, `interface_velocity.exa`
- **Numerical result**: `interface_position.num` (2-rank MPI output)
- **Reference**: Welch and Wilson (2000)

### Scriven bubble growth (`Scriven/`)

Three-dimensional growth of a spherical vapour bubble in superheated liquid.
The bubble radius grows as R(t) = 2 beta sqrt(alpha_l t), with beta = 4.06022
for 1.25 K superheat.

- **Domain**: 300 um cube, initial bubble radius R0 = 50 um
- **Properties**: water-steam at 1 bar, sigma = 0.059 N/m
- **Meshes**: structured 75^3, 100^3, 125^3, 150^3;
  polyhedral 75^3, 100^3, 125^3, 150^3
- **Time steps**: 0.1 to 10 us depending on the study — see [`RUNS.md`](RUNS.md)
- **Analytical reference**: `dT1.25_R50microns.txt`
- **Reference**: Scriven (1959)

The Scriven case is also the carrier for the anisotropy study of Section 4. Those
runs are the same case with one solver switch changed each — a uniform
interfacial mass flux, the volume-fraction transport suppressed, the CICSAM
blend weight frozen at either limit, or the surface tension scaled. `RUNS.md`
lists them against the table or figure each one supports.

#### Bench-data files

The `bench-data*.dat` files carry time, interfacial area, bubble volume and the
volume-equivalent radius for the eight convergence runs:

| File | Mesh |
|------|------|
| `bench-data-struct-075.dat` | Structured 75^3 |
| `bench-data-struct-100.dat` | Structured 100^3 |
| `bench-data-struct-125.dat` | Structured 125^3 |
| `bench-data-struct-150.dat` | Structured 150^3 |
| `bench-data-poly-075.dat`   | Polyhedral 75^3 |
| `bench-data-poly-100.dat`   | Polyhedral 100^3 |
| `bench-data-poly-125.dat`   | Polyhedral 125^3 |
| `bench-data-poly-150.dat`   | Polyhedral 150^3 |

Note that the paper reports **R_front**, the area-weighted mean distance of the
reconstructed front elements from the bubble centre, not the volume-equivalent
**R_vol** in column 4 of these files. The two differ by 3.3% (structured) and
4.9% (polyhedral), and the ratio is scheme-dependent, so the columns cannot
simply be rescaled. `Scriven/compute_front_radius.py` computes R_front from
front output.

### Static droplet and annular flow

The three-dimensional static droplet of Appendix D.1 and the turbulent annular
boiling case of Section 5 are not shipped as case directories here. Their
parameters are given in full in the paper (Table D.1 and Table 11 respectively),
and the scripts that post-process them are listed below.

## Solver instrumentation (`instrumentation/`)

`End_Of_Time_Step_mass_balance.f90` is the `User_Mod` hook that accumulates the
mass leaving through the outflow, step by step. Figure 13 needs it: the egress
term is recorded nowhere else, and inferring it as `M_0 - M_d` would make the
conservation error identically zero by algebra rather than by measurement.

## Analysis and figure scripts

Scripts fall into three categories:

- **S** — self-contained: `numpy`, `matplotlib` and `scipy` only. Runnable as-is.
- **P** — needs T-Flows PVTU or front output. Point the environment variables
  below at the run tree first.
- **S\*** — the figure is self-contained, but the script also prints a statistic
  measured on a real run, which it skips with a message when the data is absent.

Some **P** scripts additionally need [`pyvista`](https://pyvista.org/) to read
the unstructured output.

**Known gap.** `hovmoller_wave_analysis.py` (Figure 24) imports a helper module,
`PhaseVelocity`, that is not in this repository. Everything in the script up to
the Hovmöller diagram itself runs without it; the phase-speed estimate at the end
does not. Ask the authors for the module if you need that part.

[`FIGURES.md`](FIGURES.md) is the full figure-by-figure and table-by-table map.
The summary:

| Script | Produces | Type |
|--------|----------|:----:|
| `make_gradient_stencil_schematic.py` | Fig 1 | S |
| `make_stefan_sucking_schematic.py` | Fig 2 | S |
| `plot_stefan_sucking.py` | Fig 3 | P |
| `analyze_sucking_profiles.py` | Fig 4 | P |
| `plot_sucking_subcell.py` | Fig 5, Table F.3 | P |
| `analyze_scriven_fields.py` | Fig 6 | P |
| `plot_scriven_panels.py` | Fig 7 | P |
| `plot_scriven_sensitivity.py` | Figs 9, 10, 11 | P |
| `analyze_mass_conservation.py` | Fig 8 | P |
| `plot_scriven_extrapolation.py` | Fig 12, Table 4 | S |
| `plot_mass_history.py` | Fig 13 | P |
| `analyze_scriven_gradients.py` | Fig 14, Table 6 | P |
| `Scriven/analyze_stencil_anisotropy.py` | Figs 15, B.1 | S |
| `plot_uniform_mdot.py` | Fig 16 | P |
| `plot_noadvection.py` | Fig 17, Table 9 | P |
| `plot_bubble_shape.py` | Fig 18 | P |
| `plot_cicsam_schematic.py` | Fig 19 | S |
| *(not included)* | Fig 20 | - |
| `plot_hyperc.py` | Fig 21, Table 12 | P |
| `generate_mesh_figure.py` | Fig 22 | P |
| `generate_snapshot_figure.py` | Fig 23 | P |
| `hovmoller_wave_analysis.py` | Fig 24 | P |
| `generate_quantitative_vectorized.py` | Fig 25 | P |
| `plot_sigma_shapes.py` | Fig C.1 | P |
| `plot_droplet3d_slice.py` | Fig D.1 | P |

Figure 20, the limiter-branch plot, arrived in the 11 September revision and
its script is not part of this release. It replaced the ramp-sampling figure;
`plot_ramp_sampling.py` is still here and still produces the measurements the
surrounding text quotes, but the figure it draws is no longer in the
manuscript.

### Substitution study (Section 4.2, detail in Appendix C.1)

`stencil_parameter_scan.py` is the offline reimplementation of the
interface-modified gradient construction named in the Data Availability
statement. It re-evaluates the frozen simulation state with the temperature
field, the interface position, or both replaced by their analytical
counterparts, which is what Tables 7 and B.1 report. `analyze_anisotropy_attribution.py`
holds the lattice loader and the bubble fit that it and most of the
anisotropy scripts share.

### Table scripts

| Script | Produces |
|--------|----------|
| `analyze_temporal_convergence.py` | Table 2 |
| `Scriven/analyze_convergence.py`, `Scriven/compute_front_radius.py` | Table 3 |
| `plot_scriven_extrapolation.py`, `analyze_roache_gci.py` | Table 4 |
| `analyze_scriven_budgets.py`, `analyze_mass_budget.py`, `analyze_stefan_energy.py` | Table 5 |
| `analyze_scriven_gradients.py` | Table 6 |
| `stencil_parameter_scan.py` | Tables 7, B.1 |
| `analyze_extraction_error.py`, `analyze_confinement.py`, `analyze_cicsam_normal.py`, `analyze_area_anisotropy.py` | Table 8 |
| `analyze_directional_radius.py`, `plot_noadvection.py` | Table 9 |
| `analyze_cicsam_beta.py`, `analyze_beta_m4_running.py` | Table 10 |
| `analyze_alpha_tilde_orientations.py` | Table 11 |
| `analyze_hyperc_dt.py`, `plot_hyperc.py` | Table 12 |
| `analyze_confinement.py` | Table C.1 |
| `analyze_nvd_family.py`, `analyze_nvd_donor.py` | Table C.2 |
| `analyze_static_droplet.py`, `analyze_droplet_convergence.py`, `analyze_droplet_traces.py` | Table D.1 |
| `analyze_mesh_quality.py` | Table F.1 |
| `analyze_dt_series.py` | Table F.2 |
| `analyze_sucking_errors.py`, `analyze_sucking_velocity_error.py` | Table F.3 |
| `analyze_hyperc_shape_m4.py`, `analyze_area_anisotropy.py` | Table F.4 |

Tables 1, 13 and E.1 (properties, annular parameters, cost breakdown) are typed
directly in the manuscript; E.1 comes from the solver's own timing output.

### Supporting scripts (not figures in the paper)

| Script | Purpose |
|--------|---------|
| `plot_conservation.py` | Mass and energy balance closure of the phase-change model |
| `plot_spurious_currents.py` | Six-panel spurious-current survey behind Appendix D.1 |
| `plot_smoothing_sweep.py` | Curvature pre-smoothing sweep on both topologies (Appendix D.1) |
| `check_smoothing_operator.py` | Verifies the smoothing operator against its stated kernel |
| `measure_sigma_sweep.py` | Numbers behind the surface-tension sweep of Appendix C |
| `compare_idw2_gradient.py` | IDW^2 gradient weighting comparison |
| `Scriven/compute_erf_model.py` | Closed-form gradient ratios for the stencil discussion |
| `Scriven/plot_radius.py` | Quick radius-vs-time plot |
| `Scriven/analyze_overshoot_mechanism.py` | Structured-mesh overshoot investigation |
| `Scriven/diagnose_bl_thickness.py` | Thermal boundary-layer thickness diagnostics |
| `Scriven/diagnose_paradox.py` | Gradient/volume-growth consistency check |

### Superseded

These produced figures in an earlier version of the manuscript and no longer
correspond to anything in it. They are kept because they still run.

| Script | Was |
|--------|-----|
| `make_overrelaxed.py` | Non-orthogonal correction schematic |
| `gradient_overestimate_schematic.py` | Appendix C gradient-overestimate schematic |
| `generate_schematic_figure.py` | Annular flow schematic |
| `analyze_scriven_mdot.py` | Standalone mass-transfer-rate figure, now row (c,d) of Fig 7 |
| `Scriven/plot_sensitivity.py` | Sensitivity composite, superseded by `plot_scriven_sensitivity.py` |
| `plot_scriven_convergence.py` | Fixed-slope convergence panel, superseded by `plot_scriven_extrapolation.py` |

## Running the cases

Each case directory contains the minimum files needed to reproduce the
simulation with T-Flows:

1. Generate the mesh:
   ```
   T-Flows/Binaries/Generate < generate.scr
   ```
2. (For parallel runs) Divide the domain:
   ```
   T-Flows/Binaries/Divide < divide.scr
   ```
3. Run the solver:
   ```
   T-Flows/Binaries/Process
   ```

The `User_Mod/` directories contain the Fortran source files that must be
compiled with the solver (initial conditions, boundary conditions, and
post-processing routines specific to each case).

## Environment variables

Every path in every script is an environment variable with a default. The
defaults are the authors' own directories and will not exist on your machine;
set the ones your script needs.

| Variable | Points at | Used by |
|----------|-----------|---------|
| `OUTPUT_DIR` | Where figures are written | All figure scripts |
| `PAPER_ROOT` | Manuscript root, whose `Data/` holds the anisotropy campaign | Anisotropy and Hyper-C scripts |
| `SCRIVEN_DIR` | T-Flows Scriven case directory | Scriven PVTU scripts |
| `STEFAN_DIR` | T-Flows Stefan case directory | `plot_stefan_sucking.py` |
| `SUCKING_DIR` | T-Flows Sucking case directory | Sucking scripts |
| `MERLIN_DIR` | Archived `Scriven-Merlin` polyhedral runs | `plot_scriven_panels.py`, `analyze_scriven_fields.py` |
| `DROPLET_DIR` | Static droplet runs | `plot_droplet3d_slice.py`, `plot_smoothing_sweep.py` |
| `MASSBALANCE_DIR` | Runs carrying the mass-balance hook | `plot_mass_history.py` |
| `GOLD_DIR` | Reference (`_gold`) run tree | `analyze_static_droplet.py`, `plot_conservation.py` |
| `DATA_DIR` | Bureš & Sato reference data | `Scriven/plot_radius.py` |
| `ANNULAR_VTU` | Annular mesh VTU file | `generate_mesh_figure.py` |

A few scripts take `--out` instead of, or in addition to, `OUTPUT_DIR`; run them
with `--help`.

## File types

| Extension | Description |
|-----------|-------------|
| `control` | Solver configuration (properties, numerics, BCs) |
| `*.dom` | Domain definition for mesh generation |
| `generate.scr` | Script input for the T-Flows mesh generator |
| `*.ini` | Initial condition parameters (bubble/box geometry) |
| `*.f90` | Fortran source (User_Mod functions) |
| `*.dat` | Simulation results or analytical reference data |
| `*.exa` | Analytical (exact) solution data |
| `*.num` | Numerical solution data |
| `*.gnu` | Gnuplot post-processing scripts |
| `*.geo` | GMSH geometry file (polyhedral mesh) |
| `*.py` | Python analysis/post-processing scripts |
