# Paper Figures and Tables Reference

Every figure and table of *Sharp-interface VOF method for phase-change
simulations on unstructured meshes*, what it shows, and which script produces
it. Numbering follows the manuscript; appendix figures and tables number per
appendix (B.1, C.1, ...), as `elsarticle` sets them.

**S** = self-contained (numpy/scipy/matplotlib only).
**P** = needs T-Flows PVTU or front output; see the environment variables in
[`README.md`](README.md).

---

## Section 2 — Methodology

### Figure 1: Interface-modified gradient stencil
- **File**: `gradient_stencil_schematic.png`
- **Script**: `make_gradient_stencil_schematic.py` — **S**
- **Content**: The stencil on a polyhedral mesh. Displacement vectors to
  same-phase neighbours are unmodified (blue). Where the cell-centre connection
  crosses the interface the vector is shortened to the intersection point (red)
  and the neighbour temperature is replaced by T_sat.

---

## Section 3.1 — Stefan and Sucking problems

### Figure 2: Stefan and Sucking schematic
- **File**: `stefan_sucking_schematic.pdf`
- **Script**: `make_stefan_sucking_schematic.py` — **S**
- **Content**: The shared one-dimensional configuration. Stefan profile in red
  (linear in the vapour, constant in the liquid), Sucking in blue (constant in
  the vapour, error function in the liquid), meeting at T_sat.

### Figure 3: Interface position and relative error
- **File**: `stefan_sucking_interface.png`
- **Script**: `plot_stefan_sucking.py` — **P**
- **Content**: (a,b) Stefan, 100 cells; (c,d) Sucking, 400 cells. Interface
  position against the analytical solution, and the relative error, which stays
  below 1% throughout for both.
- **Data**: `Stefan/stefans_solution.dat`, `Stefan/interface_position.dat`,
  `Sucking/interface_position.num`, `Sucking/interface_position.exa`

### Figure 4: Sucking temperature and velocity profiles
- **File**: `sucking_profiles.png`
- **Script**: `analyze_sucking_profiles.py` — **P**
- **Content**: Instantaneous profiles at t = 0.2, 0.35 and 0.5 s, T-Flows
  (dashed) against analytical (solid).

### Figure 5: Origin of the Sucking velocity error
- **File**: `sucking_subcell.{pdf,png}`
- **Script**: `plot_sucking_subcell.py` — **P**
- **Content**: (a) The liquid velocity profile against distance from the
  interface at t = 0.5 s, against the analytical liquid-phase velocity. The
  shaded +-2-cell band is the interface region where the analytical solution is
  discontinuous by construction, and is excluded from the error norms.
  (b) Plateau velocity error against the interface position in a cell,
  phi = (x_int/h) mod 1, over 107 snapshots; the error collapses onto a single
  harmonic of period one cell, fitted amplitude 8.0%. This is the mechanism
  behind the oscillatory error convergence.

---

## Section 3.2 — Scriven bubble growth

### Figure 6: Mesh and temperature cross-sections
- **File**: `scriven_meshes.png`
- **Script**: `analyze_scriven_fields.py` — **P**
- **Content**: t = 0.63 ms, temperature and mesh through the bubble centre for
  (a) structured 125^3 and (b) polyhedral, numerical interface in black. Insets
  magnify the marginally resolved thermal boundary layer.

### Figure 7: Three fields, two topologies
- **File**: `scriven_panels.pdf`
- **Script**: `plot_scriven_panels.py` — **P**
- **Content**: t = 1.0 ms on z = 0, structured 125^3 (left) against polyhedral
  125^3 (right), both at dt = 2 us. (a,b) temperature with velocity vectors;
  (c,d) interfacial mass transfer rate, drawn on the cells that carry it rather
  than interpolated; (e,f) gradient magnitude. The temperature and gradient
  panels carry an inset magnifying a 26 um square centred where the interface
  crosses the +x axis — wide enough to hold the thermal layer with margin, and
  common to both columns so the topologies are magnified over one physical
  window. The mass-transfer panels carry none: that band is one to two cells
  wide and already resolves as discrete cells at full scale. The structured
  bubble is visibly four-fold in all three fields; the polyhedral one is not.
- **Replaces** three separate 2x3 arrays in the earlier manuscript
  (`analyze_scriven_mdot.py` produced one of them).

### Figure 8: Radius growth
- **File**: `scriven_mass_conservation.png`
- **Script**: `analyze_mass_conservation.py` — **P**
- **Content**: Radius growth on the 75^3 and 125^3 structured (dt = 10 us) and
  polyhedral (dt = 2 us) meshes, against the analytical solution and the
  geometric-VOF data of Bureš & Sato (2021).

### Figure 9: Time-step sensitivity
- **File**: `scriven_sensitivity_dt.pdf`
- **Script**: `plot_scriven_sensitivity.py` — **P**
- **Content**: R_front/R_Scriven against time. (a) polyhedral 75^3 at 1, 2 and
  10 us; (b) structured 75^3 at the same three steps, showing the overshoot,
  with the 2 us series marked because its front output is written at five
  instants against the 1 us run's 181. Grey dotted in (b) is the polyhedral
  75^3 at 1 us, for reference.
- **Note**: one script writes both this figure and Figure 10. The manuscript
  carried them as a single four-panel `scriven_sensitivity_new.pdf` until the
  August review split the time-step panels from the mesh ones.

### Figure 10: Mesh sensitivity
- **File**: `scriven_sensitivity_mesh.pdf`
- **Script**: `plot_scriven_sensitivity.py` — **P**
- **Content**: R_front/R_Scriven against time. (a) polyhedral mesh convergence
  at dt = 2 us; (b) interface-modified against standard LSQ gradient on the
  structured 75^3 at 1 us. Grey dotted in (b) is the polyhedral 75^3 at 1 us,
  for reference.
- **Note**: rebuilt on the d_f-corrected polyhedral campaign, so it is
  consistent with Table 3. The earlier `Scriven/plot_sensitivity.py` predates
  that correction.

---

### Figure 11: Gradient construction
- **File**: `scriven_gradient.pdf`
- **Script**: `plot_scriven_sensitivity.py` — **P**
- **Content**: Structured 75^3 at dt = 1 us, R_front/R_Scriven against time.
  The interface-modified stencil overshoots the analytical solution and the
  standard least-squares stencil underpredicts it by more, so the modification
  is necessary and is not the origin of the overshoot. Grey dotted is the
  polyhedral 75^3 at 1 us, for reference.

### Figure 12: Extrapolation to the refinement limit
- **File**: `scriven_extrapolation.pdf`
- **Script**: `plot_scriven_extrapolation.py` — **S**
- **Content**: Roache extrapolation of **R_front** at t = 1.0 ms for both
  families. (a) Mesh refinement, fitting R = R_inf - c h^b with the limit free
  rather than the exponent fixed, resolved by direction; the refinement ratios
  are 1.20-1.34, so ln r is too small to condition an exponent, and what the
  sequence answers is what value it converges to, not how fast. (b) Time-step
  refinement on the 75^3 mesh as a first-order fit, on the same basis. Same
  quantity and instant as Tables 3 and 4.
- **Note**: this panel has been through two earlier forms. It first plotted
  R_vol at t = 1.5 ms — the only figure in the paper on R_vol, which forced a
  standing caveat about the 3.3%/4.9% offset — then a fixed-slope log-log
  convergence panel drawn by `plot_scriven_convergence.py`, now superseded.
  It was also, until the 11 September revision, panel (b) of the radius-growth
  figure; the manuscript now carries the two separately.

## Section 3.3 — Mass and energy budgets

### Figure 13: Global mass conservation
- **File**: `mass_history.{pdf,png}`
- **Script**: `plot_mass_history.py` — **P**
- **Content**: After Sato & Ničeno (2013, Fig. 8), for (a) Stefan, (b) Sucking
  and (c) the polyhedral 75^3 Scriven case. Left axis, the conservation error
  M_s/M_0 - 1, where M_s is the domain mass plus the accumulated outflow egress.
  Right axis, the domain mass M_d/M_0 alone. Both measured in the solver.
- **Needs**: the `instrumentation/End_Of_Time_Step_mass_balance.f90` hook
  compiled into the run — the egress term is recorded nowhere else.

---

## Section 4 — Origin of the anisotropy

### Figure 14: Gradient polar distribution and Fourier decomposition
- **File**: `scriven_grad_polar.png`
- **Script**: `analyze_scriven_gradients.py` — **P**
- **Content**: t = 1.0 ms of simulated growth, which is t = 1.23 ms once the
  virtual origin the Scriven solution starts from is added. Top row, polar
  distribution of |grad T| around the interface for (a) structured 75^3,
  (b) structured 125^3, (c) polyhedral 125^3, with the rings labelled in units
  of 10^4 K/m and the unit stated once under each legend. Bottom row, the
  azimuthal Fourier decomposition with the m = 4 bar highlighted. The
  structured meshes carry a coherent m = 4 mode where the polyhedral mesh
  spreads its residual quasi-randomly across modes.
- **Run with**: `python3 analyze_scriven_gradients.py polar` regenerates this
  figure alone and prints the ring-label angles and the amplitudes.
- **Caution on the numbers**: the amplitudes annotated on the bottom row
  (15.4 / 11.1 / 1.3%) come from an FFT of the binned means on a one-cell
  equatorial slice, which is *not* the estimator Table 6 reports. The table
  fits the modes by least squares to the scattered liquid-side points in three
  dimensions at dt = 2 us, and gives 10.2% for the structured 75^3 at the same
  instant. Both measure the same mode; only the table's convention is the one
  the manuscript quotes, and the difference is the estimator, not the data.
- **Also produces**: Table 6.

### Figure 15: Stencil geometry at three angular positions
- **File**: `stencil_geometry.png`
- **Script**: `Scriven/analyze_stencil_anisotropy.py`
  (`plot_stencil_geometry()`) — **S**
- **Content**: Blue, unmodified displacement vectors; red, vectors shortened to
  the interface intersections. At 0 degrees one face is intersected and the
  stencil stays nearly symmetric; at 45 degrees two are, with different
  shortening ratios, giving an asymmetric gradient matrix.

### Figure 16: The deformation survives a uniform mass transfer rate
- **File**: `uniform_mdot.pdf`
- **Script**: `plot_uniform_mdot.py` — **P**
- **Content**: Structured 75^3 at t = 1 ms, liquid-side band. (a) Interface
  radius against the cubic invariant s = nx^4+ny^4+nz^4 (unity on a coordinate
  axis, 1/3 on a body diagonal); a negative slope is a bubble bulging towards
  the corners. (b) The same data as normalised radius against azimuth. Grey is
  the computed gradient-derived flux, red a spatially uniform flux with the same
  domain total. Removing all angular variation from the source removes only
  about a third of the distortion.

### Figure 17: The deformation requires the transport
- **File**: `noadvection.pdf`
- **Script**: `plot_noadvection.py` — **P**
- **Content**: Structured 75^3, dt = 1 us. (a) Radius against s at matched
  radius (and therefore matched R/h). (b) Axis-to-diagonal difference against
  radius — constant without transport, which is what a fixed discretisation
  bias looks like, and climbing monotonically with it, which is what an error
  accumulating with interface displacement looks like.
- **Also produces**: Table 9.

### Figure 18: The two interfaces
- **File**: `bubble_shape.pdf`
- **Script**: `plot_bubble_shape.py` — **P**
- **Content**: Both surfaces coloured by local departure from the mean radius,
  viewed along a cube diagonal. (a) Transport retained at R = 120 um, spanning
  -4.1 to +2.8%. (b) Transport removed, same radius and colour scale,
  featureless. (c) Panel (b) on a five-times-finer scale, where the residual is
  visibly still cubic and of the same sign, an order of magnitude smaller, and
  not growing as the interface advances.

### Figure 19: The CICSAM stencil at two orientations
- **File**: `cicsam_schematic.{pdf,png}`
- **Script**: `plot_cicsam_schematic.py` — **S**
- **Content**: (a,b) The U, D and A samples, the normal n, the cell-centre
  connection d and the angle theta_f between them, with the ramp width
  W = h||n||_1 bracketed above each row and the sample spacing delta below it.
  The mesh, not the normal, is rotated between panels, so "along n" is
  horizontal in both. (c) The same marks stacked on one axis.

### Figure 20: Limiter branches
- **File**: `limiter_branches.{pdf,png}`
- **Script**: not included in this release
- **Content**: The two branches of the blend against the normalised donor
  value at Co = 0.2, the shaded band marking where Hyper-C is saturated and the
  face value is the acceptor's.
- **Note**: introduced in the 11 September revision, replacing the
  ramp-sampling figure that `plot_ramp_sampling.py` drew. That script is still
  here, and still produces the measurements the surrounding text quotes, but
  the figure it draws is no longer in the manuscript.

### Figure 21: Declining the blend weight
- **File**: `hyperc_ladder.{pdf,png}`
- **Script**: `plot_hyperc.py` — **P**
- **Content**: dt = 1 us, t = 1 ms. (a) Radius error against cells per
  direction, with the fits R = R_inf - C n^-p of Table 12 dotted and their
  limits dashed; the two sequences are convergent in opposite directions.
  (b) The bubble at the same instant, z = 0 cross-section of the reconstructed
  front at 150^3, at true scale against the analytical Scriven radius, cut on
  face diagonals.
- **Also produces**: Table 12.
- **Note**: `plot_hyperc_ladder.py` in the manuscript tree is an earlier
  two-panel version of this figure and is not what the paper carries.

---

## Section 5 — Turbulent annular boiling flow

### Figure 22: Computational mesh
- **File**: `annular_mesh.png`
- **Script**: `generate_mesh_figure.py` — **P**
- **Content**: (a) the 45-degree sector domain with a slice coloured by radial
  coordinate; (b) cross-section at y = 26 mm through the gap; (c) detail of the
  boundary mesh at the heated inner wall.

### Figure 23: Instantaneous snapshot
- **File**: `annular_snapshot.pdf`
- **Script**: `generate_snapshot_figure.py` — **P**
- **Content**: (a) phase distribution, liquid film on the wall and gas core;
  (b) axial velocity; (c) mass transfer rate, enhanced at wave troughs where
  the film is thin.

### Figure 24: Hovmöller diagram
- **File**: `annular_hovmoller.png`
- **Script**: `hovmoller_wave_analysis.py` — **P**
- **Content**: Interface perturbation eta(y,t). Steep diagonals are structures
  advected by the gas (c ~ 12 m/s), shallow ones slower interfacial
  disturbances (c ~ 1.6 m/s), with reference velocities dashed.
- **Note**: the phase-speed estimate at the end of the script needs the
  `PhaseVelocity` helper module, which is not in this repository.

### Figure 25: Wave-modulated mass transfer
- **File**: `annular_quantitative.pdf`
- **Script**: `generate_quantitative_vectorized.py` — **P**
- **Content**: (a) film thickness against normalised mass transfer rate, binned
  means in red and the 1/delta scaling dashed, Pearson r = -0.62;
  (b) conditional rate by wave phase, troughs evaporating about four times
  faster than crests.

---

## Appendix B — Gradient stencil analysis

### Figure B.1: Synthetic stencil anisotropy
- **File**: `stencil_polar_comparison.png`
- **Script**: `Scriven/analyze_stencil_anisotropy.py`
  (`plot_stencil_analysis()`) — **S**
- **Content**: Polar plots of |grad T|_computed / |grad T|_analytical for
  (a) standard against interface-modified, (b) inverse-distance weighting,
  (c) extended stencil. The unit circle is ideal. Both stencils show four-fold
  anisotropy, at different orientations and magnitudes.
- **See also**: [`Scriven/STENCIL_ANALYSIS.md`](Scriven/STENCIL_ANALYSIS.md).

---

## Appendix C — Supporting measurement and mechanism studies of anisotropy

### Figure C.1: Bubble shape across the surface-tension sweep
- **File**: `sigma_shapes.pdf`
- **Script**: `plot_sigma_shapes.py` — **P**
- **Content**: t = 1.0 ms, structured 75^3 at dt = 0.1 us. (a) The alpha = 0.5
  interface on z = 0 for sigma/4, sigma, 2 sigma and 4 sigma, analytical radius
  dashed. (b) The same contours divided by their own mean radius, leaving the
  four-fold distortion alone: peak-to-peak 12.9%, 7.8%, 5.4% and 3.5%. Over the
  same sixteen-fold range the four-fold gradient mode falls only from 12.17% to
  9.02% — rounding the bubble does not round the gradient.

---

## Appendix D — Static droplet (surface tension verification)

### Figure D.1: Spurious currents around the static droplet
- **File**: `droplet3d_slice_arrows.{pdf,png}`
- **Script**: `plot_droplet3d_slice.py` — **P**
- **Content**: The domain mid-plane y = 1 at t = 0.12 s. Columns are the two
  mesh topologies, rows the two resolutions. Vectors are sampled on a uniform
  26 x 26 grid linearly interpolated from each mesh's velocity field, because a
  polyhedral dual has no cell layer aligned with any plane; the boundary ring of
  that grid falls outside the convex hull of the cell centres and is left blank,
  so 24 x 24 arrows are drawn. Magnitude is carried by arrow length on one scale
  across all four panels, and because the field spans five decades that length
  goes as |u|^(1/3), so the legend carries four reference arrows and no velocity
  should be read off an arrow. Red is the alpha = 1/2 contour, dashed the initial
  interface at R = 0.4, which is also the exact equilibrium.
- **Note**: the earlier `droplet3d_slice` variant put magnitude in colour and
  drew every arrow at unit length; the script no longer produces it.

---

## Tables

| Table | Content | Script |
|-------|---------|--------|
| 1 | Thermophysical properties (water-steam, 1 bar) | manual |
| 2 | Temporal convergence of R_front at fixed exponent | `analyze_temporal_convergence.py` |
| 3 | Scriven mesh convergence at t = 1.0 ms on R_front | `Scriven/analyze_convergence.py`, `Scriven/compute_front_radius.py` |
| 4 | Roache extrapolation of the mesh sequence, limit free | `plot_scriven_extrapolation.py`, `analyze_roache_gci.py` |
| 5 | Phase-change delivery budgets for the benchmarks | `analyze_scriven_budgets.py`, `analyze_mass_budget.py`, `analyze_stefan_energy.py` |
| 6 | Azimuthal m = 4 amplitude of the interfacial gradient | `analyze_scriven_gradients.py` |
| 7 | Substitution test, structured 125^3 | `stencil_parameter_scan.py` |
| 8 | Candidate explanations and the measurement excluding each | `analyze_extraction_error.py`, `analyze_confinement.py`, `analyze_cicsam_normal.py`, `analyze_area_anisotropy.py` |
| 9 | Deformation with the volume-fraction transport removed | `analyze_directional_radius.py`, `plot_noadvection.py` |
| 10 | Four-fold content of the CICSAM blending factor beta_f | `analyze_cicsam_beta.py`, `analyze_beta_m4_running.py` |
| 11 | Normalised donor value at the three symmetry orientations | `analyze_alpha_tilde_orientations.py` |
| 12 | Structured refinement, blend free against frozen at gamma_f = 1 | `analyze_hyperc_dt.py`, `plot_hyperc.py` |
| 13 | Annular flow parameters and properties | manual |
| B.1 | m = 4 amplitude from the frozen-state stencil sweep | `stencil_parameter_scan.py` |
| C.1 | Four-fold shape distortion at matched confinement | `analyze_confinement.py` |
| C.2 | Four-fold mode across the normalised-variable family | `analyze_nvd_family.py`, `analyze_nvd_donor.py` |
| D.1 | Three-dimensional static droplet, spurious currents | `analyze_static_droplet.py`, `analyze_droplet_convergence.py`, `analyze_droplet_traces.py` |
| E.1 | Cost breakdown, structured 75^3 | solver timing output |
| F.1 | Mesh statistics for the two families | `analyze_mesh_quality.py` |
| F.2 | Operator-splitting error between dt = 10 and 2 us | `analyze_dt_series.py` |
| F.3 | Sucking problem, errors at three instants | `analyze_sucking_errors.py`, `analyze_sucking_velocity_error.py` |
| F.4 | Four-fold azimuthal amplitude of the interface radius | `analyze_hyperc_shape_m4.py`, `analyze_area_anisotropy.py` |

Which simulation each row comes from is in [`RUNS.md`](RUNS.md).
