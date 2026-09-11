# Scriven run matrix

Every table and figure reporting Scriven results corresponds to one row below.
All runs use the 300 um cube, R0 = 50 um and dT = 1.25 K; mesh levels are quoted
as node counts.

| Purpose | Topology, meshes | dt | Reported in |
|---------|------------------|----|-------------|
| Temporal sensitivity | polyhedral 75^3 | 1, 2, 10 us | Tables 2, F.2, Fig 8(a) |
| Mesh convergence | polyhedral 75^3–150^3 | 2 us | Tables 3, 4, Fig 9(b) |
| Mesh convergence | structured 75^3–150^3 | 2 us | Tables 3, 4, 6, C.1 |
| Radius-growth comparison | structured 75^3, 125^3 | 10 us | Fig 9(a) |
| Time-step series | structured 75^3 | 2, 1, 0.1 us | Table 2, Fig 8(b), Section 4 (scaling) |
| Time-step check | structured 100^3, 125^3 | 2, 1 us | Section 4 (scaling) |
| Substitution, stencil sweep | structured 125^3 | 1 us | Tables 7, B.1 |
| Extended stencil, coupled | structured 75^3–150^3 | 2 us | Table C.1 |
| Uniform mass flux | structured 75^3, 125^3 | 2, 1 us | Section 4 (bounds), Fig 14 |
| Transport removed (incl. sigma/rho_l-matched variant) | structured 75^3 | 1 us | Table 9, Figs 15–16 |
| Blend frozen, gamma_f = 1 ladder | structured 75^3–150^3 | 1 us | Tables 12, F.4, Fig 19 |
| Blend frozen, gamma_f = 0 | structured 75^3 | 1 us | Section 4 (VOF transport) |
| Surface-tension sweep | structured 75^3 | 0.1 us | Appendix C, Fig C.1 |
| Mesh quality | both families, all levels | — | Table F.1 |

The 150^3 blend baseline at dt = 1 us sits in `Data/hyperc-bench/base-150-dt1.dat`
and `Data/gamma-st150`, not under `Data/campaign/DtSeries` with the other three
levels. The directory name suggests a gamma-override arm; it is not one.
`gamma-st100` and `gamma-st125` return R_front identical to the DtSeries
directories to every printed digit on identical element counts.

## Solver switches

The anisotropy runs are the Scriven case with one thing changed each. The
switches, as they appear in the solver:

| Switch | Effect | Used by |
|--------|--------|---------|
| `UNIFORM_MASS_TRANSFER` | Rescales the interfacial flux so every front element receives the same mass per unit area, with the domain total unchanged | Fig 13; both solid curves of Fig 14 |
| Volume-fraction transport suppressed | The alpha divergence term is removed, so the interface moves only by the source | Table 7, Figs 14–15 |
| `gamma_f` frozen at 1 | Pure Hyper-C, the compressive limit of the blend | Tables 12, F.4, Fig 19 |
| `gamma_f` frozen at 0 | Pure UQ, the bounded limit | Section 4 |
| `MAX_CORRECTION_CYCLES_BETA_VOF` | Cap on the boundedness corrector. Set to 1 for the Hyper-C ladder, which buys a provably pure scheme at the cost of alpha leaving [0,1] on some steps | Table 12 |

## Two radius definitions

The paper reports **R_front**, the area-weighted mean distance of the
reconstructed front element centroids from the bubble centre. The
`bench-data*.dat` files carry **R_vol**, the volume-equivalent radius, in
column 4.

They are not interchangeable. R_front exceeds R_vol by 3.3% on the structured
meshes and 4.9% on the polyhedral ones, and the ratio is scheme-dependent
(1.007 for pure Hyper-C, 1.035 for the blend, 1.077 for UQ) because the blend
weight sets the effective interface thickness. It also drifts in time, running
1.00 to 1.05 over the record, so a bench trace cannot be rescaled by a constant.
`Scriven/compute_front_radius.py` computes R_front from front output.

Mixing the two looks like an error in the Scriven constant. Dividing tabulated
R_vol radii by a relative error measured on R_front and back-solving gives
beta ≈ 3.89 against the correct 4.06022, and the 3.3% "discrepancy" recovered
that way is exactly the R_front/R_vol offset. Solving Scriven's relation
directly from the property table of Table 1 gives beta = 4.0647, 0.11% from the
value used throughout, which moves R by 0.2%.
