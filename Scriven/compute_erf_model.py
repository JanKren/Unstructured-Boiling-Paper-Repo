#!/usr/bin/env python3
"""
Erf temperature model for Appendix C gradient overestimation analysis.

Computes all gradient ratios cited in the paper (Appendix C):
  - g_std / g_true   (standard central difference vs true interface gradient)
  - g_fm  / g_true   (front-modified gradient with analytical T_P)
  - g_fm  / g_true   (front-modified gradient in heat-sink limit T_P → T_sat)
  - g_fm  / g_std    (amplification factor)

The model uses an erf temperature profile:
    T(x) = T_sat + ΔT · erf(x / δ_T)

where δ_T is the Scriven-consistent thermal penetration depth, obtained by
matching the erf derivative at x=0 to the analytical Scriven interface gradient.

Usage:
    cd Scriven
    python3 compute_erf_model.py

References:
    - Appendix C of the paper (gradient magnitude overestimate)
    - Scriven (1959) for the bubble growth solution
"""

import numpy as np
from scipy.special import erf

# ===========================================================================
#  Physical parameters (water-steam at 1 bar, ΔT = 1.25 K)
# ===========================================================================
rho_l   = 958.4       # liquid density [kg/m^3]
rho_v   = 0.597       # vapour density [kg/m^3]
cp_l    = 4216.0       # liquid specific heat [J/(kg·K)]
k_l     = 0.677        # liquid thermal conductivity [W/(m·K)]
h_lv    = 2.257e6      # latent heat of vaporisation [J/kg]
beta    = 4.06022      # Scriven growth constant [-]
dT_sup  = 1.25         # superheat [K]
alpha_l = k_l / (rho_l * cp_l)   # thermal diffusivity [m^2/s]
R0      = 5.0e-5       # initial bubble radius [m]
t0      = R0**2 / (4.0 * beta**2 * alpha_l)   # virtual time origin [s]


def scriven_radius(t_sim):
    """Analytical Scriven radius R(t) at simulation time t_sim."""
    return 2.0 * beta * np.sqrt(alpha_l * (t_sim + t0))


def scriven_dRdt(t_sim):
    """Analytical Scriven growth rate dR/dt at simulation time t_sim."""
    return beta * np.sqrt(alpha_l / (t_sim + t0))


def scriven_interface_gradient(t_sim):
    """True interface temperature gradient from Stefan condition [K/m].

    From the Stefan condition: ṁ = k_l · g / h_lv, and ṁ = ρ_v · dR/dt:
        g_true = ρ_v · h_lv · dR/dt / k_l
    """
    return rho_v * h_lv * scriven_dRdt(t_sim) / k_l


def scriven_delta_T(t_sim):
    """Scriven-consistent thermal penetration depth δ_T [m].

    Defined by matching the erf derivative at x=0 to the analytical gradient:
        g_true = 2·ΔT / (√π · δ_T)
    =>  δ_T = 2·ΔT / (√π · g_true)
    """
    g_true = scriven_interface_gradient(t_sim)
    return 2.0 * dT_sup / (np.sqrt(np.pi) * g_true)


# ===========================================================================
#  1D gradient model (Eqs. 3–6 in Appendix C)
# ===========================================================================
def g_true_from_delta(delta_T):
    """True interface gradient from erf profile: g = 2·ΔT / (√π · δ_T)."""
    return 2.0 * dT_sup / (np.sqrt(np.pi) * delta_T)


def g_std(h, d, delta_T):
    """Standard symmetric central difference gradient (Eq. 3).

    g_std = (T_far - T_sat) / (2h)

    where T_far = ΔT · erf((d + h) / δ_T).
    Independent of T_P.
    """
    T_far = dT_sup * erf((d + h) / delta_T)
    return T_far / (2.0 * h)


def g_fm_physical(h, d, delta_T):
    """Front-modified gradient with analytical T_P (Eq. 4).

    g_fm = [(T_P - T_sat)·d + (T_far - T_P)·h] / (d² + h²)

    where T_P and T_far are from the erf profile.
    """
    T_P   = dT_sup * erf(d / delta_T)
    T_far = dT_sup * erf((d + h) / delta_T)
    return (T_P * d + (T_far - T_P) * h) / (d**2 + h**2)


def g_fm_heatsink(h, d, delta_T):
    """Front-modified gradient in heat-sink limit T_P → T_sat (Eq. 6).

    g_fm = ΔT_far · h / (d² + h²)

    where T_far = ΔT · erf((d + h) / δ_T), and T_P = T_sat.
    """
    T_far = dT_sup * erf((d + h) / delta_T)
    return T_far * h / (d**2 + h**2)


def boost_factor(d_over_h):
    """Pure geometric amplification g_fm/g_std in heat-sink limit (Eq. 6).

    boost = 2h² / (d² + h²) = 2 / (1 + (d/h)²)

    Independent of temperature profile.
    """
    return 2.0 / (1.0 + d_over_h**2)


# ===========================================================================
#  Mesh parameters
# ===========================================================================
h_cell = 4.0e-6   # cell size for 75^3 mesh [m]
d_over_h_median = 0.44   # median front intersection distance

# Range of d/h values
d_over_h_range = np.array([0.3, 0.44, 0.5, 0.7])

# Time range for Scriven problem
t_range_ms = np.array([0.5, 1.0, 1.5])   # ms
t_range = t_range_ms * 1e-3               # s


# ===========================================================================
#  Main computation
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  ERF MODEL — GRADIENT RATIOS FOR APPENDIX C")
    print("=" * 72)

    # --- Scriven-consistent δ_T ---
    print("\n  Scriven-consistent thermal penetration depth:")
    print(f"  {'t [ms]':>8s}  {'R [μm]':>8s}  {'dR/dt [m/s]':>12s}  "
          f"{'g_true [K/m]':>12s}  {'δ_T [μm]':>10s}  {'δ_T/h':>6s}  "
          f"{'2√(α·t) [μm]':>14s}  {'ratio':>6s}")
    print("  " + "-" * 90)
    for t_sim in t_range:
        R = scriven_radius(t_sim)
        dRdt = scriven_dRdt(t_sim)
        g_true_val = scriven_interface_gradient(t_sim)
        dT = scriven_delta_T(t_sim)
        dT_diff = 2.0 * np.sqrt(alpha_l * (t_sim + t0))  # pure diffusion
        print(f"  {t_sim*1e3:8.1f}  {R*1e6:8.1f}  {dRdt:12.4e}  "
              f"{g_true_val:12.1f}  {dT*1e6:10.1f}  {dT/h_cell:6.2f}  "
              f"{dT_diff*1e6:14.1f}  {dT/dT_diff:6.3f}")

    # --- Gradient ratios at d/h = 0.44 (median) ---
    print("\n" + "=" * 72)
    print("  GRADIENT RATIOS AT d/h = 0.44 (MEDIAN)")
    print("=" * 72)
    d_h = d_over_h_median
    d = d_h * h_cell

    print(f"\n  {'t [ms]':>8s}  {'δ_T [μm]':>10s}  "
          f"{'g_std/g_true':>13s}  {'g_fm/g_true':>13s}  "
          f"{'g_fm_hs/g_true':>15s}  {'g_fm/g_std':>11s}  "
          f"{'g_fm_hs/g_std':>14s}")
    print("  " + "-" * 100)
    for t_sim in t_range:
        dT_val = scriven_delta_T(t_sim)
        gt = g_true_from_delta(dT_val)
        gs = g_std(h_cell, d, dT_val)
        gf = g_fm_physical(h_cell, d, dT_val)
        gh = g_fm_heatsink(h_cell, d, dT_val)

        print(f"  {t_sim*1e3:8.1f}  {dT_val*1e6:10.1f}  "
              f"{gs/gt:13.4f}  {gf/gt:13.4f}  "
              f"{gh/gt:15.4f}  {gf/gs:11.4f}  "
              f"{gh/gs:14.4f}")

    # --- Summary of ranges (matching paper text) ---
    print("\n" + "=" * 72)
    print("  SUMMARY — RANGES CITED IN PAPER")
    print("=" * 72)

    gs_gt_vals = []
    gf_gt_vals = []
    gh_gt_vals = []
    gf_gs_vals = []
    gh_gs_vals = []

    for t_sim in t_range:
        dT_val = scriven_delta_T(t_sim)
        gt = g_true_from_delta(dT_val)
        gs = g_std(h_cell, d, dT_val)
        gf = g_fm_physical(h_cell, d, dT_val)
        gh = g_fm_heatsink(h_cell, d, dT_val)
        gs_gt_vals.append(gs / gt)
        gf_gt_vals.append(gf / gt)
        gh_gt_vals.append(gh / gt)
        gf_gs_vals.append(gf / gs)
        gh_gs_vals.append(gh / gs)

    print(f"\n  g_std / g_true  = {min(gs_gt_vals):.2f} – {max(gs_gt_vals):.2f}"
          f"   (paper: 0.68–0.69)")
    print(f"  g_fm  / g_true  = {min(gf_gt_vals):.2f} – {max(gf_gt_vals):.2f}"
          f"   (paper: 0.93–0.95, with analytical T_P)")
    print(f"  g_fm_hs / g_true = {min(gh_gt_vals):.2f} – {max(gh_gt_vals):.2f}"
          f"   (paper: 1.14–1.16, heat-sink limit)")
    print(f"  g_fm  / g_std   = {min(gf_gs_vals):.2f} – {max(gf_gs_vals):.2f}"
          f"   (amplification with analytical T_P)")
    print(f"  g_fm_hs / g_std = {min(gh_gs_vals):.2f} – {max(gh_gs_vals):.2f}"
          f"   (amplification in heat-sink limit)")

    # --- Boost factor for d/h range (Eq. 6 limit) ---
    print(f"\n  Boost factor 2h²/(d²+h²) for d/h = 0.3–0.7:")
    for d_h in d_over_h_range:
        bf = boost_factor(d_h)
        print(f"    d/h = {d_h:.2f}:  boost = {bf:.3f}")
    print(f"  Paper: 1.34–1.83 for d/h = 0.3–0.7")

    # --- Effective distance argument ---
    print(f"\n  Effective distance argument (d/h = 0.44):")
    d = d_over_h_median * h_cell
    d_eff = (d**2 + h_cell**2) / h_cell
    d_actual = d + h_cell
    print(f"    d           = {d*1e6:.2f} μm")
    print(f"    h           = {h_cell*1e6:.2f} μm")
    print(f"    d + h       = {d_actual*1e6:.2f} μm  (actual distance)")
    print(f"    (d²+h²)/h   = {d_eff*1e6:.2f} μm  (effective distance)")
    print(f"    inflation   = {d_actual/d_eff:.3f}  "
          f"({(d_actual/d_eff - 1)*100:.0f}% overshoot)")

    # --- Mesh refinement analysis (new paragraph) ---
    print("\n" + "=" * 72)
    print("  MESH REFINEMENT BEHAVIOUR")
    print("=" * 72)
    print(f"\n  At fixed t = 1.0 ms, varying mesh size:")
    t_ref = 1.0e-3
    dT_ref = scriven_delta_T(t_ref)
    gt_ref = g_true_from_delta(dT_ref)

    for N in [75, 100, 125, 150, 200, 300]:
        h_n = 300e-6 / N
        d_n = d_over_h_median * h_n
        T_P_phys = dT_sup * erf(d_n / dT_ref)
        gs_n = g_std(h_n, d_n, dT_ref)
        gf_n = g_fm_physical(h_n, d_n, dT_ref)
        gh_n = g_fm_heatsink(h_n, d_n, dT_ref)
        bf_n = boost_factor(d_over_h_median)  # same for all meshes

        print(f"  N={N:3d}: h={h_n*1e6:.2f} μm, d={d_n*1e6:.2f} μm, "
              f"T_P-T_sat={T_P_phys:.4f} K, "
              f"g_std/g_true={gs_n/gt_ref:.3f}, "
              f"g_fm/g_true={gf_n/gt_ref:.3f}, "
              f"g_fm_hs/g_true={gh_n/gt_ref:.3f}")

    print("\n  Key observation: g_fm_hs/g_true remains nearly constant (~1.15)")
    print("  because boost = 2/(1+(d/h)²) depends only on d/h, not on h itself.")
    print("  Meanwhile, T_P → T_sat on finer meshes (erf(d/δ_T) → 0),")
    print("  making the heat-sink limit increasingly realistic.")

    print("\nDone.")
