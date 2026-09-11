#!/usr/bin/env python3
"""
Closed energy budget for the one-dimensional Stefan problem  (WP-B1, R2 #5).

The mass budget (analyze_mass_budget.py) works for every benchmark, but the
ENERGY budget only closes where the net boundary flux is a single, well
defined term.  Stefan is that case:

    x = 0        superheated wall, Dirichlet T_w  (the only energy inlet)
    x = L        outflow with q = 0
    y, z         symmetry with q = 0

so all heat entering the domain crosses one plane of known area, and

    int_0^t Q_wall dt   =   dH_sensible(t)  +  h_lv * m_evaporated(t)

Scriven (unbounded superheated liquid) and Sucking (outflow carrying
enthalpy) have no such single term, which is why this script is Stefan-only.

Q_wall is evaluated from the wall-adjacent cell rather than from the stored
"Temperature Gradients" field: at a boundary cell the least-squares gradient
is one-sided and is itself part of what the paper is assessing, so using it
here would make the check partly circular.  The Dirichlet difference

    q = k * (T_w - T_P) / x_P

is the same expression the solver applies for the boundary condition.

Usage:
  python3 analyze_stefan_energy.py <case_dir> [--t-wall 10.1] [--t-sat 0.1]
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.exit("pyvista is required")

# Stefan case properties (control file, OneD/Stefan)
RHO_V, RHO_L = 0.597, 958.4
CP_V, CP_L = 2030.0, 4216.0
H_LV = 2.26e6


def timestep(fn):
    return int(os.path.basename(fn).split("-ts")[1].split(".")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--t-wall", type=float, default=10.1)
    ap.add_argument("--t-sat", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=5.0e-3)
    a = ap.parse_args()

    files = sorted(
        (f for ext in ("pvtu", "vtu")
         for f in glob.glob(os.path.join(a.case, f"*-ts*.{ext}"))
         if "front" not in os.path.basename(f)
         and "plane" not in os.path.basename(f)),
        key=timestep)
    if not files:
        sys.exit("no full-domain snapshots found")

    rows = []
    for fn in files:
        mesh = pv.read(fn)
        cd = mesh.cell_data
        vol = np.asarray(cd["Grid Cell Volume [m^3]"], dtype=np.float64)
        t_c = np.asarray(cd["Temperature [K]"], dtype=np.float64)
        alpha = np.asarray(cd["Vof Sharp [1]"], dtype=np.float64)
        kcond = np.asarray(cd["Physical Conductivity [W/m/K]"],
                           dtype=np.float64)
        xc = np.asarray(mesh.cell_centers().points[:, 0], dtype=np.float64)

        # wall-adjacent layer: the cells sharing the smallest x-centre
        x_min = xc.min()
        wall = np.isclose(xc, x_min, rtol=0.0, atol=0.25 * x_min)
        # cross-sectional area of that layer, from volume / thickness
        thick = 2.0 * x_min                      # centre at half a cell
        area = vol[wall].sum() / thick

        q_wall = (kcond[wall] * (a.t_wall - t_c[wall]) / x_min
                  * vol[wall] / thick).sum()     # W, area-weighted

        # vapour volume: same convention as analyze_mass_budget.py, so the
        # mass and energy budgets are measured consistently
        v_vap = ((1.0 - alpha) * vol).sum()

        # Sensible enthalpy relative to T_sat, with a SHARP phase split.
        #
        # A volume-blended rho*cp is wrong here and by a large factor.  In an
        # interface cell the liquid fraction is at T_sat by definition, but
        # the cell-averaged T belongs to the vapour side of the profile;
        # since rho_l*cp_l is 3300x rho_v*cp_v, weighting liquid capacity by
        # that average attributes vapour superheat to liquid mass and
        # overstates the stored energy by ~50x.  The premise is checkable in
        # the data: liquid cells sit at T_sat to within 1.6e-4 K, and their
        # total contribution is -5.3e-6 J against a 0.19 J budget.
        vap = alpha < 0.5
        h_sens = ((RHO_V * CP_V) * (t_c[vap] - a.t_sat) * vol[vap]).sum() \
               + ((RHO_L * CP_L) * (t_c[~vap] - a.t_sat) * vol[~vap]).sum()

        rows.append((timestep(fn) * a.dt, q_wall, v_vap, h_sens, area))

    t = np.array([r[0] for r in rows])
    q = np.array([r[1] for r in rows])
    v_vap = np.array([r[2] for r in rows])
    h_sens = np.array([r[3] for r in rows])

    # cumulative wall heat, trapezoid on the snapshot times
    q_cum = np.concatenate(([0.0], np.cumsum(0.5 * (q[1:] + q[:-1])
                                            * np.diff(t))))
    latent = H_LV * RHO_V * (v_vap - v_vap[0])
    d_h = h_sens - h_sens[0]
    stored = d_h + latent

    print(f"\n{'='*76}\n  STEFAN ENERGY BUDGET  --  {os.path.basename(a.case)}"
          f"\n{'='*76}")
    print(f"  wall T = {a.t_wall} , T_sat = {a.t_sat} "
          f"(superheat {a.t_wall - a.t_sat:g} K),  area = {rows[0][4]:.4e} m2")
    print(f"  {len(files)} snapshots, t = {t[0]:g} to {t[-1]:g} s\n")
    print(f"  {'t [s]':>8s} {'Q_wall dt [J]':>15s} {'dH_sens [J]':>13s}"
          f" {'latent [J]':>12s} {'closure':>9s}")
    print("  " + "-" * 62)
    for i in range(0, len(t), max(1, len(t) // 7)):
        if i == 0:
            continue
        clo = (stored[i] - q_cum[i]) / q_cum[i] * 100.0
        print(f"  {t[i]:8.3f} {q_cum[i]:15.6e} {d_h[i]:13.6e}"
              f" {latent[i]:12.6e} {clo:8.2f}%")
    i = len(t) - 1
    clo = (stored[i] - q_cum[i]) / q_cum[i] * 100.0
    print(f"  {t[i]:8.3f} {q_cum[i]:15.6e} {d_h[i]:13.6e}"
          f" {latent[i]:12.6e} {clo:8.2f}%")

    print(f"\n  final closure error : {clo:+.2f}%"
          f"   (stored {stored[i]:.6e} J vs supplied {q_cum[i]:.6e} J)")
    print(f"  latent fraction     : {latent[i] / stored[i] * 100:.1f}% "
          f"of the stored energy")
    half = len(t) // 2
    g1 = abs(stored[half] - q_cum[half])
    g2 = abs(stored[i] - q_cum[i])
    print(f"  absolute gap        : {g1:.4e} J at t={t[half]:g}"
          f"  ->  {g2:.4e} J at t={t[i]:g}")
    print(f"  gap growth (2nd half): {g2 / g1:.2f}x"
          f"   {'(no secular drift)' if g2 / g1 < 1.5 else '(DRIFTING)'}\n")


if __name__ == "__main__":
    main()
