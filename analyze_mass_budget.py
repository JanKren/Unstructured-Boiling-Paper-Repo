#!/usr/bin/env python3
"""
Global mass budget for any case with full-domain output  (WP-B1, R2 #5).

Compares the time-integrated interfacial mass transfer with the change in
phase mass, which is the comparison R2 asked for:

    int_0^t int_V mdot dV dt      vs      rho_v * [ V_vap(t) - V_vap(0) ]

Geometry-independent, so the same routine serves the planar Sucking and
Stefan problems and the spherical Scriven problem.

Two conventions that matter and are easy to get wrong:

  * Vapour is alpha < 0.5 (Mass_Transfer_Estimate.f90 treats fun < 0.5 as
    vapour), so V_vap = sum (1 - alpha) * V_cell.

  * "Vof MassTransfer [kg/m^3/s]" is MISLABELLED.  Save_Vtu_Fields.f90
    writes Vof % m_dot raw, and Mass_Transfer_Estimate.f90 sets it to
    (q_1 - q_0)/h_lv with q in watts -- so it is kg/s PER CELL, already
    integrated over the cell.  The domain total is a plain sum; weighting
    by cell volume under-predicts by ~16 orders of magnitude.

Plane-extract output (files named *-x_plane_*) cannot be used: a slice has
no cell volumes and no closed control volume.

Usage:
  python3 analyze_mass_budget.py <case_dir> [--pattern PREFIX] [--stride N]
                                 [--rho-v RHO]
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.exit("pyvista required")


def scan(case, pattern, stride):
    # Parallel runs write a .pvtu wrapper, serial runs a plain .vtu; prefer
    # the wrapper when both are present so the pieces are read only once.
    files = sorted(glob.glob(os.path.join(case, f"{pattern}-ts*.pvtu")))
    if not files:
        files = sorted(glob.glob(os.path.join(case, f"{pattern}-ts*.vtu")))
    files = [f for f in files
             if "front" not in os.path.basename(f)
             and "plane" not in os.path.basename(f)]
    if not files:
        return None
    files = files[::stride]

    rows = []
    for fn in files:
        try:
            mesh = pv.read(fn)
        except Exception:
            continue
        cd = mesh.cell_data
        if "Vof MassTransfer [kg/m^3/s]" not in cd:
            continue
        t = float(mesh.field_data["TIME"][0]) \
            if "TIME" in mesh.field_data else np.nan
        vol = cd["Grid Cell Volume [m^3]"]
        vof = cd["Vof Sharp [1]"]
        mdot = cd["Vof MassTransfer [kg/m^3/s]"]      # kg/s per cell

        # sensible enthalpy referred to T_sat, for the energy budget
        h_sens = np.nan
        if all(k in cd for k in ("Physical Density [kg/m^3]",
                                 "Physical Capacity [J/kg/K]",
                                 "Temperature [K]")):
            rho = cd["Physical Density [kg/m^3]"]
            cp = cd["Physical Capacity [J/kg/K]"]
            T = cd["Temperature [K]"]
            t_sat = float(np.median(T[vof < 0.5])) if (vof < 0.5).any() \
                else float(T.min())
            h_sens = float(np.sum(rho * cp * (T - t_sat) * vol))

        rows.append((t,
                     float(np.sum(mdot)),
                     float(np.sum((1.0 - vof) * vol)),
                     h_sens))
    return np.array(rows) if rows else None


def report(case, data, rho_v, H_LV):
    print("=" * 76)
    print(f"  {os.path.basename(case.rstrip('/'))}")
    print("=" * 76)
    if data is None or len(data) < 3:
        print("  no usable full-domain snapshots\n")
        return

    t, rate, v_vap = data[:, 0], data[:, 1], data[:, 2]
    srt = np.argsort(t)
    t, rate, v_vap = t[srt], rate[srt], v_vap[srt]

    cum = np.concatenate([[0.0],
                          np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    dm = rho_v * (v_vap - v_vap[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(np.abs(dm) > 1e-20, (cum - dm) / np.abs(dm), np.nan)

    print(f"  {len(t)} snapshots, t = {t[0]:.4g} to {t[-1]:.4g} s"
          f"   (rho_v = {rho_v})")
    print(f"  {'t':>12s} {'transferred [kg]':>18s} {'phase dM [kg]':>16s}"
          f" {'closure':>10s} {'gap [kg]':>12s}")
    print("  " + "-" * 72)
    for i in np.linspace(1, len(t) - 1, min(7, len(t) - 1)).astype(int):
        print(f"  {t[i]:12.5e} {cum[i]:18.6e} {dm[i]:16.6e} "
              f"{100*err[i]:9.2f}% {cum[i]-dm[i]:12.4e}")

    gap = cum - dm
    n2 = len(t) // 2
    print(f"\n  final closure error : {100*err[-1]:+.2f}%")
    print(f"  absolute gap        : {gap[n2]:.3e} kg at t={t[n2]:.3g}"
          f"  ->  {gap[-1]:.3e} kg at t={t[-1]:.3g}")
    growth = abs(gap[-1]) / abs(gap[n2]) if abs(gap[n2]) > 1e-30 else np.nan
    print(f"  gap growth (2nd half): {growth:.2f}x   "
          f"({'no secular drift' if growth < 1.5 else 'DRIFTING'})")
    print("  (early closure error is dominated by coarse time sampling of"
          " the initial transient)")

    # ---- energy budget -------------------------------------------------
    # Latent heat absorbed by evaporation must come out of the sensible
    # enthalpy of the domain, plus whatever crosses the boundaries:
    #
    #     dH_sensible(t) + h_lv * m(t)  =  integral of Q_boundary dt
    #
    # For a domain whose thermal layer has not reached the far boundary the
    # right-hand side is negligible, so the residual measures how well
    # energy is conserved.  Where it is not negligible the residual IS the
    # net boundary flux, which is the meaningful quantity for a case with a
    # heated wall (Stefan).
    if data.shape[1] > 3 and np.isfinite(data[:, 3]).all():
        h = data[srt, 3]
        dh = h - h[0]
        latent = H_LV * cum
        resid = dh + latent
        scale = np.maximum(np.abs(latent), 1e-30)
        print(f"\n  ENERGY BUDGET (h_lv = {H_LV:.3g} J/kg, T_ref = T_sat)")
        print(f"  {'t':>12s} {'dH_sens [J]':>15s} {'latent [J]':>14s}"
              f" {'residual':>13s} {'/latent':>9s}")
        print("  " + "-" * 68)
        for i in np.linspace(1, len(t) - 1, min(6, len(t) - 1)).astype(int):
            print(f"  {t[i]:12.5e} {dh[i]:15.5e} {latent[i]:14.5e} "
                  f"{resid[i]:13.4e} {100*resid[i]/scale[i]:8.2f}%")
        print(f"\n  final residual / latent : {100*resid[-1]/scale[-1]:+.2f}%")
        print("  a residual near zero means the domain is effectively closed;")
        print("  a systematic non-zero residual is the net boundary flux.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--pattern", default=None,
                    help="file prefix; inferred if omitted")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rho-v", type=float, default=0.597)
    ap.add_argument("--h-lv", type=float, default=2.26e6)
    a = ap.parse_args()

    pat = a.pattern
    if pat is None:
        cands = {os.path.basename(f).split("-ts")[0]
                 for ext in ("pvtu", "vtu")
                 for f in glob.glob(os.path.join(a.case, f"*-ts*.{ext}"))
                 if "plane" not in f and "front" not in f}
        if not cands:
            sys.exit("  no full-domain *-ts*.[p]vtu found "
                     "(plane extracts cannot be used for a volume budget)")
        pat = sorted(cands, key=len)[0]
        print(f"\n  (inferred file prefix: {pat})")

    report(a.case, scan(a.case, pat, a.stride), a.rho_v, a.h_lv)
