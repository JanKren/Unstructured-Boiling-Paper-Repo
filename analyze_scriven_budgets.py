#!/usr/bin/env python3
"""
Scriven diagnostics: conservation budget, evaporation rate, corrected m=4.

Covers three referee demands in one pass over the VTU series, because they
all read the same fields:

  WP-B1 (R1 MC5/SC2, R2 #5)  time-integrated interfacial mass transfer
                             against the change in phase mass:
                                 int_0^t int_V mdot dV dt   vs   rho_v dV_v
  WP-C2 (R2 #4)              evaporation rate int_V mdot dV and the
                             interfacial heat flux against the analytical
                             Scriven rate
  m=4 re-measurement         with the LIQUID-SIDE band, not the manuscript's
                             0.3 < alpha < 0.7 band, which is ~52% vapour
                             cells and halves the mean gradient

Vapour volume and radius come from bench-data.dat (columns: time,
interfacial area, vapour volume, radius) where available; otherwise they
are integrated from the VTU alpha field.

Usage:
  python3 analyze_scriven_budgets.py [case_dir ...] [--stride N]
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

# ----------------------------------------------------------------------
#  Physical parameters (control file / analyze_convergence.py)
# ----------------------------------------------------------------------
RHO_V, RHO_L = 0.597, 958.4
K_V, K_L = 0.025, 0.679
CP_L = 4216.0
H_LV = 2.26e6
BETA = 4.06022
ALPHA_L = K_L / (RHO_L * CP_L)
R0 = 5.0e-5
T0 = R0**2 / (4.0 * BETA**2 * ALPHA_L)      # virtual origin

DEFAULT_CASES = [
    os.path.join(os.environ.get("MERLIN_DIR", "/home/jan/archive/Scriven-Merlin"), "Scriven-Poly-75"),
    os.path.join(os.environ.get("MERLIN_DIR", "/home/jan/archive/Scriven-Merlin"), "Scriven-Poly-100"),
    os.path.join(os.environ.get("MERLIN_DIR", "/home/jan/archive/Scriven-Merlin"), "Scriven-Poly-125"),
    os.path.join(os.environ.get("MERLIN_DIR", "/home/jan/archive/Scriven-Merlin"), "Scriven-Poly-150"),
]


def scriven_radius(t):
    """Analytical radius, shifted by the virtual origin."""
    return 2.0 * BETA * np.sqrt(ALPHA_L * (t + T0))


def scriven_rate(t):
    """Analytical dM/dt = rho_v * 4 pi R^2 dR/dt  [kg/s]."""
    tau = t + T0
    R = 2.0 * BETA * np.sqrt(ALPHA_L * tau)
    dRdt = BETA * np.sqrt(ALPHA_L / tau)
    return RHO_V * 4.0 * np.pi * R**2 * dRdt


def fourier_lsq(theta, g, m_max=8):
    cols = [np.ones_like(theta)]
    for m in range(1, m_max + 1):
        cols += [np.cos(m * theta), np.sin(m * theta)]
    coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), g, rcond=None)
    a0 = coef[0]
    if abs(a0) < 1e-30:
        return {m: np.nan for m in range(1, m_max + 1)}, a0
    return ({m: float(np.hypot(coef[2*m-1], coef[2*m]) / abs(a0))
             for m in range(1, m_max + 1)}, float(a0))


def scan_case(case, stride=1):
    """One pass over the VTU series; returns per-snapshot diagnostics."""
    files = sorted(f for f in glob.glob(os.path.join(case, "*-ts*.pvtu"))
                   if "front" not in os.path.basename(f))
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

        t = float(mesh.field_data["TIME"][0]) if "TIME" in mesh.field_data \
            else np.nan
        vol = cd["Grid Cell Volume [m^3]"]
        vof = cd["Vof Sharp [1]"]
        mdot = cd["Vof MassTransfer [kg/m^3/s]"]

        # NOTE ON UNITS.  Save_Vtu_Fields.f90:691 labels this field
        # "[kg/m^3/s]" but writes Vof % m_dot raw, and
        # Mass_Transfer_Estimate.f90:302 sets
        #     m_dot(c) = (q_1 - q_0) / h_lv,     q in [W]
        # so it is kg/s PER CELL -- an extensive quantity, already
        # integrated over the cell.  The total rate is therefore a plain
        # sum, NOT sum(mdot * vol).  Multiplying by the volume
        # under-predicts by ~16 orders of magnitude.
        m_int = float(np.sum(mdot))
        # vapour volume from alpha (vof<0.5 is vapour; use 1-alpha)
        v_vap = float(np.sum((1.0 - vof) * vol))

        row = dict(t=t, m_int=m_int, v_vap=v_vap)

        # --- corrected m=4 on the liquid-side interface band -----------
        if "Temperature Gradients [K/m]" in cd:
            cc = mesh.cell_centers().points
            vap = vof < 0.5
            if vap.sum() > 10:
                ctr = np.average(cc[vap], axis=0, weights=vol[vap])
                p = cc - ctr
                r = np.linalg.norm(p, axis=1)
                Req = (3.0 * v_vap / (4.0 * np.pi)) ** (1.0 / 3.0)
                # liquid cells within one cell size of the interface
                h = float(np.cbrt(np.median(vol)))
                band = (vof > 0.5) & (np.abs(r - Req) < 1.5 * h)
                if band.sum() > 200:
                    q = p[band]
                    rr = np.linalg.norm(q, axis=1)
                    nrm = q / rr[:, None]
                    grad = cd["Temperature Gradients [K/m]"][band]
                    g_n = np.abs(np.sum(grad * nrm, axis=1))
                    amps, a0 = fourier_lsq(np.arctan2(q[:, 1], q[:, 0]), g_n)
                    row.update(m2=amps[2], m4=amps[4], m6=amps[6],
                               m8=amps[8], gmean=a0, nband=int(band.sum()))
                    # interfacial heat flux  q = k_l * grad . n * A
                    area = 4.0 * np.pi * Req**2
                    row["q_int"] = K_L * a0 * area
        rows.append(row)
    return rows


def report(case, rows):
    name = os.path.basename(case)
    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)
    if not rows or len(rows) < 3:
        print("  insufficient snapshots\n")
        return

    t = np.array([r["t"] for r in rows])
    m_int = np.array([r["m_int"] for r in rows])
    v_vap = np.array([r["v_vap"] for r in rows])
    ok = np.isfinite(t) & np.isfinite(m_int)
    t, m_int, v_vap = t[ok], m_int[ok], v_vap[ok]
    srt = np.argsort(t)
    t, m_int, v_vap = t[srt], m_int[srt], v_vap[srt]

    # ---- WP-B1: cumulative transferred mass vs change in vapour mass ---
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (m_int[1:] + m_int[:-1])
                                           * np.diff(t))])
    dm_phase = RHO_V * (v_vap - v_vap[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        closure = np.where(np.abs(dm_phase) > 1e-18,
                           (cum - dm_phase) / np.abs(dm_phase), np.nan)

    print("  WP-B1  mass budget:  int int mdot dV dt   vs   rho_v * dV_vap")
    print(f"  {'t [ms]':>8s} {'transferred [kg]':>18s} {'phase dM [kg]':>15s}"
          f" {'closure err':>12s}")
    print("  " + "-" * 58)
    idx = np.linspace(1, len(t) - 1, min(6, len(t) - 1)).astype(int)
    for i in idx:
        print(f"  {1e3*t[i]:8.3f} {cum[i]:18.6e} {dm_phase[i]:15.6e} "
              f"{100*closure[i]:11.2f}%")
    fin = closure[-1]
    print(f"\n  final closure error: {100*fin:.2f}%"
          f"   (note: sampled every {len(t)} snapshots, so the time"
          " integral is coarse)")

    # ---- WP-C2: evaporation rate vs analytical ------------------------
    ana = scriven_rate(t)
    print("\n  WP-C2  evaporation rate  int mdot dV   vs analytical")
    print(f"  {'t [ms]':>8s} {'sim [kg/s]':>14s} {'analytic [kg/s]':>16s}"
          f" {'ratio':>8s}")
    print("  " + "-" * 50)
    for i in idx:
        rr = m_int[i] / ana[i] if ana[i] != 0 else np.nan
        print(f"  {1e3*t[i]:8.3f} {m_int[i]:14.5e} {ana[i]:16.5e} {rr:8.3f}")

    # ---- radius check --------------------------------------------------
    Req = (3.0 * v_vap / (4.0 * np.pi)) ** (1.0 / 3.0)
    Ran = scriven_radius(t)
    print("\n  radius  R_sim / R_Scriven")
    for i in idx:
        print(f"  {1e3*t[i]:8.3f} {1e6*Req[i]:10.2f} um  /"
              f" {1e6*Ran[i]:8.2f} um  = {Req[i]/Ran[i]:6.3f}")

    # ---- corrected m=4 -------------------------------------------------
    m4 = [r.get("m4") for r in rows if r.get("m4") is not None]
    if m4:
        m4 = np.array(m4, dtype=float)
        m2 = np.array([r.get("m2", np.nan) for r in rows
                       if r.get("m4") is not None], dtype=float)
        nb = [r.get("nband") for r in rows if r.get("m4") is not None]
        print(f"\n  m=4 on the LIQUID-SIDE band ({int(np.median(nb))} cells "
              "median):")
        print(f"    mean {100*np.nanmean(m4):.2f}%   "
              f"range {100*np.nanmin(m4):.2f}-{100*np.nanmax(m4):.2f}%"
              f"   (m=2 mean {100*np.nanmean(m2):.2f}% -- should be ~0)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    print("\n" + "=" * 78)
    print("  SCRIVEN BUDGETS AND DIAGNOSTICS  (WP-B1, WP-C2, corrected m=4)")
    print("=" * 78 + "\n")
    for c in (a.cases or DEFAULT_CASES):
        report(c, scan_case(c, a.stride))
