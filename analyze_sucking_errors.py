#!/usr/bin/env python3
"""Quantified velocity errors for the Sucking problem, and the sub-cell oscillation.

    python3 analyze_sucking_errors.py

Answers the JCP referee's request on Figure 5(b): the velocity profile shows
local oscillations at the discontinuity and a plateau that sits slightly off the
analytical value, and the referee asked for those to be quantified rather than
described -- specifically the plateau error away from the interface, the error
in the interfacial jump condition, an L2 error excluding a band around the
discontinuity, and whether any of it feeds through to the mass-transfer rate,
the interface speed or the global mass balance.

It also tests the explanation for the oscillatory error convergence: that the
error depends on where the interface sits *inside* its cell. On a fixed mesh the
interface sweeps through each cell, and a scheme that reconstructs it from a
cell-averaged colour function has no reason to be equally accurate at every
sub-cell position. If that is the mechanism the error should correlate with the
phase

    phi = (x_int / h) mod 1

and repeat with period one as the interface crosses successive cells, rather
than drift.

WHICH DATA IS USABLE.  The Sucking directory holds several runs written on top
of one another: of 401 volume snapshots, 201 are on the production 400-cell mesh
and 200 on a 199-cell mesh, spread over three write dates, and 100 of 400
consecutive pairs move the interface backwards.  Only the 400-cell snapshots are
read here, and the trend analysis is further restricted to the single coherent
block (ts >= 9400, written 2026-02-03).  Screening is reported so the reader can
see what was dropped rather than having to trust it.

Conventions follow analyze_sucking_profiles.py, which produced Figure 5:
vof = 0 in the vapour (x < x_int) and 1 in the liquid, the analytical clock is
t = 0.1 s + ts * 2.5e-5 s, and the analytical plateau is

    u_l = v_int (1 - rho_v/rho_l),   v_int = beta sqrt(alpha_v / t)

so the velocity jump across the interface is exactly m_dot (1/rho_v - 1/rho_l).
That identity is what makes the jump a check rather than a restatement: the
plateau velocity and the interface speed are measured independently here, and
the mass flux implied by each is compared.
"""

import glob
import os
import re
import sys

import numpy as np
import pyvista as pv
from scipy.special import erfc

CASE = os.environ.get("SUCKING_DIR", "/home/jan/runs/tflows-vof/Validation/Sucking")
RHO_V, RHO_L = 0.597, 958.4
K_V, K_L = 0.025, 0.679
CP_V, CP_L = 2030.0, 4216.0
H_LV = 2.26e6
T_SAT, T_INF = 10.0, 15.0
ALPHA_V = K_V / (RHO_V * CP_V)
ALPHA_L = K_L / (RHO_L * CP_L)
DT = 2.5e-5              # solver time step [s]
T0 = 0.1                 # analytical time of the initial condition [s]
BAND = 2                 # cells excluded either side, matching tab:sucking_errors
CLEAN_TS = 9400          # first step of the single coherent block


def beta_root():
    """Growth constant from the Sucking transcendental relation."""
    def f(b):
        inb = b**2 * (ALPHA_V * RHO_V**2) / (ALPHA_L * RHO_L**2)
        return (b - ((T_INF - T_SAT) * CP_V * K_L * np.sqrt(ALPHA_V) * np.exp(-inb))
                / (H_LV * K_V * np.sqrt(np.pi * ALPHA_L) * erfc(np.sqrt(inb))))
    a, b = 0.0, 1.0
    for _ in range(200):
        m = 0.5 * (a + b)
        a, b = (m, b) if f(m) < 0 else (a, m)
    return 0.5 * (a + b)


def profile(path):
    """Axis profile (x, vof, u, T) sorted by x, or None if unreadable."""
    try:
        m = pv.read(path)
    except Exception:
        return None
    c = m.cell_centers().points
    s = (np.abs(c[:, 1]) < 1e-4) & (np.abs(c[:, 2]) < 1e-4)
    if s.sum() < 10:
        return None
    o = np.argsort(c[s, 0])
    return (c[s, 0][o],
            np.asarray(m.cell_data["Vof Sharp [1]"])[s][o],
            np.asarray(m.cell_data["Velocity [m/s]"])[s][o, 0],
            np.asarray(m.cell_data["Temperature [K]"])[s][o])


def interface(x, vof):
    """Linear alpha = 1/2 crossing, and the index of the cell holding it."""
    k = np.where(np.diff(np.sign(vof - 0.5)) != 0)[0]
    if len(k) == 0:
        return np.nan, -1
    i = k[0]
    return np.interp(0.5, [vof[i], vof[i + 1]], [x[i], x[i + 1]]), i


def metrics(path, beta):
    """Every quantity the referee asked for, for one snapshot."""
    p = profile(path)
    if p is None:
        return None
    x, vof, u, T = p
    if len(x) != 400:                       # production mesh only
        return None
    h = x[1] - x[0]
    ts = int(re.search(r"ts0*(\d+)", os.path.basename(path)).group(1))
    t = T0 + ts * DT

    xi, k = interface(x, vof)
    if not np.isfinite(xi):
        return None
    xi_a = 2.0 * beta * np.sqrt(ALPHA_V * t)
    v_int_a = beta * np.sqrt(ALPHA_V / t)
    u_plate_a = v_int_a * (1.0 - RHO_V / RHO_L)

    # liquid, excluding BAND cells next to the interface and the last cell
    liq = (np.arange(len(x)) > k + BAND) & (np.arange(len(x)) < len(x) - 1)
    if liq.sum() < 20:
        return None
    u_plate = float(np.mean(u[liq]))
    u_scatter = float(np.std(u[liq]))

    # vapour side, excluding the band: should be identically zero
    vap = (np.arange(len(x)) < k - BAND)
    u_vap = float(np.mean(u[vap])) if vap.sum() > 5 else np.nan

    # L2 of velocity over the whole axis, and with the band excluded
    u_a = np.where(x > xi_a, u_plate_a, 0.0)
    keep = np.abs(np.arange(len(x)) - k) > BAND
    l2_all = float(np.sqrt(np.mean((u - u_a) ** 2)) / u_plate_a)
    l2_cut = float(np.sqrt(np.mean((u[keep] - u_a[keep]) ** 2)) / u_plate_a)

    # peak overshoot inside the excluded band, as a fraction of the plateau
    over = float(np.max(np.abs(u[~keep] - u_a[~keep])) / u_plate_a) if (~keep).sum() else np.nan

    # Mass flux two ways.  The jump route is NOT independent of the plateau --
    # u_l - u_v IS the jump -- so it is reported to show how much m_dot the
    # velocity error would imply, and compared against the interface route,
    # which is measured from the interface position instead.
    mdot_jump = (u_plate - u_vap) / (1.0 / RHO_V - 1.0 / RHO_L)
    mdot_anal = RHO_V * v_int_a

    return dict(ts=ts, t=t, h=h, xi=xi, xi_a=xi_a,
                e_xi=100.0 * (xi - xi_a) / xi_a,
                phase=(xi / h) % 1.0,
                u_plate=u_plate, u_plate_a=u_plate_a,
                e_plate=100.0 * (u_plate - u_plate_a) / u_plate_a,
                u_scatter=100.0 * u_scatter / u_plate_a,
                u_vap=u_vap, l2_all=100.0 * l2_all, l2_cut=100.0 * l2_cut,
                over=100.0 * over,
                mdot_jump=mdot_jump, mdot_anal=mdot_anal,
                e_mdot=100.0 * (mdot_jump - mdot_anal) / mdot_anal)


def main():
    if not os.path.isdir(CASE):
        sys.exit(f"no case at {CASE}")
    beta = beta_root()
    print(f"beta = {beta:.8f}\n")

    files = sorted((f for f in glob.glob(os.path.join(CASE, "sucking-ts*.vtu"))
                    if "front" not in f),
                   key=lambda f: int(re.search(r"ts0*(\d+)", f).group(1)))
    rows = [r for r in (metrics(f, beta) for f in files) if r]
    clean = [r for r in rows if r["ts"] >= CLEAN_TS]
    print(f"{len(files)} volume snapshots, {len(rows)} on the 400-cell mesh, "
          f"{len(clean)} in the coherent block (ts >= {CLEAN_TS})\n")

    print("Figure 5(b) instants")
    print(f"  {'t [s]':>6} {'plateau u':>11} {'exact':>10} {'err %':>8} "
          f"{'scatter %':>10} {'L2 all %':>9} {'L2 cut %':>9} {'peak %':>8} {'mdot err %':>11}")
    for r in rows:
        if r["ts"] in (4000, 10000, 16000):
            print(f"  {r['t']:6.2f} {r['u_plate']:11.6f} {r['u_plate_a']:10.6f} "
                  f"{r['e_plate']:+8.2f} {r['u_scatter']:10.3f} {r['l2_all']:9.2f} "
                  f"{r['l2_cut']:9.2f} {r['over']:8.1f} {r['e_mdot']:+11.2f}")

    if clean:
        a = lambda k: np.array([r[k] for r in clean])
        print(f"\nCoherent block, {len(clean)} snapshots, t = "
              f"{clean[0]['t']:.3f}-{clean[-1]['t']:.3f} s")
        for k, lab in (("e_plate", "plateau velocity error [%]"),
                       ("e_xi", "interface position error [%]"),
                       ("e_mdot", "mass-flux error from the jump [%]"),
                       ("l2_cut", "L2 velocity error, band excluded [%]"),
                       ("l2_all", "L2 velocity error, full profile [%]"),
                       ("over", "peak overshoot in the band [%]")):
            v = a(k)
            print(f"  {lab:<38} mean {v.mean():+8.3f}   sd {v.std():7.3f}   "
                  f"range {v.min():+8.3f} to {v.max():+8.3f}")

        # does the error track the sub-cell position of the interface?
        print("\nSub-cell dependence (phase = (x_int/h) mod 1)")
        ph = a("phase")
        # least squares on A cos(2 pi phi) + B sin(2 pi phi) + C, which measures
        # a one-per-cell oscillation of any phase; a bare cosine correlation
        # misses it whenever the extremum does not sit at phi = 0.
        M = np.column_stack([np.cos(2*np.pi*ph), np.sin(2*np.pi*ph),
                             np.cos(4*np.pi*ph), np.sin(4*np.pi*ph),
                             np.ones_like(ph)])
        for k, lab in (("e_plate", "plateau error"), ("e_xi", "interface error"),
                       ("l2_cut", "L2 (band excluded)")):
            v = a(k)
            c, *_ = np.linalg.lstsq(M, v, rcond=None)
            amp1 = np.hypot(c[0], c[1]); amp2 = np.hypot(c[2], c[3])
            resid = v - M @ c
            r2 = 1.0 - resid.var() / v.var()
            print(f"  {lab:<22} 1/cell amplitude {amp1:6.3f}   2/cell {amp2:6.3f}   "
                  f"sd {v.std():6.3f}   R2 {r2:5.3f}")
        nb = 8
        idx = np.minimum((ph * nb).astype(int), nb - 1)
        print(f"\n  {'phase bin':>12} {'n':>4} {'plateau err %':>14} {'iface err %':>13}")
        for b in range(nb):
            m = idx == b
            if m.sum():
                print(f"  {b/nb:5.2f}-{(b+1)/nb:4.2f} {m.sum():>4} "
                      f"{a('e_plate')[m].mean():>14.3f} {a('e_xi')[m].mean():>13.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
