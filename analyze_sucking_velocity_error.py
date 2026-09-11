#!/usr/bin/env python3
"""
T2.2 -- quantitative velocity errors for the Sucking problem (Reviewer 2, point 2).

The manuscript shows the velocity profile agrees "apart from a small discrepancy
near the interface" but never quantifies it.  Three metrics, at t = 0.2, 0.35, 0.5 s:

  1. plateau error     -- the liquid is pushed at a uniform u_l away from the
                          interface; compared with the analytical plateau.
  2. jump-condition    -- u_l should equal x_int_dot (1 - rho_v/rho_l).  Evaluated
                          with the SIMULATION's own interface speed, so it tests
                          the internal consistency of the coupling rather than the
                          accumulated interface-position error.
  3. L2 excluding the  -- the discontinuity is spread over the cells straddling the
     interface band       interface by construction; a norm that includes them
                          measures the mesh, not the solution.  Excluded band is
                          +-2 cells.

Reuses the analytical solution of analyze_sucking_profiles.py unchanged.

Usage:  python3 analyze_sucking_velocity_error.py [--case DIR]
"""

import argparse
import glob
import os

import numpy as np
import pyvista as pv

from analyze_sucking_profiles import (
    RHO_G, RHO_L, ALPHA_G, compute_beta,
    analytical_interface_position, analytical_velocity,
)

DT = 2.5e-5          # s, from the case control
TARGETS = (0.35, 0.45, 0.55)   # analytical times inside the clean subset
T_START = 0.1        # the initial 2.2 mm vapour layer corresponds to
                     # t_analytical = 0.1 s (analyze_sucking_profiles.py:136),
                     # so t_analytical = T_START + t_sim.  The manuscript's
                     # "t = 0.2, 0.35, 0.5 s" are ANALYTICAL times.


def profile(fn):
    """Cell-centred 1-D profile along x: (x, u, alpha, h)."""
    m = pv.read(fn)
    cc = m.cell_centers().points
    u = np.asarray(m.cell_data["Velocity [m/s]"], float)[:, 0]
    a = np.asarray(m.cell_data["Vof Sharp [1]"], float)
    vol = np.asarray(m.cell_data["Grid Cell Volume [m^3]"], float)
    x = cc[:, 0]
    xs = np.unique(np.round(x, 12))
    h = float(np.median(np.diff(xs)))
    ub = np.array([u[np.isclose(x, xv)].mean() for xv in xs])
    ab = np.array([a[np.isclose(x, xv)].mean() for xv in xs])
    return xs, ub, ab, h


def interface_x(xs, a):
    """Interface position from the integrated vapour fraction.

    In 1-D the vapour layer thickness is exactly int (1-alpha) dx, which is
    noise-free.  The alpha=0.5 crossing is not usable for a time derivative
    here: the interface advances ~0.6 cells between stored snapshots, so the
    sub-cell interpolation error dominates and produces spurious velocities
    (it gave x_int_dot = -0.43 m/s at t = 0.35 s, i.e. the wrong sign and
    two orders of magnitude too large).
    """
    h = np.median(np.diff(xs))
    return float(np.sum(1.0 - a) * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case",
                    default="/home/jan/runs/tflows-vof/Validation/Sucking")
    a = ap.parse_args()

    beta = compute_beta()
    files = sorted((f for f in glob.glob(os.path.join(a.case, "*-ts*.vtu"))
                    + glob.glob(os.path.join(a.case, "*-ts*.pvtu"))
                    if "plane" not in os.path.basename(f)
                    and "front" not in os.path.basename(f)),
                   key=lambda f: int(f.split("-ts")[1].split(".")[0]))
    if not files:
        raise SystemExit("no snapshots")
    tmap = {int(f.split("-ts")[1].split(".")[0]) * DT: f for f in files}
    times = np.array(sorted(tmap))

    print(f"\n{'='*94}\n  T2.2  Sucking velocity errors  (beta = {beta:.6f})"
          f"\n{'='*94}")
    print(f"  {len(files)} snapshots, t = {times[0]:.4f} .. {times[-1]:.4f} s\n")
    print("  %-7s %11s %11s %9s | %11s %11s %9s | %10s" %
          ("t_an[s]", "u_pl sim", "u_pl exact", "err", "x_int sim",
           "x_int exact", "err", "L2 (excl.)"))
    print("  " + "-" * 96)

    rows = []
    for t_an in TARGETS:
        k = times[np.argmin(np.abs(times - (t_an - T_START)))]
        t_anal = k + T_START
        xs, u, al, h = profile(tmap[k])
        xi_sim = interface_x(xs, al)
        xi_ex = analytical_interface_position(beta, t_anal)
        u_ex = analytical_velocity(beta, xs, t_anal)
        u_pl_ex = u_ex[xs > xi_ex].max() if (xs > xi_ex).any() else np.nan

        # plateau: liquid side, at least 10 cells beyond the interface
        pl = xs > xi_sim + 10 * h
        u_pl = u[pl].mean() if pl.any() else np.nan

        # L2 excluding +-2 cells about the SIMULATED interface
        keep = np.abs(xs - xi_sim) > 2 * h
        l2 = np.sqrt(np.mean((u[keep] - u_ex[keep]) ** 2)) / abs(u_pl_ex)

        print("  %-7.3f %11.6e %11.6e %8.3f%% | %11.6e %11.6e %8.3f%% | %9.4f%%"
              % (t_anal, u_pl, u_pl_ex, 100 * (u_pl / u_pl_ex - 1),
                 xi_sim, xi_ex, 100 * (xi_sim / xi_ex - 1), 100 * l2))
        rows.append((k, xi_sim, u_pl, u_pl_ex, t_anal))

    # jump condition with the simulation's OWN interface speed
    print("\n  Interfacial jump condition  u_l = x_int_dot (1 - rho_v/rho_l),")
    print("  evaluated with the simulation's own interface speed (central"
          " difference over +-1 stored snapshot):\n")
    print("  %-7s %13s %13s %13s %9s" %
          ("t [s]", "x_int_dot", "u_l predicted", "u_l measured", "err"))
    print("  " + "-" * 62)
    for t_an in TARGETS:
        k = times[np.argmin(np.abs(times - (t_an - T_START)))]
        i = int(np.argmin(np.abs(times - k)))
        if i == 0 or i == len(times) - 1:
            continue
        def xi_at(j):
            xs_, u_, al_, h_ = profile(tmap[times[j]])
            return interface_x(xs_, al_)
        x0, x1 = xi_at(i - 1), xi_at(i + 1)
        xdot = (x1 - x0) / (times[i + 1] - times[i - 1])
        u_pred = xdot * (1.0 - RHO_G / RHO_L)
        u_meas = [r[2] for r in rows if abs(r[0] - k) < 1e-12][0]
        print("  %-7.3f %13.6e %13.6e %13.6e %8.3f%%" %
              (k + T_START, xdot, u_pred, u_meas,
               100 * (u_meas / u_pred - 1)))


if __name__ == "__main__":
    main()
