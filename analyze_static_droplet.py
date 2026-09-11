#!/usr/bin/env python3
"""
Static droplet / Laplace benchmark  (WP-E, referee R2 #7).

Zero gravity, no phase change, matched densities and viscosities: the only
active physics is surface tension, so the exact solution is a droplet at
rest with a uniform pressure jump across the interface.  Any motion is
numerical -- the "spurious" or "parasitic" currents that a balanced-force
formulation is supposed to suppress.

Two reported quantities:

  spurious currents   Ca = mu * max|u| / sigma          (should -> 0)
  pressure jump       dp = p_in - p_out   vs   sigma * kappa

The T-Flows Tests/Vof/Spurious case is quasi-2D: a cylinder of radius R
with its axis along y, so the exact jump is the CYLINDRICAL Laplace value
sigma/R, not the spherical 2*sigma/R.

In/out regions are selected geometrically rather than from the volume
fraction, so the measurement does not depend on the interface being in the
right place -- which is part of what is being tested.

Usage:
  python3 analyze_static_droplet.py [case_dir] [--radius R] [--sigma S] ...
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

DEFAULT_CASE = os.path.join(os.environ.get("GOLD_DIR", "/home/jan/runs/RisingBubbleCases_gold"), "Spurious_gold")


def analyse(case, R, sigma, mu, cx, cz, axis, stride):
    files = sorted(f for f in glob.glob(os.path.join(case, "*-ts*.pvtu"))
                   if "bnd" not in os.path.basename(f)
                   and "front" not in os.path.basename(f))
    if not files:
        return []
    files = files[::stride]

    dp_exact = sigma / R          # cylindrical Laplace
    rows = []
    for fn in files:
        try:
            mesh = pv.read(fn)
        except Exception:
            continue
        cd = mesh.cell_data
        cc = mesh.cell_centers().points

        vname = next((k for k in cd if k.startswith("Velocity")), None)
        pname = next((k for k in cd if k.startswith("Pressure [")), None)
        if vname is None or pname is None:
            continue

        t = float(mesh.field_data["TIME"][0]) \
            if "TIME" in mesh.field_data else np.nan
        u = cd[vname]
        p = cd[pname]
        umag = np.linalg.norm(u, axis=1)

        # radial distance from the cylinder axis (axis is 'y' -> use x,z)
        if axis == "y":
            rad = np.hypot(cc[:, 0] - cx, cc[:, 2] - cz)
        elif axis == "z":
            rad = np.hypot(cc[:, 0] - cx, cc[:, 1] - cz)
        else:
            rad = np.hypot(cc[:, 1] - cx, cc[:, 2] - cz)

        inner = rad < 0.5 * R           # well inside
        outer = rad > 1.6 * R           # well outside
        if inner.sum() < 5 or outer.sum() < 5:
            continue

        dp = float(np.mean(p[inner]) - np.mean(p[outer]))
        rows.append(dict(
            t=t,
            umax=float(umag.max()),
            ca=float(mu * umag.max() / sigma),
            urms=float(np.sqrt(np.mean(umag**2))),
            dp=dp,
            dp_err=(dp - dp_exact) / dp_exact,
            p_in_sd=float(np.std(p[inner])),
            p_out_sd=float(np.std(p[outer])),
            n_in=int(inner.sum()), n_out=int(outer.sum()),
        ))
    return rows, dp_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=DEFAULT_CASE)
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--sigma", type=float, default=15.0e3)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--cx", type=float, default=1.0)
    ap.add_argument("--cz", type=float, default=1.0)
    ap.add_argument("--axis", default="y")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    print("\n" + "=" * 76)
    print("  STATIC DROPLET / LAPLACE BENCHMARK   (WP-E, R2 #7)")
    print("=" * 76)
    print(f"  case   : {a.case}")
    print(f"  R = {a.radius}, sigma = {a.sigma:g}, mu = {a.mu:g}, "
          f"axis = {a.axis}")

    out = analyse(a.case, a.radius, a.sigma, a.mu, a.cx, a.cz,
                  a.axis, a.stride)
    if not out or not out[0]:
        sys.exit("  no usable snapshots (need Velocity and Pressure fields)")
    rows, dp_exact = out

    La = a.sigma * 1.0 * (2 * a.radius) / a.mu**2
    print(f"  Laplace number La = sigma*rho*D/mu^2 = {La:.0f}")
    print(f"  exact jump  sigma/R = {dp_exact:,.1f} Pa   "
          f"(cylindrical; a sphere would give {2*dp_exact:,.1f})")
    print(f"  cells sampled: {rows[0]['n_in']} inside, "
          f"{rows[0]['n_out']} outside\n")

    print(f"  {'t':>10s} {'max|u|':>11s} {'Ca':>11s} {'u_rms':>11s}"
          f" {'dp [Pa]':>11s} {'dp err':>9s}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r['t']:10.4f} {r['umax']:11.4e} {r['ca']:11.4e} "
              f"{r['urms']:11.4e} {r['dp']:11.1f} {100*r['dp_err']:8.2f}%")

    ca = np.array([r["ca"] for r in rows])
    err = np.array([r["dp_err"] for r in rows])
    late = slice(len(rows) // 2, None)
    print("\n  " + "-" * 68)
    print(f"  Ca      : initial {ca[0]:.3e}   max {ca.max():.3e}   "
          f"final {ca[-1]:.3e}")
    print(f"            late-time mean {ca[late].mean():.3e}"
          f"  ({'decaying' if ca[-1] < ca.max() else 'NOT decaying'})")
    print(f"  dp error: final {100*err[-1]:.2f}%   "
          f"late-time mean {100*err[late].mean():.2f}%")
    print("\n  Kumar & Premachandran (2022) Table 5, same benchmark,")
    print("  independent CLSVOF code:")
    print("      structured  E_total,dp = 5.567%   E_o,dp = 0.725%")
    print("      triangular  E_total,dp = 4.633%   E_o,dp = 0.450%")
    print("      polyhedral  E_total,dp = 3.818%   E_o,dp = 0.373%")
    print("  (polyhedral lowest in their code too -- independent support")
    print("   for the isotropy argument, from a group with no stake in it)\n")


if __name__ == "__main__":
    main()
