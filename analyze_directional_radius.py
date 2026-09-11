#!/usr/bin/env python3
"""Direction-resolved front radius: is the radius error a single number?

The convergence tables report one relative error per mesh and instant, which
presumes the interface is spherical enough for a single radius to describe it.
On structured meshes it is not: the bubble matches the analytical radius along
the coordinate axes and overshoots towards the cube corners, so the tabulated
error is a mean over directions that disagree by more than the error itself.

Measured on the SAME quantity the tables use -- the reconstructed front, whose
element centroids give R_front as an area-weighted mean distance -- so that the
directional values and the tabulated mean are the same measurement resolved
differently, not two different measurements.

For each front element the unit direction from the area-weighted centroid gives
the cubic invariant s = nx^4+ny^4+nz^4, which is 1 on a coordinate axis, 1/2 on
a face diagonal and 1/3 on a body diagonal.  Regressing the element radius on s
and evaluating at those three values resolves the mean into directions.  The
regression is area-weighted, so it reduces exactly to R_front when the slope
vanishes.

Usage:  python3 analyze_directional_radius.py [--t 1.0e-3]
"""

import argparse
import glob
import os

import numpy as np
import pyvista as pv

BETA, ALPHA_L, R0 = 4.06022, 1.6753e-7, 50.0e-6
T0 = R0 ** 2 / (4 * BETA ** 2 * ALPHA_L)

STRUCT = [("Structured $75^3$", "Data/Aniso/aniso-075-off", 4.05e-6),
          ("Structured $100^3$", "Data/Aniso/aniso-100-off", 3.03e-6),
          ("Structured $125^3$", "Data/Aniso/aniso-125-off", 2.42e-6),
          ("Structured $150^3$", "Data/Aniso/aniso-150-off", 2.01e-6)]
POLY = [("Polyhedral $75^3$", "Data/campaign/front-75", 4.41e-6),
        ("Polyhedral $100^3$", "Data/campaign/front-100", 3.32e-6),
        ("Polyhedral $125^3$", "Data/campaign/front-125", 2.67e-6),
        ("Polyhedral $150^3$", "Data/campaign/front-150", 2.23e-6)]


def r_scriven(t):
    return 2.0 * BETA * np.sqrt(ALPHA_L * (t + T0))


def directional(path):
    """Area-weighted R_front, and the fit resolved onto axis / face / body diagonal."""
    m = pv.read(path)
    if m.n_cells == 0:
        return None
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    a = np.asarray(sized.cell_data["Area"], float)
    c = np.asarray(sized.cell_centers().points, float)
    ctr = (a[:, None] * c).sum(0) / a.sum()
    q = c - ctr
    r = np.linalg.norm(q, axis=1)
    n = q / r[:, None]
    s = (n ** 4).sum(-1)
    # area-weighted least squares, so the intercept form reduces to R_front
    W = np.sqrt(a)
    A = np.vstack([W, W * s]).T
    c0, c1 = np.linalg.lstsq(A, W * r, rcond=None)[0]
    mean = float((a * r).sum() / a.sum())
    return dict(mean=mean, axis=c0 + c1, face=c0 + c1 / 2, body=c0 + c1 / 3,
                n=len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=1.0e-3)
    ap.add_argument("--dt", type=float, default=2.0e-6)
    a = ap.parse_args()
    ts = int(round(a.t / a.dt))
    Ra = r_scriven(a.t) * 1e6

    print(f"\n  t = {a.t*1e3:.2f} ms (ts{ts:06d}),  R_Scriven = {Ra:.2f} um")
    print(f"  {'mesh':<20s}{'R_front':>9s}{'err':>8s} | {'axis':>8s}{'err':>8s} | "
          f"{'body dg':>8s}{'err':>8s} | {'dg-axis':>8s}")
    print("  " + "-" * 84)
    for group in (STRUCT, POLY):
        for lab, d, h in group:
            f = [x for x in glob.glob(os.path.join(d, f"*front-ts{ts:06d}.pvtu"))]
            if not f:
                f = [x for x in glob.glob(os.path.join(d, "Sub", "*",
                                                       f"*front-ts{ts:06d}.vtu"))]
            if not f:
                print(f"  {lab:<20s}  (no front output at this instant)")
                continue
            r = directional(f[0])
            if r is None:
                print(f"  {lab:<20s}  (empty front)")
                continue
            e = lambda v: 100 * (v * 1e6 / Ra - 1)
            print(f"  {lab:<20s}{r['mean']*1e6:9.2f}{e(r['mean']):+7.2f}% | "
                  f"{r['axis']*1e6:8.2f}{e(r['axis']):+7.2f}% | "
                  f"{r['body']*1e6:8.2f}{e(r['body']):+7.2f}% | "
                  f"{100*(r['body']/r['axis']-1):+7.2f}%")
        print()


if __name__ == "__main__":
    main()
