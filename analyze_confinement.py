#!/usr/bin/env python3
"""
Is the Scriven bubble deformation caused by the domain boundary?

The Scriven domain is a 300 um cube with OUTFLOW on all six sides (the
boundary is *named* "wall" in bubble.dom but its TYPE is outflow -- the run
log reports 0 wall faces and 92256 outflow faces).  Because a cube has
four-fold symmetry about every axis, confinement would imprint the same
m = 4 azimuthal mode that the paper attributes to mesh alignment, so the
two explanations must be separated rather than assumed.

Three independent measurements:

  1. ORIENTATION of the deformation.  For each interface cell, take the
     unit direction n from the bubble centre and form

         s = n_x^4 + n_y^4 + n_z^4

     which is 1 along a face normal (axis), 1/2 along an edge diagonal and
     1/3 along a body diagonal (corner).  Regressing the interface radius
     on s says whether the bubble bulges towards the faces or the corners.
     Both candidate mechanisms happen to predict a bulge towards the faces,
     so this alone does not discriminate -- but it establishes what the
     deformation actually is.

  2. IS THE THERMAL LAYER CLIPPED?  Scriven assumes T -> T_inf far away.
     If the thermal boundary layer reaches the outflow plane, the driving
     temperature field is truncated by the boundary and the growth is
     directly boundary-affected.  Measured as the temperature on the
     boundary-adjacent cells relative to T_inf.

  3. SCALING.  This is the discriminator.  Confinement is set by R/(L/2),
     which at a given physical time is IDENTICAL on every mesh, since all
     meshes discretise the same 300 um cube.  Mesh-induced anisotropy is
     set by R/h, which differs between meshes by construction.  So:
     comparing meshes at MATCHED TIME holds confinement fixed and varies
     only h.  Any difference in deformation is then not confinement.

NOT the source of Table tab:shape_anisotropy.  The dR/ds printed here is an
UNWEIGHTED fit over cells with 0.35 < alpha < 0.65, which is a different
quantity from the paper's shape metric (the area-weighted regression on the
extracted front) and comes out 39 to 86 per cent low against it.  It is fine
for the three questions above, which need only the sign and the scaling.  For
anything that feeds that table use analyze_shape_confinement.py.

Usage:
  python3 analyze_confinement.py <case_dir> [<case_dir> ...] [--t-inf 101.25]
"""

import argparse
import glob
import os

import numpy as np

try:
    import pyvista as pv
except ImportError:
    raise SystemExit("pyvista is required")


def snapshots(case):
    fs = [f for f in glob.glob(os.path.join(case, "*-ts*.pvtu"))
          + glob.glob(os.path.join(case, "*-ts*.vtu"))
          if "bnd" not in os.path.basename(f)
          and "front" not in os.path.basename(f)]
    return sorted(fs, key=lambda f: int(f.split("-ts")[1].split(".")[0]))


def analyse(fn, t_inf):
    mesh = pv.read(fn)
    cd = mesh.cell_data
    alpha = np.asarray(cd["Vof Sharp [1]"], dtype=np.float64)
    vol = np.asarray(cd["Grid Cell Volume [m^3]"], dtype=np.float64)
    temp = np.asarray(cd["Temperature [K]"], dtype=np.float64)
    P = np.asarray(mesh.cell_centers().points, dtype=np.float64)

    b = mesh.bounds
    L = b[1] - b[0]
    half = L / 2.0

    # bubble centre from the vapour distribution, radius from its volume
    w = 1.0 - alpha
    ctr = (P * (w * vol)[:, None]).sum(axis=0) / (w * vol).sum()
    v_vap = (w * vol).sum()
    r_eq = (3.0 * v_vap / (4.0 * np.pi)) ** (1.0 / 3.0)

    # interface band: liquid-side cells adjacent to vapour (CLAUDE.md mask)
    d = P - ctr
    rad = np.linalg.norm(d, axis=1)
    band = (alpha > 0.35) & (alpha < 0.65)

    n = d[band] / rad[band][:, None]
    s = (n ** 4).sum(axis=1)                 # 1 face, 1/2 edge, 1/3 corner
    r = rad[band]

    # least-squares slope of r on s, normalised by the mean radius
    A = np.vstack([np.ones_like(s), s]).T
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    slope = coef[1] / r.mean()

    # binned radii at the three symmetry directions
    def near(target, tol=0.03):
        m = np.abs(s - target) < tol
        return r[m].mean() / r.mean() if m.sum() > 20 else np.nan

    r_face, r_edge, r_corner = near(1.0), near(0.5), near(1.0 / 3.0)

    # is the thermal layer clipped?  temperature on boundary-adjacent cells.
    # The cell size is taken from the mesh itself rather than assumed, so the
    # same test works on every resolution.
    h = float(np.median(vol) ** (1.0 / 3.0))
    edge_cell = (np.abs(np.abs(P) - half).min(axis=1) < 0.75 * h)
    if not edge_cell.any():                      # fall back to a wider shell
        edge_cell = (np.abs(np.abs(P) - half).min(axis=1) < 2.0 * h)
    t_bnd = temp[edge_cell]

    return dict(L=L, h=h, r_eq=r_eq, conf=r_eq / half, n_band=int(band.sum()),
                slope=slope, r_face=r_face, r_edge=r_edge,
                r_corner=r_corner,
                t_bnd_max=t_bnd.max(), t_bnd_mean=t_bnd.mean(),
                dT_bnd=(t_bnd.max() - t_inf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--t-inf", type=float, default=101.25)
    ap.add_argument("--every", type=int, default=200)
    a = ap.parse_args()

    for case in a.cases:
        fs = snapshots(case)
        if not fs:
            print(f"\n  {case}: no snapshots")
            continue
        print(f"\n{'='*84}\n  {os.path.basename(case.rstrip('/'))}"
              f"   ({len(fs)} snapshots)\n{'='*84}")
        print(f"  {'ts':>6s} {'R[um]':>8s} {'R/(L/2)':>8s} {'band':>6s}"
              f" {'h[um]':>6s} {'R/h':>6s}"
              f" {'dR/ds':>9s} {'r_face':>8s} {'r_edge':>8s} {'r_corn':>8s}"
              f" {'T_bnd-Tinf':>11s}")
        print("  " + "-" * 80)
        for f in fs:
            ts = int(f.split("-ts")[1].split(".")[0])
            if ts % a.every:
                continue
            try:
                d = analyse(f, a.t_inf)
            except KeyError as e:
                print(f"  {ts:6d}   missing field {e}")
                continue
            print(f"  {ts:6d} {d['r_eq']*1e6:8.2f} {d['conf']:8.3f}"
                  f" {d['n_band']:6d} {d['h']*1e6:6.2f}"
                  f" {d['r_eq']/d['h']:6.1f} {d['slope']:+9.4f}"
                  f" {d['r_face']:8.4f} {d['r_edge']:8.4f}"
                  f" {d['r_corner']:8.4f} {d['dT_bnd']:+11.2e}")
        print("\n  dR/ds > 0 : bubble bulges towards the FACES (axes)")
        print("  dR/ds < 0 : bubble bulges towards the CORNERS (diagonals)")
        print("  r_face/r_edge/r_corner are radii normalised by the mean")
        print("  T_bnd-Tinf ~ 0 means the thermal layer does NOT reach the "
              "boundary")


if __name__ == "__main__":
    main()
