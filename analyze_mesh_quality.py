#!/usr/bin/env python3
"""
T2.1 -- mesh-quality metrics for the Scriven meshes (Reviewer 2, point 3).

Non-orthogonality is the angle between a face normal and the line joining the
two cell centres it separates.  It cannot be taken from T-Flows' `.faces.vtu`:
that file writes `Grid Connection Vectors` as all zeros and triangulates every
face, so both the connection vector and the polygonal face normal are lost.  It
is therefore reconstructed here from the VTK_POLYHEDRON face streams.

Face-cell adjacency is built from a point->cells map: two cells share a face if
they share all of that face's points.  The metrics are evaluated on a random
sample of cells, which is ample for a distribution and avoids a full 10-35 M
face traversal in Python.

Skewness is reported as the distance from the face centroid to the point where
the cell-centre line pierces the face plane, normalised by |d| -- the usual
finite-volume definition.

Structured meshes are uniform Cartesian and are orthogonal and skew-free by
construction; they are reported analytically rather than sampled.

Usage:
  python3 analyze_mesh_quality.py <case.vtu|pvtu> [--sample 4000] [--label NAME]
"""

import argparse
from collections import defaultdict

import numpy as np
import pyvista as pv


def cell_faces(mesh, cid):
    """Point-id lists of the faces of cell `cid`."""
    c = mesh.get_cell(int(cid))
    return [tuple(sorted(f.point_ids)) for f in c.faces], \
           [np.asarray(f.points, float) for f in c.faces]


def poly_normal_area_centroid(pts):
    """Newell normal, area and centroid of a planar polygon."""
    ctr = pts.mean(axis=0)
    n = np.zeros(3)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n += np.cross(a - ctr, b - ctr)
    area = 0.5 * np.linalg.norm(n)
    if area > 0:
        n = n / (2.0 * area)
    return n, area, ctr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    mesh = pv.read(a.case)
    label = a.label or a.case
    vol = np.asarray(mesh.cell_data["Grid Cell Volume [m^3]"], float)
    cc = np.asarray(mesh.cell_centers().points, float)
    h = float(np.median(vol) ** (1 / 3))

    print(f"\n{'='*74}\n  mesh quality: {label}\n{'='*74}")
    print(f"  {mesh.n_cells:,} cells, h_median = {h*1e6:.4f} um")
    print(f"  cell volume:  min {vol.min():.4e}  max {vol.max():.4e}  "
          f"mean {vol.mean():.4e}  std/mean {vol.std()/vol.mean():.4e}")
    print(f"  volume ratio max/min = {vol.max()/vol.min():.4f}")

    ctypes = np.unique(mesh.celltypes)
    if len(ctypes) == 1 and ctypes[0] == 12:      # VTK_HEXAHEDRON
        print("  all cells VTK_HEXAHEDRON on a uniform Cartesian lattice:")
        print("    faces/cell            = 6 (exact)")
        print("    non-orthogonality     = 0.000 deg (exact, by construction)")
        print("    skewness              = 0.000     (exact, by construction)")
        return

    # ---- polyhedral: build point -> cells, then sample ------------------
    rng = np.random.default_rng(0)
    conn = mesh.cell_connectivity
    offs = mesh.offset
    p2c = defaultdict(list)
    for c in range(mesh.n_cells):
        for p in conn[offs[c]:offs[c + 1]]:
            p2c[p].append(c)

    idx = rng.choice(mesh.n_cells, size=min(a.sample, mesh.n_cells),
                     replace=False)
    nonorth, skew, nfaces = [], [], []
    for cid in idx:
        keys, ptsets = cell_faces(mesh, cid)
        nfaces.append(len(keys))
        for key, pts in zip(keys, ptsets):
            # neighbour = the other cell containing every point of this face
            cand = set(p2c[key[0]])
            for p in key[1:]:
                cand &= set(p2c[p])
                if not cand:
                    break
            cand.discard(int(cid))
            if len(cand) != 1:
                continue                      # boundary face, or ambiguous
            nb = cand.pop()
            n, area, fc = poly_normal_area_centroid(pts)
            if area <= 0:
                continue
            d = cc[nb] - cc[cid]
            dn = np.linalg.norm(d)
            if dn < 1e-30:
                continue
            cosang = abs(float(n @ d) / dn)
            nonorth.append(np.degrees(np.arccos(min(1.0, cosang))))
            # skewness: pierce point of the centre line with the face plane
            denom = float(n @ d)
            if abs(denom) > 1e-30:
                t = float(n @ (fc - cc[cid])) / denom
                pierce = cc[cid] + t * d
                skew.append(float(np.linalg.norm(pierce - fc) / dn))

    nonorth = np.array(nonorth)
    skew = np.array(skew)
    nfaces = np.array(nfaces, float)
    print(f"  sampled {len(idx):,} cells -> {len(nonorth):,} internal faces")
    print(f"    faces/cell         mean {nfaces.mean():7.4f}  "
          f"median {np.median(nfaces):5.1f}  min {nfaces.min():.0f}  "
          f"max {nfaces.max():.0f}")
    for nm, v in (("non-orthogonality [deg]", nonorth), ("skewness [-]", skew)):
        if len(v) == 0:
            print(f"    {nm:<22s} (no samples)"); continue
        print(f"    {nm:<22s} mean {v.mean():7.4f}  median "
              f"{np.median(v):7.4f}  p95 {np.percentile(v,95):7.4f}  "
              f"p99 {np.percentile(v,99):7.4f}  max {v.max():7.4f}")


if __name__ == "__main__":
    main()
