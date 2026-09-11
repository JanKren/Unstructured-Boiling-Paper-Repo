#!/usr/bin/env python3
"""Is the reconstructed interfacial AREA orientation-dependent?

The source term deposits mass in proportion to the reconstructed interfacial
area of a cell, and the interface then advances by however far that mass
carries it.  The two areas are supposed to cancel:

    a cell gains  dV/dt = mdot" A_recon / rho,
    advancing the plane by d sweeps  A_true d,
    so                  v_n = mdot" / rho   (A cancels)

which is why a uniform flux ought to move every point of a sphere at the same
speed.  The cancellation is exact only if A_recon is the area actually swept.
If the reconstruction over- or under-counts area by a factor that depends on
orientation, the cancellation fails and v_n picks up a cubic modulation -- with
no advection scheme involved at all.

That is a mechanism nothing in this project has tested.  Interface *position*
was checked (0.20% vs 0.23% on an exact sphere, `analyze_extraction_error.py`);
area was not.  It is a plausible candidate because a surface tiled on a
Cartesian mesh over-counts by a factor set by its angle to the axes -- 1 for a
face-normal plane, sqrt(2) for a face diagonal, sqrt(3) for a body diagonal --
which is four-fold and cubic by construction.

Method.  The reconstructed front is a tiling of the interface, so the area it
assigns to a cone of directions can be compared with what an exact sphere of
the same radius would put there: R^2 times the solid angle of the cone.  Binning
on the cubic invariant s = nx^4+ny^4+nz^4 (1 on an axis, 1/2 on a face
diagonal, 1/3 on a body diagonal) and dividing by the Monte-Carlo solid angle of
each bin gives an area density that must be flat if the reconstruction is
isotropic.

A TOTAL area close to the sphere's proves nothing here: a directional excess
and deficit cancel in the sum, which is exactly the case that would leave a
uniform flux anisotropic while looking correct globally.

Usage:  python3 analyze_area_anisotropy.py [--ts 500]
"""

import argparse
import glob
import os

import numpy as np
import pyvista as pv

NBIN = 12


def solid_angle_weights(nbin=NBIN, n_mc=4_000_000, seed=0):
    """Solid angle of each cubic-invariant bin, by Monte Carlo on the sphere."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_mc, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    s = (v ** 4).sum(-1)
    edges = np.linspace(1.0 / 3.0, 1.0, nbin + 1)
    cnt, _ = np.histogram(s, bins=edges)
    return edges, 4.0 * np.pi * cnt / n_mc


def front_area_density(path, edges, omega):
    """Reconstructed area per unit solid angle, per cubic-invariant bin."""
    m = pv.read(path)
    if m.n_cells == 0:
        return None
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    a = np.asarray(sized.cell_data["Area"], float)
    c = np.asarray(sized.cell_centers().points, float)
    ctr = (a[:, None] * c).sum(0) / a.sum()
    q = c - ctr
    r = np.linalg.norm(q, axis=1)
    ok = r > 0
    a, q, r = a[ok], q[ok], r[ok]
    n = q / r[:, None]
    s = (n ** 4).sum(-1)
    R = float((a * r).sum() / a.sum())

    idx = np.clip(np.digitize(s, edges) - 1, 0, len(omega) - 1)
    area_in_bin = np.bincount(idx, weights=a, minlength=len(omega))

    # CONFOUND: the bubble is itself deformed, sitting further out toward the
    # diagonals.  Area subtended per steradian goes as the LOCAL r^2, so
    # normalising by the mean R^2 would report r_body^2/r_axis^2 -- about +17%
    # for the measured 8% radius spread -- as an area error when it is only the
    # shape.  Normalise by the area-weighted <r^2> of each bin instead, so a
    # perfectly reconstructed deformed surface comes out flat and only genuine
    # over/under-tiling survives.
    r2_in_bin = np.bincount(idx, weights=a * r ** 2, minlength=len(omega))
    w_in_bin = np.bincount(idx, weights=a, minlength=len(omega))
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_mean = r2_in_bin / w_in_bin
        dens = area_in_bin / (r2_mean * omega)
        dens_naive = area_in_bin / (R ** 2 * omega)
    return dict(R=R, dens=dens, dens_naive=dens_naive, total=a.sum(), n=len(a),
                ratio_total=a.sum() / (4 * np.pi * R ** 2))


def report(label, path, edges, omega):
    got = front_area_density(path, edges, omega)
    if got is None:
        print(f"  {label:<22} (empty front)")
        return
    d, dn = got["dens"], got["dens_naive"]
    fin = np.isfinite(d) & (d > 0)
    if fin.sum() < 4:
        print(f"  {label:<22} (too few populated bins)")
        return
    axis, body = d[fin][-1], d[fin][0]
    axn, bdn = dn[fin][-1], dn[fin][0]
    mean = d[fin].mean()
    print(f"  {label:<22}{got['R']*1e6:8.2f}{got['ratio_total']:9.4f}"
          f"{100*(bdn/axn-1):+11.2f}%{body:9.4f}{axis:9.4f}"
          f"{100*(body/axis-1):+11.2f}%{100*(d[fin].max()-d[fin].min())/mean:8.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=int, default=500)
    a = ap.parse_args()
    edges, omega = solid_angle_weights()

    cases = [("Structured $75^3$", "Data/Aniso/aniso-075-off"),
             ("Structured $100^3$", "Data/Aniso/aniso-100-off"),
             ("Structured $125^3$", "Data/Aniso/aniso-125-off"),
             ("Structured $150^3$", "Data/Aniso/aniso-150-off"),
             ("Polyhedral $125^3$", "Data/campaign/front-125"),
             ("Polyhedral $150^3$", "Data/campaign/front-150")]

    print(f"\n  Reconstructed interfacial area per unit solid angle, ts{a.ts:06d}")
    print("  Flat means isotropic.  'total' is the global area / exact sphere area:")
    print("  close to 1 there while the columns differ is exactly the hidden case.\n")
    print(f"  {'case':<22}{'R [um]':>8}{'total':>9}{'RAW b-a':>12}"
          f"{'body dg':>9}{'axis':>9}{'CORRECTED':>12}{'spread':>8}")
    print("  " + "-" * 90)
    for lab, d in cases:
        f = glob.glob(os.path.join(d, f"*front-ts{a.ts:06d}.pvtu"))
        if not f:
            print(f"  {lab:<22} (no front output at ts{a.ts:06d})")
            continue
        report(lab, f[0], edges, omega)
    print("\n  RAW b-a normalises by the mean R^2 and therefore still contains the")
    print("  bubble's own deformation (r_body^2/r_axis^2).  CORRECTED divides by the")
    print("  local <r^2> per bin, so a perfectly reconstructed deformed surface reads")
    print("  0.00% and only genuine orientation-dependent over-tiling survives.\n")


if __name__ == "__main__":
    main()
