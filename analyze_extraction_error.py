#!/usr/bin/env python3
"""
Does the interface EXTRACTION manufacture the anisotropy, or the ADVECTION?

The 2x2 substitution of analyze_anisotropy_attribution.py bundles two
different things under "interface = actual (alpha)":

  (a) the ADVECTION error -- CICSAM's alpha field is not the exact volume
      fraction of the true interface, and its error is mesh-aligned;
  (b) the EXTRACTION error -- T-Flows locates the interface by interpolating
      alpha LINEARLY between cell centres to find alpha = 0.5
      (Front_Mod/Mark_Cells_And_Faces.f90:84, whose own comment calls it
      "a ludicrously simple way ... by a linear interpolation :-/").

Those have different remedies.  (a) needs a geometric advection scheme;
(b) is a contained fix -- feed the intersection point from the isoap PLIC
plane that the solver already computes.

This script separates them by removing (a) entirely.  It builds the EXACT
volume fraction field of a sphere on the same Cartesian lattice, by
sub-sampling each near-interface cell, so the alpha field carries no
advection error whatsoever.  It then runs the two extractions on it:

    exact      : intersect the axis-aligned cell-centre segment with the
                 analytical sphere (what case 4 of the 2x2 does)
    linear     : interpolate alpha to 0.5 (what T-Flows does)

and feeds both to the same interface-modified stencil with the same
analytical erf temperature field.  Any anisotropy in the "linear" column
is manufactured by the extraction from a perfect input.

Why it might be: for a locally planar interface cutting a cubic cell, the
map from alpha to signed distance depends on the plane's orientation
relative to the cell axes -- linear when the normal is axis-aligned,
piecewise cubic when it is diagonal.  That orientation dependence has
cubic symmetry, which is the symmetry of the mode being measured.

Usage:  python3 analyze_extraction_error.py [case.pvtu] [--sub 12]
"""

import argparse
import sys

import numpy as np

from analyze_anisotropy_attribution import (
    DEFAULT_CASE, load_lattice, fit_bubble, fourier_lsq, M_MAX,
)

# symmetric 3x3 packing, identical to analyze_anisotropy_attribution
IJ = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 0, 1), (4, 0, 2), (5, 1, 2)]


def exact_sphere_vof(d, ctr, R, sub, band):
    """Exact volume fraction of a sphere, by sub-sampling cells in a band.

    Cells fully inside/outside are set analytically; only the band is
    sub-sampled, which is where alpha is not 0 or 1.
    """
    h = d["h"]
    X, Y, Z = d["X"], d["Y"], d["Z"]
    r = np.sqrt((X - ctr[0]) ** 2 + (Y - ctr[1]) ** 2 + (Z - ctr[2]) ** 2)

    vof = np.where(r < R, 0.0, 1.0)          # liquid = 1 (T-Flows convention)

    sel = np.abs(r - R) < band * h
    n_sel = int(sel.sum())
    print(f"  sub-sampling {n_sel:,} interface cells at {sub}^3 "
          f"= {sub**3} points each")

    # sub-cell offsets, cell-centred
    o = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy, oz = np.meshgrid(o, o, o, indexing="ij")
    off = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1) * h  # (S,3)

    xs, ys, zs = X[sel], Y[sel], Z[sel]
    frac = np.empty(n_sel, dtype=np.float64)
    CH = 20000                                # chunk to bound memory
    for i0 in range(0, n_sel, CH):
        i1 = min(i0 + CH, n_sel)
        px = xs[i0:i1, None] + off[None, :, 0]
        py = ys[i0:i1, None] + off[None, :, 1]
        pz = zs[i0:i1, None] + off[None, :, 2]
        rr = np.sqrt((px - ctr[0]) ** 2 + (py - ctr[1]) ** 2
                     + (pz - ctr[2]) ** 2)
        frac[i0:i1] = (rr >= R).mean(axis=1)   # liquid fraction
    vof[sel] = frac
    return vof


def gradient(d, ctr, R, vof, T, T_sat, mode):
    """Interface-modified LSQ gradient; `mode` selects the extraction."""
    h = d["h"]
    X, Y, Z = d["X"], d["Y"], d["Z"]
    r = np.sqrt((X - ctr[0]) ** 2 + (Y - ctr[1]) ** 2 + (Z - ctr[2]) ** 2)

    G = [np.zeros(d["shape"]) for _ in range(6)]
    G0 = [np.zeros(d["shape"]) for _ in range(6)]   # unmodified, for fallback
    b = [np.zeros(d["shape"]) for _ in range(3)]

    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)

        conn = np.zeros(3)
        conn[axis] = h

        a1, a2 = vof[lo], vof[hi]
        T1, T2 = T[lo], T[hi]

        if mode == "exact":
            s1, s2 = r[lo] - R, r[hi] - R
            crosses = (s1 * s2) < 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(crosses, s1 / (s1 - s2), 0.0)
        elif mode == "linear":
            crosses = ((a1 - 0.5) * (a2 - 0.5)) < 0.0
            den = np.abs(a1 - a2)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(crosses & (den > 1e-15),
                                np.abs(a1 - 0.5)
                                / np.where(den > 1e-15, den, 1.0), 0.0)
        else:
            raise ValueError(mode)
        frac = np.clip(np.nan_to_num(frac), 0.0, 1.0)

        f = frac[..., None]
        r1 = np.where(crosses[..., None], f * conn, conn)
        r2 = np.where(crosses[..., None], (1.0 - f) * conn, conn)
        d1 = np.where(crosses, T_sat - T1, T2 - T1)
        d2 = np.where(crosses, T2 - T_sat, T2 - T1)

        for k, i, j in IJ:
            G[k][lo] += r1[..., i] * r1[..., j]
            G[k][hi] += r2[..., i] * r2[..., j]
            G0[k][lo] += conn[i] * conn[j]
            G0[k][hi] += conn[i] * conn[j]
        for i in range(3):
            b[i][lo] += d1 * r1[..., i]
            b[i][hi] += d2 * r2[..., i]

    # T-Flows' explicit determinant and its fallback to the unmodified matrix
    def jac_of(g):
        return (g[0] * g[1] * g[2] - g[0] * g[5] ** 2 - g[3] ** 2 * g[2]
                + 2.0 * g[3] * g[4] * g[5] - g[4] ** 2 * g[1])
    jac = jac_of(G)
    bad = jac <= 0.0
    for k in range(6):
        G[k] = np.where(bad, G0[k], G[k])
    jac = jac_of(G)
    jac = np.where(np.abs(jac) < 1e-300, 1.0, jac)
    inv = [(G[1]*G[2] - G[5]**2) / jac,
           (G[0]*G[2] - G[4]**2) / jac,
           (G[0]*G[1] - G[3]**2) / jac,
           -(G[3]*G[2] - G[4]*G[5]) / jac,
           (G[3]*G[5] - G[4]*G[1]) / jac,
           -(G[0]*G[5] - G[3]*G[4]) / jac]
    MAP = [[0, 3, 4], [3, 1, 5], [4, 5, 2]]
    return np.stack([sum(inv[MAP[i][j]] * b[j] for j in range(3))
                     for i in range(3)], axis=-1)


def measure(d, ctr, R, grad, mask, g_exact):
    X, Y, Z = d["X"], d["Y"], d["Z"]
    dx, dy, dz = X - ctr[0], Y - ctr[1], Z - ctr[2]
    rr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    n = np.stack([dx / rr, dy / rr, dz / rr], -1)
    g_n = np.abs((grad * n).sum(-1))[mask]
    theta = np.arctan2(dy, dx)[mask]
    amps, a0 = fourier_lsq(theta, g_n, M_MAX)   # returns (dict, mean)
    return amps, a0, a0 / g_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=DEFAULT_CASE)
    ap.add_argument("--sub", type=int, default=12)
    a = ap.parse_args()

    print(f"\n{'='*78}\n  EXTRACTION vs ADVECTION: which one makes the "
          f"anisotropy?\n{'='*78}")
    d = load_lattice(a.case)
    fit = fit_bubble(d)
    ctr = np.asarray(fit["ctr"], dtype=np.float64)
    R, T_sat, dT, dlt = fit["R"], fit["T_sat"], fit["dT"], fit["delta_T"]
    print(f"  R = {R*1e6:.2f} um, h = {d['h']*1e6:.3f} um, "
          f"R/h = {R/d['h']:.1f}, delta_T = {dlt*1e6:.2f} um")

    # exact volume fraction of a sphere -- zero advection error by construction
    vof_ex = exact_sphere_vof(d, ctr, R, a.sub, band=3.0)

    # analytical erf temperature field about the same sphere
    X, Y, Z = d["X"], d["Y"], d["Z"]
    rr = np.sqrt((X-ctr[0])**2 + (Y-ctr[1])**2 + (Z-ctr[2])**2)
    x = rr - R
    from scipy.special import erf
    T_ideal = np.where(x > 0.0, T_sat + dT * erf(x / dlt), T_sat)
    g_exact = dT * 2.0 / (np.sqrt(np.pi) * dlt)

    # liquid-side interface band about the analytical sphere
    mask = (rr > R) & (rr < R + 1.5 * d["h"])
    print(f"  liquid-side band: {int(mask.sum()):,} cells\n")

    print(f"  {'extraction':<34s} {'m2':>6s} {'m4':>7s} {'m6':>6s} "
          f"{'m8':>6s} {'<|gn|>/exact':>13s}")
    print("  " + "-" * 76)
    out = {}
    for mode, label in (("exact",  "exact sphere intersection"),
                        ("linear", "linear alpha=0.5 interpolation")):
        g = gradient(d, ctr, R, vof_ex, T_ideal, T_sat, mode)
        amps, a0, ratio = measure(d, ctr, R, g, mask, g_exact)
        out[mode] = amps[4] * 100
        print(f"  {label:<34s} {100*amps[2]:6.2f} {100*amps[4]:7.2f} "
              f"{100*amps[6]:6.2f} {100*amps[8]:6.2f} {ratio:13.3f}")

    print(f"\n  Both columns use an EXACT sphere volume fraction, so neither")
    print(f"  carries any advection error. The only difference is how the")
    print(f"  interface position is recovered from it.")
    print(f"\n  m4 exact  extraction : {out['exact']:.2f}%")
    print(f"  m4 linear extraction : {out['linear']:.2f}%"
          f"   ({out['linear']/max(out['exact'],1e-9):.1f}x)")
    if out["linear"] > 3.0 * max(out["exact"], 1e-9) and out["linear"] > 1.0:
        print("\n  => the LINEAR EXTRACTION manufactures the anisotropy from a")
        print("     perfect volume-fraction field. This is a contained fix:")
        print("     take the intersection point from the isoap PLIC plane.")
    else:
        print("\n  => the extraction is essentially innocent on a perfect")
        print("     alpha field; the anisotropy must enter through the")
        print("     advected alpha field itself (geometric advection needed).")


if __name__ == "__main__":
    main()
