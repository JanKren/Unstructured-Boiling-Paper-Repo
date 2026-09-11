#!/usr/bin/env python3
"""
Is CICSAM's blending factor mesh-quantised?  (advection-side diagnostic)

Predict_Beta.f90 blends the compressive CBC and the diffusive UQ scheme with

    gamma_f = cos^2(theta),   theta = angle( grad(alpha) , cell connection )

so gamma_f decides how much interface compression each face receives.  The
gradient it uses is `fun % x/y/z`, the gradient of the RAW, sharp volume
fraction.  On a Cartesian mesh a sharp alpha jumps over one or two cells and
that gradient is dominated by whichever axis carries the jump, so its
direction is quantised towards the six axis normals.  Faces aligned with the
grid then receive more compression than faces at 45 degrees, which is a
four-fold modulation of the interface sharpening -- the symmetry of the mode
measured in the paper.

Meanwhile the solver already computes a much better normal in the same step,
`Vof % nx/ny/nz = grad(smooth)/|grad(smooth)|`, from the SMOOTHED volume
fraction (Smooth_Vof_And_Compute_Surface_Normals.f90:60).  It is used for
curvature and for the mass transfer, but not by the advection, and because
Compute_Vof runs before the smoothing (Main_Vof.f90:48 vs :73) the previous
step's value is sitting there unused when Predict_Beta needs it.

This script measures, on the real simulation state and against the exact
radial normal of the fitted sphere:

  * the angular error of the raw-alpha normal and of the smoothed normal;
  * the resulting error in gamma_f itself, Fourier-decomposed in the
    azimuthal angle, which is what actually feeds the scheme.

Usage:  python3 analyze_cicsam_normal.py [case.pvtu] [--nsmooth 2]
"""

import argparse

import numpy as np

from analyze_anisotropy_attribution import (
    DEFAULT_CASE, load_lattice, fit_bubble, fourier_lsq, M_MAX,
)


def grad_central(f, h):
    """Second-order central difference; matches a Gauss gradient on a
    uniform Cartesian lattice, which is what T-Flows reduces to here."""
    g = np.empty(f.shape + (3,))
    for ax in range(3):
        g[..., ax] = np.gradient(f, h, axis=ax)
    return g


def smooth_vof(f, n_iter):
    """Face-area-weighted averaging, iterated -- on a uniform lattice this is
    the 6-neighbour average that Smooth_Vof does."""
    s = f.copy()
    for _ in range(n_iter):
        acc = np.zeros_like(s)
        for ax in range(3):
            acc += np.roll(s, 1, axis=ax) + np.roll(s, -1, axis=ax)
        s = acc / 6.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=DEFAULT_CASE)
    ap.add_argument("--nsmooth", type=int, default=2)
    a = ap.parse_args()

    d = load_lattice(a.case)
    fit = fit_bubble(d)
    ctr = np.asarray(fit["ctr"], float)
    R, h = fit["R"], d["h"]
    X, Y, Z = d["X"], d["Y"], d["Z"]
    dx, dy, dz = X - ctr[0], Y - ctr[1], Z - ctr[2]
    rr = np.sqrt(dx**2 + dy**2 + dz**2)

    print(f"\n{'='*78}\n  CICSAM BLENDING FACTOR: is the normal mesh-quantised?"
          f"\n{'='*78}")
    print(f"  R = {R*1e6:.2f} um, h = {h*1e6:.3f} um, R/h = {R/h:.1f}")

    vof = d["vof"]
    n_ex = np.stack([dx / rr, dy / rr, dz / rr], -1)      # exact radial

    variants = {
        "raw alpha (what CICSAM uses)": grad_central(vof, h),
        f"smoothed alpha, {a.nsmooth} cycles (already computed)":
            grad_central(smooth_vof(vof, a.nsmooth), h),
        "smoothed alpha, 4 cycles":
            grad_central(smooth_vof(vof, 4), h),
    }

    band = (np.abs(rr - R) < 1.5 * h)
    theta = np.arctan2(dy, dx)[band]
    print(f"  interface band: {int(band.sum()):,} cells\n")

    print(f"  {'normal used for gamma_f':<44s} {'ang.err':>8s} "
          f"{'m4 of err':>10s} {'m4 of gamma_f':>14s}")
    print("  " + "-" * 80)

    for label, g in variants.items():
        mag = np.linalg.norm(g, axis=-1)
        ok = band & (mag > 1e-30)
        n = np.zeros_like(g)
        n[ok] = g[ok] / mag[ok][..., None]
        # sign: alpha increases into the liquid, so -grad points outward
        cosang = np.abs((n * n_ex).sum(-1))
        ang = np.degrees(np.arccos(np.clip(cosang[ok], -1, 1)))

        # gamma_f for the x-connection, the quantity Predict_Beta forms
        gam_c = (n[..., 0] ** 2)[ok]          # cos^2 with the x axis
        gam_e = (n_ex[..., 0] ** 2)[ok]
        err = gam_c - gam_e
        th = np.arctan2(dy, dx)[ok]
        amps_e, _ = fourier_lsq(th, err - err.mean() + 1.0, M_MAX)
        amps_g, _ = fourier_lsq(th, gam_c, M_MAX)
        print(f"  {label:<44s} {ang.mean():7.2f}d {100*amps_e[4]:9.2f}% "
              f"{100*amps_g[4]:13.2f}%")

    print("\n  ang.err = mean angle between the computed normal and the exact")
    print("  radial direction. m4 of err = four-fold content of the gamma_f")
    print("  error. gamma_f sets the CBC/UQ blend, so a four-fold modulation")
    print("  means grid-aligned faces are compressed more than diagonal ones.")


if __name__ == "__main__":
    main()
